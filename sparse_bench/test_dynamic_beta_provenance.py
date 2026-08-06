from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import cearf
import pasgr
from audit_dynamic_beta_provenance import (
    ProvenanceAuditError,
    _artifact_identity,
    _verify_result_entry,
    audit_memory_cache,
    context_query_fingerprint,
    labeled_query_fingerprint,
    load_frozen_pasgr,
    ranking_view,
    reconstruct_declared_validation,
    reconstruct_training_oof_split,
    regenerate_pasgr_topk_and_compare,
    replay_profile_lock,
    session_fingerprint,
    stable_fraction,
)
from run_cearfn_evidence import build_memory_arrays


class DynamicBetaProvenanceAuditTests(unittest.TestCase):
    def test_result_entry_must_point_to_canonical_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            ranks = root / "ranks.npz"
            manifest.touch()
            ranks.touch()
            results = {
                "Video_Games": {
                    "runs": [{
                        "seed": 42,
                        "manifest": str(manifest),
                        "rank_artifact": str(ranks),
                    }]
                }
            }
            verified = _verify_result_entry(
                results, "Video_Games", 42, manifest, ranks)
            self.assertEqual(
                Path(verified["resolved_manifest"]), manifest.resolve())
            with self.assertRaises(ProvenanceAuditError):
                _verify_result_entry(
                    results,
                    "Video_Games",
                    42,
                    manifest,
                    root / "different.npz",
                )

    def test_declared_validation_is_stable_hash_subset(self):
        queries = {
            f"q{index}": {
                "context": [index + 1],
                "targets": [index + 2],
            }
            for index in range(12)
        }
        reversed_queries = dict(reversed(list(queries.items())))
        expected = sorted(
            queries, key=lambda uid: stable_fraction(uid))[:5]
        first = reconstruct_declared_validation(queries, 5)
        second = reconstruct_declared_validation(reversed_queries, 5)
        self.assertEqual(list(first), expected)
        self.assertEqual(list(second), expected)
        self.assertEqual(first, second)

    def test_oof_split_is_disjoint_and_exactly_reconstructable(self):
        sessions = {
            f"s{index}": [index + 1, index + 2, index + 3]
            for index in range(20)
        }
        validation_sources = {"s2", "s7"}
        inner, profile, gate, report = reconstruct_training_oof_split(
            sessions,
            validation_sources,
            fraction=0.5,
            cap=8,
            profile_cap=2,
        )
        held = {
            uid.split("::", 1)[1] for uid in (*profile, *gate)
        }
        expected_eligible = [
            uid for uid in sessions if uid not in validation_sources]
        expected_held = set(sorted(
            expected_eligible,
            key=lambda uid: stable_fraction(f"dynamic-beta::{uid}"),
        )[:8])
        self.assertEqual(held, expected_held)
        self.assertEqual(set(inner), set(sessions) - expected_held)
        self.assertFalse(set(profile) & set(gate))
        self.assertFalse(held & validation_sources)
        self.assertEqual(report["profile_source_sessions"], 2)
        self.assertEqual(report["gate_source_sessions"], 6)
        self.assertEqual(report["declared_validation_source_overlap"], 0)

    def test_ranking_view_removes_labels_and_context_hash_ignores_them(self):
        first = {
            "q": {"context": [1, 2], "targets": [3]},
        }
        second = {
            "q": {"context": [1, 2], "targets": [99]},
        }
        view = ranking_view(first)
        self.assertEqual(view, {"q": {"context": [1, 2]}})
        self.assertNotIn("targets", view["q"])
        view["q"]["context"].append(7)
        self.assertEqual(first["q"]["context"], [1, 2])
        self.assertEqual(
            context_query_fingerprint(first),
            context_query_fingerprint(second),
        )
        self.assertNotEqual(
            labeled_query_fingerprint(first),
            labeled_query_fingerprint(second),
        )

    def test_memory_audit_checks_every_component_exactly(self):
        sessions = {
            "a": [1, 2, 3, 4],
            "b": [2, 5, 6],
            "c": [1, 5, 7],
        }
        queries = {
            "q1": {"context": [1, 2], "targets": [3]},
            "q2": {"context": [2], "targets": [5]},
        }
        profiles = {
            "short": [0.6, 0.4, 0.0],
            "long": [0.35, 0.60, 0.05],
        }
        index = cearf.CEARFIndex(
            sessions, 9, cearf.CEARFConfig(component_topn=5))
        arrays = build_memory_arrays(
            index, queries, profiles, width=5, label="unit")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.npz"
            np.savez_compressed(
                path,
                **arrays,
                fingerprint=np.asarray(labeled_query_fingerprint(queries)),
                profiles=np.asarray(json.dumps(profiles, sort_keys=True)),
            )
            report = audit_memory_cache(
                path, index, queries, profiles, width=5)
            self.assertTrue(all(report["arrays_exact"].values()))

            arrays["session"] = arrays["session"].copy()
            arrays["session"][1, 0] += 1
            np.savez_compressed(
                path,
                **arrays,
                fingerprint=np.asarray(labeled_query_fingerprint(queries)),
                profiles=np.asarray(json.dumps(profiles, sort_keys=True)),
            )
            with self.assertRaisesRegex(
                    ProvenanceAuditError, "exact mismatch"):
                audit_memory_cache(
                    path, index, queries, profiles, width=5)

    def test_profile_replay_matches_reference_without_label_rank_input(self):
        sessions = {
            "a": [1, 2, 3, 4],
            "b": [2, 5, 6],
            "c": [1, 5, 7],
        }
        queries = {
            "q1": {"context": [1, 2], "targets": [3]},
            "q2": {"context": [2], "targets": [5]},
        }
        index = cearf.CEARFIndex(sessions, 9)
        expected_profiles, expected_report = cearf.tune_profiles(
            index, queries)
        actual_profiles, actual_report = replay_profile_lock(index, queries)
        self.assertEqual(actual_profiles, expected_profiles)
        self.assertEqual(actual_report, expected_report)

    def test_pasgr_replay_is_exact_and_target_invariant(self):
        config = pasgr.PASGRConfig(
            dim=4,
            prototypes=2,
            max_seq=4,
            epochs=1,
            batch_size=2,
            hard_negatives=1,
            seed=7,
            top_k=4,
        )
        embeddings = np.asarray([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 0.0],
        ], dtype=np.float32)
        torch.manual_seed(11)
        model = pasgr.PASGRModel(embeddings, config).cpu().eval()
        queries = {
            "q1": {"context": [1, 2], "targets": [3]},
            "q2": {"context": [3], "targets": [4]},
        }
        keys, rankings = pasgr.predict_pasgr_array(
            model, ranking_view(queries), 6, top_k=4, exclude_seen=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.npz"
            np.savez_compressed(
                path,
                keys=np.asarray(keys),
                rankings=rankings,
                fingerprint=np.asarray(labeled_query_fingerprint(queries)),
            )
            report = regenerate_pasgr_topk_and_compare(
                model,
                queries,
                6,
                path,
                width=4,
                exclude_seen=True,
                batch_size=2,
            )
            self.assertTrue(report["topk_ids_exact"])
            self.assertFalse(report["ranking_used_target_labels"])

            tampered = rankings.copy()
            tampered[0, 0], tampered[0, 1] = (
                tampered[0, 1], tampered[0, 0])
            np.savez_compressed(
                path,
                keys=np.asarray(keys),
                rankings=tampered,
                fingerprint=np.asarray(labeled_query_fingerprint(queries)),
            )
            with self.assertRaisesRegex(
                    ProvenanceAuditError, "exact PASGR"):
                regenerate_pasgr_topk_and_compare(
                    model,
                    queries,
                    6,
                    path,
                    width=4,
                    exclude_seen=True,
                    batch_size=2,
                )

    def test_checkpoint_replay_requires_exact_training_sessions(self):
        config = pasgr.PASGRConfig(
            dim=4, prototypes=2, seed=7, top_k=4)
        model = pasgr.PASGRModel(
            np.zeros((6, 4), dtype=np.float32), config)
        sessions = {"s": [1, 2, 3]}
        manifest = {
            "candidate_width": 4,
            "pasgr_config": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save({
                "config": config.__dict__,
                "state_dict": model.state_dict(),
                "sessions_fingerprint": session_fingerprint(sessions),
                "protocol": "unit-test",
            }, path)
            replayed, identity = load_frozen_pasgr(
                path, sessions, 7, manifest, torch.device("cpu"))
            self.assertEqual(
                replayed.item.num_embeddings, model.item.num_embeddings)
            self.assertEqual(
                identity["training_sessions_sha256"],
                session_fingerprint(sessions),
            )
            with self.assertRaisesRegex(
                    ProvenanceAuditError, "training-session fingerprint"):
                load_frozen_pasgr(
                    path,
                    {"s": [1, 2, 4]},
                    7,
                    manifest,
                    torch.device("cpu"),
                )

    def test_file_and_session_identities_are_content_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(b"alpha")
            first = _artifact_identity(path)
            path.write_bytes(b"beta")
            second = _artifact_identity(path)
            self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertNotEqual(
            session_fingerprint({"a": [1, 2]}),
            session_fingerprint({"a": [1, 3]}),
        )


if __name__ == "__main__":
    unittest.main()
