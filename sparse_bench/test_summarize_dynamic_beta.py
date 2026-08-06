import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from summarize_dynamic_beta import (
    DOMAINS,
    _per_query_value,
    validate_raw_results,
)


class PerQueryMetricTest(unittest.TestCase):
    def test_ndcg_respects_cutoff_and_rank_discount(self):
        ranks = np.asarray([0, 1, 2, 7, 20], dtype=np.uint8)
        values = _per_query_value(ranks, "ndcg@6")
        np.testing.assert_allclose(
            values,
            np.asarray([
                0.0,
                1.0,
                1.0 / np.log2(3.0),
                0.0,
                0.0,
            ]),
        )

    def test_utility_averages_early_and_broad_hits(self):
        ranks = np.asarray([0, 2, 9, 20, 21], dtype=np.uint8)
        np.testing.assert_array_equal(
            _per_query_value(ranks, "utility"),
            np.asarray([0.0, 1.0, 0.5, 0.5, 0.0]),
        )


class ResultIdentityTest(unittest.TestCase):
    def _raw(self, root: Path):
        output = {}
        for domain in DOMAINS:
            manifest = root / f"{domain}.manifest.json"
            ranks = root / f"{domain}.ranks.npz"
            manifest.write_text(json.dumps({
                "protocol": "test-protocol",
                "domain": domain,
                "seed": 42,
            }))
            np.savez_compressed(ranks, placeholder=np.asarray([1]))
            output[domain] = {
                "protocol": "test-protocol",
                "runs": [{
                    "seed": 42,
                    "manifest": str(manifest),
                    "rank_artifact": str(ranks),
                }],
            }
        return output

    def test_accepts_same_protocol_and_seed_set(self):
        with tempfile.TemporaryDirectory() as directory:
            validate_raw_results(self._raw(Path(directory)), None)

    def test_rejects_mixed_seed_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = self._raw(Path(directory))
            raw["Baby_Products"]["runs"][0]["seed"] = 123
            with self.assertRaisesRegex(ValueError, "manifest identity mismatch"):
                validate_raw_results(raw, None)


if __name__ == "__main__":
    unittest.main()
