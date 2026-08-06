"""Unit tests for the per-query adaptive β router.

These do NOT touch real datasets; they verify:
  1. Features extract deterministically and live in expected ranges.
  2. The bucketed router falls back to the coarse regime β for empty buckets.
  3. The continuous router's β is always clipped to [0, 1].
  4. Neither router consumes test labels (the target query is fed only via the
     bucketed/ridge fit on validation queries).
"""
import unittest
from collections import Counter

import numpy as np

from perquery_router import (
    BucketedRouter, ContinuousRouter, QueryFeatures, bucket_of,
    extract_features, feature_vector,
)


class FeatureTests(unittest.TestCase):
    def test_feature_vector_shape_and_ranges(self):
        f = QueryFeatures(length=4, log_last_pop=2.5, last_is_tail=1,
                          agreement=2, transition_mass_norm=0.1,
                          session_mass_norm=0.2)
        v = feature_vector(f)
        self.assertEqual(v.shape, (8,))
        self.assertTrue(np.isfinite(v).all())

    def test_agreement_counts_top5_overlap(self):
        # Three components all recommending item 7 in top-5 → agreement 3.
        comps = ([7, 1, 2, 3, 4], [7, 5, 6], [7, 8, 9, 10])
        f = extract_features(context=[7], memory_components=comps,
                             item_freq=Counter({7: 100}), head_items={7})
        self.assertEqual(f.agreement, 3)
        # Head item, so last_is_tail should be 0.
        self.assertEqual(f.last_is_tail, 0)


class BucketedRouterTests(unittest.TestCase):
    def test_empty_bucket_falls_back_to_regime(self):
        # Build features that map into a single bucket, then query a bucket
        # the router has never seen during fit.
        router = BucketedRouter(fallback_beta={"short": 0.0, "long": 0.7})
        keys = ["u1"]
        queries = {"u1": {"context": [1], "targets": [2]}}
        memory = {"u1": [2, 3, 4]}
        neural = {"u1": [9, 8, 7]}
        feats = {"u1": QueryFeatures(length=1, log_last_pop=1.0,
                                     last_is_tail=0, agreement=1,
                                     transition_mass_norm=0.2,
                                     session_mass_norm=0.2)}
        router.fit(queries, memory, neural, keys, feats)
        # Bucket seen during fit (short_head_low) must return a fitted β.
        self.assertIn(router.beta_for(feats["u1"]), set(round(b, 2) for b in __import__("run_cearfn").BETAS))
        # A long_tail_high bucket never seen during fit falls back to long regime.
        unseen = QueryFeatures(length=10, log_last_pop=2.0, last_is_tail=1,
                               agreement=3, transition_mass_norm=0.1,
                               session_mass_norm=0.1)
        self.assertAlmostEqual(router.beta_for(unseen), 0.7)


class ContinuousRouterTests(unittest.TestCase):
    def test_beta_is_clipped_to_unit_interval(self):
        router = ContinuousRouter(alpha=1.0)
        keys = [f"u{i}" for i in range(20)]
        queries = {k: {"context": [1, 2], "targets": [3]} for k in keys}
        memory = {k: [3, 4, 5] for k in keys}
        neural = {k: [10, 11, 12] for k in keys}
        feats = {k: QueryFeatures(length=2, log_last_pop=1.5, last_is_tail=0,
                                  agreement=2, transition_mass_norm=0.2,
                                  session_mass_norm=0.2) for k in keys}
        router.fit(queries, memory, neural, keys, feats)
        # β must be valid for an arbitrary feature vector.
        for length in (0, 1, 5, 20):
            f = QueryFeatures(length=length, log_last_pop=3.0, last_is_tail=1,
                              agreement=3, transition_mass_norm=0.1,
                              session_mass_norm=0.1)
            beta = router.beta_for(f)
            self.assertGreaterEqual(beta, 0.0)
            self.assertLessEqual(beta, 1.0)

    def test_router_does_not_consume_test_targets(self):
        # The continuous router's public API only uses feature vectors at
        # inference; the targets are referenced only inside `fit` on validation.
        router = ContinuousRouter(alpha=1.0)
        keys = [f"u{i}" for i in range(10)]
        queries = {k: {"context": [1], "targets": [2]} for k in keys}
        memory = {k: [2, 3] for k in keys}
        neural = {k: [4, 5] for k in keys}
        feats = {k: QueryFeatures(length=1, log_last_pop=1.0, last_is_tail=0,
                                  agreement=1, transition_mass_norm=0.2,
                                  session_mass_norm=0.2) for k in keys}
        router.fit(queries, memory, neural, keys, feats)
        # Inference only needs features — no target needed.
        beta = router.beta_for(feats["u0"])
        self.assertIsInstance(beta, float)


if __name__ == "__main__":
    unittest.main()
