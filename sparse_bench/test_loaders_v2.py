"""Regression tests for the extended loader set (Phase 1).

These are smoke tests: they verify each loader returns a well-formed dict with
the expected keys and basic invariants, without asserting exact counts (which
would couple the tests to the underlying data files).
"""
import unittest

import loaders


class LoaderShapeTests(unittest.TestCase):
    REQUIRED_KEYS = (
        "domain", "n_items", "train_sessions", "test_queries", "valid_queries",
        "item_freq", "item_categories", "item_texts", "visit_counts",
        "reference_groups",
    )

    def _assert_shape(self, data: dict, expect_valid: bool = True):
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, data, f"missing key {key}")
        self.assertGreater(data["n_items"], 1)
        self.assertGreater(len(data["train_sessions"]), 0)
        self.assertGreater(len(data["test_queries"]), 0)
        if expect_valid:
            self.assertGreater(len(data["valid_queries"]), 0,
                               "valid_queries must be non-empty for router tuning")
        # Every query has a non-empty context and a single target.
        for uid, q in list(data["test_queries"].items())[:50]:
            self.assertIn("context", q)
            self.assertIn("targets", q)
            self.assertIsInstance(q["targets"], list)
            self.assertEqual(len(q["targets"]), 1)
            # Context items must be in [1, n_items).
            for item in q["context"]:
                self.assertTrue(0 < item < data["n_items"],
                                f"out-of-vocab context item {item} in {uid}")
            self.assertTrue(0 < q["targets"][0] < data["n_items"])

    def test_diginetica_hid(self):
        data = loaders.load_diginetica_hid()
        self._assert_shape(data, expect_valid=True)
        # Diginetica HID is the canonical short-context benchmark; assert that
        # a meaningful fraction of test queries actually exercise short contexts,
        # otherwise the router claim is not testable on this dataset.
        short = sum(1 for q in data["test_queries"].values()
                    if len(q["context"]) <= 2)
        self.assertGreater(short, 1000,
                           "Diginetica_HID must have many short-context queries")
        # Product text metadata should be loaded (raw products.csv is on disk).
        self.assertGreater(len(data["item_texts"]), 10000,
                           "Diginetica product-name texts failed to load")

    def test_tmall(self):
        data = loaders.load_tmall()
        self._assert_shape(data, expect_valid=True)
        self.assertGreater(len(data["test_queries"]), 1000)

    def test_amazon_brand_metadata(self):
        # Brand is a strong, low-noise signal we now extract; sanity-check it
        # actually appears in the concatenated text for the Amazon domain that
        # ships meta_*.jsonl with `details.Brand`.
        data = loaders.load_amazon("Video_Games")
        branded = [t for t in data["item_texts"].values() if t.startswith("Brand:")]
        # At least half the catalog should carry a brand string.
        self.assertGreater(len(branded), len(data["item_texts"]) // 2,
                           "brand extraction regressed: too few Brand: prefixes")
        self.assertGreater(len(data["item_texts"]), 0)

    def test_registry_includes_new_loaders(self):
        for name in ("Diginetica_HID", "Tmall", "Video_Games", "Baby_Products"):
            self.assertIn(name, loaders.ALL_LOADERS,
                          f"{name} missing from ALL_LOADERS registry")


if __name__ == "__main__":
    unittest.main()
