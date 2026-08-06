import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from sparse_bench.semantic_teacher import build_semantic_teacher


class SemanticTeacherTest(unittest.TestCase):
    def test_tail_uses_more_semantics_than_head(self):
        behavior = np.zeros((3, 2), dtype=np.float32)
        behavior[1:] = [1.0, 0.0]
        semantic = np.zeros((3, 2), dtype=np.float32)
        semantic[1:] = [0.0, 1.0]
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "semantic.npy"
            np.save(cache, semantic)
            fused = build_semantic_teacher({1: "a", 2: "b"}, 3, behavior,
                                           Counter({1: 1000, 2: 1}), cache)
        self.assertGreater(fused[2, 1], fused[1, 1])


if __name__ == "__main__":
    unittest.main()
