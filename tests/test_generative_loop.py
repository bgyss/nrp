import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.generative_loop import fixture_masks, masked_psnr  # noqa: E402


class GenerativeLoopTests(unittest.TestCase):
    def test_fixture_masks_have_objective_and_protect_regions(self):
        objective, protect = fixture_masks(12, 10)
        self.assertEqual(objective.shape, (12, 10))
        self.assertEqual(protect.shape, (12, 10))
        self.assertGreater(float(objective.max()), 1.0)
        self.assertEqual(float(protect[:8].sum()), 0.0)
        self.assertGreater(float(protect[9:].sum()), 0.0)

    def test_masked_psnr_uses_only_selected_pixels(self):
        a = np.zeros((2, 2, 3))
        b = np.zeros((2, 2, 3))
        b[0, 0] = 10.0
        mask = np.array([[False, True], [False, False]])
        self.assertGreater(masked_psnr(a, b, mask), 300.0)


if __name__ == "__main__":
    unittest.main()
