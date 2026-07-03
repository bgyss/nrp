import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.quality_tiers import quality_metrics  # noqa: E402


class QualityTierReportTests(unittest.TestCase):
    def test_quality_metrics_include_psnr_ssim_flip(self):
        image = np.ones((4, 4, 3))
        metrics = quality_metrics(image, image)
        self.assertEqual(metrics["psnr_vs_final_db"], "inf")
        self.assertAlmostEqual(metrics["ssim_vs_final"], 1.0)
        self.assertEqual(metrics["flip_vs_final"], 0.0)


if __name__ == "__main__":
    unittest.main()
