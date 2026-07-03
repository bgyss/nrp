import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.quality_tiers import quality_metrics, supervisor_trust_verdict  # noqa: E402


class QualityTierReportTests(unittest.TestCase):
    def test_quality_metrics_include_psnr_ssim_flip(self):
        image = np.ones((4, 4, 3))
        metrics = quality_metrics(image, image)
        self.assertEqual(metrics["psnr_vs_final_db"], "inf")
        self.assertAlmostEqual(metrics["ssim_vs_final"], 1.0)
        self.assertEqual(metrics["flip_vs_final"], 0.0)

    def test_supervisor_trust_verdict_limits_radius_after_first_failure(self):
        verdict = supervisor_trust_verdict(
            1e-16,
            [
                {"center_dx": 0.0, "psnr_db_vs_cached_gather": "inf"},
                {"center_dx": 0.05, "psnr_db_vs_cached_gather": 24.0},
            ],
            psnr_threshold_db=25.0,
        )
        self.assertTrue(verdict["approved_config_exact"])
        self.assertEqual(verdict["trusted_center_dx_radius"], 0.0)
        self.assertEqual(verdict["first_untrusted_sample"]["center_dx"], 0.05)
        self.assertEqual(
            verdict["verdict"],
            "trust approved frame only; re-bake residual after any measured light move",
        )

    def test_supervisor_trust_verdict_rejects_failed_identity(self):
        verdict = supervisor_trust_verdict(
            1e-3,
            [{"center_dx": 0.0, "psnr_db_vs_cached_gather": "inf"}],
        )
        self.assertFalse(verdict["approved_config_exact"])
        self.assertIsNone(verdict["trusted_center_dx_radius"])
        self.assertEqual(verdict["verdict"], "do not trust approval frame")


if __name__ == "__main__":
    unittest.main()
