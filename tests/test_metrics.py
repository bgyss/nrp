import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.metrics import flip, flip_error_map, psnr, smape, ssim, tonemap_srgb  # noqa: E402


class SSIMTests(unittest.TestCase):
    def test_identical_images_score_one(self):
        rng = np.random.default_rng(0)
        img = rng.random((32, 32, 3))
        self.assertAlmostEqual(ssim(img, img, data_range=1.0), 1.0, places=12)
        hdr = img * 37.0  # default data_range convention must also hold for HDR
        self.assertAlmostEqual(ssim(hdr, hdr), 1.0, places=12)

    def test_monotone_degradation_under_noise(self):
        rng = np.random.default_rng(1)
        ref = rng.random((48, 48, 3))
        scores = []
        for sigma in (0.02, 0.05, 0.1, 0.2, 0.4):
            noisy = np.clip(ref + rng.normal(0, sigma, ref.shape), 0, 1)
            scores.append(ssim(noisy, ref, data_range=1.0))
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertLess(scores[-1], 0.9)
        self.assertGreater(scores[0], 0.95)

    def test_grayscale_2d_accepted(self):
        rng = np.random.default_rng(2)
        img = rng.random((16, 16))
        self.assertAlmostEqual(ssim(img, img, data_range=1.0), 1.0, places=12)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ssim(np.zeros((4, 4, 3)), np.zeros((5, 5, 3)))


class FLIPTests(unittest.TestCase):
    def test_identical_images_score_zero(self):
        rng = np.random.default_rng(0)
        img = rng.random((32, 32, 3))
        self.assertEqual(flip(img, img), 0.0)

    def test_uniform_fixtures_match_reference_implementation(self):
        # Hand-checked fixtures: on spatially uniform images the CSF prefilter is an
        # identity (unit-sum filter, replicate padding), the feature difference is 0
        # (derivative kernels), so FLIP reduces to the redistributed Hunt-adjusted
        # HyAB color difference — a single closed-form number per color pair. The
        # expected values below were cross-checked against NVIDIA's official
        # `flip-evaluator` package (LDR mode, default 67.02 ppd), which agrees to
        # <1e-4 on these fixtures.
        fixtures = [
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.96738),  # black vs white
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.98666),  # red vs green
            ((0.2, 0.2, 0.2), (0.4, 0.4, 0.4), 0.49994),  # mid-gray step
        ]
        for a, b, expected in fixtures:
            pa = np.full((16, 16, 3), a, dtype=np.float64)
            pb = np.full((16, 16, 3), b, dtype=np.float64)
            err = flip_error_map(pa, pb)
            self.assertAlmostEqual(float(err.mean()), expected, places=4)
            # Uniform inputs must give a uniform error map (border handling included).
            self.assertLess(float(err.max() - err.min()), 1e-12)

    def test_monotone_degradation_under_noise(self):
        rng = np.random.default_rng(3)
        ref = rng.random((48, 48, 3))
        scores = []
        for sigma in (0.02, 0.05, 0.1, 0.2, 0.4):
            noisy = np.clip(ref + rng.normal(0, sigma, ref.shape), 0, 1)
            scores.append(flip(noisy, ref))
        self.assertEqual(scores, sorted(scores))

    def test_range_and_symmetry(self):
        rng = np.random.default_rng(4)
        a, b = rng.random((24, 24, 3)), rng.random((24, 24, 3))
        err = flip_error_map(a, b)
        self.assertTrue(bool((err >= 0).all() and (err <= 1).all()))
        self.assertAlmostEqual(flip(a, b), flip(b, a), places=12)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            flip(np.zeros((4, 4, 3)), np.zeros((4, 4)))


class TonemapTests(unittest.TestCase):
    def test_range_and_monotonicity(self):
        x = np.array([0.0, 0.5, 1.0, 10.0, 1e6])
        y = tonemap_srgb(x)
        self.assertTrue(bool((y >= 0).all() and (y <= 1.0 + 1e-12).all()))
        self.assertTrue(bool((np.diff(y) > 0).all()))
        self.assertEqual(y[0], 0.0)

    def test_display_metrics_compose_with_tonemap(self):
        # The intended usage for HDR radiance: identical HDR images stay identical
        # through the tonemap, so SSIM/FLIP hit their ideal values.
        rng = np.random.default_rng(5)
        hdr = rng.gamma(2.0, 2.0, (16, 16, 3))
        self.assertEqual(flip(tonemap_srgb(hdr), tonemap_srgb(hdr)), 0.0)
        self.assertAlmostEqual(
            ssim(tonemap_srgb(hdr), tonemap_srgb(hdr), data_range=1.0), 1.0, places=12
        )


class ExistingMetricsTests(unittest.TestCase):
    def test_psnr_and_smape_still_behave(self):
        ref = np.full((8, 8, 3), 2.0)
        self.assertEqual(psnr(ref, ref), float("inf"))
        self.assertEqual(smape(ref, ref), 0.0)


if __name__ == "__main__":
    unittest.main()
