import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.metrics import psnr  # noqa: E402
from nrp.torch_backend.shot import (  # noqa: E402
    delta_stats,
    flickering_baseline_frame,
    noise_sigma_for_psnr,
    temporal_check,
    temporal_flip_delta,
)


def smooth_sequence(n_frames=6, size=32, seed=0):
    """A smoothly drifting HDR gradient: consecutive frames differ a little,
    the same way a smoothly keyframed light's frames do."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) / size
    frames = []
    for i in range(n_frames):
        shift = 0.02 * i
        img = np.stack([x + shift, y, 0.5 * (x + y)], axis=-1) + 0.1
        frames.append(img)
    return frames


class TemporalMetricTests(unittest.TestCase):
    def test_identical_frames_have_zero_delta(self):
        frame = smooth_sequence(1)[0]
        self.assertEqual(temporal_flip_delta(frame, frame), 0.0)

    def test_delta_stats_empty_and_basic(self):
        self.assertEqual(delta_stats([]), {"count": 0})
        stats = delta_stats([0.1, 0.2, 0.3])
        self.assertEqual(stats["count"], 3)
        self.assertAlmostEqual(stats["mean"], 0.2)
        self.assertAlmostEqual(stats["max"], 0.3)
        self.assertAlmostEqual(stats["p50"], 0.2)

    def test_temporal_check_passes_within_excess(self):
        result = temporal_check([0.10, 0.11], [0.10, 0.10], excess_max=0.02)
        self.assertTrue(result["passed"])
        self.assertEqual(result["excess_max_allowed"], 0.02)
        self.assertIn("pass", result["verdict"])

    def test_temporal_check_fails_beyond_excess(self):
        result = temporal_check([0.10, 0.20], [0.10, 0.10], excess_max=0.02)
        self.assertFalse(result["passed"])
        self.assertIn("fail", result["verdict"])

    def test_temporal_check_requires_aligned_sequences(self):
        with self.assertRaises(ValueError):
            temporal_check([0.1], [0.1, 0.2], excess_max=0.02)

    def test_noise_sigma_hits_target_psnr(self):
        rng = np.random.default_rng(0)
        reference = smooth_sequence(1, size=128)[0]
        sigma = noise_sigma_for_psnr(reference, 30.0)
        noised = reference + rng.normal(0.0, sigma, size=reference.shape)
        # statistical: large frame, expect PSNR within 0.5 dB of the target
        self.assertAlmostEqual(psnr(noised, reference), 30.0, delta=0.5)

    def test_flickering_baseline_fails_check_smooth_sequence_passes(self):
        frames = smooth_sequence()
        reference_deltas = [
            temporal_flip_delta(a, b) for a, b in zip(frames, frames[1:], strict=False)
        ]
        # the reference sequence trivially passes against itself
        self.assertTrue(
            temporal_check(reference_deltas, reference_deltas, excess_max=0.02)["passed"]
        )
        rng = np.random.default_rng(1)
        noised = [flickering_baseline_frame(f, rng, psnr_db=15.0) for f in frames]
        noised_deltas = [
            temporal_flip_delta(a, b) for a, b in zip(noised, noised[1:], strict=False)
        ]
        self.assertFalse(temporal_check(noised_deltas, reference_deltas, excess_max=0.02)["passed"])


if __name__ == "__main__":
    unittest.main()
