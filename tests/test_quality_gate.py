import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_lights  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.quality.gate import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    TierThresholds,
    evaluate_gate,
    gate_metrics,
    load_thresholds,
    re_emit_report,
)
from nrp.toy_tracer import trace_path_cache  # noqa: E402


class GateMetricsTests(unittest.TestCase):
    def test_pass_when_all_metrics_clear_thresholds(self):
        result = gate_metrics({"psnr_db": 45.0, "ssim": 0.995, "flip": 0.01}, "final")
        self.assertTrue(result["passed"])
        self.assertEqual(result["verdict"], "pass at final tier")
        self.assertEqual(result["metrics_evaluated"], 3)
        for m in ("psnr_db", "ssim", "flip"):
            self.assertEqual(result["metrics"][m]["verdict"], "pass")

    def test_single_failing_metric_fails_gate(self):
        result = gate_metrics({"psnr_db": 45.0, "ssim": 0.995, "flip": 0.05}, "final")
        self.assertFalse(result["passed"])
        self.assertEqual(result["metrics"]["flip"]["verdict"], "fail")
        self.assertIn("flip", result["verdict"])

    def test_thresholds_are_inclusive(self):
        t = DEFAULT_THRESHOLDS["draft"]
        result = gate_metrics(
            {"psnr_db": t.psnr_db_min, "ssim": t.ssim_min, "flip": t.flip_max}, "draft"
        )
        self.assertTrue(result["passed"])

    def test_tier_ordering_preview_looser_than_final(self):
        metrics = {"psnr_db": 25.0, "ssim": 0.85, "flip": 0.10}
        self.assertTrue(gate_metrics(metrics, "preview")["passed"])
        self.assertFalse(gate_metrics(metrics, "draft")["passed"])
        self.assertFalse(gate_metrics(metrics, "final")["passed"])

    def test_inf_psnr_string_passes_every_tier(self):
        # JSON reports store infinite PSNR as the string "inf" (finite_or_inf).
        for tier in ("preview", "draft", "final"):
            result = gate_metrics({"psnr_db": "inf", "ssim": 1.0, "flip": 0.0}, tier)
            self.assertTrue(result["passed"])

    def test_missing_metrics_are_skipped_not_failed(self):
        result = gate_metrics({"psnr_db": 45.0}, "final")
        self.assertTrue(result["passed"])
        self.assertEqual(result["metrics_evaluated"], 1)
        self.assertEqual(result["metrics"]["ssim"]["verdict"], "skipped")

    def test_no_metrics_at_all_does_not_pass(self):
        result = gate_metrics({}, "final")
        self.assertFalse(result["passed"])
        self.assertEqual(result["verdict"], "no metrics evaluated")

    def test_unknown_tier_raises(self):
        with self.assertRaises(ValueError):
            gate_metrics({"psnr_db": 45.0}, "cinematic")

    def test_threshold_override_table(self):
        table = dict(DEFAULT_THRESHOLDS)
        table["draft"] = TierThresholds(psnr_db_min=5.0, ssim_min=0.1, flip_max=0.9)
        metrics = {"psnr_db": 10.0, "ssim": 0.5, "flip": 0.5}
        self.assertFalse(gate_metrics(metrics, "draft")["passed"])
        self.assertTrue(gate_metrics(metrics, "draft", table)["passed"])

    def test_load_thresholds_partial_file_keeps_other_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thresholds.json"
            path.write_text(
                json.dumps({"draft": {"psnr_db_min": 1.0, "ssim_min": 0.0, "flip_max": 1.0}})
            )
            table = load_thresholds(str(path))
        self.assertEqual(table["draft"].psnr_db_min, 1.0)
        self.assertEqual(table["final"], DEFAULT_THRESHOLDS["final"])


class EvaluateGateTests(unittest.TestCase):
    """The T3 'deliberately degraded render' criterion, on a real toy render."""

    @classmethod
    def setUpClass(cls):
        cache = trace_path_cache(24, 24, 16, 2, seed=3)
        light = SphereLight(center=[0.0, 0.6, 0.0], radius=0.25, rgb=[1.2, 1.0, 0.8])
        cls.reference = gather_lights(cache, [light])

    def test_identical_render_passes_final_tier(self):
        result = evaluate_gate(self.reference, self.reference, "final")
        self.assertTrue(result["passed"])
        self.assertGreater(result["evaluation_seconds"], 0.0)

    def test_deliberately_degraded_render_fails_gate(self):
        rng = np.random.default_rng(7)
        scale = max(float(self.reference.max()), 1.0)
        degraded = np.clip(
            self.reference + rng.normal(0.0, 0.5 * scale, self.reference.shape), 0.0, None
        )
        result = evaluate_gate(degraded, self.reference, "preview")
        self.assertFalse(result["passed"])
        # and a fortiori at the stricter tiers
        self.assertFalse(evaluate_gate(degraded, self.reference, "final")["passed"])

    def test_mild_degradation_passes_preview_but_fails_final(self):
        rng = np.random.default_rng(8)
        scale = max(float(self.reference.max()), 1.0)
        mild = np.clip(
            self.reference + rng.normal(0.0, 0.005 * scale, self.reference.shape), 0.0, None
        )
        self.assertTrue(evaluate_gate(mild, self.reference, "preview")["passed"])
        self.assertFalse(evaluate_gate(mild, self.reference, "final")["passed"])


class ReEmitReportTests(unittest.TestCase):
    def test_attaches_gates_without_touching_existing_content(self):
        report = {
            "tiers": {
                "draft": {"psnr_db": 35.0, "ssim": 0.95, "flip": 0.05, "ms": 1.0},
                "final": {"psnr_db": "inf", "ssim": 1.0, "flip": 0.0, "ms": 2.0},
            },
            "notes": "untouched",
        }
        original = json.loads(json.dumps(report))
        gated, attached = re_emit_report(report, "draft")
        self.assertEqual(report, original)  # input not mutated
        self.assertEqual(len(attached), 2)
        self.assertTrue(all(g["passed"] for g in attached))
        # existing keys unchanged in the copy, gate attached as a sibling
        for tier in ("draft", "final"):
            for key, value in original["tiers"][tier].items():
                self.assertEqual(gated["tiers"][tier][key], value)
            self.assertIn("quality_gate", gated["tiers"][tier])
        self.assertEqual(gated["notes"], "untouched")

    def test_custom_metric_keys_and_lists(self):
        report = {"cells": [{"psnr_db_mean": 12.0, "ssim_mean": 0.4, "flip_mean": 0.4}]}
        gated, attached = re_emit_report(
            report,
            "preview",
            psnr_key="psnr_db_mean",
            ssim_key="ssim_mean",
            flip_key="flip_mean",
        )
        self.assertEqual(len(attached), 1)
        self.assertFalse(attached[0]["passed"])
        self.assertEqual(attached[0]["path"], "cells[0]")
        self.assertIn("quality_gate", gated["cells"][0])

    def test_dicts_without_metrics_are_left_alone(self):
        gated, attached = re_emit_report({"config": {"iters": 100}}, "draft")
        self.assertEqual(attached, [])
        self.assertNotIn("quality_gate", gated["config"])


class GateCLITests(unittest.TestCase):
    def _run(self, *argv):
        return subprocess.run(
            [sys.executable, "-m", "nrp.quality.gate", *argv],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )

    def test_images_mode_exit_codes(self):
        rng = np.random.default_rng(11)
        ref = rng.random((16, 16, 3))
        with tempfile.TemporaryDirectory() as tmp:
            ref_path, bad_path = Path(tmp) / "ref.npy", Path(tmp) / "bad.npy"
            np.save(ref_path, ref)
            np.save(bad_path, np.clip(ref + rng.normal(0, 0.5, ref.shape), 0, None))
            ok = self._run("images", str(ref_path), str(ref_path), "--tier", "final")
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertTrue(json.loads(ok.stdout)["passed"])
            bad = self._run("images", str(bad_path), str(ref_path), "--tier", "final")
            self.assertEqual(bad.returncode, 1, bad.stderr)

    def test_report_mode_re_emits(self):
        report = {"tiers": {"draft": {"psnr_db": 35.0, "ssim": 0.95, "flip": 0.05}}}
        with tempfile.TemporaryDirectory() as tmp:
            in_path, out_path = Path(tmp) / "report.json", Path(tmp) / "gated.json"
            in_path.write_text(json.dumps(report))
            proc = self._run("report", str(in_path), "--tier", "draft", "--out", str(out_path))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            gated = json.loads(out_path.read_text())
            self.assertTrue(gated["tiers"]["draft"]["quality_gate"]["passed"])

    def test_report_mode_fails_when_no_metrics_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "report.json"
            in_path.write_text(json.dumps({"config": {"iters": 3}}))
            proc = self._run("report", str(in_path), "--tier", "draft")
            self.assertEqual(proc.returncode, 1)


if __name__ == "__main__":
    unittest.main()
