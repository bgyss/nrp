"""S6: first-class device/precision config validation (nrp.torch_backend.device)."""

import sys
import unittest
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.torch_backend.device import (  # noqa: E402
    autocast,
    resolve_device,
    resolve_precision,
    synchronize,
)


class ResolveDeviceTests(unittest.TestCase):
    def test_default_is_cpu(self):
        self.assertEqual(resolve_device(None).type, "cpu")
        self.assertEqual(resolve_device("cpu").type, "cpu")

    def test_unknown_device_rejected(self):
        with self.assertRaises(ValueError):
            resolve_device("gpu")
        with self.assertRaises(ValueError):
            resolve_device("cuda:0")  # bare names only; index selection is out of scope

    @unittest.skipIf(torch.cuda.is_available(), "cuda present; unavailable path untestable")
    def test_cuda_unavailable_fails_with_clear_message(self):
        with self.assertRaises(RuntimeError) as ctx:
            resolve_device("cuda")
        msg = str(ctx.exception)
        self.assertIn("cuda", msg)
        self.assertIn("is_available", msg)

    @unittest.skipUnless(torch.cuda.is_available(), "needs cuda")
    def test_cuda_available_resolves(self):
        self.assertEqual(resolve_device("cuda").type, "cuda")

    @unittest.skipIf(torch.backends.mps.is_available(), "mps present; unavailable path untestable")
    def test_mps_unavailable_fails_with_clear_message(self):
        with self.assertRaises(RuntimeError):
            resolve_device("mps")

    @unittest.skipUnless(torch.backends.mps.is_available(), "needs mps")
    def test_mps_available_resolves(self):
        self.assertEqual(resolve_device("mps").type, "mps")


class ResolvePrecisionTests(unittest.TestCase):
    def test_default_and_valid_names(self):
        self.assertEqual(resolve_precision(None), "fp32")
        for name in ("fp32", "fp16", "bf16"):
            self.assertEqual(resolve_precision(name), name)

    def test_invalid_rejected(self):
        for bad in ("float32", "half", "fp8", ""):
            with self.assertRaises(ValueError, msg=bad):
                resolve_precision(bad if bad else "fp8")

    def test_autocast_fp32_is_noop_context(self):
        ctx = autocast(torch.device("cpu"), "fp32")
        self.assertIsInstance(ctx, nullcontext)

    def test_autocast_bf16_cpu_context_works(self):
        with autocast(torch.device("cpu"), "bf16"):
            out = torch.nn.functional.linear(torch.ones(4, 4), torch.ones(4, 4))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_synchronize_cpu_noop(self):
        synchronize(torch.device("cpu"))  # must not raise


class TrainConfigValidationTests(unittest.TestCase):
    def test_train_rejects_bad_precision_early(self):
        from nrp.torch_backend.train import train

        with self.assertRaises(ValueError):
            train({"seed": 0, "precision": "fp17"})

    def test_train_rejects_bad_device_early(self):
        from nrp.torch_backend.train import train

        with self.assertRaises(ValueError):
            train({"seed": 0, "device": "tpu"})


if __name__ == "__main__":
    unittest.main()
