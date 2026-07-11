"""F2 summit demo: the F1 shot rendered at final tier with residual-identity
frames, encoded as a committed MP4 plus a per-frame report.

Final tier on the single-cache T1 kitchen = denoised cached GATHERLIGHT (the
supervision-class reference — same mapping as F1/G2). Each frame stores what a
production pipeline would store: one shared proxy (model.pt per shot) plus a
per-frame residual against the final-tier reference, quantized fp16 in a
compressed .npz (the cache's own Sec. 4.2 precision convention). Checks per
frame: exact float64 residual identity (proxy + residual == final by
construction, recorded), the fp16-stored reconstruction within a stated
tolerance, and the T3 *final*-tier gate on the stored reconstruction vs the
final reference — a failing frame is flagged with its cause, never dropped.

Storage compares like-for-like containers: (model + sum of residual .npz) vs
the same final frames as fp16 compressed .npz. Wall-clock compares the
measured cache-reuse shot against re-path-tracing every frame, estimated from
the committed T2 export report's measured wall_seconds (a no-cache pipeline
re-runs SAMPLEPATHS per frame); the estimate's provenance is recorded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.metrics import tonemap_srgb  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.quality.gate import evaluate_gate  # noqa: E402
from nrp.torch_backend.animate import frame_times, lights_at  # noqa: E402
from nrp.torch_backend.denoise import denoise_image, oidn_available  # noqa: E402
from nrp.torch_backend.gather import TorchPathCache  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.relight import relight  # noqa: E402
from nrp.torch_backend.shot import delta_stats, light_specs_at  # noqa: E402

GATE_TIER = "final"


def store_residual(path: Path, residual: np.ndarray) -> int:
    """fp16-quantize and store one residual frame; returns the file size in bytes."""
    np.savez_compressed(path, residual=residual.astype(np.float16))
    return path.stat().st_size


def encode_mp4(rgb_path: Path, width: int, height: int, fps: int, out_path: Path) -> None:
    """Encode an appended raw-RGB24 frame stream to H.264 MP4 via ffmpeg."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            str(rgb_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )


def render_final_shot(
    model: TorchNRP,
    cache: PathCache,
    spec: dict,
    out_dir: Path,
    denoise_method: str = "bilateral",
    device: str = "cpu",
    fps: int = 24,
    identity_rtol: float = 1e-3,
    export_report: str | None = None,
    model_path: str | None = None,
    encode: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    residual_dir = out_dir / "residuals"
    raw_dir = out_dir / "raw_frames"
    residual_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    rgb_path = out_dir / "frames.rgb"

    times = frame_times(int(spec["frames"]))
    torch_cache = TorchPathCache(cache, torch.device(device))
    rows: list[dict] = []
    frame_seconds: list[float] = []
    residual_bytes_total = 0
    raw_bytes_total = 0
    shot_t0 = time.perf_counter()

    with open(rgb_path, "wb") as rgb_stream:
        for idx, t in enumerate(times):
            t_f = float(t)
            f0 = time.perf_counter()
            lights = lights_at(spec, t_f)

            proxy = relight(model, cache, lights)
            gather_t = torch_cache.gather_light(lights[0])
            for light in lights[1:]:
                gather_t = gather_t + torch_cache.gather_light(light)
            draft = gather_t.cpu().numpy().astype(np.float64)
            final_ref = (
                draft
                if denoise_method == "none"
                else denoise_image(
                    draft, cache.albedo, cache.normal, cache.depth, method=denoise_method
                )
            )

            residual = final_ref - proxy
            exact_identity = float(np.max(np.abs((proxy + residual) - final_ref)))
            residual_path = residual_dir / f"frame_{idx:04d}.npz"
            residual_bytes = store_residual(residual_path, residual)
            residual_bytes_total += residual_bytes
            stored = np.load(residual_path)["residual"].astype(np.float64)
            reconstruction = proxy + stored
            stored_identity = float(np.max(np.abs(reconstruction - final_ref)))
            # fp16 quantization error is relative to the residual's own magnitude
            identity_tolerance = identity_rtol * max(float(np.max(np.abs(residual))), 1e-12)
            identity_ok = stored_identity <= identity_tolerance

            np.savez_compressed(
                raw_dir / f"frame_{idx:04d}.npz", frame=final_ref.astype(np.float16)
            )
            raw_bytes = (raw_dir / f"frame_{idx:04d}.npz").stat().st_size
            raw_bytes_total += raw_bytes

            gate = evaluate_gate(reconstruction, final_ref, GATE_TIER)
            flag = None
            if not np.isfinite(reconstruction).all():
                flag = "non-finite pixels in reconstruction"
            elif not identity_ok:
                flag = (
                    f"fp16-stored residual identity {stored_identity:.3e} exceeds "
                    f"tolerance {identity_tolerance:.3e}"
                )
            elif not gate["passed"]:
                flag = f"final-tier gate: {gate['verdict']} (fp16 residual quantization)"

            ldr = np.clip(tonemap_srgb(reconstruction), 0.0, 1.0)
            rgb_stream.write((ldr * 255.0 + 0.5).astype(np.uint8).tobytes())

            frame_seconds.append(time.perf_counter() - f0)
            rows.append(
                {
                    "index": idx,
                    "t": t_f,
                    "lights": light_specs_at(spec, t_f),
                    "seconds": frame_seconds[-1],
                    "exact_identity_max_abs": exact_identity,
                    "stored_identity_max_abs": stored_identity,
                    "stored_identity_tolerance": identity_tolerance,
                    "stored_identity_within_tolerance": identity_ok,
                    "quality_gate": gate,
                    "flag": flag,
                    "residual_bytes": residual_bytes,
                    "raw_frame_bytes": raw_bytes,
                }
            )

    shot_seconds = time.perf_counter() - shot_t0

    mp4_bytes = None
    mp4_note = None
    if encode:
        if shutil.which("ffmpeg") is None:
            mp4_note = "ffmpeg not on PATH; frames.rgb kept for later encoding"
        else:
            encode_mp4(rgb_path, cache.width, cache.height, fps, out_dir / "shot.mp4")
            mp4_bytes = (out_dir / "shot.mp4").stat().st_size

    rerender = None
    if export_report:
        exported = json.loads(Path(export_report).read_text())
        per_frame = float(exported["wall_seconds"])
        rerender = {
            "source": export_report,
            "export_wall_seconds_per_frame": per_frame,
            "estimated_total_seconds": per_frame * len(times),
            "amortization_ratio": (per_frame * len(times)) / shot_seconds,
            "note": (
                "estimate from the committed T2 export's measured wall-clock: a "
                "no-cache pipeline re-runs SAMPLEPATHS (full path trace + cache "
                "write) every frame; the shot instead reuses one cache"
            ),
        }

    if model_path:
        model_bytes = Path(model_path).stat().st_size
        model_bytes_source = "model .pt on disk"
    else:
        model_bytes = sum(
            t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers())
        )
        model_bytes_source = "in-memory parameter/buffer bytes (no .pt supplied)"
    flagged = [row["index"] for row in rows if row["flag"]]
    report = {
        "rung": "F2",
        "scope": (
            "F1 shot at final tier with fp16-stored residual-identity frames, "
            "per-frame final-tier gate, MP4 encode, and storage/wall-clock accounting"
        ),
        "frames": len(times),
        "resolution": [cache.width, cache.height],
        "fps": fps,
        "denoise": denoise_method,
        "final_tier_definition": (
            "raw cached GATHERLIGHT"
            if denoise_method == "none"
            else f"{denoise_method}-denoised cached GATHERLIGHT "
            "(single-cache scene; supervision-class reference, Sec. 4.4)"
        ),
        "gate_tier": GATE_TIER,
        "all_frames_pass_final_gate": not flagged,
        "flagged_frames": flagged,
        "residual_storage_precision": "fp16 (np.savez_compressed), Sec. 4.2 convention",
        "per_frame": rows,
        "storage": {
            "container": "fp16 np.savez_compressed on both sides (like-for-like)",
            "model_bytes": model_bytes,
            "model_bytes_source": model_bytes_source,
            "residual_bytes_total": residual_bytes_total,
            "proxy_plus_residuals_bytes": model_bytes + residual_bytes_total,
            "raw_frames_bytes_total": raw_bytes_total,
            "proxy_plus_residuals_over_raw": (
                (model_bytes + residual_bytes_total) / raw_bytes_total if raw_bytes_total else None
            ),
            "mp4_bytes": mp4_bytes,
            "mp4_note": mp4_note,
        },
        "wall_clock": {
            "shot_total_seconds": shot_seconds,
            "per_frame_seconds": delta_stats(frame_seconds),
            "rerender_estimate": rerender,
        },
        "outputs": {
            "mp4": "shot.mp4" if mp4_bytes else None,
            "residuals_dir": "residuals/",
            "raw_frames_dir": "raw_frames/",
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--keyframes", required=True, help="shot keyframe JSON (E1 format)")
    parser.add_argument("--out-dir", default="out/f2-shot")
    parser.add_argument(
        "--denoise",
        default="oidn",
        choices=["oidn", "bilateral", "none"],
        help="final-tier denoiser (oidn needs the nix devshell for libtbb)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--identity-rtol", type=float, default=1e-3)
    parser.add_argument(
        "--export-report",
        default="out/t2-streaming/export_512x512_64spp.json",
        help="committed export report supplying the per-frame re-render estimate",
    )
    args = parser.parse_args()

    if args.denoise == "oidn" and not oidn_available():
        raise SystemExit(
            "oidn unavailable — run under `nix develop --command` (libtbb), or pass "
            "--denoise bilateral/none (the final tier then differs from the proxy's "
            "supervision target)"
        )

    model = TorchNRP.load(args.model)
    cache = PathCache.load(args.cache)
    with open(args.keyframes) as f:
        spec = json.load(f)
    report = render_final_shot(
        model,
        cache,
        spec,
        Path(args.out_dir),
        denoise_method=args.denoise,
        device=args.device,
        fps=args.fps,
        identity_rtol=args.identity_rtol,
        export_report=args.export_report if Path(args.export_report).exists() else None,
        model_path=args.model,
    )
    ok = report["all_frames_pass_final_gate"]
    print(
        f"{'PASS' if ok else 'FAIL'}: {report['frames'] - len(report['flagged_frames'])}"
        f"/{report['frames']} frames pass final tier; flagged {report['flagged_frames']} — "
        f"wrote {Path(args.out_dir) / 'report.json'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
