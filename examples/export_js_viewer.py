"""E6's WebGPU/engine-backend gap: export a TorchNRP as raw weights + aux tensors so a
non-PyTorch backend (here: plain JavaScript, standing in for a WebGPU/engine port) can
run inference and drive a real interactive GUI slider.

This is deliberately the smallest honest step toward "structure the exporter so a
WebGPU/engine port is a backend, not a rewrite": the model is trained with
`use_encoding=False` (raw pixel xy, no hashgrid) so the forward pass is just
Linear -> ReLU -> ... -> Linear -> softplus, which a ~40-line JS function can
replicate exactly from the same weight matrices. It writes:
  - `model_weights.json`: layer weights/biases, light type, aux/xy tensors for the
    fixed toy cache the model was trained on.
  - `reference.json`: the Python forward pass's own output image for a default light,
    used as the parity check the HTML page runs against its own JS forward pass at
    load time (the E6 "matches the PyTorch module" criterion, now for a JS backend
    instead of the TorchScript one).
  - `viewer.html`: a self-contained page with light-position/radius sliders; every
    change re-runs the JS forward pass over all pixels and redraws a canvas. This is
    a real GUI (renders in a browser), unlike the headless frame-dump loop in
    `examples/engine_runtime.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.relight import relight  # noqa: E402
from nrp.torch_backend.sampling import sample_light  # noqa: E402
from nrp.torch_backend.train import light_param_vector, pixel_tensors  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402


def train_toy_proxy(cache, iters: int = 600, lr: float = 5e-3, seed: int = 0) -> TorchNRP:
    torch.manual_seed(seed)
    model = TorchNRP(
        light_type="sphere",
        hidden_width=32,
        hidden_layers=2,
        use_encoding=False,
    )
    xy, aux = pixel_tensors(cache, "cpu")
    n_px = xy.shape[0]
    rng = np.random.default_rng(seed)
    bounds = {"radius_min": 0.08, "radius_max": 0.25}
    n_pool = 16
    pool_params, pool_targets = [], []
    for _ in range(n_pool):
        light = sample_light(cache, rng, "sphere", bounds, "segments")
        pool_params.append(light_param_vector(light))
        pool_targets.append(gather_light(cache, light).reshape(-1, 3))
    params_t = torch.as_tensor(np.stack(pool_params), dtype=torch.float32)
    targets_t = torch.as_tensor(np.stack(pool_targets), dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    batch = min(512, n_px)
    for _ in range(iters):
        k = int(torch.randint(0, n_pool, (1,), generator=gen).item())
        pixel_ids = torch.randint(0, n_px, (batch,), generator=gen)
        pred = model(xy[pixel_ids], aux[pixel_ids], params_t[k].expand(batch, -1))
        loss = torch.mean((pred - targets_t[k, pixel_ids]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def extract_weights(model: TorchNRP) -> list[dict]:
    """MLP layer weights in the order the forward pass applies them."""
    layers = []
    for module in model.mlp:
        if isinstance(module, torch.nn.Linear):
            layers.append(
                {
                    "weight": module.weight.detach().numpy().astype(np.float64).tolist(),
                    "bias": module.bias.detach().numpy().astype(np.float64).tolist(),
                }
            )
    return layers


def render_viewer_html(weights: dict, reference: dict) -> str:
    """Self-contained HTML/JS viewer: embeds weights + reference data directly (no
    fetch, no external requests) so it works as a standalone file or Artifact."""
    width, height = weights["resolution"]
    data_json = json.dumps({"weights": weights, "reference": reference})
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>NRP JS viewer (E6)</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; padding: 16px; }}
canvas {{ border: 1px solid #444; image-rendering: pixelated; width: 384px; height: 384px; }}
label {{ display: block; margin-top: 8px; }}
#parity {{ margin-top: 12px; font-family: monospace; }}
</style></head>
<body>
<h2>NRP proxy — JS backend, real-time light sliders (E6)</h2>
<canvas id="c" width="{width}" height="{height}"></canvas>
<div>
<label>center x <input id="cx" type="range" min="0" max="1" step="0.01" value="0.5"></label>
<label>center y <input id="cy" type="range" min="0" max="1" step="0.01" value="0.6"></label>
<label>center z <input id="cz" type="range" min="0" max="1" step="0.01" value="0.4"></label>
<label>radius <input id="cr" type="range" min="0.03" max="0.3" step="0.01" value="0.15"></label>
</div>
<div id="parity"></div>
<script>
const DATA = {data_json};
const W = DATA.weights;
const RES = W.resolution;
const NX = RES[0], NY = RES[1];

function relu(v) {{ return v > 0 ? v : 0; }}
function softplus(v) {{ return Math.log(1 + Math.exp(v)); }}

function forwardPixel(xy, aux, lightParams) {{
  let input = xy.concat(aux, lightParams);
  for (let li = 0; li < W.layers.length; li++) {{
    const layer = W.layers[li];
    const out = new Array(layer.weight.length).fill(0);
    for (let o = 0; o < layer.weight.length; o++) {{
      let s = layer.bias[o];
      const row = layer.weight[o];
      for (let i = 0; i < row.length; i++) s += row[i] * input[i];
      out[o] = (li < W.layers.length - 1) ? relu(s) : s;
    }}
    input = out;
  }}
  return input.map(softplus);
}}

function renderImage(center, radius) {{
  const canvas = document.getElementById('c');
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(NX, NY);
  let maxAbsDiff = 0;
  const useRef = (
    Math.abs(center[0] - W.default_light.center[0]) < 1e-9 &&
    Math.abs(center[1] - W.default_light.center[1]) < 1e-9 &&
    Math.abs(center[2] - W.default_light.center[2]) < 1e-9 &&
    Math.abs(radius - W.default_light.radius) < 1e-9
  );
  for (let p = 0; p < NX * NY; p++) {{
    const rgb = forwardPixel(W.xy[p], W.aux[p], center.concat([radius]));
    const idx = p * 4;
    for (let c = 0; c < 3; c++) {{
      const tonemapped = rgb[c] / (1 + rgb[c]);
      img.data[idx + c] = Math.max(0, Math.min(255, Math.round(tonemapped * 255)));
      if (useRef) {{
        const ref = DATA.reference.reference_image[Math.floor(p / NX)][p % NX][c];
        maxAbsDiff = Math.max(maxAbsDiff, Math.abs(ref - rgb[c]));
      }}
    }}
    img.data[idx + 3] = 255;
  }}
  ctx.putImageData(img, 0, 0);
  const parityEl = document.getElementById('parity');
  if (useRef) {{
    const diffStr = maxAbsDiff.toExponential(3);
    parityEl.textContent = 'parity vs Python reference: max abs diff = ' + diffStr;
  }} else {{
    parityEl.textContent = '';
  }}
}}

function update() {{
  const center = [
    parseFloat(document.getElementById('cx').value),
    parseFloat(document.getElementById('cy').value),
    parseFloat(document.getElementById('cz').value),
  ];
  const radius = parseFloat(document.getElementById('cr').value);
  renderImage(center, radius);
}}

const sliderIds = ['cx', 'cy', 'cz', 'cr'];
sliderIds.forEach(id => document.getElementById(id).addEventListener('input', update));
update();
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="out/engine-runtime/js_viewer")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = trace_path_cache(args.width, args.height, spp=6, max_bounces=2, seed=17)
    model = train_toy_proxy(cache)
    model.eval()

    default_light = SphereLight(center=[0.5, 0.6, 0.4], radius=0.15)
    reference_image = relight(model, cache, [default_light])
    default_direct = gather_light(cache, default_light)
    reference_psnr = psnr(reference_image, default_direct)

    xy, aux = pixel_tensors(cache, "cpu")
    weights = {
        "light_type": "sphere",
        "resolution": [args.width, args.height],
        "layers": extract_weights(model),
        "xy": xy.numpy().astype(np.float64).tolist(),
        "aux": aux.numpy().astype(np.float64).tolist(),
        "default_light": {"center": default_light.center.tolist(), "radius": default_light.radius},
        "parameter_count": model.parameter_count,
    }
    reference = {
        "reference_image": reference_image.tolist(),
        "reference_vs_gather_psnr_db": reference_psnr if np.isfinite(reference_psnr) else "inf",
    }
    (out_dir / "model_weights.json").write_text(json.dumps(weights))
    (out_dir / "reference.json").write_text(json.dumps(reference))
    (out_dir / "viewer.html").write_text(render_viewer_html(weights, reference))

    report = {
        "extension": "E6",
        "scope": "JS-backend export + interactive GUI slider viewer (WebGPU-shaped step)",
        "resolution": [args.width, args.height],
        "parameter_count": model.parameter_count,
        "reference_vs_gather_psnr_db": reference_psnr if np.isfinite(reference_psnr) else "inf",
        "artifact_dir": str(out_dir),
        "viewer_html": str(out_dir / "viewer.html"),
        "notes": [
            "Model trained with use_encoding=False so the JS forward pass is plain "
            "Linear->ReLU->...->Linear->softplus, replicable without a hashgrid port.",
            "viewer.html runs a JS-side parity check against reference.json at load "
            "time and displays the max abs diff, standing in for a committed parity "
            "test for this new (non-TorchScript) backend.",
            "This is a step toward E6's WebGPU criterion, not the criterion itself: "
            "the backend is plain JS/canvas, not compiled WebGPU shaders.",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
