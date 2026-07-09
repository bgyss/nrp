import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.export_js_viewer import (  # noqa: E402
    extract_weights,
    render_viewer_html,
    train_toy_proxy,
)
from nrp.toy_tracer import trace_path_cache  # noqa: E402

HAVE_NODE = shutil.which("node") is not None

# Mirrors the forward pass in `render_viewer_html`'s embedded JS: plain
# Linear -> ReLU -> ... -> Linear -> softplus, run once for a fixed light against the
# embedded reference image, reporting the max abs diff.
_NODE_PARITY_SCRIPT = """
const DATA = %s;
const W = DATA.weights;
function relu(v) { return v > 0 ? v : 0; }
function softplus(v) { return Math.log(1 + Math.exp(v)); }
function forwardPixel(xy, aux, lightParams) {
  let input = xy.concat(aux, lightParams);
  for (let li = 0; li < W.layers.length; li++) {
    const layer = W.layers[li];
    const out = new Array(layer.weight.length).fill(0);
    for (let o = 0; o < layer.weight.length; o++) {
      let s = layer.bias[o];
      const row = layer.weight[o];
      for (let i = 0; i < row.length; i++) s += row[i] * input[i];
      out[o] = (li < W.layers.length - 1) ? relu(s) : s;
    }
    input = out;
  }
  return input.map(softplus);
}
const center = W.default_light.center;
const radius = W.default_light.radius;
let maxDiff = 0;
const NX = W.resolution[0];
for (let p = 0; p < W.xy.length; p++) {
  const rgb = forwardPixel(W.xy[p], W.aux[p], center.concat([radius]));
  const ref = DATA.reference.reference_image[Math.floor(p / NX)][p %% NX];
  for (let c = 0; c < 3; c++) maxDiff = Math.max(maxDiff, Math.abs(ref[c] - rgb[c]));
}
console.log(JSON.stringify({max_abs_diff: maxDiff}));
"""


class ExportJsViewerTests(unittest.TestCase):
    def test_viewer_html_embeds_data_and_is_self_contained(self):
        cache = trace_path_cache(8, 8, spp=2, max_bounces=1, seed=2)
        model = train_toy_proxy(cache, iters=20)
        weights = {
            "light_type": "sphere",
            "resolution": [8, 8],
            "layers": extract_weights(model),
            "xy": [[0.0, 0.0]] * 64,
            "aux": [[0.0] * 7] * 64,
            "default_light": {"center": [0.5, 0.5, 0.5], "radius": 0.1},
            "parameter_count": model.parameter_count,
        }
        reference = {"reference_image": [[[0.0, 0.0, 0.0]] * 8] * 8}
        html = render_viewer_html(weights, reference)
        self.assertIn("<canvas", html)
        self.assertIn("input", html)
        self.assertNotIn("fetch(", html)
        self.assertIn("const DATA = ", html)

    @unittest.skipUnless(HAVE_NODE, "node is not installed")
    def test_js_forward_pass_matches_python_reference(self):
        cache = trace_path_cache(10, 10, spp=4, max_bounces=1, seed=5)
        model = train_toy_proxy(cache, iters=50)
        model.eval()
        from nrp.gather_light import gather_light  # local import: avoid unused otherwise
        from nrp.lights import SphereLight
        from nrp.torch_backend.relight import relight
        from nrp.torch_backend.train import pixel_tensors

        light = SphereLight(center=[0.5, 0.6, 0.4], radius=0.15)
        reference_image = relight(model, cache, [light])
        gather_light(cache, light)  # sanity: cache/light combination is valid
        xy, aux = pixel_tensors(cache, "cpu")
        weights = {
            "light_type": "sphere",
            "resolution": [10, 10],
            "layers": extract_weights(model),
            "xy": xy.numpy().tolist(),
            "aux": aux.numpy().tolist(),
            "default_light": {"center": light.center.tolist(), "radius": light.radius},
            "parameter_count": model.parameter_count,
        }
        reference = {"reference_image": reference_image.tolist()}
        html = render_viewer_html(weights, reference)
        data_json = re.search(r"const DATA = (.*?);\n", html, re.S).group(1)
        script = _NODE_PARITY_SCRIPT % data_json
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "parity.js"
            script_path.write_text(script)
            result = subprocess.run(
                ["node", str(script_path)], capture_output=True, text=True, check=True
            )
        parity = json.loads(result.stdout.strip())
        self.assertLess(parity["max_abs_diff"], 1e-4)


if __name__ == "__main__":
    unittest.main()
