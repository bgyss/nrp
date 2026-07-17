#!/usr/bin/env bash
# S4: rebuild the 8-light rig with H3 kernel-conditioned textured-quad proxies,
# re-run the additivity gate + V2 art-direction loop, export to the WebGPU rig
# compositor, and run the H4 Chrome bench against the rebuilt rig.
# Needs OIDN (run under `nix develop --command`) and the kitchen-512 cache.
set -euo pipefail
cd "$(dirname "$0")/.."

CACHE=out/kitchen-512/path_cache.npz

echo "== S4 stage 1: rig rebuild (v1_rig, kernel-conditioned textured quads) =="
uv run python examples/v1_rig.py \
  --cache "$CACHE" --out-dir out/s4-rig \
  --iters 1600 --textured-quad-iters 3200 --texture-conditioning kernel \
  --gate-tier preview --denoise oidn

echo "== S4 stage 2: V2 art-direction loop on the rebuilt rig =="
uv run python examples/v2_art_loop.py \
  --rig out/s4-rig/rig.json --models-dir out/s4-rig/models \
  --cache "$CACHE" --out-dir out/s4-artloop

echo "== S4 stage 3: WebGPU rig export (kernel-head WGSL) =="
uv run python examples/export_webgpu_rig.py \
  --rig out/s4-rig/rig.json --models-dir out/s4-rig/models \
  --cache "$CACHE" --out-dir out/s4-rig-export

echo "== S4 stage 4: H4 Chrome bench against the rebuilt rig =="
(cd webgpu && npm install --silent && npx playwright install chrome >/dev/null 2>&1 || true)
(cd webgpu && node bench_h4.mjs --export-dir ../out/s4-rig-export --report ../out/s4-rig/webgpu_report.json)

echo "S4 pipeline complete"
