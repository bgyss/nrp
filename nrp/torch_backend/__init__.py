"""PyTorch backend: the paper's architecture on modern scientific-Python tooling.

This package follows the paper's implementation (§4) as closely as feasible on CPU/MPS:
a 2D multiresolution hash encoding of pixel coordinates [MESK22] feeding an MLP that
also receives auxiliary pixel features (albedo, depth, normal — 7D) and the light's
shape parameters (sphere: 4, quad: 8); the relative-MSE HDR loss of Eq. 4 with a
stop-gradient denominator; segment-based light-position sampling (§4.4); and a pool of
denoised target images with periodic replacement. tiny-cuda-nn and the fused Triton
GATHERLIGHT kernel are not used — the hash encoding is plain PyTorch and gathering is
the numpy reference — so absolute speed is not comparable to the paper's GPU numbers.

The numpy backend in the parent package stays as the dependency-light,
finite-difference-checked reference implementation.
"""
