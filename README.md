# Flash Hog
<p align="center">
<img src="assets/logo.png" alt="Flash Hog Logo" width="256" />
</p>

This repo contains the code for Flash Higher-Order-Gradients, aka. Flash Hog.
This kernel achieves around a 3.7x speedup over an XLA optimized kernel, with linear memory scaling instead of quadratic scaling.

<p align="center">
<img src="assets/speedup.png" alt="Hog Speedup" width="512"/>
</p>

## Installation
```sh
uv add flash-hog
```

## Optional: ThunderKittens kernels (Hopper)
Opt-in TK kernels for the double-backward, ~1.65x faster end-to-end on H200
(causal, head_dim 64, seq % 128 == 0). Off by default; unsupported shapes fall back to Pallas.
```sh
uv add 'flash-hog[tk]'   # optional: CUDA build tools from PyPI (needs a host C++ compiler)
```
```python
from flash_hog.jax import _tk_gpu as tk
tk.enable()   # JIT-builds the plugin on first use (cached); then TK kernels are live
```
Set `THUNDERKITTENS_PATH` to use a local ThunderKittens checkout instead of the auto-fetched one.

## Method
Flash Hog does 4 recomputation passes to avoid any atomics or saving any intermediary tensors of shape `(N_Q, N_K)`.
This shakes out to be thread-wise tiling across Q in 3 passes first, once to compute `dd`, then once for `b`, then once for both `dQ'` and `ddO`.
Finally we do another pass tiled over K, producing `dK'` and `dV'`.
The equations we implement are the following:


<p align="center">
<img src="assets/handwritten_equations.png" alt="Equations" width="512"/>
</p>

## Citation
If you use Flash Hog in your work, please cite it as:

```bibtex
@software{roed2025flashhog,
  author = {Marcel R{\o}d and Herman Brunborg and Neil Band and Kesavan Ramakrishnan and Stephen Ge},
  title = {{F}lash {H}og: Memory-Efficient Kernels for Higher-Order Gradients of Flash Attention},
  url = {https://github.com/marcelroed/flash-hog},
  year = {2025},
}
```
