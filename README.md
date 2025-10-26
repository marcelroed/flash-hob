# Flash Hog
<p align="center">
<img src="logo.png" alt="Flash Hog Logo" width="256" />
</p>

This repo contains the code for Flash Higher-Order-Gradients, aka. Flash Hog.
This kernel achieves around a 3.7x speedup over an XLA optimized kernel, with linear memory scaling instead of quadratic scaling.

<p align="center">
<img src="speedup.png" alt="Hog Speedup" width="512"/>
</p>

## Installation
TODO

## Method
Flash Hog does 4 recomputation passes to avoid any atomics or saving any intermediary tensors of shape `(N_Q, N_K)`.
The equations we implement are the following:


<p align="center">
<img src="handwritten_equations.png" alt="Equations" width="512"/>
</p>