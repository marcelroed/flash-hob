"""Benchmark FlashAttention-4 (CuTe-DSL) backward vs cuDNN backward, from JAX.

Target shape: batch=16, seq=512, 20 heads, head_dim=64, bf16, non-causal.
Both paths are pure JAX (no torch):
  - FA4 backward: fa4_jax.fa4_bwd (preprocess -> bwd -> postprocess; 3 kernels),
    with out/lse precomputed by fa4_fwd and a random dout.
  - cuDNN backward: jax.vjp of jax.nn.dot_product_attention(implementation="cudnn"),
    applied to dout (times just the backward).

Wall-clock is min/median over windows after a long warmup (GPU clock ramp). For
overhead-free on-device kernel time use nsys (see --nsys-mode); FA4 is 3 kernels
so sum their cuda_gpu_kern_sum rows.

Run:  CUDA_VISIBLE_DEVICES=0 uv run --group experiments python experiments/bench_backward.py
nsys: CUDA_VISIBLE_DEVICES=0 nsys profile --stats=true -o /tmp/fa4_bwd \
        uv run --group experiments python experiments/bench_backward.py --nsys-mode fa4
"""
import argparse
import time

import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np

import fa4_jax


def _time(fn, *args, iters=100, warmup=200, windows=10):
    for _ in range(warmup):
        out = fn(*args)
    jax.block_until_ready(out)
    times = []
    for _ in range(windows):
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) / iters)
    times.sort()
    return times[0], times[len(times) // 2]  # min, median


def bwd_flops(b, s, h, d, causal=False):
    # Forward (non-causal) = 4*b*h*s*s*d MACs-as-FLOPs (QK^T + P@V).
    # Backward attention is ~2.5x forward (dV, dP, dQ, dK matmuls).
    fwd = 4.0 * b * h * s * s * d
    f = 2.5 * fwd
    return f * 0.5 if causal else f


def make_inputs(B, S, H, D, seed=0):
    scale = float(1.0 / np.sqrt(D))
    ks = jax.random.split(jax.random.key(seed), 4)
    q = (jax.random.normal(ks[0], (B, S, H, D), jnp.float32) * 0.5).astype(jnp.bfloat16)
    k = (jax.random.normal(ks[1], (B, S, H, D), jnp.float32) * 0.5).astype(jnp.bfloat16)
    v = (jax.random.normal(ks[2], (B, S, H, D), jnp.float32) * 0.5).astype(jnp.bfloat16)
    dout = (jax.random.normal(ks[3], (B, S, H, D), jnp.float32) * 0.5).astype(jnp.bfloat16)
    return q, k, v, dout, scale


def build_fns(B, S, H, D):
    q, k, v, dout, scale = make_inputs(B, S, H, D)
    out, lse = jax.block_until_ready(fa4_jax.fa4_fwd(q, k, v, scale))

    fa4_bwd = jax.jit(lambda q, k, v, out, dout, lse: fa4_jax.fa4_bwd(q, k, v, out, dout, lse, scale))

    def _cudnn_fwd(q, k, v):
        return jnn.dot_product_attention(q, k, v, scale=scale, implementation="cudnn")

    def _cudnn_bwd(q, k, v, dout):
        _, vjp = jax.vjp(_cudnn_fwd, q, k, v)
        return vjp(dout)

    cudnn_bwd = jax.jit(_cudnn_bwd)
    return (q, k, v, out, dout, lse), fa4_bwd, cudnn_bwd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--heads", type=int, default=20)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--nsys-mode", choices=["off", "fa4", "cudnn"], default="off",
                   help="run one impl in a tight loop for nsys profiling (no timing).")
    args = p.parse_args()

    B, S, H, D = args.batch, args.seq, args.heads, args.head_dim
    (q, k, v, out, dout, lse), fa4_bwd, cudnn_bwd = build_fns(B, S, H, D)

    if args.nsys_mode != "off":
        # Warm up (compile), then a tight loop for nsys to trace kernels.
        if args.nsys_mode == "fa4":
            f = lambda: fa4_bwd(q, k, v, out, dout, lse)
        else:
            f = lambda: cudnn_bwd(q, k, v, dout)
        for _ in range(50):
            o = f()
        jax.block_until_ready(o)
        for _ in range(200):
            o = f()
        jax.block_until_ready(o)
        return

    print(f"shape: B={B} S={S} H={H} D={D} (d_model={H*D}) bf16, non-causal backward\n")

    # correctness vs cuDNN backward
    dq, dk, dv = jax.block_until_ready(fa4_bwd(q, k, v, out, dout, lse))
    dq_r, dk_r, dv_r = jax.block_until_ready(cudnn_bwd(q, k, v, dout))
    for name, a, b in [("dq", dq, dq_r), ("dk", dk, dk_r), ("dv", dv, dv_r)]:
        a = a.astype(jnp.float32); b = b.astype(jnp.float32)
        mad = float(jnp.max(jnp.abs(a - b))); mref = float(jnp.mean(jnp.abs(b)))
        print(f"  {name}: max|diff|={mad:.3e}  mean|ref|={mref:.3e}  ratio={mad/(mref+1e-30):.3f}")
    print()

    flops = bwd_flops(B, S, H, D)
    print(f"backward FLOPs = 2.5 * (4*B*H*S*S*D) = {flops/1e9:.2f} GFLOP\n")
    print(f"{'impl':22s}  {'min us':>8s} {'med us':>8s}   {'TFLOP/s (min)':>13s}")
    for name, fn, fargs in [
        ("FA4 bwd (3 kernels)", fa4_bwd, (q, k, v, out, dout, lse)),
        ("cuDNN bwd (vjp)", cudnn_bwd, (q, k, v, dout)),
    ]:
        tmin, tmed = _time(fn, *fargs, iters=args.iters)
        print(f"{name:22s}  {tmin*1e6:8.1f} {tmed*1e6:8.1f}   {flops/tmin/1e12:13.1f}")


if __name__ == "__main__":
    main()
