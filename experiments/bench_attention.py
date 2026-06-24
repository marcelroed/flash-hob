"""Benchmark FlashAttention-4 (CuTe-DSL) vs cuDNN for attention, from JAX.

Target shape: batch=16, seq=512, d_model=20*64 (20 heads), head_dim=64, bf16.
Both paths are pure JAX: FA4 via cutlass.jax.cutlass_call, cuDNN via
jax.nn.dot_product_attention(implementation="cudnn"). No torch.

Run:  uv run --group experiments python experiments/bench_attention.py
"""
import argparse
import time

import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np

import fa4_jax


def _time(fn, *args, iters=200, warmup=200, windows=10):
    # Long warmup so the GPU reaches steady-state clocks; report min/median over
    # several windows (min ~ contention-free steady state).
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


def fwd_flops(b, s, h, d, causal=False):
    # QK^T + P@V, each 2*b*h*s*s*d mul-adds; halve for causal.
    f = 4.0 * b * h * s * s * d
    return f * 0.5 if causal else f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--heads", type=int, default=20)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    B, S, H, D = args.batch, args.seq, args.heads, args.head_dim
    scale = 1.0 / np.sqrt(D)
    print(f"shape: B={B} S={S} H={H} D={D} (d_model={H*D}) bf16, non-causal\n")

    ks = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(ks[0], (B, S, H, D), jnp.bfloat16) * 0.5
    k = jax.random.normal(ks[1], (B, S, H, D), jnp.bfloat16) * 0.5
    v = jax.random.normal(ks[2], (B, S, H, D), jnp.bfloat16) * 0.5

    fa4 = jax.jit(lambda q, k, v: fa4_jax.fa4_fwd(q, k, v, scale, return_lse=False))
    cudnn = jax.jit(lambda q, k, v: jnn.dot_product_attention(q, k, v, scale=float(scale), implementation="cudnn"))

    # correctness
    o_fa4 = jax.block_until_ready(fa4(q, k, v)).astype(jnp.float32)
    o_cud = jax.block_until_ready(cudnn(q, k, v)).astype(jnp.float32)
    md = float(jnp.max(jnp.abs(o_fa4 - o_cud)))
    print(f"max abs diff FA4 vs cuDNN: {md:.3e}  (mean|cuDNN|={float(jnp.mean(jnp.abs(o_cud))):.3e})\n")

    flops = fwd_flops(B, S, H, D)
    print(f"{'impl':18s}  {'min us':>8s} {'med us':>8s}   {'TFLOP/s (min)':>13s}")
    for name, fn in [("FA4 (CuTe-DSL)", fa4), ("cuDNN", cudnn)]:
        tmin, tmed = _time(fn, q, k, v, iters=args.iters)
        print(f"{name:18s}  {tmin*1e6:8.1f} {tmed*1e6:8.1f}   {flops/tmin/1e12:13.1f}")


if __name__ == "__main__":
    main()
