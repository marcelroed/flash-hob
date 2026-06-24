"""Benchmark attention across all four execution phases, FA4 (CuTe-DSL) vs cuDNN.

Cases (the standard custom_vjp phases):
  1. pure forward       -- inference: out only, nothing saved for backward
  2. forward rule       -- training forward: also saves residuals (lse) for backward
  3. backward only      -- given the saved residuals (out, lse) + dout -> dq, dk, dv
  4. forward + backward -- a full training step (forward-with-save then backward)

FA4 separates these cleanly (fa4_fwd return_lse=False/True, fa4_bwd). cuDNN
(jax.nn.dot_product_attention, implementation="cudnn") does not expose its softmax
stat, so:
  - "forward rule" is timed as the same inference call (the only extra work cuDNN's
    training forward does is writing the LSE stat -- negligible); reported with a note.
  - "backward only" is jax.vjp's backward closure, which RECOMPUTES the forward
    (flash_fprop) -- a cuDNN-vjp user pays this, so it is reported as-is.
  - "forward + backward" is jax.vjp value + vjp(dout).

Both paths are pure JAX (no torch). Shape defaults: B=16 S=512 H=20 D=64 bf16, non-causal.

Run:  CUDA_VISIBLE_DEVICES=0 uv run --group experiments python experiments/bench_all.py
"""
import argparse
import time

import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np

import fa4_jax


def _time(fn, *args, iters=100, warmup=200, windows=10):
    # Long warmup so the GPU reaches steady-state clocks; min ~ contention-free steady state.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--heads", type=int, default=20)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    B, S, H, D = args.batch, args.seq, args.heads, args.head_dim
    scale = float(1.0 / np.sqrt(D))
    fwd_flops = 4.0 * B * H * S * S * D            # QK^T + P@V
    bwd_flops = 2.5 * fwd_flops                    # dV, dP, dQ, dK
    print(f"shape: B={B} S={S} H={H} D={D} (d_model={H*D}) bf16, non-causal")
    print(f"fwd FLOPs = {fwd_flops/1e9:.2f} GFLOP, bwd ~ {bwd_flops/1e9:.2f} GFLOP\n")

    ks = jax.random.split(jax.random.key(0), 4)
    q = (jax.random.normal(ks[0], (B, S, H, D), jnp.float32) * 0.5).astype(jnp.bfloat16)
    k = (jax.random.normal(ks[1], (B, S, H, D), jnp.float32) * 0.5).astype(jnp.bfloat16)
    v = (jax.random.normal(ks[2], (B, S, H, D), jnp.float32) * 0.5).astype(jnp.bfloat16)
    dout = (jax.random.normal(ks[3], (B, S, H, D), jnp.float32) * 0.5).astype(jnp.bfloat16)

    # Precomputed residuals for the backward-only case (FA4).
    out0, lse0 = jax.block_until_ready(fa4_jax.fa4_fwd(q, k, v, scale))

    # --- FA4 callables ---
    fa4_fwd_nosave = jax.jit(lambda q, k, v: fa4_jax.fa4_fwd(q, k, v, scale, return_lse=False))
    fa4_fwd_save = jax.jit(lambda q, k, v: fa4_jax.fa4_fwd(q, k, v, scale, return_lse=True))
    fa4_bwd_only = jax.jit(lambda q, k, v, out, dout, lse: fa4_jax.fa4_bwd(q, k, v, out, dout, lse, scale))

    def _fa4_fwdbwd(q, k, v, dout):
        out, lse = fa4_jax.fa4_fwd(q, k, v, scale, return_lse=True)
        return out, fa4_jax.fa4_bwd(q, k, v, out, dout, lse, scale)
    fa4_fwdbwd = jax.jit(_fa4_fwdbwd)

    # --- cuDNN callables ---
    def _cudnn_fwd(q, k, v):
        return jnn.dot_product_attention(q, k, v, scale=scale, implementation="cudnn")
    cudnn_fwd = jax.jit(_cudnn_fwd)

    def _cudnn_bwd_only(q, k, v, dout):
        _, vjp = jax.vjp(_cudnn_fwd, q, k, v)
        return vjp(dout)
    cudnn_bwd_only = jax.jit(_cudnn_bwd_only)

    def _cudnn_fwdbwd(q, k, v, dout):
        out, vjp = jax.vjp(_cudnn_fwd, q, k, v)
        return out, vjp(dout)
    cudnn_fwdbwd = jax.jit(_cudnn_fwdbwd)

    # case name, FA4 (fn, args, flops), cuDNN (fn, args, flops, note)
    cases = [
        ("1. pure forward (no save)",
            (fa4_fwd_nosave, (q, k, v), fwd_flops),
            (cudnn_fwd, (q, k, v), fwd_flops, "")),
        ("2. forward rule (save lse)",
            (fa4_fwd_save, (q, k, v), fwd_flops),
            (cudnn_fwd, (q, k, v), fwd_flops, "(no separate save path; ~=pure fwd)")),
        ("3. backward only",
            (fa4_bwd_only, (q, k, v, out0, dout, lse0), bwd_flops),
            (cudnn_bwd_only, (q, k, v, dout), bwd_flops, "(vjp recomputes fwd)")),
        ("4. forward + backward",
            (fa4_fwdbwd, (q, k, v, dout), fwd_flops + bwd_flops),
            (cudnn_fwdbwd, (q, k, v, dout), fwd_flops + bwd_flops, "")),
    ]

    hdr = f"{'case':28s} {'impl':6s}  {'min us':>8s} {'med us':>8s} {'TFLOP/s':>8s}  note"
    print(hdr)
    print("-" * len(hdr))
    for name, (ffn, fargs, fflop), (cfn, cargs, cflop, cnote) in cases:
        tmin, tmed = _time(ffn, *fargs, iters=args.iters)
        print(f"{name:28s} {'FA4':6s}  {tmin*1e6:8.1f} {tmed*1e6:8.1f} {fflop/tmin/1e12:8.1f}")
        tmin, tmed = _time(cfn, *cargs, iters=args.iters)
        print(f"{'':28s} {'cuDNN':6s}  {tmin*1e6:8.1f} {tmed*1e6:8.1f} {cflop/tmin/1e12:8.1f}  {cnote}")
        print()


if __name__ == "__main__":
    main()
