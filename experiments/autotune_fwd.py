"""Quick autotuning sweep for the FA4 forward: try several SM90 tile/config
choices for a fixed shape, validate vs cuDNN, and time each (min over windows).

interface.py picks ONE config from offline (H100-tuned) heuristics; this checks
whether a per-shape sweep beats that default on this H200 shape.

Run: CUDA_VISIBLE_DEVICES=0 uv run --group experiments python experiments/autotune_fwd.py
"""
import time
import jax, jax.numpy as jnp, jax.nn as jnn, numpy as np
import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import fa4_jax
from fa4_jax import FlashAttentionForwardSm90, _DTYPE_MAP


def build_fwd_cfg(dtype_name, d, dv, tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap, num_threads):
    return FlashAttentionForwardSm90(
        _DTYPE_MAP[dtype_name], d, dv, 1,
        is_causal=False, is_local=False, pack_gqa=False,
        tile_m=tile_m, tile_n=tile_n, num_stages=2, num_threads=num_threads,
        Q_in_regs=False, intra_wg_overlap=intra_wg_overlap, mma_pv_is_rs=mma_pv_is_rs,
        mask_mod=None, score_mod=None, has_aux_tensors=False,
        q_subtile_factor=1, paged_kv_non_tma=False,
    )


def fwd_with(fa_fwd, q, k, v, scale):
    b, s, h, d = q.shape
    dv = v.shape[-1]

    @cute.jit
    def launch(stream, mQ, mK, mV, mO, mLSE):
        fa_fwd(mQ, mK, mV, mO, mLSE, float(scale), stream=stream)

    return cjax.cutlass_call(
        launch,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((b, s, h, dv), q.dtype),
            jax.ShapeDtypeStruct((b, h, s), jnp.float32),
        ),
        use_static_tensors=True,
    )(q, k, v)


def time_fn(fn, *a, iters=200, warmup=200, windows=8):
    for _ in range(warmup):
        o = fn(*a)
    jax.block_until_ready(o)
    ts = []
    for _ in range(windows):
        t0 = time.perf_counter()
        for _ in range(iters):
            o = fn(*a)
        jax.block_until_ready(o)
        ts.append((time.perf_counter() - t0) / iters)
    return min(ts)


def main():
    B, S, H, D = 16, 512, 20, 64
    scale = 1.0 / np.sqrt(D)
    ks = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(ks[0], (B, S, H, D), jnp.bfloat16) * 0.5
    k = jax.random.normal(ks[1], (B, S, H, D), jnp.bfloat16) * 0.5
    v = jax.random.normal(ks[2], (B, S, H, D), jnp.bfloat16) * 0.5
    ref = jnn.dot_product_attention(q, k, v, scale=float(scale), implementation="cudnn").astype(jnp.float32)
    flops = 4.0 * B * H * S * S * D
    print(f"shape B={B} S={S} H={H} D={D} bf16; cuDNN ref ready\n")

    # (tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap, num_threads)
    configs = [
        (192, 128, True, True, 384),   # interface heuristic default (hd<=64)
        (128, 128, True, True, 384),
        (128, 128, True, True, 256),
        (128, 64, True, True, 384),
        (192, 144, False, True, 384),
        (192, 128, False, True, 384),
        (128, 128, False, True, 256),
        (64, 128, True, True, 256),
    ]
    print(f"{'tile_m':>6} {'tile_n':>6} {'RS':>5} {'OL':>5} {'thr':>4}  {'us':>7} {'TFLOP/s':>8}  {'maxdiff':>9}")
    best = None
    for (tm, tn, rs, ol, nt) in configs:
        try:
            fa = build_fwd_cfg("bfloat16", D, D, tm, tn, rs, ol, nt)
            fn = jax.jit(lambda q, k, v, fa=fa: fwd_with(fa, q, k, v, scale)[0])
            o = jax.block_until_ready(fn(q, k, v)).astype(jnp.float32)
            md = float(jnp.max(jnp.abs(o - ref)))
            t = time_fn(fn, q, k, v)
            tfl = flops / t / 1e12
            print(f"{tm:>6} {tn:>6} {str(rs):>5} {str(ol):>5} {nt:>4}  {t*1e6:7.1f} {tfl:8.1f}  {md:9.2e}")
            if md < 5e-3 and (best is None or t < best[0]):
                best = (t, (tm, tn, rs, ol, nt))
        except Exception as e:
            print(f"{tm:>6} {tn:>6} {str(rs):>5} {str(ol):>5} {nt:>4}  FAILED: {str(e)[:50]}")
    if best:
        print(f"\nbest config: {best[1]}  @ {best[0]*1e6:.1f} us ({flops/best[0]/1e12:.1f} TFLOP/s)")
        print("(cuDNN on this shape ~55.8 us on-device / ~63 us wall-clock)")


if __name__ == "__main__":
    main()
