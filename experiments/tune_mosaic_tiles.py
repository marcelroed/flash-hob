"""Autotune the Mosaic Hopper double-backward tiles and report speedups.

Compares, on one identical rule-level harness:
  * default Pallas (non-specialized) double-backward
  * Mosaic with the kernel's built-in default tiles
  * Mosaic with tune-jax-selected tiles

Run:  python experiments/tune_mosaic_tiles.py
"""

import time
from math import sqrt

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax._src.cudnn.fused_attention_stablehlo import MaskType

import flash_hog.jax._attention_impl as impl
from flash_hog.jax import _mosaic_gpu as mosaic
from flash_hog.jax import _mosaic_gpu_tune as tuner

BATCH, SEQ, NHEADS, HD = 16, 512, 20, 64
SCALE = 1.0 / sqrt(HD)


def bench(fn, *args, warmup=10, iters=100):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    print(f"Autotuning Mosaic tiles for b{BATCH} seq{SEQ} {NHEADS}x{HD} (causal)...")
    best = tuner.autotune_tiles(
        batch=BATCH, seq=SEQ, num_heads=NHEADS, head_dim=HD, scale=SCALE, install=True
    )
    print(f"  best stage1 = {best['stage1']}  "
          f"({best['n_stage1_correct']}/{best['n_stage1_screened']} configs correct, "
          f"{best['stage1_time_s']*1e3:.3f} ms)")
    print(f"  best stage2 = {best['stage2']}  "
          f"({best['n_stage2_correct']}/{best['n_stage2_screened']} configs correct, "
          f"{best['stage2_time_s']*1e3:.3f} ms)")

    # ---- rule-level harness, identical inputs for all three -----------------
    sh = (BATCH, SEQ, NHEADS, HD)
    ks = jrandom.split(jrandom.PRNGKey(7), 8)
    q, k, v, do = (jrandom.normal(ks[i], sh, dtype=jnp.bfloat16) for i in range(4))
    ddq, ddk, ddv = (jrandom.normal(ks[i], sh, dtype=jnp.bfloat16) for i in range(4, 7))
    out, act = impl.cuda_dot_product_attention(
        q, k, v, mask_type=MaskType.CAUSAL, scale=SCALE, return_residual=True)
    res, g = (q, k, v, out, act, do), (ddq, ddk, ddv)

    def rule_fn():
        rule = impl.dot_product_attention_bwd_rule_bwd_rule  # swapped by enable/disable
        return jax.jit(lambda res, g: rule(MaskType.CAUSAL, SCALE, res, g))

    mosaic.disable()
    f_pallas = rule_fn()
    ref = jax.block_until_ready(f_pallas(res, g))
    t_pallas = bench(f_pallas, res, g)

    mosaic.enable()
    mosaic.set_tuned_tiles({}, {})
    f_def = rule_fn()
    jax.block_until_ready(f_def(res, g))
    t_def = bench(f_def, res, g)

    mosaic.set_tuned_tiles(best["stage1"], best["stage2"])
    f_tuned = rule_fn()
    out_tuned = jax.block_until_ready(f_tuned(res, g))
    t_tuned = bench(f_tuned, res, g)
    mosaic.set_tuned_tiles({}, {})
    mosaic.disable()

    # ---- correctness gate ---------------------------------------------------
    def flat(o):
        dres, ddO = o
        return [dres[0], dres[1], dres[2], ddO]

    print("\n--- correctness: tuned Mosaic vs default Pallas ---")
    ok = True
    for nm, a, b in zip(["dQ2", "dK2", "dV2", "ddO"], flat(ref), flat(out_tuned)):
        a32, b32 = a.astype(jnp.float32), b.astype(jnp.float32)
        rel = float(jnp.abs(a32 - b32).mean() / jnp.maximum(jnp.abs(a32).mean(), 1e-6))
        ok = ok and rel < 5e-3
        print(f"  {nm}: mean_rel_err={rel:.3e}")
    print(f"  => {'PASS' if ok else 'FAIL'}")

    print(f"\n========= RESULTS  b{BATCH} seq{SEQ} {NHEADS}x{HD} causal =========")
    print("  (isolated double-backward rule; identical harness for every row)")
    print(f"  default Pallas (non-specialized) : {t_pallas:.3f} ms   1.00x")
    print(f"  Mosaic, default tiles            : {t_def:.3f} ms   "
          f"{t_pallas/t_def:.2f}x")
    print(f"  Mosaic, tuned tiles              : {t_tuned:.3f} ms   "
          f"{t_pallas/t_tuned:.2f}x")
    print(f"  tuning gain over default tiles   : {t_def/t_tuned:.2f}x")


if __name__ == "__main__":
    main()
