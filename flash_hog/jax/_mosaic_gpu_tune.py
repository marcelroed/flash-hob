"""Tile-size autotuning for the Mosaic GPU double-backward kernels.

Uses tune-jax (https://github.com/rdyro/tune-jax) to time the two Mosaic stages
over a grid of tile sizes / pipeline depths and pick the fastest.

Two things make the feasible space smaller than the raw grid:

  * Shared memory is capped (~228 KB on Hopper SM90), so wide tiles / deep
    pipelines overflow smem and fail to compile.
  * wgmma needs both operands of a matmul to share one swizzle, so the score
    tile and the K/V/Q operands use a single per-stage swizzle sized to the
    smallest participating minor dim (see ``_mosaic_gpu_kernel._tr``).  This
    lets 16/32-wide score tiles compile, but the kernel's per-row/col statistic
    loads (Lv/Dv/dDv/Bv) currently assume a 64-wide score tile, so tiles with
    BK<64 (stage 1) or BQ2<64 (stage 2) compile yet produce wrong numerics.

Because a fast-but-wrong tile would otherwise win on pure timing, every
candidate is screened for numerical correctness against a trusted reference
(the kernel defaults, validated against the Pallas path) BEFORE it is timed.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax._src.cudnn.fused_attention_stablehlo import MaskType

import flash_hog.jax._attention_impl as impl
from flash_hog.jax import _mosaic_gpu as mosaic
from flash_hog.jax import _mosaic_gpu_kernel as mk

# ------------------------------------------------------------------ grids
# Stage 1: BQ rows (wgmma M, must be >=64), BK score-tile width, pipeline depth.
_S1_BQ = [64, 128]
_S1_BK = [16, 32, 64, 128]
_S1_DEPTH = [2, 3, 4]

# Stage 2: (BQ2 score-tile width, BK2 rows).  BK2 is wgmma M (>=64); BQ2 must
# divide 2*BK2 so the causal q-start lands on a query-tile boundary.
_S2_TILES = [(bq2, bk2) for bk2 in (64, 128) for bq2 in (16, 32, 64, 128)
             if (2 * bk2) % bq2 == 0]
_S2_DEPTH = [2, 3, 4, 6]

_REL_TOL = 5e-3  # mean relative error vs the reference (matches the bf16 gate)


def _randn(key, shape, dtype):
    return jrandom.normal(key, shape, dtype=dtype)


def _make_inputs(batch, seq, num_heads, head_dim, seed, scale):
    """Full-kernel inputs in BHTD layout.

    O and L come from a real causal forward so the softmax statistics are
    self-consistent -- random L/O would make exp2(S - L) overflow differently
    per tile size and produce spurious screening mismatches.
    """
    ks = jrandom.split(jrandom.PRNGKey(seed), 12)
    shB = (batch, seq, num_heads, head_dim)  # BTNH for the cuDNN forward
    bf16 = jnp.bfloat16
    q, k, v, dO, ddQ, ddK, ddV = (_randn(ks[i], shB, bf16) for i in range(7))
    out, act = impl.cuda_dot_product_attention(
        q, k, v, mask_type=MaskType.CAUSAL, scale=scale, return_residual=True)

    def bhtd(x):
        return jnp.transpose(x, (0, 2, 1, 3))

    Q, K, V, O, dO, ddQ, ddK, ddV = (
        bhtd(x) for x in (q, k, v, out, dO, ddQ, ddK, ddV))
    L = act.reshape(batch, num_heads, seq).astype(jnp.float32)
    return Q, K, V, O, dO, ddQ, ddK, ddV, L


def _full(inp, scale, s1, s2):
    Q, K, V, O, dO, ddQ, ddK, ddV, L = inp
    return jax.jit(lambda: mk.flash_bwdbwd_mosaic(
        Q, K, V, O, dO, ddQ, ddK, ddV, L, scale=scale,
        stage1_cfg=s1, stage2_cfg=s2))()


def _mean_rel_err(a, b):
    a32, b32 = a.astype(jnp.float32), b.astype(jnp.float32)
    return float(jnp.abs(a32 - b32).mean() / jnp.maximum(jnp.abs(a32).mean(), 1e-6))


def _screen(inp, scale, ref, configs, which):
    """Keep configs that compile AND match the reference output numerically."""
    # stage 1 owns dQ2 (idx 0) and ddO (idx 3); stage 2 owns dK2/dV2 (idx 1,2).
    idxs = [0, 3] if which == "stage1" else [1, 2]
    survivors = []
    for cfg in configs:
        s1, s2 = (cfg, {}) if which == "stage1" else ({}, cfg)
        try:
            out = jax.block_until_ready(_full(inp, scale, s1, s2))
        except Exception:
            continue
        if all(_mean_rel_err(ref[i], out[i]) <= _REL_TOL for i in idxs):
            survivors.append(cfg)
    return survivors


def autotune_tiles(
    *,
    batch: int,
    seq: int,
    num_heads: int,
    head_dim: int = 64,
    scale: float | None = None,
    seed: int = 1,
    install: bool = True,
    logger_level: str = "WARNING",
):
    """Screen + time the Mosaic tiles for a shape and return the best correct one.

    Returns a dict with the best ``stage1``/``stage2`` configs, their per-stage
    kernel time, and how many candidates were screened/passed.  When ``install``
    is True the winning tiles are registered via ``mosaic.set_tuned_tiles``.
    """
    from tune_jax import tune, tune_logger

    tune_logger.setLevel(logger_level)
    if scale is None:
        scale = head_dim ** -0.5

    inp = _make_inputs(batch, seq, num_heads, head_dim, seed, scale)
    ref = jax.block_until_ready(_full(inp, scale, {}, {}))  # trusted defaults

    # ---- stage 1 ----------------------------------------------------------
    s1_grid = [dict(BQ=bq, BK=bk, K_PIPELINE_DEPTH=d)
               for bq, bk, d in itertools.product(_S1_BQ, _S1_BK, _S1_DEPTH)]
    s1_ok = _screen(inp, scale, ref, s1_grid, "stage1")
    best1, t1 = _time_best(tune, inp, scale, s1_ok, "stage1")

    # ---- stage 2 ----------------------------------------------------------
    s2_grid = [dict(BQ2=bq2, BK2=bk2, Q_PIPELINE_DEPTH=d)
               for (bq2, bk2), d in itertools.product(_S2_TILES, _S2_DEPTH)]
    s2_ok = _screen(inp, scale, ref, s2_grid, "stage2")
    best2, t2 = _time_best(tune, inp, scale, s2_ok, "stage2")

    if install:
        mosaic.set_tuned_tiles(best1, best2)

    return {
        "stage1": best1, "stage2": best2,
        "stage1_time_s": t1, "stage2_time_s": t2,
        "n_stage1_screened": len(s1_grid), "n_stage1_correct": len(s1_ok),
        "n_stage2_screened": len(s2_grid), "n_stage2_correct": len(s2_ok),
    }


def _time_best(tune, inp, scale, configs, which):
    """Time correct `configs` with tune-jax and return (best_cfg, best_time_s)."""
    if not configs:
        return {}, float("nan")
    # Pack each config as a hashable tuple of (key,value) pairs; the wrapper
    # rebuilds the kwargs.  A single packed hyperparam avoids tune-jax taking a
    # cartesian product that would re-introduce screened-out combinations.
    packed = [tuple(sorted(c.items())) for c in configs]
    Q, K, V, O, dO, ddQ, ddK, ddV, L = inp

    def fn(*, cfg):
        c = dict(cfg)
        s1, s2 = (c, {}) if which == "stage1" else ({}, c)
        return mk.flash_bwdbwd_mosaic(Q, K, V, O, dO, ddQ, ddK, ddV, L,
                                      scale=scale, stage1_cfg=s1, stage2_cfg=s2)

    tuned = tune(fn, hyperparams={"cfg": packed})
    jax.block_until_ready(jax.jit(tuned)())
    best = dict(tuned.optimal_hyperparams["cfg"])
    best = {k: int(v) for k, v in best.items()}
    times = [r.t_mean for r in tuned.timing_results.values() if r.t_mean == r.t_mean]
    return best, (min(times) if times else float("nan"))
