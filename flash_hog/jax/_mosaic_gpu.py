"""Pallas:MGPU (Mosaic GPU, Hopper/SM90) double-backward for causal attention.

The default flash-hog path is the portable Pallas kernel; this is an opt-in
JAX-native Mosaic GPU implementation of the same op:

    from flash_hog.jax import _mosaic_gpu as mosaic
    mosaic.enable()     # route the double-backward through the Mosaic kernels
    mosaic.disable()    # back to the default Pallas path

enable() swaps flash-hog's double-backward rule for the Mosaic kernels where
supported() (causal, head_dim 64, seq % 128 == 0, no GQA, Hopper) and falls back
to default when not supported.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


@functools.cache
def _kernel():
    """The Mosaic bwd-of-bwd entry point, or None if Mosaic GPU is unavailable."""
    try:
        from flash_hog.jax._mosaic_gpu_kernel import flash_bwdbwd_mosaic
    except Exception:
        return None
    return flash_bwdbwd_mosaic


@functools.cache
def _on_hopper() -> bool:
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        return False
    return all(getattr(d, "compute_capability", "") == "9.0" for d in devices)


def supported(*, is_causal: bool, seq_len: int, head_dim: int, num_q_heads: int, num_kv_heads: int) -> bool:
    """True iff the Mosaic kernels can serve this shape on this machine."""
    return (
        is_causal
        and head_dim == 64
        and seq_len % 128 == 0
        and num_q_heads == num_kv_heads  # no GQA
        and _on_hopper()
        and _kernel() is not None
    )


# Tuned tile-size configs for the two Mosaic stages (set via set_tuned_tiles()).
# Empty dicts mean "use the kernel module's compiled-in defaults".
_TUNED_STAGE1_CFG: dict = {}
_TUNED_STAGE2_CFG: dict = {}


def set_tuned_tiles(stage1_cfg: dict | None = None, stage2_cfg: dict | None = None) -> None:
    """Install tuned tile-size overrides for the Mosaic kernels.

    stage1_cfg keys: BQ, BK, K_PIPELINE_DEPTH
    stage2_cfg keys: BQ2, BK2, Q_PIPELINE_DEPTH
    Pass None/empty to clear and fall back to the kernel defaults.
    """
    global _TUNED_STAGE1_CFG, _TUNED_STAGE2_CFG
    _TUNED_STAGE1_CFG = dict(stage1_cfg or {})
    _TUNED_STAGE2_CFG = dict(stage2_cfg or {})
    jax.clear_caches()


def flash_bwdbwd(*, Q, K, V, O, dO, ddQ, ddK, ddV, L, scale: float):
    """Causal-attention double-backward. Arguments are BTNH; returns dQ2, dK2, dV2, ddO."""
    kernel = _kernel()
    assert kernel is not None, "Mosaic GPU kernel unavailable"
    B, T, N, Hd = Q.shape

    def to_bhtd(x):
        return jnp.transpose(x, (0, 2, 1, 3))

    Qb, Kb, Vb, Ob, dOb, ddQb, ddKb, ddVb = (to_bhtd(x).astype(jnp.bfloat16) for x in (Q, K, V, O, dO, ddQ, ddK, ddV))
    Lf = L.reshape(B, N, T).astype(jnp.float32)

    dQ2, dK2, dV2, ddO = kernel(
        Qb, Kb, Vb, Ob,
        dOb, ddQb, ddKb, ddVb,
        Lf,
        scale=scale,
        stage1_cfg=_TUNED_STAGE1_CFG, stage2_cfg=_TUNED_STAGE2_CFG,
    )  # fmt: skip
    return tuple(to_bhtd(x).astype(Q.dtype) for x in (dQ2, dK2, dV2, ddO))


_original_rule = None


def enable() -> None:
    """Opt in: route the double-backward through the Mosaic kernels (per-call fallback)."""
    global _original_rule
    if _kernel() is None:
        raise RuntimeError("Mosaic GPU kernel unavailable (needs jax mosaic_gpu on Hopper)")
    if _original_rule is not None:
        return

    from jax._src.cudnn.fused_attention_stablehlo import MaskType

    from flash_hog.jax import _attention_impl as impl

    default_rule = impl.dot_product_attention_bwd_rule_bwd_rule

    def mosaic_rule(mask_type, scale, res, g):
        query, key, value, out, activation, dO = res
        if not supported(
            is_causal=(mask_type == MaskType.CAUSAL),
            seq_len=query.shape[1],
            head_dim=query.shape[3],
            num_q_heads=query.shape[2],
            num_kv_heads=key.shape[2],
        ):
            return default_rule(mask_type, scale, res, g)
        ddQ, ddK, ddV = g
        dQ2, dK2, dV2, ddO = flash_bwdbwd(
            Q=query, K=key, V=value, O=out, dO=dO,
            ddQ=ddQ, ddK=ddK, ddV=ddV, L=activation, scale=scale,
        )  # fmt: skip
        return (dQ2, dK2, dV2, None, None), ddO

    _original_rule = default_rule
    impl.dot_product_attention_bwd_rule_bwd_rule = mosaic_rule
    jax.clear_caches()


def disable() -> None:
    """Restore the default Pallas double-backward rule."""
    global _original_rule
    if _original_rule is None:
        return
    from flash_hog.jax import _attention_impl as impl

    impl.dot_product_attention_bwd_rule_bwd_rule = _original_rule
    _original_rule = None
    jax.clear_caches()


def is_enabled() -> bool:
    return _original_rule is not None
