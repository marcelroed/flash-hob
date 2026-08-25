"""ThunderKittens (Hopper/SM90) double-backward for causal attention.

Default path is Pallas; opt in to TK:

    pip install 'flash-hog[tk]'        # CUDA build tools from PyPI

    from flash_hog.jax import _tk_gpu as tk
    tk.enable()    # JIT-builds the plugin on first use (cached); TK kernels live
    tk.disable()   # back to Pallas

enable() swaps flash-hog's double-backward rule for the TK kernels where
supported() (causal, head_dim 64, seq % 128 == 0, no GQA, Hopper) and falls back
to the Pallas rule per-call otherwise. FLASH_HOG_TK_LIB overrides the plugin path.
"""

from __future__ import annotations

import ctypes
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

_LIB_ENV = "FLASH_HOG_TK_LIB"
_FFI_TARGET = "tk_bwdbwd"


def _preload_cuda_libs() -> None:
    # plugin is built with `-cudart none`: resolve driver/cudart symbols from the
    # process by loading them RTLD_GLOBAL first
    from flash_hog.jax import _tk_build

    try:
        ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    for libdir in _tk_build.nvidia_wheel_paths("lib"):
        for so in sorted(libdir.glob("libcudart.so.*")):
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue


@functools.cache
def _lib() -> ctypes.CDLL | None:
    from flash_hog.jax import _tk_build

    path = os.environ.get(_LIB_ENV) or _tk_build.cached_so()
    if path is None or not os.path.exists(path):
        return None
    _preload_cuda_libs()
    lib = ctypes.CDLL(str(path))
    jax.ffi.register_ffi_target(_FFI_TARGET, jax.ffi.pycapsule(lib.TkBwdBwd), platform="CUDA")
    return lib


@functools.cache
def _on_hopper() -> bool:
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        return False
    return all(getattr(d, "compute_capability", "") == "9.0" for d in devices)


def supported(*, is_causal: bool, seq_len: int, head_dim: int,
              num_q_heads: int, num_kv_heads: int) -> bool:
    """True iff the TK kernels can serve this shape on this machine."""
    return (
        is_causal
        and head_dim == 64
        and seq_len % 128 == 0
        and num_q_heads == num_kv_heads   # no GQA
        and _on_hopper()
        and _lib() is not None
    )


def flash_bwdbwd(*, Q, K, V, O, dO, ddQ, ddK, ddV, L, scale: float):
    """Causal-attention double-backward. Arguments are BTNH; returns dQ2, dK2, dV2, ddO."""
    B, T, N, Hd = Q.shape

    def to_bhtd(x):
        return jnp.transpose(x, (0, 2, 1, 3))

    Qb, Kb, Vb, dOb, ddQb, ddKb, ddVb = (
        to_bhtd(x).astype(jnp.bfloat16) for x in (Q, K, V, dO, ddQ, ddK, ddV)
    )
    D = jnp.sum(to_bhtd(dO).astype(jnp.float32) * to_bhtd(O).astype(jnp.float32), axis=-1)
    Lf = L.reshape(B, N, T).astype(jnp.float32)

    outs = jax.ffi.ffi_call(
        _FFI_TARGET,
        [jax.ShapeDtypeStruct((B, N, T, Hd), jnp.bfloat16)] * 4    # dQ2, ddO, dK2, dV2
        + [jax.ShapeDtypeStruct((B, N, T), jnp.float32)] * 2,      # dD, B (scratch)
    )(Qb, Kb, Vb, dOb, ddQb, ddKb, ddVb, Lf, D, scale=np.float32(scale))

    dQ2, ddO, dK2, dV2 = (to_bhtd(x).astype(Q.dtype) for x in outs[:4])
    return dQ2, dK2, dV2, ddO


_original_rule = None   # non-None iff the TK rule is installed


def enable() -> None:
    """Opt in: route the double-backward through the TK kernels (per-call fallback)."""
    global _original_rule
    if _lib() is None:
        from flash_hog.jax import _tk_build

        _tk_build.build()
        _lib.cache_clear()
        if _lib() is None:
            raise RuntimeError("TK plugin built but failed to load")
    if _original_rule is not None:
        return

    from jax._src.cudnn.fused_attention_stablehlo import MaskType

    from flash_hog.jax import _attention_impl as impl

    pallas_rule = impl.dot_product_attention_bwd_rule_bwd_rule

    def tk_rule(mask_type, scale, res, g):
        query, key, value, out, activation, dO = res
        if not supported(
            is_causal=(mask_type == MaskType.CAUSAL),
            seq_len=query.shape[1],
            head_dim=query.shape[3],
            num_q_heads=query.shape[2],
            num_kv_heads=key.shape[2],
        ):
            return pallas_rule(mask_type, scale, res, g)
        ddQ, ddK, ddV = g
        dQ2, dK2, dV2, ddO = flash_bwdbwd(
            Q=query, K=key, V=value, O=out, dO=dO,
            ddQ=ddQ, ddK=ddK, ddV=ddV, L=activation, scale=scale,
        )
        return (dQ2, dK2, dV2, None, None), ddO

    _original_rule = pallas_rule
    impl.dot_product_attention_bwd_rule_bwd_rule = tk_rule
    jax.clear_caches()   # traced double-backwards captured the old rule


def disable() -> None:
    """Restore the stock Pallas double-backward rule."""
    global _original_rule
    if _original_rule is None:
        return
    from flash_hog.jax import _attention_impl as impl

    impl.dot_product_attention_bwd_rule_bwd_rule = _original_rule
    _original_rule = None
    jax.clear_caches()


def is_enabled() -> bool:
    return _original_rule is not None
