"""Thin JAX wrappers around the FlashAttention-4 CuTe-DSL SM90 kernels.

The kernels are driven entirely from JAX via ``cutlass.jax.cutlass_call`` -- JAX
arrays go straight into the CuTe kernel and XLA owns the CUDA stream and output
buffers. No torch on the data path (see ``fa4_loader`` for the import shim).

Currently wraps the forward kernel for the dense, non-causal, MHA bf16 case
(head_dim == head_dim_v, one KV head per Q head), which is what the benchmark
exercises. Tile/config selection mirrors flash_attn.cute.interface for SM90.
"""
from functools import lru_cache

import jax
import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax

import fa4_loader

_FA4 = fa4_loader.load()
FlashAttentionForwardSm90 = _FA4.fwd.FlashAttentionForwardSm90
FlashAttentionBackwardSm90 = _FA4.bwd.FlashAttentionBackwardSm90
FlashAttentionBackwardPreprocess = _FA4.pre.FlashAttentionBackwardPreprocess
FlashAttentionBackwardPostprocess = _FA4.post.FlashAttentionBackwardPostprocess

_DTYPE_MAP = {"bfloat16": cutlass.BFloat16, "float16": cutlass.Float16}


def _tile_size_fwd_sm90(head_dim):
    # Mirrors flash_attn.cute.interface._tile_size_fwd_sm90 for the dense
    # non-causal / non-local case.
    if head_dim <= 64:
        return dict(tile_m=192, tile_n=128, mma_pv_is_rs=True, intra_wg_overlap=True)
    elif head_dim <= 128:
        return dict(tile_m=128, tile_n=128, mma_pv_is_rs=True, intra_wg_overlap=True)
    raise NotImplementedError(f"head_dim={head_dim} not wired up in this benchmark wrapper")


@lru_cache(maxsize=None)
def _build_fwd(dtype_name, head_dim, head_dim_v, num_threads=384):
    cfg = _tile_size_fwd_sm90(head_dim)
    fa_fwd = FlashAttentionForwardSm90(
        _DTYPE_MAP[dtype_name],
        head_dim,
        head_dim_v,
        1,                      # qhead_per_kvhead (MHA)
        is_causal=False,
        is_local=False,
        pack_gqa=False,
        tile_m=cfg["tile_m"],
        tile_n=cfg["tile_n"],
        num_stages=2,
        num_threads=num_threads,
        Q_in_regs=False,
        intra_wg_overlap=cfg["intra_wg_overlap"],
        mma_pv_is_rs=cfg["mma_pv_is_rs"],
        mask_mod=None,
        score_mod=None,
        has_aux_tensors=False,
        q_subtile_factor=1,
        paged_kv_non_tma=False,
    )
    return fa_fwd


def fa4_fwd(q, k, v, scale, *, return_lse=True):
    """Forward attention. q,k,v: (B, S, H, D) bf16. Returns (out, lse) where
    out is (B, S, H, Dv) bf16 and lse is (B, H, S) f32."""
    b, s, h, d = q.shape
    dv = v.shape[-1]
    dtype_name = str(q.dtype)
    fa_fwd = _build_fwd(dtype_name, d, dv)
    scale = float(scale)

    @cute.jit
    def launch(stream, mQ, mK, mV, mO, mLSE):
        fa_fwd(mQ, mK, mV, mO, mLSE, scale, stream=stream)

    out_spec = jax.ShapeDtypeStruct((b, s, h, dv), q.dtype)
    lse_spec = jax.ShapeDtypeStruct((b, h, s), jax.numpy.float32)
    out, lse = cjax.cutlass_call(
        launch,
        output_shape_dtype=(out_spec, lse_spec),
        use_static_tensors=True,
    )(q, k, v)
    return (out, lse) if return_lse else out


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------
# SM90 backward tile/config for the dense non-causal MHA case, mirroring
# flash_attn.cute.interface._tile_size_bwd_sm90 (head_dim <= 64 branch).
def _bwd_cfg_sm90(head_dim):
    if head_dim <= 64:
        return dict(
            tile_m=128, tile_n=128, Q_stage=2, dO_stage=2, PdS_stage=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=2, num_wg=2,
            dQ_single_wg=False,
        )
    raise NotImplementedError(f"head_dim={head_dim} not wired up in this benchmark wrapper")


@lru_cache(maxsize=None)
def _build_bwd(dtype_name, head_dim, head_dim_v):
    cfg = _bwd_cfg_sm90(head_dim)
    dtype = _DTYPE_MAP[dtype_name]
    num_threads = (cfg["num_wg"] + 1) * 128
    pre = FlashAttentionBackwardPreprocess(
        dtype, head_dim, head_dim_v, cfg["tile_m"],
        use_padded_offsets=True, nheads_major=False, pack_gqa=False,
        qhead_per_kvhead=1, nheads_kv=1,
    )
    bwd = FlashAttentionBackwardSm90(
        dtype, head_dim, head_dim_v, 1, False,  # qhead_per_kvhead=1, causal=False
        is_local=False, deterministic=False,
        tile_m=cfg["tile_m"], tile_n=cfg["tile_n"],
        Q_stage=cfg["Q_stage"], dO_stage=cfg["dO_stage"], PdS_stage=cfg["PdS_stage"],
        SdP_swapAB=cfg["SdP_swapAB"], dKV_swapAB=cfg["dKV_swapAB"], dQ_swapAB=cfg["dQ_swapAB"],
        AtomLayoutMSdP=cfg["AtomLayoutMSdP"], AtomLayoutNdKV=cfg["AtomLayoutNdKV"],
        AtomLayoutMdQ=cfg["AtomLayoutMdQ"], num_threads=num_threads, V_in_regs=False,
        score_mod=None, score_mod_bwd=None, mask_mod=None, has_aux_tensors=False,
        q_subtile_factor=2, dQ_single_wg=cfg["dQ_single_wg"],
    )
    num_threads_post = 128 if cfg["dQ_single_wg"] else cfg["num_wg"] * 128
    post = FlashAttentionBackwardPostprocess(
        dtype, head_dim, 90, tile_m=cfg["tile_m"], num_threads=num_threads_post,
        AtomLayoutMdQ=cfg["AtomLayoutMdQ"], dQ_swapAB=cfg["dQ_swapAB"],
        use_2cta_instrs=False, cluster_size=1,
    )
    return pre, bwd, post, cfg


def fa4_bwd(q, k, v, out, dout, lse, scale):
    """Backward attention. q,k,v,out,dout: (B,S,H,D) bf16; lse: (B,H,S) f32.
    Returns (dq, dk, dv) bf16. Runs FA4's 3-kernel pipeline:
    preprocess (dpsum, lse*log2e, zero dq_accum) -> main bwd (dk, dv, accumulate
    dq_accum in fp32) -> postprocess (dq_accum f32 -> dq bf16)."""
    b, s, h, d = q.shape
    dv = v.shape[-1]
    f32 = jax.numpy.float32
    pre, bwd, post, cfg = _build_bwd(str(q.dtype), d, dv)
    scale = float(scale)
    tile_m = cfg["tile_m"]
    s_round = -(-s // tile_m) * tile_m            # ceil(s/tile_m)*tile_m
    d_round = -(-d // 32) * 32                     # ceil(d/32)*32

    # --- preprocess: out, dout, lse -> dpsum, lse_log2, dq_accum(zeroed) ---
    @cute.jit
    def launch_pre(stream, mO, mdO, mLSE, mPdPsum, mLSElog2, mdQaccum):
        pre(mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum, None, None, None, None, None,
            scale, stream=stream)

    dpsum, lse_log2, dq_accum = cjax.cutlass_call(
        launch_pre,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((b, h, s_round), f32),          # dpsum
            jax.ShapeDtypeStruct((b, h, s_round), f32),          # lse_log2
            jax.ShapeDtypeStruct((b, h, s_round * d_round), f32),  # dq_accum
        ),
        use_static_tensors=True,
    )(out, dout, lse)

    # --- main bwd: dk, dv (fresh) + accumulate dq_accum (alias input->output) ---
    @cute.jit
    def launch_bwd(stream, mQ, mK, mV, mdO, mLSElog2, mPdPsum, mdQaccum, mdK, mdV):
        # mdQaccum is the zeroed dq_accum buffer, aliased to the dq_accum output
        # (input_output_aliases below); the kernel atomic-adds into it.
        bwd(mQ, mK, mV, mdO, mLSElog2, mPdPsum, mdQaccum, mdK, mdV, scale, stream=stream)

    dk, dv, dq_accum2 = cjax.cutlass_call(
        launch_bwd,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((b, s, h, d), q.dtype),         # dk
            jax.ShapeDtypeStruct((b, s, h, dv), q.dtype),        # dv
            jax.ShapeDtypeStruct((b, h, s_round * d_round), f32),  # dq_accum (accumulated)
        ),
        input_output_aliases={6: 2},  # dq_accum input (idx 6) -> dq_accum output (idx 2)
        use_static_tensors=True,
    )(q, k, v, dout, lse_log2, dpsum, dq_accum)

    # --- postprocess: dq_accum f32 -> dq bf16 ---
    @cute.jit
    def launch_post(stream, mdQaccum, mdQ):
        # cutlass 4.6.0.dev0 tags cutlass_call inputs as `generic` address space;
        # the postprocess loads dq_accum with a cp.async (G2S) copy whose verifier
        # requires a `global` source. Re-tag the (genuinely global) input gmem.
        mdQaccum = fa4_loader._as_gmem_tensor(mdQaccum)
        post(mdQaccum, mdQ, scale, None, None, stream=stream)

    (dq,) = cjax.cutlass_call(
        launch_post,
        output_shape_dtype=(jax.ShapeDtypeStruct((b, s, h, d), q.dtype),),
        use_static_tensors=True,
    )(dq_accum2)
    return dq, dk, dv
