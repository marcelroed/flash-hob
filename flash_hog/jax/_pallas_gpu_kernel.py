import dataclasses
from functools import partial

import jax
import jax.experimental.mosaic.gpu  # noqa: F401
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.experimental.pallas.triton as tlgpu
import jax.numpy as jnp
import jax.random as jrandom
from einops import einsum, rearrange
from jax import Array
from jax._src.cudnn.fused_attention_stablehlo import MaskType
from jax.extend import backend
from jaxtyping import PyTree

from flash_hog._utils.jax_utils import tree_rearrange

# TODO: Implement autotuning using Tokamax


@dataclasses.dataclass(frozen=True)
class TuningConfig:
    tile_q: int
    tile_k: int
    max_concurrent_steps: int
    epilogue_tile_n: int = 64
    grid_minor_dim: int = 0
    grid_tile_width: int = 1


# DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)
# DEFAULT_MASK_VALUE = -1e13
DEFAULT_MASK_VALUE = -jnp.inf


def maybe_causal_mask(A_ij, qi, kj, causal_mask):
    if causal_mask:
        mask = qi[:, None] >= kj[None, :]
        A_ij = jnp.where(mask, A_ij, DEFAULT_MASK_VALUE)
    return A_ij


# @flash_bwdbwd.def_vmap
# def _flash_bwdbwd0_vmap(axis_size, in_batched, Q, K, V, O, dO, ddQ, ddK, ddV, L, mask_type: MaskType, scale: float, config: TuningConfig):
#     return


def flash_bwdbwd0(
    Q: Array,
    K: Array,
    V: Array,
    O: Array,
    dO: Array,
    ddQ: Array,
    ddK: Array,
    ddV: Array,
    L: Array,
    mask_type: MaskType,
    scale: float | None,
    config: TuningConfig,
):
    """
    Flash Higher-Order-Gradients (Flash Hog) backward pass.
    This implementation is the most generic pallas implementation, and should work on both GPU and TPU.

    Args:
        Q: Query tensor of shape (batch_size, N_QUERIES, Q_HEADS, HIDDEN_DIM)
        K: Key tensor of shape (batch_size, N_KEYS, KV_HEADS, HIDDEN_DIM)
        V: Value tensor of shape (batch_size, N_KEYS, KV_HEADS, HIDDEN_DIM)
        O: Output tensor of shape (batch_size, N_QUERIES, Q_HEADS, HIDDEN_DIM)
        dO: Gradient of the output tensor with respect to the input tensor of shape (batch_size, N_QUERIES, Q_HEADS, HIDDEN_DIM)
        ddQ: Gradient of the query tensor with respect to the input tensor of shape (batch_size, N_QUERIES, Q_HEADS, HIDDEN_DIM)
        ddK: Gradient of the key tensor with respect to the input tensor of shape (batch_size, N_KEYS, KV_HEADS, HIDDEN_DIM)
        ddV: Gradient of the value tensor with respect to the input tensor of shape (batch_size, N_KEYS, KV_HEADS, HIDDEN_DIM)
    Returns:
        dQ2: Higher order gradients of Q (batch_size, N_QUERIES, Q_HEADS, HIDDEN_DIM)
        dK2: Higher order gradients of K (batch_size, N_KEYS, KV_HEADS, HIDDEN_DIM)
        dV2: Higher order gradients of V (batch_size, N_KEYS, KV_HEADS, HIDDEN_DIM)
        ddO: Higher order gradients of dO (batch_size, N_QUERIES, Q_HEADS, HIDDEN_DIM)
    """
    dtype = Q.dtype
    batch_size, n_queries, q_heads, hidden_dim = Q.shape
    n_keys, kv_heads = K.shape[1], K.shape[2]
    assert n_queries % config.tile_q == 0, f"{n_queries=} must be divisible by {config.tile_q=} for now"
    assert n_keys % config.tile_k == 0, f"{n_keys=} must be divisible by {config.tile_k=} for now"
    # Assert the remaining shapes are consistent
    # if batch_size == 8:
    #     breakpoint()
    assert K.shape == (batch_size, n_keys, kv_heads, hidden_dim)
    assert V.shape == (batch_size, n_keys, kv_heads, hidden_dim)
    assert O.shape == (batch_size, n_queries, q_heads, hidden_dim)
    assert dO.shape == (batch_size, n_queries, q_heads, hidden_dim)
    assert ddQ.shape == (batch_size, n_queries, q_heads, hidden_dim)
    assert ddK.shape == (batch_size, n_keys, kv_heads, hidden_dim)
    assert ddV.shape == (batch_size, n_keys, kv_heads, hidden_dim)
    assert L.shape == (batch_size, q_heads, n_queries)  # This weird shape comes from the cuDNN attention kernel
    assert config.tile_q > 0 and n_queries % config.tile_q == 0, "Haven't tested that tile_q supports non-divisible n_queries"
    assert q_heads % kv_heads == 0, "GQA is not yet implemented"
    assert mask_type in [MaskType.NO_MASK, MaskType.CAUSAL], "Only NO_MASK and CAUSAL are supported"
    assert n_keys >= n_queries, "n_keys must be greater than or equal to n_queries"

    if scale is None:
        scale = hidden_dim**-0.5

    group_size = q_heads // kv_heads
    (Q, O, dO, ddQ, L) = tree_rearrange((Q, O, dO, ddQ, L), "... (kv_heads group) final_dim -> ... kv_heads group final_dim", group=group_size)

    # Compute D before the kernel
    D = einsum(dO, O, "batch query kv_head group head_dim, batch query kv_head group head_dim -> batch query kv_head group")  # (batch_size, n_queries, kv_heads, group_size)

    # Allocate space for stage 1 outputs
    # dQ2 = jnp.empty_like(Q)
    # ddO = jnp.empty_like(O)
    # dD = jnp.empty_like(D)
    # B = jnp.empty_like(D)
    # Instead just define their shapes and dtypes
    dQ2_shape_dtype = jax.ShapeDtypeStruct(Q.shape, dtype, vma=jax.typeof(Q).vma)
    ddO_shape_dtype = jax.ShapeDtypeStruct(O.shape, dtype, vma=jax.typeof(O).vma)
    dD_shape_dtype = jax.ShapeDtypeStruct(D.shape, jnp.float32, vma=jax.typeof(D).vma)
    B_shape_dtype = jax.ShapeDtypeStruct(D.shape, jnp.float32, vma=jax.typeof(D).vma)

    dK2_shape_dtype = jax.ShapeDtypeStruct(K.shape, dtype, vma=jax.typeof(K).vma)
    dV2_shape_dtype = jax.ShapeDtypeStruct(V.shape, dtype, vma=jax.typeof(V).vma)

    def bwd_bwd_kernel_stage1(
        Q_ref,  # (tile_q, hidden_dim)
        K_ref,  # (n_keys, hidden_dim)
        V_ref,  # (n_keys, hidden_dim)
        D_ref,  # (tile_q,)
        dO_ref,  # (tile_q, hidden_dim)
        ddQ_ref,  # (tile_q, hidden_dim)
        ddK_ref,  # (n_keys, hidden_dim)
        ddV_ref,  # (n_keys, hidden_dim)
        L_ref,  # (tile_q,)
        # Outputs
        dQ2_ref,  # (tile_q, hidden_dim)
        ddO_ref,  # (tile_q, hidden_dim)
        dD_ref,  # (tile_q,)
        B_ref,  # (tile_q,)
    ):  # fmt: off
        Q_i = Q_ref[:, :]
        L_i = L_ref[:]
        ddQ_i = ddQ_ref[:, :]
        q_tile_index = pl.program_id(0)
        # qslice = pl.ds(q_tile_index * config.tile_q, config.tile_q).
        q_indices = q_tile_index * config.tile_q + jnp.arange(config.tile_q)

        dO_i = dO_ref[:, :]

        D_i = D_ref[:]

        def dd_loop_body(k_tile_index, carry, causal_mask=False):
            # 2-pass fusion: accumulate dD and R in one sweep, then close B in closed form
            # (was a separate K pass).
            #
            # B = sum_k dP'_ik P_ik, with dP'_ik = dPa_ik - dP_ik*dD_i - ddS_ik*D_i + dP_ik*ddS_ik.
            # The per-row scalars dD_i, D_i factor out of the sum:
            #   B = r1 - dD_i*r2 - D_i*dD_i + r3,  r1=sum dPa*P, r2=sum dP*P, r3=sum dP*ddS*P.
            # Two simplifications:
            #   - r2 = sum_k dP_ik P_ik = dO_i . (sum_k P_ik V_k) = dO_i . O_i = D_i (precomputed).
            #   - r1 and r3 share coefficient +1, so accumulate them together as R = r1 + r3.
            # Hence B = R - 2*D_i*dD_i.
            dD_i_acc, R_acc = carry
            kslice = pl.ds(k_tile_index * config.tile_k, config.tile_k)
            k_indices = kslice.start + jnp.arange(kslice.size)

            K_j = K_ref[kslice, :]
            V_j = V_ref[kslice, :]
            ddK_j = ddK_ref[kslice, :]
            ddV_j = ddV_ref[kslice, :]

            S_ij = pl.dot(Q_i, K_j.T) * scale
            S_ij = maybe_causal_mask(S_ij, q_indices, k_indices, causal_mask)

            P_ij = jnp.exp(S_ij - L_i[:, None])

            dP_ij = pl.dot(dO_i, V_j.T)
            ddS_ij = (pl.dot(ddQ_i, K_j.T) + pl.dot(Q_i, ddK_j.T)) * scale
            dPa_ij = pl.dot(dO_i, ddV_j.T)

            dD_i_acc += jnp.sum(ddS_ij * P_ij, axis=1)
            R_acc += jnp.sum((dPa_ij + dP_ij * ddS_ij) * P_ij, axis=1)
            return (dD_i_acc, R_acc)

        zeros = (jnp.zeros((config.tile_q,), dtype=jnp.float32),) * 2
        if mask_type == MaskType.CAUSAL:
            num_unmasked_k_tiles = pl.cdiv(q_tile_index * config.tile_q - config.tile_k + 1, config.tile_k)
            num_required_k_tiles = jnp.minimum(pl.cdiv((q_tile_index + 1) * config.tile_q - 1, config.tile_k) + 1, pl.cdiv(n_keys, config.tile_k))
            carry = jax.lax.fori_loop(0, num_unmasked_k_tiles, dd_loop_body, zeros)
            carry = jax.lax.fori_loop(num_unmasked_k_tiles, num_required_k_tiles, partial(dd_loop_body, causal_mask=True), carry)
        else:
            carry = jax.lax.fori_loop(0, pl.cdiv(n_keys, config.tile_k), dd_loop_body, zeros)
        dD_i, R_i = carry
        dD_ref[:] = dD_i

        # B in closed form from the fused pass (uses the r2 = D_i identity).
        B_i = R_i - 2.0 * D_i * dD_i
        B_ref[:] = B_i

        def dQ2_ddO_loop_body(k_tile_index, carry, causal_mask=False):
            dQ2_i_acc, ddO_i_acc = carry
            kslice = pl.ds(k_tile_index * config.tile_k, config.tile_k)
            k_indices = k_tile_index * config.tile_k + jnp.arange(config.tile_k)

            K_j = K_ref[kslice, :]
            V_j = V_ref[kslice, :]
            ddK_j = ddK_ref[kslice, :]
            ddV_j = ddV_ref[kslice, :]

            # Compute attention scores
            S_ij = pl.dot(Q_i, K_j.T) * scale
            S_ij = maybe_causal_mask(S_ij, q_indices, k_indices, causal_mask)
            P_ij = jnp.exp(S_ij - L_i[:, None])

            dP_ij = pl.dot(dO_i, V_j.T)

            ddS_ij = (pl.dot(ddQ_i, K_j.T) + pl.dot(Q_i, ddK_j.T)) * scale

            # ddS centered by its P-weighted row reduction dD_i; shared by dP2 and ddP.
            ddS_centered_ij = ddS_ij - dD_i[:, None]
            scaled_P_ij = scale * P_ij

            dP2_ij = pl.dot(dO_i, ddV_j.T) - ddS_ij * D_i[:, None] + dP_ij * ddS_centered_ij
            dS2_ij = scaled_P_ij * (dP2_ij - B_i[:, None])

            dS_ij = scaled_P_ij * (dP_ij - D_i[:, None])

            dQ2_i_acc += pl.dot(dS_ij.astype(ddK_j.dtype), ddK_j)
            dQ2_i_acc += pl.dot(dS2_ij.astype(K_j.dtype), K_j)

            ddP_ij = P_ij * ddS_centered_ij
            ddO_i_acc += pl.dot(ddP_ij.astype(V_j.dtype), V_j)
            ddO_i_acc += pl.dot(P_ij.astype(ddV_j.dtype), ddV_j)
            return (dQ2_i_acc, ddO_i_acc)

        if mask_type == MaskType.CAUSAL:
            num_unmasked_k_tiles = pl.cdiv(q_tile_index * config.tile_q - config.tile_k + 1, config.tile_k)

            num_required_k_tiles = jnp.minimum(pl.cdiv((q_tile_index + 1) * config.tile_q - 1, config.tile_k) + 1, pl.cdiv(n_keys, config.tile_k))
            dQ2_i, ddO_i = jax.lax.fori_loop(
                0,
                num_unmasked_k_tiles,
                dQ2_ddO_loop_body,
                (jnp.zeros((config.tile_q, hidden_dim), dtype=jnp.float32), jnp.zeros((config.tile_q, hidden_dim), dtype=jnp.float32)),
            )
            dQ2_i, ddO_i = jax.lax.fori_loop(num_unmasked_k_tiles, num_required_k_tiles, partial(dQ2_ddO_loop_body, causal_mask=True), (dQ2_i, ddO_i))
        else:
            dQ2_i, ddO_i = jax.lax.fori_loop(
                0,
                pl.cdiv(n_keys, config.tile_k),
                dQ2_ddO_loop_body,
                (jnp.zeros((config.tile_q, hidden_dim), dtype=jnp.float32), jnp.zeros((config.tile_q, hidden_dim), dtype=jnp.float32)),
            )
        dQ2_ref[:] = dQ2_i.astype(dQ2_ref.dtype)
        ddO_ref[:] = ddO_i.astype(ddO_ref.dtype)

    bwd_bwd_stage1 = pl.pallas_call(
        bwd_bwd_kernel_stage1,
        out_shape=(dQ2_shape_dtype, ddO_shape_dtype, dD_shape_dtype, B_shape_dtype),
        grid=(pl.cdiv(n_queries, config.tile_q), batch_size, q_heads),
        in_specs=[
            pl.BlockSpec(
                (None, config.tile_q, None, None, hidden_dim), lambda q, b, h: (b, q, h // group_size, h % group_size, 0)
            ),  # Q, None indicates to squeeze the index (assuming sliced)
            pl.BlockSpec((None, n_keys, None, hidden_dim), lambda _q, b, h: (b, 0, h // group_size, 0)),  # K
            pl.BlockSpec((None, n_keys, None, hidden_dim), lambda _q, b, h: (b, 0, h // group_size, 0)),  # V
            pl.BlockSpec((None, config.tile_q, None, None), lambda q, b, h: (b, q, h // group_size, h % group_size)),  # D
            pl.BlockSpec((None, config.tile_q, None, None, hidden_dim), lambda q, b, h: (b, q, h // group_size, h % group_size, 0)),  # dO
            pl.BlockSpec((None, config.tile_q, None, None, hidden_dim), lambda q, b, h: (b, q, h // group_size, h % group_size, 0)),  # ddQ
            pl.BlockSpec((None, n_keys, None, hidden_dim), lambda _q, b, h: (b, 0, h // group_size, 0)),  # ddK
            pl.BlockSpec((None, n_keys, None, hidden_dim), lambda _q, b, h: (b, 0, h // group_size, 0)),  # ddV
            pl.BlockSpec((None, None, None, config.tile_q), lambda q, b, h: (b, h // group_size, h % group_size, q)),  # L
        ],
        out_specs=[
            pl.BlockSpec((None, config.tile_q, None, None, hidden_dim), lambda q, b, h: (b, q, h // group_size, h % group_size, 0)),  # dQ2
            pl.BlockSpec((None, config.tile_q, None, None, hidden_dim), lambda q, b, h: (b, q, h // group_size, h % group_size, 0)),  # ddO
            pl.BlockSpec((None, config.tile_q, None, None), lambda q, b, h: (b, q, h // group_size, h % group_size)),  # dD
            pl.BlockSpec((None, config.tile_q, None, None), lambda q, b, h: (b, q, h // group_size, h % group_size)),  # B
        ],
        compiler_params=tlgpu.CompilerParams(num_stages=2, num_warps=4),
        name="hog_bwd_bwd_kernel_stage1",
    )

    dQ2, ddO, dD, B = bwd_bwd_stage1(Q, K, V, D, dO, ddQ, ddK, ddV, L)

    def bwd_bwd_kernel_stage2(
        # Inputs
        Q_ref,
        K_ref,
        V_ref,
        D_ref,
        dO_ref,
        ddQ_ref,
        ddK_ref,
        ddV_ref,
        L_ref,
        dD_ref,
        B_ref,
        # Outputs
        dK2_ref,
        dV2_ref,
    ):
        K_j = K_ref[...]
        V_j = V_ref[...]
        ddK_j = ddK_ref[...]
        ddV_j = ddV_ref[...]
        k_tile_index = pl.program_id(0)
        k_indices = k_tile_index * config.tile_k + jnp.arange(config.tile_k)

        def dk2_dv2_loop_body(qi, carry, causal_mask=False):
            def inner_loop_body(group_index, carry):
                qslice = pl.ds(qi * config.tile_q, config.tile_q)
                q_indices = qi * config.tile_q + jnp.arange(config.tile_q)
                dV2_j_acc, dK2_j_acc = carry

                Q_i = Q_ref[qslice, group_index, :]
                L_i = L_ref[group_index, qslice]
                dO_i = dO_ref[qslice, group_index, :]
                ddQ_i = ddQ_ref[qslice, group_index, :]
                dD_i = dD_ref[qslice, group_index]
                B_i = B_ref[qslice, group_index]
                D_i = D_ref[qslice, group_index]

                S_ij = pl.dot(Q_i, K_j.T) * scale
                S_ij = maybe_causal_mask(S_ij, q_indices, k_indices, causal_mask)
                P_ij = jnp.exp(S_ij - L_i[:, None])

                dP_ij = pl.dot(dO_i, V_j.T)

                ddS_ij = (pl.dot(ddQ_i, K_j.T) + pl.dot(Q_i, ddK_j.T)) * scale

                # ddS centered by its P-weighted row reduction dD_i; shared by dP2 and ddP.
                ddS_centered_ij = ddS_ij - dD_i[:, None]
                scaled_P_ij = scale * P_ij

                dP2_ij = pl.dot(dO_i, ddV_j.T) - ddS_ij * D_i[:, None] + dP_ij * ddS_centered_ij

                dS2_ij = scaled_P_ij * (dP2_ij - B_i[:, None])

                dS_ij = scaled_P_ij * (dP_ij - D_i[:, None])

                ddP_ij = P_ij * ddS_centered_ij
                dV2_j_acc += pl.dot(ddP_ij.astype(dO_i.dtype).T, dO_i)

                dK2_j_acc += pl.dot(dS_ij.astype(ddQ_i.dtype).T, ddQ_i)
                dK2_j_acc += pl.dot(dS2_ij.astype(Q_i.dtype).T, Q_i)

                return (dV2_j_acc, dK2_j_acc)

            return jax.lax.fori_loop(0, group_size, inner_loop_body, carry)

        if mask_type == MaskType.CAUSAL:
            start_masked_q_tile = jnp.maximum((k_tile_index * config.tile_k + 1) // config.tile_q - 1, 0)
            end_masked_q_tile = pl.cdiv((k_tile_index + 1) * config.tile_k - 1, config.tile_q)
            num_required_q_tiles = pl.cdiv(n_queries, config.tile_q)
            dV2_j, dK2_j = jax.lax.fori_loop(
                start_masked_q_tile,
                end_masked_q_tile,
                partial(dk2_dv2_loop_body, causal_mask=True),
                (jnp.zeros((config.tile_k, hidden_dim), dtype=jnp.float32), jnp.zeros((config.tile_k, hidden_dim), dtype=jnp.float32)),
            )
            dV2_j, dK2_j = jax.lax.fori_loop(end_masked_q_tile, num_required_q_tiles, dk2_dv2_loop_body, (dV2_j, dK2_j))
        else:
            dV2_j, dK2_j = jax.lax.fori_loop(
                0,
                pl.cdiv(n_queries, config.tile_q),
                dk2_dv2_loop_body,
                (jnp.zeros((config.tile_k, hidden_dim), dtype=jnp.float32), jnp.zeros((config.tile_k, hidden_dim), dtype=jnp.float32)),
            )
        dK2_ref[:] = dK2_j.astype(dK2_ref.dtype)
        dV2_ref[:] = dV2_j.astype(dV2_ref.dtype)

    bwd_bwd_stage2 = pl.pallas_call(
        bwd_bwd_kernel_stage2,
        out_shape=(dK2_shape_dtype, dV2_shape_dtype),
        grid=(pl.cdiv(n_keys, config.tile_k), batch_size, kv_heads),
        in_specs=[
            pl.BlockSpec((None, n_queries, None, group_size, hidden_dim), lambda _k, b, h: (b, 0, h, 0, 0)),  # Q
            pl.BlockSpec((None, config.tile_k, None, hidden_dim), lambda k, b, h: (b, k, h, 0)),  # K
            pl.BlockSpec((None, config.tile_k, None, hidden_dim), lambda k, b, h: (b, k, h, 0)),  # V
            pl.BlockSpec((None, n_queries, None, group_size), lambda _k, b, h: (b, 0, h, 0)),  # D
            pl.BlockSpec((None, n_queries, None, group_size, hidden_dim), lambda _k, b, h: (b, 0, h, 0, 0)),  # dO
            pl.BlockSpec((None, n_queries, None, group_size, hidden_dim), lambda _k, b, h: (b, 0, h, 0, 0)),  # ddQ
            pl.BlockSpec((None, config.tile_k, None, hidden_dim), lambda k, b, h: (b, k, h, 0)),  # ddK
            pl.BlockSpec((None, config.tile_k, None, hidden_dim), lambda k, b, h: (b, k, h, 0)),  # ddV
            pl.BlockSpec((None, None, group_size, n_queries), lambda _k, b, h: (b, h, 0, 0)),  # L (weird shape from cuDNN)
            pl.BlockSpec((None, n_queries, None, group_size), lambda _k, b, h: (b, 0, h, 0)),  # dD
            pl.BlockSpec((None, n_queries, None, group_size), lambda _k, b, h: (b, 0, h, 0)),  # B
        ],
        out_specs=[
            pl.BlockSpec((None, config.tile_k, None, hidden_dim), lambda k, b, h: (b, k, h, 0)),  # dK2
            pl.BlockSpec((None, config.tile_k, None, hidden_dim), lambda k, b, h: (b, k, h, 0)),  # dV2
        ],
        compiler_params=tlgpu.CompilerParams(num_stages=2, num_warps=4),
        name="hog_bwd_bwd_kernel_stage2",
    )

    dK2, dV2 = bwd_bwd_stage2(Q, K, V, D, dO, ddQ, ddK, ddV, L, dD, B)

    dQ2, ddO = tree_rearrange((dQ2, ddO), "... kv_heads group_size final_dim -> ... (kv_heads group_size) final_dim", group_size=group_size)

    return dQ2, dK2, dV2, ddO
