"""
Implementations for the functions used in flash_hog.jax.attention.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core
from jax._src.cudnn.fused_attention_stablehlo import (
    AttentionLayout,
    MaskType,
    _dot_product_attention_bwd_lower,
    _dot_product_attention_bwd_p,
    _dot_product_attention_bwd_p_wrapper,
    _dot_product_attention_bwd_rule,
    _dot_product_attention_fwd_p,
    _dot_product_attention_fwd_p_wrapper,
    get_max_seg_per_batch,
)
from jax._src.cudnn.fused_attention_stablehlo import (
    dot_product_attention as cuda_dot_product_attention,
)
from jax.experimental.custom_partitioning import ArrayMapping, SdyShardingRule
from jax.experimental.layout import Layout, with_layout_constraint

from flash_hog.jax._pallas_gpu import TuningConfig, flash_bwdbwd

# ---------------------------------------------------------------------------
# Monkey-patch: fix the cuDNN backward sharding rule
# ---------------------------------------------------------------------------
# JAX's _bwd_shardy_rule creates independent dimension names for each operand
# (e.g. '…0', '…1', '…2', …) so the Shardy partitioner cannot infer that they
# share batch/head/seq dimensions.  This causes SPMD sharding mismatches when
# the backward is called inside a sharded computation.
#
# The fix below replaces the sharding rule with one that uses *shared* factor
# names so Shardy propagates sharding consistently across all operands.
# ---------------------------------------------------------------------------
# def _fixed_dot_product_attention_bwd_shardy_rule(
#     scale,
#     seed,
#     dropout_rate,
#     variadic_args,
#     mask_type,
#     layout,
#     sliding_window_length,
#     mesh,
#     value_types,
#     result_types,
# ):
#     _, has_dbias = variadic_args
#     num_args = len(value_types)
#     if layout == AttentionLayout.BTNH.value:
#         q_map = ArrayMapping("batch", "qseq", "qheads", "head")
#         k_map = ArrayMapping("batch", "kvseq", "kvheads", "head")
#         v_map = ArrayMapping("batch", "kvseq", "kvheads", "vhead")
#         act_map = ArrayMapping("batch", "qheads", "qseq")
#         out_map = ArrayMapping("batch", "qseq", "qheads", "vhead")
#         grd_map = ArrayMapping("batch", "qseq", "qheads", "vhead")
#     elif layout == AttentionLayout.BNTH.value:
#         q_map = ArrayMapping("batch", "qheads", "qseq", "head")
#         k_map = ArrayMapping("batch", "kvheads", "kvseq", "head")
#         v_map = ArrayMapping("batch", "kvheads", "kvseq", "vhead")
#         act_map = ArrayMapping("batch", "qheads", "qseq")
#         out_map = ArrayMapping("batch", "qheads", "qseq", "vhead")
#         grd_map = ArrayMapping("batch", "qheads", "qseq", "vhead")
#     else:
#         # Unknown layout – fall back to the (broken) original behaviour so we
#         # don't silently produce wrong sharding for untested layouts.
#         from jax._src.cudnn.fused_attention_stablehlo import _bwd_shardy_rule
#         return _bwd_shardy_rule(num_args, has_dbias, is_fp8=False)
#     # Dynamic args order (static args are already stripped by custom_partitioning):
#     #   0: query, 1: key, 2: value,
#     #   3: bias, 4: q_seqlen, 5: kv_seqlen, 6: q_offsets, 7: kv_offsets,
#     #   8: page_table_k, 9: page_table_v,
#     #   10: activation, 11: fwd_output, 12: grad_output
#     input_sharding = [q_map, k_map, v_map]
#     num_unused = num_args - 6  # everything except Q,K,V, act, out, grad
#     for i in range(num_unused):
#         input_sharding.append(ArrayMapping(f"unused{i}"))
#     input_sharding.extend([act_map, out_map, grd_map])
#     output_sharding = (q_map, k_map, v_map)
#     if has_dbias:
#         # dBias has the same mapping as the bias input (index 3), but bias is
#         # unused in our case so we just give it its own independent name.
#         output_sharding = output_sharding + (ArrayMapping("dbias"),)
#     return SdyShardingRule(tuple(input_sharding), output_sharding)
# _dot_product_attention_bwd_lower.sharding_rule = _fixed_dot_product_attention_bwd_shardy_rule


def _dot_product_attention_fwd_abstract(
    query,
    key,
    value,
    bias,
    q_seqlen,
    kv_seqlen,
    q_offsets,
    kv_offsets,
    page_table_k,
    page_table_v,
    *,
    scale,
    seed,
    dropout_rate,
    variadic_args,
    mask_type,
    layout,
    sliding_window_length,
    is_training,
):
    if layout == AttentionLayout.BNTH.value:
        B, N, T, _ = query.shape
        _, _, S, H = value.shape
        output_shape = (B, N, T, H)
    else:
        B, T, N, _ = query.shape
        _, S, _, H = value.shape
        output_shape = (B, T, N, H)

    max_seg_per_batch = get_max_seg_per_batch(q_offsets)
    softmax_stat_shape = (B * max_seg_per_batch, N, T)

    mat = core.ManualAxisType(varying=query.mat.varying | key.mat.varying | value.mat.varying)

    output_sharding = query.sharding

    if is_training:
        softmax_stat_sharding = jax.NamedSharding(mesh=query.sharding.mesh, spec=jax.P(*output_sharding.spec[:-1]))
        return (
            core.ShapedArray(output_shape, query.dtype, sharding=output_sharding, manual_axis_type=mat),  # output
            core.ShapedArray(softmax_stat_shape, np.float32, sharding=softmax_stat_sharding, manual_axis_type=mat),  # softmax_stat
        )
    else:
        return (
            core.ShapedArray(output_shape, query.dtype, sharding=output_sharding, manual_axis_type=mat),  # output
        )


_dot_product_attention_fwd_p.def_abstract_eval(_dot_product_attention_fwd_abstract)
_dot_product_attention_fwd_p_wrapper.def_abstract_eval(_dot_product_attention_fwd_abstract)


def _dot_product_attention_bwd_abstract(
    query,
    key,
    value,
    bias,
    q_seqlen,
    kv_seqlen,
    q_offsets,
    kv_offsets,
    page_table_k,
    page_table_v,
    activation,
    fwd_output,
    grad_output,
    *,
    scale,
    seed,
    dropout_rate,
    variadic_args,
    mask_type,
    layout,
    sliding_window_length,
):
    _, has_dbias = variadic_args
    mat = core.ManualAxisType(varying=query.mat.varying | key.mat.varying | value.mat.varying)

    if has_dbias:
        # cuDNN supports bias for this case
        return (
            core.ShapedArray(query.shape, query.dtype, sharding=query.sharding, manual_axis_type=mat),  # grad query
            core.ShapedArray(key.shape, key.dtype, sharding=key.sharding, manual_axis_type=mat),  # grad key
            core.ShapedArray(value.shape, value.dtype, sharding=value.sharding, manual_axis_type=mat),  # grad value
            core.ShapedArray(bias.shape, bias.dtype, sharding=bias.sharding, manual_axis_type=mat),  # grad bias
        )
    else:
        return (
            core.ShapedArray(query.shape, query.dtype, sharding=query.sharding, manual_axis_type=mat),  # grad query
            core.ShapedArray(key.shape, key.dtype, sharding=key.sharding, manual_axis_type=mat),  # grad key
            core.ShapedArray(value.shape, value.dtype, sharding=value.sharding, manual_axis_type=mat),  # grad value
        )


_dot_product_attention_bwd_p.def_abstract_eval(_dot_product_attention_bwd_abstract)
_dot_product_attention_bwd_p_wrapper.def_abstract_eval(_dot_product_attention_bwd_abstract)


# Static parameters for our use case (no bias, no seqlen masking, no dropout)
_SEED = 42
_DROPOUT_RATE = 0.0
_VARIADIC_ARGS = (False, False)  # (has_bias, has_dbias)
_LAYOUT = AttentionLayout.BTNH.value
_SLIDING_WINDOW_LENGTH = None


def _make_bwd_residual(query, key, value, activation, out):
    """
    Build the residual tuple expected by _dot_product_attention_bwd_rule:
      (query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
       page_table_k, page_table_v, activation, fwd_output)
    Unused optional fields are filled with empty arrays.
    """
    _not_used = jnp.zeros(0, dtype=query.dtype)
    return (
        query,
        key,
        value,
        _not_used,
        _not_used,
        _not_used,
        _not_used,
        _not_used,
        _not_used,
        _not_used,
        activation,
        out,
    )


def dot_product_attention_fwd(query, key, value, mask_type: MaskType, scale: float):
    """
    Forward pass, no saving.
    Only needs to return the output.
    """
    return cuda_dot_product_attention(query, key, value, mask_type=mask_type, scale=scale)


def dot_product_attention_fwd_rule(query, key, value, mask_type: MaskType, scale: float):
    """
    Forward pass, saving Q, K, V, activation (logsumexp) and O as an explicit
    residual tuple for the backward pass.
    """
    out, activation = cuda_dot_product_attention(query, key, value, mask_type=mask_type, scale=scale, return_residual=True)
    res = _make_bwd_residual(query, key, value, activation, out)
    return out, res


def dot_product_attention_bwd_rule(mask_type: MaskType, scale: float, res, g):
    """
    Backward pass, no saving.
    """
    query, key, value, activation, out = res
    cuda_res = _make_bwd_residual(query, key, value, activation, out)
    grads = _dot_product_attention_bwd_rule(
        scale=scale,
        seed=_SEED,
        dropout_rate=_DROPOUT_RATE,
        variadic_args=_VARIADIC_ARGS,
        mask_type=mask_type,
        layout=_LAYOUT,
        sliding_window_length=_SLIDING_WINDOW_LENGTH,
        is_training=None,
        return_residual=False,
        res=cuda_res,
        grad_output=g,
    )
    return grads[0], grads[1], grads[2]


def dot_product_attention_bwd_rule_fwd_rule(mask_type: MaskType, scale: float, res, g):
    """
    Backward pass, saving for higher order backward.
    """
    # query, key, value = res[0], res[1], res[2]
    # activation, out = res[10], res[11]
    query, key, value, activation, out = res
    dO = g

    dQ, dK, dV = dot_product_attention_bwd_rule(mask_type=mask_type, scale=scale, res=res, g=dO)
    residual = (query, key, value, out, activation, dO)
    return (dQ, dK, dV), residual


def dot_product_attention_bwd_rule_bwd_rule(mask_type: MaskType, scale: float, res, g):
    """
    Backward pass through the backward pass.
    """
    query, key, value, out, activation, dO = res
    ddQ, ddK, ddV = g

    dQ2, dK2, dV2, ddO = flash_bwdbwd(
        Q=query,
        K=key,
        V=value,
        O=out,
        dO=dO,
        ddQ=ddQ,
        ddK=ddK,
        ddV=ddV,
        L=activation,
        mask_type=mask_type,
        scale=scale,
        config=TuningConfig(tile_q=32, tile_k=32, max_concurrent_steps=4),
    )
    # Return gradients matching the structure of (res, g).
    # res = _make_bwd_residual(query, key, value, activation, out) — a 12-tuple.
    # Gradients w.r.t. the unused placeholder fields and activation/out are None.
    d_res = (dQ2, dK2, dV2, None, None)
    return d_res, ddO

    # Dummy return for debugging
    # d_res = (query, key, value, None, None)
    # return d_res, dO
