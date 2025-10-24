import math
from functools import partial

import torch
from einops import einsum


# @partial(torch.vmap, in_dims=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None))
def attn_bwd_bwd(
    stats,
    q,
    k,
    v,
    o,
    do,
    ddq,
    ddk,
    ddv,
    scale=1.0,
    is_causal=False,
    # sliding_window_length=float("inf"), #TODO: add sliding window length
):
    scale = scale.to(q.dtype)

    # precision = jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    # einsum = partial(jnp.einsum, precision=precision) #TODO precision
    # d, dd, stats2 = precompute_values(
    #     stats, q, k, v, o, do, ddq, ddk, ddv, scale, is_causal, sliding_window_length
    # )
    # d = dd = stats2 = 1

    s = (
        torch.einsum(
            "qd,kd->qk",
            q,
            k,  # TODO precision
        )
        * scale
    )

    # if is_causal:
    #     qs, ks = q.shape[0], k.shape[0]  # -2 or 0?
    #     mask = torch.ones(qs, ks, dtype=torch.bool, device=q.device).triu(diagonal=1)
    #     s = torch.where(mask, s, -torch.inf)

    p = torch.exp(s - stats[:, None]).to(q.dtype)

    dp_bwd = do @ v.T
    dds = scale * (ddq @ k.T + q @ ddk.T)

    # d = sum_k (dP * P)_qk
    # => dd = sum_k (ddS * P)_qk
    # dds = ddq k^T + q ddk^T
    # => dd = sum_k ((ddq k^T) * p) + sum_k ((q ddk^T) * p)
    d = torch.einsum("qd,qd->q", o, do)[:, None]
    dd = torch.einsum("qk,qk->q", dds, p)[:, None]

    # dp2 = (dds * dp_bwd - dds * d) - dp_bwd * dd + do @ ddv.T
    dp2 = do @ ddv.T - dp_bwd * dd - dds * d + dp_bwd * dds
    ddp = p * (dds - dd)

    ds2 = scale * p * (dp2 - torch.einsum("qk,qk->q", dp2, p)[:, None])
    # ds2 = scale * p * (dp2 - stats2)
    ds = scale * p * (dp_bwd - d)

    dq2 = ds @ ddk + ds2 @ k
    dk2 = ds.T @ ddq + ds2.T @ q

    dv2 = ddp.T @ do

    ddo = p @ ddv + ddp @ v

    return dq2, dk2, dv2, ddo


@partial(torch.vmap, in_dims=(0, 0, None, 0, 0, 0, 0))
def attn_bwd_torch(
    O,
    L,
    scale,
    Q,
    K,
    V,
    dO,
):
    S = Q @ K.T * scale
    P = torch.exp(S - L[:, None])
    dV = P.T @ dO
    dP = dO @ V.T
    D = einsum(O, dO, "q d, q d -> q")
    dS = P * (dP - D[:, None])
    dQ = (dS @ K) * scale
    dK = (dS.T @ Q) * scale
    return dQ, dK, dV


# def simple_sdpa_backward_torch(ctx, dO: Float[Tensor, "... N_q d"]):
#     Q, K, V, O, L = ctx.saved_tensors
#     N_q = Q.size(-2)
#     N_k = K.size(-2)
#     d = Q.size(-1)
#     is_causal = ctx.is_causal
#     S = einsum(Q, K, "... N_q d, ... N_k d -> ... N_q N_k") / math.sqrt(d)
#     P = (S - L.unsqueeze(-1)).exp()

#     dV = einsum(P, dO, "... N_q N_k, ... N_q d -> ... N_k d")
#     dP = einsum(dO, V, "... N_q d, ... N_k d -> ... N_q N_k")
#     D = einsum(O, dO, "... N_q_o d, ... N_q_do d -> ... N_q_o N_q_do").diagonal(
#         dim1=1, dim2=2
#     )
#     dS = P * (dP - D.unsqueeze(-1))
#     if is_causal:
#         dS = dS.masked_fill(mask.unsqueeze(0).expand_as(dS), 0)
#     dQ = einsum(dS, K, "... N_q N_kv, ... N_kv d -> ... N_q d") / math.sqrt(d)
#     dK = einsum(dS, Q, "... N_q N_k, ... N_q d -> ... N_k d") / math.sqrt(d)
#     return dQ, dK, dV, None


def attn_bwd_bwd_torch(
    Q,
    K,
    V,
    dO,
    L,
    scale,
    ddQ,
    ddK,
    ddV,
    O,
):
    breakpoint()
    (dQ, dK, dV), vjp_fun = torch.func.vjp(
        partial(attn_bwd_torch, O, L, scale), Q, K, V, dO
    )
    dQ2, dK2, dV2, ddO = vjp_fun((ddQ, ddK, ddV))
    return dQ2, dK2, dV2, ddO


def test_attn_bwd_bwd_torch():
    batch_size = 3
    n_queries = 4
    n_keys = 8
    d = 16
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(batch_size, n_queries, d, device="cuda", dtype=torch.float32)
    k = torch.randn(batch_size, n_keys, d, device="cuda", dtype=torch.float32)
    v = torch.randn(batch_size, n_keys, d, device="cuda", dtype=torch.float32)
    o = torch.randn(batch_size, n_queries, d, device="cuda", dtype=torch.float32)
    dO = torch.randn(batch_size, n_queries, d, device="cuda", dtype=torch.float32)
    L = torch.randn(batch_size, n_queries, device="cuda", dtype=torch.float32)
    ddQ = torch.randn(batch_size, n_queries, d, device="cuda", dtype=torch.float32)
    ddK = torch.randn(batch_size, n_keys, d, device="cuda", dtype=torch.float32)
    ddV = torch.randn(batch_size, n_keys, d, device="cuda", dtype=torch.float32)
    results_torch = attn_bwd_bwd(
        L,
        q,
        k,
        v,
        o,
        dO,
        ddQ,
        ddK,
        ddV,
        scale,
        False,
    )

    results_torch_vjp = attn_bwd_bwd_torch(q, k, v, dO, L, scale, ddQ, ddK, ddV, o)

    for t1, t2 in zip(results_torch, results_torch_vjp):
        print(torch.max(t1), torch.max(t2))
        torch.testing.assert_close(t1, t2)

    print(results_torch)
    print(results_torch_vjp)
