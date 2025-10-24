import torch


def attn_bwd_bwd(*,
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
    #sliding_window_length=float("inf"), #TODO: add sliding window length
):

    scale = scale.to(q.dtype)

    #precision = jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    #einsum = partial(jnp.einsum, precision=precision) #TODO precision
    # d, dd, stats2 = precompute_values(
    #     stats, q, k, v, o, do, ddq, ddk, ddv, scale, is_causal, sliding_window_length
    # )
    # d = dd = stats2 = 1

    s = (
        torch.einsum(
            "qd,kd->qk", q, k,  #TODO precision
        )
        * scale
    )

    if is_causal:
        qs, ks = q.shape[0], k.shape[0] # -2 or 0?
        mask = torch.ones(qs, ks, dtype=torch.bool, device=q.device).triu(diagonal=1)
        s = torch.where(mask, s, -torch.inf)

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
