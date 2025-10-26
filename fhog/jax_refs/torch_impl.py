import math
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import torch
from einops import einsum

# from fhog.triton_bwdbwd import use_bwdbwd
# from fhog.triton_flash import produce_L, run_regular_attention


# @partial(torch.vmap, in_dims=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None))
def attn_bwd_bwd_torch(
    *,
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


# def attn_bwd_bwd_torch(
#     Q,
#     K,
#     V,
#     dO,
#     L,
#     scale,
#     ddQ,
#     ddK,
#     ddV,
#     O,
# ):
#     breakpoint()
#     (dQ, dK, dV), vjp_fun = torch.func.vjp(
#         partial(attn_bwd_torch, O, L, scale), Q, K, V, dO
#     )
#     dQ2, dK2, dV2, ddO = vjp_fun((ddQ, ddK, ddV))
#     return dQ2, dK2, dV2, ddO


def main():
    test_data_folder = Path("test_data")
    (input_tensors_path := test_data_folder / "input_items").mkdir(
        parents=True, exist_ok=True
    )
    (output_tensors_path := test_data_folder / "output_items").mkdir(
        parents=True, exist_ok=True
    )

    # torch.save(torch.tensor(q), input_tensors_path / "q.pt")
    # torch.save(torch.tensor(k), input_tensors_path / "k.pt")
    # torch.save(torch.tensor(v), input_tensors_path / "v.pt")
    # torch.save(torch.tensor(do), input_tensors_path / "do.pt")
    # torch.save(torch.tensor(ddq), input_tensors_path / "ddq.pt")
    # torch.save(torch.tensor(ddk), input_tensors_path / "ddk.pt")
    # torch.save(torch.tensor(ddv), input_tensors_path / "ddv.pt")

    q = torch.load(input_tensors_path / "q.pt").to(torch.bfloat16).cuda()
    k = torch.load(input_tensors_path / "k.pt").to(torch.bfloat16).cuda()
    v = torch.load(input_tensors_path / "v.pt").to(torch.bfloat16).cuda()
    o = torch.load(input_tensors_path / "o.pt").to(torch.bfloat16).cuda()
    l = torch.load(input_tensors_path / "L.pt").to(torch.bfloat16).cuda()
    do = torch.load(input_tensors_path / "do.pt").to(torch.bfloat16).cuda()
    ddq = torch.load(input_tensors_path / "ddq.pt").to(torch.bfloat16).cuda()
    ddk = torch.load(input_tensors_path / "ddk.pt").to(torch.bfloat16).cuda()
    ddv = torch.load(input_tensors_path / "ddv.pt").to(torch.bfloat16).cuda()

    nq, d_in = q.shape
    nkv, _ = k.shape
    _, d_out = v.shape

    # scale = jnp.array(1.0 / jnp.sqrt(d_in), dtype=jnp.float32)

    assert nq == 128
    assert nkv == 256
    assert d_in == 64
    assert d_out == 64

    expected_dq2 = torch.load(output_tensors_path / "dq2.pt").to(torch.bfloat16).cuda()
    expected_dk2 = torch.load(output_tensors_path / "dk2.pt").to(torch.bfloat16).cuda()
    expected_dv2 = torch.load(output_tensors_path / "dv2.pt").to(torch.bfloat16).cuda()
    expected_ddo = torch.load(output_tensors_path / "ddo.pt").to(torch.bfloat16).cuda()

    q, k, v, do, ddq, ddk, ddv = (
        q,  # .unsqueeze(0),
        k,  ##.unsqueeze(0),
        v,  ##.unsqueeze(0),
        do,  ##.unsqueeze(0),
        ddq,  ##.unsqueeze(0),
        ddk,  ##.unsqueeze(0),
        ddv,  ##.unsqueeze(0),
    )
    expected_dq2, expected_ddo = expected_dq2, expected_ddo

    o = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=1 / d_in**0.5, is_causal=False
    )
    # L = produce_L(q, k, is_causal=False)
    L = torch.load(input_tensors_path / "L.pt").to(torch.bfloat16).cuda()

    torch_dq2, torch_dk2, torch_dv2, torch_ddo = attn_bwd_bwd_torch(
        stats=L,
        q=q,
        k=k,
        v=v,
        o=o,
        do=do,
        ddq=ddq,
        ddk=ddk,
        ddv=ddv,
        scale=torch.tensor(1 / d_in**0.5).to(torch.bfloat16),
        is_causal=False,
        # sliding_window_length=float("inf"), #TODO: add sliding window length
    )
    # triton_dq2, triton_dk2, triton_dv2, triton_ddo = use_bwdbwd(
    #     q, k, v, do, o, ddq, ddk, ddv, L, 1 / d_in**0.5
    # )

    print(torch_dq2)
    print(expected_dq2)
    torch.testing.assert_close(torch_dq2, expected_dq2)
    torch.testing.assert_close(torch_ddo, expected_ddo)

    # TODO: change this to use the jax implementation!
    # from fhog.jax_refs.jax_impls import attn_bwd_bwd
    # from fhog.jax_refs.jax_impl import attn_bwd_bwd as non_flash_attn_bwd_bwd

    # fhog.jax_refs.jax_impls
    # return

    # dq2, dk2, dv2, ddo = non_flash_attn_bwd_bwd(
    #     jnp.asarray(q), k, v, do, ddq, ddk, ddv, scale=scale, is_causal=False
    # )

    # nq = 100
    # nkv = 150
    # d_in = 256
    # d_out = 512
    # dtype = jnp.bfloat16
    # scale = jnp.array(1.0 / jnp.sqrt(d_in), dtype=jnp.float32)

    # print(q)


if __name__ == "__main__":
    main()
