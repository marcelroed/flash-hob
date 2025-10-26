from pathlib import Path

import jax.numpy as jnp
import torch

from fhog.triton_bwdbwd import flash_bwdbwd
from fhog.triton_flash import produce_L, run_regular_attention


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
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        do.unsqueeze(0),
        ddq.unsqueeze(0),
        ddk.unsqueeze(0),
        ddv.unsqueeze(0),
    )
    expected_dq2, expected_ddo, expected_dk2, expected_dv2 = (
        expected_dq2.unsqueeze(0),
        expected_ddo.unsqueeze(0),
        expected_dk2.unsqueeze(0),
        expected_dv2.unsqueeze(0),
    )

    o = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=1 / d_in**0.5, is_causal=False
    )
    L = produce_L(q, k, is_causal=False)
    triton_dq2, triton_dk2, triton_dv2, triton_ddo = flash_bwdbwd(
        q, k, v, o, do, ddq, ddk, ddv, L, 1 / d_in**0.5
    )

    print(triton_dq2)
    print(expected_dq2)
    # torch.testing.assert_close(triton_dq2, expected_dq2)
    # torch.testing.assert_close(triton_ddo, expected_ddo)
    # torch.testing.assert_close(triton_dk2, expected_dk2)
    torch.testing.assert_close(triton_dv2, expected_dv2)

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
