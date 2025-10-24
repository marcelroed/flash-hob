from pathlib import Path

import jax.numpy as jnp
import torch


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

    q = torch.load(input_tensors_path / "q.pt")
    k = torch.load(input_tensors_path / "k.pt")
    v = torch.load(input_tensors_path / "v.pt")
    do = torch.load(input_tensors_path / "do.pt")
    ddq = torch.load(input_tensors_path / "ddq.pt")
    ddk = torch.load(input_tensors_path / "ddk.pt")
    ddv = torch.load(input_tensors_path / "ddv.pt")

    nq, d_in = q.shape
    nkv, _ = k.shape
    _, d_out = v.shape

    scale = jnp.array(1.0 / jnp.sqrt(d_in), dtype=jnp.float32)

    assert nq == 100
    assert nkv == 150
    assert d_in == 256
    assert d_out == 512

    expected_dq2 = torch.load(output_tensors_path / "dq2.pt")
    expected_dk2 = torch.load(output_tensors_path / "dk2.pt")
    expected_dv2 = torch.load(output_tensors_path / "dv2.pt")
    expected_ddo = torch.load(output_tensors_path / "ddo.pt")

    # TODO: change this to use the jax implementation!
    # from fhob.jax_refs.jax_impls import attn_bwd_bwd
    from fhob.jax_refs.jax_impl import attn_bwd_bwd as non_flash_attn_bwd_bwd

    # fhob.jax_refs.jax_impls
    jnp.asarray(q)
    return

    dq2, dk2, dv2, ddo = non_flash_attn_bwd_bwd(
        jnp.asarray(q), k, v, do, ddq, ddk, ddv, scale=scale, is_causal=False
    )

    # nq = 100
    # nkv = 150
    # d_in = 256
    # d_out = 512
    # dtype = jnp.bfloat16
    # scale = jnp.array(1.0 / jnp.sqrt(d_in), dtype=jnp.float32)

    print(q)


if __name__ == "__main__":
    main()
