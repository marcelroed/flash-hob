import numpy as np

import fhob.jax_refs.jax_impl as jax_impl
import fhob.jax_refs.torch_impl as torch_impl

print("Loading libraries...")

import jax.numpy as jnp
import jax.random as jrandom
import torch

print("Imported libraries.")


def compare_jax_and_torch(
    *, stats, q, k, v, o, do, ddq, ddk, ddv, scale=1.0, is_causal=False
):
    jax_out = jax_impl.attn_bwd_bwd_stats(
        stats=stats,
        q=q,
        k=k,
        v=v,
        o=o,
        do=do,
        ddq=ddq,
        ddk=ddk,
        ddv=ddv,
        scale=scale,
        is_causal=is_causal,
    )
    torch_out = torch_impl.attn_bwd_bwd(
        stats=torch.tensor(np.asarray(stats), dtype=torch.float32),
        q=torch.tensor(np.asarray(q), dtype=torch.float32),
        k=torch.tensor(np.asarray(k), dtype=torch.float32),
        v=torch.tensor(np.asarray(v), dtype=torch.float32),
        o=torch.tensor(np.asarray(o), dtype=torch.float32),
        do=torch.tensor(np.asarray(do), dtype=torch.float32),
        ddq=torch.tensor(np.asarray(ddq), dtype=torch.float32),
        ddk=torch.tensor(np.asarray(ddk), dtype=torch.float32),
        ddv=torch.tensor(np.asarray(ddv), dtype=torch.float32),
        scale=torch.tensor(np.asarray(scale), dtype=torch.float32),
        is_causal=is_causal,
    )
    return jax_out, torch_out


if __name__ == "__main__":
    key1, key2, key3, key4, key5, key6, key7, key8, key9 = jrandom.split(
        jrandom.PRNGKey(42), 9
    )
    nq = 128
    nkv = 256
    d_in = 64
    d_out = 64
    dtype = jnp.float32
    scale = jnp.array(1.0 / jnp.sqrt(d_in), dtype=jnp.float32)
    q = jrandom.normal(key1, (nq, d_in), dtype=dtype)
    k = jrandom.normal(key2, (nkv, d_in), dtype=dtype)
    v = jrandom.normal(key3, (nkv, d_out), dtype=dtype)
    o = jrandom.normal(key8, (nq, d_out), dtype=dtype)
    do = jrandom.normal(key4, (nq, d_out), dtype=dtype)
    ddq = jrandom.normal(key5, (nq, d_in), dtype=dtype)
    ddk = jrandom.normal(key6, (nkv, d_in), dtype=dtype)
    ddv = jrandom.normal(key7, (nkv, d_out), dtype=dtype)
    stats = jrandom.normal(key9, (nq,), dtype=dtype)
    is_causal = False

    jax_out, torch_out = compare_jax_and_torch(
        stats=stats,
        q=q,
        k=k,
        v=v,
        o=o,
        do=do,
        ddq=ddq,
        ddk=ddk,
        ddv=ddv,
        scale=scale,
        is_causal=is_causal,
    )
    print(jax_out)
    print(torch_out)

    for tensor1, tensor2 in zip(jax_out, torch_out):
        if jnp.allclose(tensor1, jnp.asarray(tensor2.numpy())):
            print("success")
        else:
            print(torch.std(torch.tensor(np.asarray(tensor1)) - tensor2))
            print("failure")
