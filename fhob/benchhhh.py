from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom
import torch
import triton

from fhob.jax_refs.jax_impl import attn_bwd_bwd_stats
from fhob.triton_bwdbwd import use_bwdbwd

N_REPS = 500
N_WARMUP = 50

nq = 512
nkv = 1024
d_in = 64
d_out = 64

q = torch.randn((nq, d_in), device="cuda", dtype=torch.float16)
k = torch.randn((nkv, d_in), device="cuda", dtype=torch.float16)
v = torch.randn((nkv, d_out), device="cuda", dtype=torch.float16)
o = torch.randn((nq, d_out), device="cuda", dtype=torch.float16)
l = torch.randn((nq,), device="cuda", dtype=torch.float16)
do = torch.randn((nq, d_out), device="cuda", dtype=torch.float16)
ddq = torch.randn((nq, d_in), device="cuda", dtype=torch.float16)
ddk = torch.randn((nkv, d_in), device="cuda", dtype=torch.float16)
ddv = torch.randn((nkv, d_out), device="cuda", dtype=torch.float16)


def jax_benchmark():
    # nq, d_in = q.shape
    # nkv, _ = k.shape
    # _, d_out = v.shape

    scale = jnp.array(1.0 / jnp.sqrt(d_in), dtype=jnp.float32)

    # assert nq == 128
    # assert nkv == 256
    # assert d_in == 64
    # assert d_out == 64

    def do_run_jax():
        attn_bwd_bwd_stats(
            stats=jnp.asarray(l),
            q=jnp.asarray(q),
            k=jnp.asarray(k),
            v=jnp.asarray(v),
            o=jnp.asarray(o),
            do=jnp.asarray(do),
            ddq=jnp.asarray(ddq),
            ddk=jnp.asarray(ddk),
            ddv=jnp.asarray(ddv),
            scale=jnp.asarray(scale),
            is_causal=False,
            sliding_window_length=jnp.inf,
        )

    results = triton.testing.do_bench(do_run_jax, rep=N_REPS, warmup=N_WARMUP)
    print(f"jax result: {results}s")


def trition_benchmark():
    def do_run_triton():
        triton_dq2, triton_dk2, triton_dv2, triton_ddo = use_bwdbwd(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            o.unsqueeze(0),
            do.unsqueeze(0),
            ddq.unsqueeze(0),
            ddk.unsqueeze(0),
            ddv.unsqueeze(0),
            l.unsqueeze(0),
            1 / d_in**0.5,
        )

    results = triton.testing.do_bench(do_run_triton, rep=N_REPS, warmup=N_WARMUP)
    print(f"triton result: {results}s")


if __name__ == "__main__":
    jax_benchmark()
    trition_benchmark()
