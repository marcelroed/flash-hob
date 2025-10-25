from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom
import torch
import triton

from fhob.jax_refs.jax_impl import attn_bwd_bwd
from fhob.triton_bwdbwd import use_bwdbwd

N_REPS = 5000
N_WARMUP = 300


def full_benchmark(nq, nkv):
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
            # attn_bwd_bwd_stats(
            attn_bwd_bwd(
                # stats=jnp.asarray(l),
                q=jnp.asarray(q),
                k=jnp.asarray(k),
                v=jnp.asarray(v),
                # o=jnp.asarray(o),
                do=jnp.asarray(do),
                ddq=jnp.asarray(ddq),
                ddk=jnp.asarray(ddk),
                ddv=jnp.asarray(ddv),
                scale=jnp.asarray(scale),
                is_causal=False,
                sliding_window_length=jnp.inf,
            )

        results = triton.testing.do_bench(do_run_jax, rep=N_REPS, warmup=N_WARMUP)
        return results

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
        return results

    try:
        jax_time = jax_benchmark()
    except Exception as e:
        print(f"JAX benchmark failed for nq={nq}, nkv={nkv} with error: {e}")
        jax_time = float("inf")
    triton_time = trition_benchmark()
    print(f"{nq}: {jax_time=} {triton_time=} => speedup: {jax_time / triton_time:.2f}x")
    return jax_time, triton_time


if __name__ == "__main__":
    # nq = 512
    # nkv = 1024
    # for siz in
    for size in [
        # 128,
        # 512,
        # 1024,
        # 2048,
        # 4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,  # 2^19
        1048576,  # 2^20
        2097152,
    ]:
        # for size in [128, 512]:
        full_benchmark(size, size)
