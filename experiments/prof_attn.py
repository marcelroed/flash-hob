"""Run a single attention implementation for ncu profiling.

Usage:  ncu --metrics gpu__time_duration.sum --launch-count N \
            uv run --group experiments python experiments/prof_attn.py {fa4|cudnn} [--n N]

Does warmup (compile + run) then `--n` measured launches of just the attention
kernel, so ncu captures clean GPU kernel durations.
"""
import argparse
import jax, jax.numpy as jnp, jax.nn as jnn, numpy as np
import fa4_jax

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["fa4", "cudnn"])
p.add_argument("--n", type=int, default=5)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--seq", type=int, default=512)
p.add_argument("--heads", type=int, default=20)
p.add_argument("--head-dim", type=int, default=64)
a = p.parse_args()

B, S, H, D = a.batch, a.seq, a.heads, a.head_dim
scale = float(1.0 / np.sqrt(D))
ks = jax.random.split(jax.random.key(0), 3)
q = jax.random.normal(ks[0], (B, S, H, D), jnp.bfloat16) * 0.5
k = jax.random.normal(ks[1], (B, S, H, D), jnp.bfloat16) * 0.5
v = jax.random.normal(ks[2], (B, S, H, D), jnp.bfloat16) * 0.5

if a.mode == "fa4":
    fn = jax.jit(lambda q, k, v: fa4_jax.fa4_fwd(q, k, v, scale, return_lse=False))
else:
    fn = jax.jit(lambda q, k, v: jnn.dot_product_attention(q, k, v, scale=scale, implementation="cudnn"))

jax.block_until_ready(fn(q, k, v))  # warmup / compile
for _ in range(a.n):
    out = fn(q, k, v)
jax.block_until_ready(out)
