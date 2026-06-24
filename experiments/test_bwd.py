"""Validate fa4_bwd against cuDNN backward via jax.vjp."""
import sys
import jax, jax.numpy as jnp, jax.nn as jnn, numpy as np
import fa4_jax


def run(B, S, H, D):
    scale = float(1.0 / np.sqrt(D))
    ks = jax.random.split(jax.random.key(0), 4)
    q = (jax.random.normal(ks[0], (B, S, H, D), jnp.float32) * 0.5).astype(jnp.bfloat16)
    k = (jax.random.normal(ks[1], (B, S, H, D), jnp.float32) * 0.5).astype(jnp.bfloat16)
    v = (jax.random.normal(ks[2], (B, S, H, D), jnp.float32) * 0.5).astype(jnp.bfloat16)
    dout = (jax.random.normal(ks[3], (B, S, H, D), jnp.float32) * 0.5).astype(jnp.bfloat16)

    out, lse = fa4_jax.fa4_fwd(q, k, v, scale)
    out = jax.block_until_ready(out)
    dq, dk, dv = fa4_jax.fa4_bwd(q, k, v, out, dout, lse, scale)
    dq, dk, dv = jax.block_until_ready((dq, dk, dv))

    def f(q, k, v):
        return jnn.dot_product_attention(q, k, v, scale=scale, implementation="cudnn")
    o_ref, vjp = jax.vjp(f, q, k, v)
    dq_r, dk_r, dv_r = vjp(dout)

    def cmp(name, a, b):
        a = a.astype(jnp.float32); b = b.astype(jnp.float32)
        mad = float(jnp.max(jnp.abs(a - b)))
        mref = float(jnp.mean(jnp.abs(b)))
        rel = mad / (mref + 1e-30)
        print(f"  {name}: max|diff|={mad:.3e}  mean|ref|={mref:.3e}  ratio={rel:.3f}")
        return rel

    print(f"shape B={B} S={S} H={H} D={D}")
    of = float(jnp.max(jnp.abs(out.astype(jnp.float32) - o_ref.astype(jnp.float32))))
    print(f"  out(fwd): max|diff|={of:.3e}")
    cmp("dq", dq, dq_r)
    cmp("dk", dk, dk_r)
    cmp("dv", dv, dv_r)


if __name__ == "__main__":
    run(2, 512, 4, 64)
    print()
    run(16, 512, 20, 64)
