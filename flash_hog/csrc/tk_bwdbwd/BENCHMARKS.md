# ThunderKittens double-backward: benchmarks vs the Pallas path

End-to-end timing of the full causal-attention **HVP** (forward + backward +
double-backward, one jitted graph) through `flash_hog.jax.attention.dot_product_attention`,
comparing the stock Pallas double-backward against the opt-in ThunderKittens path
(`flash_hog.jax._tk_gpu.enable()`). Only the double-backward differs between the two
columns — the cuDNN forward and first backward are identical.

Setup: NVIDIA H200, CUDA 12.8.1, `jax[cuda13]==0.10.1`, 12 heads, head_dim 64, causal,
bf16 kernels with fp32 inputs/outputs. ThunderKittens pinned at `34b15f7e`. Times are
means over 10 iterations (5 at seq ≥ 16k, 3 at ≥ 64k) after warmup; expect ~±5%
machine-to-machine variance.

## B = 1 (latency)

| batch | seq | Pallas (ms) | TK (ms) | speedup | faithfulness |
|--:|--:|--:|--:|--:|:--|
| 1 | 512 | 0.18 | 0.22 | 0.82x | cos 1.0000 vs fp32 (Pallas: 1.0000) |
| 1 | 1024 | 0.33 | 0.28 | 1.17x | cos 1.0000 vs fp32 (Pallas: 1.0000) |
| 1 | 2048 | 0.88 | 0.66 | 1.34x | cos 1.0000 vs fp32 (Pallas: 1.0000) |
| 1 | 4096 | 2.72 | 1.76 | 1.55x | cos 1.0000 vs Pallas |
| 1 | 8192 | 9.46 | 5.56 | 1.70x | cos 1.0000 vs Pallas |
| 1 | 16384 | 34.94 | 19.21 | 1.82x | cos 1.0000 vs Pallas |
| 1 | 32768 | 137.54 | 73.63 | 1.87x | cos 1.0000 vs Pallas |
| 1 | 65536 | 555.30 | 289.03 | 1.92x | cos 1.0000 vs Pallas |
| 1 | 131072 | 2282.74 | 1143.72 | **2.00x** | cos 1.0000 vs Pallas |

## Constant token budget (B × S = 262,144)

GPU saturated at every row; the throughput view.

| batch | seq | Pallas (ms) | TK (ms) | speedup | faithfulness |
|--:|--:|--:|--:|--:|:--|
| 512 | 512 | 27.72 | 18.34 | 1.51x | cos 1.0000 vs Pallas |
| 256 | 1024 | 44.73 | 27.16 | 1.65x | cos 1.0000 vs Pallas |
| 128 | 2048 | 78.34 | 44.83 | 1.75x | cos 1.0000 vs Pallas |
| 64 | 4096 | 146.21 | 80.18 | 1.82x | cos 1.0000 vs Pallas |
| 32 | 8192 | 281.33 | 150.76 | 1.87x | cos 1.0000 vs Pallas |
| 16 | 16384 | 557.66 | 291.53 | 1.91x | cos 1.0000 vs Pallas |
| 8 | 32768 | 1104.21 | 577.58 | 1.91x | cos 1.0000 vs Pallas |
| 4 | 65536 | 2227.75 | 1137.58 | 1.96x | cos 1.0000 vs Pallas |
| 2 | 131072 | 4602.62 | 2287.29 | **2.01x** | cos 1.0000 vs Pallas |

The speedup grows with sequence length because the double-backward is the dominant
O(S²) term of the HVP: as it takes over the runtime, the kernel-level advantage
(stage1 ~1.9x, stage2 ~1.3x over the Pallas kernels) shows through fully. The only
regression is tiny single-sequence shapes (1×512), where launch overhead dominates —
exactly the regime `enable()`'s per-call fallback leaves available to Pallas anyway
for unsupported shapes.

Faithfulness: where the dense fp32 reference fits in memory (B=1, seq ≤ 2048), both
paths give cos = 1.0000 against it; the TK path's relative error (~4e-3) is slightly
tighter than Pallas (~5e-3). At larger shapes the two paths agree with each other to
cos = 1.0000 (~2e-3 rel).

## Reproducing

```python
import jax, jax.numpy as jnp
import flash_hog.jax.attention as fa
from flash_hog.jax import _tk_gpu as tk

def attn(q, k, v, scale):
    qb, kb, vb = (x.astype(jnp.bfloat16) for x in (q, k, v))
    return fa.dot_product_attention(qb, kb, vb, is_causal=True, scale=scale).astype(jnp.float32)

def tree_dot(a, b):
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))

def make_hvp(cot, tan, scale):           # grad of <grad(loss), tan> == HVP
    def loss(x):
        return jnp.vdot(attn(*x, scale), cot)
    return jax.jit(lambda x: jax.grad(lambda y: tree_dot(jax.grad(loss)(y), tan))(x))

# time make_hvp(...)(qkv) with tk.disable() vs tk.enable()
```

Install `flash-hog[tk]`; the plugin JIT-builds (cached) on first `tk.enable()`.
