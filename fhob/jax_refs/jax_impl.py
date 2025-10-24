from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from einops import einsum
from equinox import tree_pprint as pp
from jax import core, lax
from jax._src.ad_checkpoint import name_p
from jax._src.interpreters import ad
from jax.ad_checkpoint import checkpoint_name
from jax.interpreters.partial_eval import dce_jaxpr


def name_jvp(primals, tangents, *, name):
    (x,), (xdot,) = primals, tangents
    return name_p.bind(x, name=name), name_p.bind(
        xdot, name=f"d{name}"
    )  # name the tangent value!


def name_transpose(cotangent, *_args, name, **_kwargs):
    # print(name, type(name))
    # print(cotangent, type(cotangent))
    return [
        name_p.bind(cotangent, name=f"({name})^T"),
    ]


ad.primitive_jvps[name_p] = name_jvp
ad.primitive_transposes[name_p] = name_transpose

jax.config.update("jax_default_matmul_precision", "F32_F32_F32")


def random_like_tree(tree, rng):
    def random_like_fn(x):
        nonlocal rng
        if isinstance(x, jnp.ndarray):
            local_rng, rng = jrandom.split(rng)
            return jrandom.normal(local_rng, x.shape, dtype=x.dtype)
        else:
            return x

    return jax.tree.map(random_like_fn, tree)


def diff_metrics(tree1, tree2):
    def diff_fn(x, y):
        if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
            absdiff = jnp.abs(x - y)
            return {
                "max": jnp.max(absdiff).item(),
                "mean": jnp.mean(absdiff).item(),
                "std": jnp.std(x - y).item(),
            }
        else:
            return None

    return jax.tree.map(diff_fn, tree1, tree2)


def safe_sum(x, axis=None, keepdims=True):
    return jnp.sum(x, axis=axis, dtype=jnp.float32, keepdims=keepdims).astype(x.dtype)


def softmax(x, axis=-1):
    x_max = jnp.max(x, axis=axis, keepdims=True)
    unnormalized = jnp.exp(x - lax.stop_gradient(x_max))
    denom = jnp.sum(unnormalized, axis=axis, keepdims=True)
    denom = checkpoint_name(denom, "denom")
    return unnormalized / denom


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _softmax(x, axis: int | tuple[int, ...] | None = -1, where=None, initial=-jnp.inf):
    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x_safe = x if where is None else jnp.where(where, x, initial)
    unnormalized = jnp.exp(x_safe - x_max)
    result = unnormalized / jnp.sum(
        unnormalized, axis, where=where, keepdims=True, dtype=jnp.float32
    ).astype(x.dtype)
    if where is not None:
        result = jnp.where(where, result, 0)
    return result


# def _softmax_fwd(
#     x,
#     axis: int | tuple[int, ...] | None = -1,
#     where = None,
#     initial = -jnp.inf
# ):
#     result = _softmax(x, axis, where, initial)
#     return result, (x, where, initial)


@_softmax.defjvp
def _softmax_jvp(axis, primals, tangents):
    (x, where, initial), (x_dot, _, _) = primals, tangents
    y = _softmax(x, axis, where, initial)
    x_dot = checkpoint_name(x_dot, "x_dot")
    return y, y * (
        x_dot - jnp.sum(y * x_dot, axis, where=where, keepdims=True, dtype=jnp.float32)
    ).astype(x_dot.dtype)


def get_mask_bottom_right(qs: int, ks: int, sliding_window_length: int):
    qi = jnp.arange(qs)[:, None]
    kj = jnp.arange(ks)[None, :]
    qi = qi + ks - qs
    mask = (qi >= kj) & (qi - kj < sliding_window_length)
    return mask


def attn_fwd(q, k, v, scale=1.0, is_causal=False, sliding_window_length=jnp.inf):
    scale = scale.astype(q.dtype)
    q = checkpoint_name(q, "q")
    k = checkpoint_name(k, "k")
    v = checkpoint_name(v, "v")
    qs, ks = q.shape[0], k.shape[0]
    # s = (q @ k.T) * scale
    s = (
        jnp.einsum(
            "qd, kd -> qk",
            q,
            k,
            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
            preferred_element_type=jnp.float32,
        )
        * scale
    )
    if is_causal:
        mask = get_mask_bottom_right(qs, ks, sliding_window_length)
        s = jnp.where(mask, s, -jnp.inf)
    s = checkpoint_name(s, "s")

    p = _softmax(s, axis=-1).astype(q.dtype)
    p = checkpoint_name(p, "p")

    o = p @ v
    o = checkpoint_name(o, "o")
    return o


def attn_bwd(q, k, v, do, scale=1.0, is_causal=False, sliding_window_length=jnp.inf):
    qs, ks = q.shape[0], k.shape[0]
    scale = scale.astype(q.dtype)

    q = checkpoint_name(q, "q")
    k = checkpoint_name(k, "k")
    v = checkpoint_name(v, "v")
    do = checkpoint_name(do, "do")

    # s = (q @ k.T) * scale
    s = (
        jnp.einsum(
            "qd, kd -> qk",
            q,
            k,
            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
            preferred_element_type=jnp.float32,
        )
        * scale
    )

    if is_causal:
        mask = get_mask_bottom_right(qs, ks, sliding_window_length)
        s = jnp.where(mask, s, -jnp.inf)

    s = checkpoint_name(s, "s")

    # s = s - jnp.max(s, axis=-1, keepdims=True)

    p = _softmax(s, axis=-1).astype(q.dtype)
    p = checkpoint_name(p, "p")
    # s_max = jnp.max(s, axis=-1, keepdims=True)
    # unnormalized = jnp.exp(s - lax.stop_gradient(s_max))
    # denom = jnp.sum(unnormalized, axis=-1, keepdims=True)

    dp = do @ v.T
    dp = checkpoint_name(dp, "dp")
    pdp = p * dp
    d = safe_sum(pdp, axis=-1)
    d = checkpoint_name(d, "d")

    ds = scale * (pdp - p * d)
    ds = checkpoint_name(ds, "ds")
    dq = ds @ k
    dq = checkpoint_name(dq, "dq")
    dk = ds.T @ q
    dk = checkpoint_name(dk, "dk")
    dv = p.T @ do
    dv = checkpoint_name(dv, "dv")
    return dq, dk, dv


def attn_bwd_bwd(
    q,
    k,
    v,
    do,
    ddq,
    ddk,
    ddv,
    scale=1.0,
    is_causal=False,
    sliding_window_length=jnp.inf,
):
    qs, ks = q.shape[0], k.shape[0]
    scale = scale.astype(q.dtype)

    # s = (q @ k.T) * scale
    s = (
        jnp.einsum(
            "qd, kd -> qk",
            q,
            k,
            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
            preferred_element_type=jnp.float32,
        )
        * scale
    )

    if is_causal:
        mask = get_mask_bottom_right(qs, ks, sliding_window_length)
        s = jnp.where(mask, s, -jnp.inf)

    p = _softmax(s, axis=-1).astype(q.dtype)

    dp_bwd = do @ v.T
    dds = scale * (ddq @ k.T + q @ ddk.T)
    d = safe_sum(p * dp_bwd, axis=-1)
    dd = safe_sum(dds * p, axis=-1)

    # dp2 = (dds * dp_bwd - dds * d) - dp_bwd * dd + do @ ddv.T
    dp2 = ((do @ ddv.T - dp_bwd * dd) - dds * d) + dp_bwd * dds
    ddp = p * dds - p * dd

    ds2 = scale * (p * dp2 - p * safe_sum(dp2 * p, axis=-1))
    ds = scale * (p * dp_bwd - p * d)
    dq2 = ds @ ddk + ds2 @ k
    dk2 = ds.T @ ddq + ds2.T @ q

    dv2 = ddp.T @ do

    ddo = p @ ddv + ddp @ v

    # dq2 = dk2 = ddo = None

    return dq2, dk2, dv2, ddo


def precompute_values(
    stats, q, k, v, o, do, ddq, ddk, ddv, scale, is_causal, sliding_window_length
):
    precision = jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    einsum = partial(jnp.einsum, precision=precision)
    d = einsum("qd,qd->q", o, do)[:, None]

    s = (
        jnp.einsum(
            "qd,kd->qk", q, k, precision=precision, preferred_element_type=jnp.float32
        )
        * scale
    )

    if is_causal:
        qs, ks = q.shape[0], k.shape[0]
        mask = get_mask_bottom_right(qs, ks, sliding_window_length)
        s = jnp.where(mask, s, -jnp.inf)

    p = jnp.exp(s - stats[:, None]).astype(jnp.bfloat16)
    dds = scale * (ddq @ k.T + q @ ddk.T)
    dd = einsum("qk,qk->q", dds, p)[:, None]

    dP = do @ v.T
    dds = scale * (ddq @ k.T + q @ ddk.T)
    dp2 = do @ ddv.T - dP * dd - dds * d + dP * dds
    stats2 = einsum("qk,qk->q", dp2, p)[:, None]
    return d, dd, stats2


def attn_bwd_bwd_stats(
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
    sliding_window_length=jnp.inf,
):
    # print(f"Running bwdbwd with stats with {is_causal=}, {sliding_window_length=}")
    # RCP_LN2 = 1.4426950408889634
    # LN2 = 0.6931471824645996
    print(scale.dtype)
    scale = scale.astype(q.dtype)

    precision = jax.lax.DotAlgorithmPreset.F32_F32_F32
    einsum = partial(jnp.einsum, precision=precision)
    # d, dd, stats2 = precompute_values(
    #     stats, q, k, v, o, do, ddq, ddk, ddv, scale, is_causal, sliding_window_length
    # )
    # d = dd = stats2 = 1

    s = (
        jnp.einsum(
            "qd,kd->qk", q, k, precision=precision, preferred_element_type=jnp.float32
        )
        * scale
    )

    if is_causal:
        qs, ks = q.shape[0], k.shape[0]
        mask = get_mask_bottom_right(qs, ks, sliding_window_length)
        s = jnp.where(mask, s, -jnp.inf)

    p = jnp.exp(s - stats[:, None]).astype(jnp.float32)

    dp_bwd = do @ v.T
    dds = scale * (ddq @ k.T + q @ ddk.T)

    # d = sum_k (dP * P)_qk
    # => dd = sum_k (ddS * P)_qk
    # dds = ddq k^T + q ddk^T
    # => dd = sum_k ((ddq k^T) * p) + sum_k ((q ddk^T) * p)
    d = einsum("qd,qd->q", o, do)[:, None]
    dd = einsum("qk,qk->q", dds, p)[:, None]

    # dp2 = (dds * dp_bwd - dds * d) - dp_bwd * dd + do @ ddv.T
    dp2 = do @ ddv.T - dp_bwd * dd - dds * d + dp_bwd * dds
    ddp = p * (dds - dd)

    ds2 = scale * p * (dp2 - einsum("qk,qk->q", dp2, p)[:, None])
    # ds2 = scale * p * (dp2 - stats2)
    ds = scale * p * (dp_bwd - d)

    dq2 = ds @ ddk + ds2 @ k
    dk2 = ds.T @ ddq + ds2.T @ q

    dv2 = ddp.T @ do

    ddo = p @ ddv + ddp @ v
    # dq2 = dk2 = ddo = None

    return dq2, dk2, dv2, ddo


def attn_bwd_bwd_original(q, k, v, do, ddq, ddk, ddv, scale=1.0, is_causal=False):
    qs, ks = q.shape[0], k.shape[0]
    s = einsum(q, k, "q d, k d -> q k") * scale

    if is_causal:
        qi = jnp.arange(qs)[:, None]
        kj = jnp.arange(ks)[None, :]
        qi = qi + ks - qs
        mask = qi >= kj
        s = jnp.where(mask, s, -jnp.inf)

    p = _softmax(s, axis=-1)

    dp_bwd = do @ v.T
    dds = scale * (ddq @ k.T + q @ ddk.T)
    d = jnp.sum(p * dp_bwd, axis=-1, keepdims=True, dtype=jnp.float32).astype(q.dtype)
    ddsp_red = jnp.sum(dds * p, axis=-1, keepdims=True)

    dp2 = dds * (dp_bwd - d) - dp_bwd * ddsp_red + do @ ddv.T
    ddp = p * dds - p * ddsp_red

    ds2 = scale * (p * (dp2 - jnp.sum(p * dp2, axis=-1, keepdims=True)))
    ds = scale * (p * (dp_bwd - d))
    dq2 = ds @ ddk + ds2 @ k
    dk2 = ds.T @ ddq + ds2.T @ q

    dv2 = ddp.T @ do

    ddo = p @ ddv + ddp @ v

    return dq2, dk2, dv2, ddo


def check_backward_matched(forward_fn, backward_fn, inp, **settings):
    random_input = random_like_tree(inp, jrandom.PRNGKey(0))

    forward_out, auto_vjp = jax.vjp(partial(forward_fn, **settings), *random_input)
    random_dout = random_like_tree(forward_out, jrandom.PRNGKey(1))

    print(f"{len(random_dout)=}")

    q, k, v, do = inp
    ddq, ddk, ddv = random_dout

    import torch

    print(torch.tensor(q))
    #     q,
    #     k,
    #     v,
    #     do,
    #     ddq,
    #     ddk,
    #     ddv,
    print(f"{len(inp)=}")

    (input_items / "q.npy")

    auto_din = auto_vjp(random_dout)
    if not isinstance(random_dout, tuple):
        random_dout = (random_dout,)
    bwd_din = backward_fn(*random_input, *random_dout, **settings)
    # pp(diff_metrics(auto_din, bwd_din))


def scuffed_save_bwd_bwd(forward_fn, backward_fn, inp, **settings):
    random_input = random_like_tree(inp, jrandom.PRNGKey(0))

    forward_out, auto_vjp = jax.vjp(partial(forward_fn, **settings), *random_input)
    random_dout = random_like_tree(forward_out, jrandom.PRNGKey(1))

    q, k, v, do = inp
    ddq, ddk, ddv = random_dout

    import torch

    S = (q @ k.T) * scale
    P = softmax(S)
    _O = P @ v
    L = jax.nn.logsumexp(S, axis=-1)

    torch.save(
        torch.from_numpy(np.asarray(q, dtype=np.float32)), input_tensors_path / "q.pt"
    )
    torch.save(
        torch.from_numpy(np.asarray(k, dtype=np.float32)), input_tensors_path / "k.pt"
    )
    torch.save(
        torch.from_numpy(np.asarray(v, dtype=np.float32)), input_tensors_path / "v.pt"
    )
    torch.save(
        torch.from_numpy(np.asarray(_O, dtype=np.float32)), input_tensors_path / "o.pt"
    )
    torch.save(
        torch.from_numpy(np.asarray(do, dtype=np.float32)), input_tensors_path / "do.pt"
    )
    torch.save(
        torch.from_numpy(np.asarray(ddq, dtype=np.float32)),
        input_tensors_path / "ddq.pt",
    )
    torch.save(
        torch.from_numpy(np.asarray(ddk, dtype=np.float32)),
        input_tensors_path / "ddk.pt",
    )
    torch.save(
        torch.from_numpy(np.asarray(ddv, dtype=np.float32)),
        input_tensors_path / "ddv.pt",
    )
    torch.save(
        torch.from_numpy(np.asarray(L, dtype=np.float32)),
        input_tensors_path / "L.pt",
    )

    auto_din = auto_vjp(random_dout)
    if not isinstance(random_dout, tuple):
        random_dout = (random_dout,)
    # bwd_din = backward_fn(*random_input, *random_dout, **settings)
    bwd_din = backward_fn(
        stats=L,
        q=q,
        k=k,
        v=v,
        o=_O,
        do=do,
        ddq=ddq,
        ddk=ddk,
        ddv=ddv,
        scale=scale,
        is_causal=False,
    )
    print(f"{len(bwd_din)=}")
    # pp(diff_metrics(auto_din, bwd_din))
    dq2, dk2, dv2, ddo = bwd_din
    torch.save(
        torch.from_numpy(np.asarray(dq2, dtype=np.float32)),
        output_tensors_path / "dq2.pt",
    )
    torch.save(
        torch.from_numpy(np.asarray(dk2, dtype=np.float32)),
        output_tensors_path / "dk2.pt",
    )
    torch.save(
        torch.from_numpy(np.asarray(dv2, dtype=np.float32)),
        output_tensors_path / "dv2.pt",
    )
    torch.save(
        torch.from_numpy(np.asarray(ddo, dtype=np.float32)),
        output_tensors_path / "ddo.pt",
    )


def bwdbwd_from_vjp(
    stats,
    q,
    k,
    v,
    o,
    do,
    ddq,
    ddk,
    ddv,
    scale,
    is_causal,
):
    assert not is_causal
    out, bwd_vjp = jax.vjp(
        jax.remat(
            partial(attn_bwd, scale=scale),
            policy=jax.checkpoint_policies.save_only_these_names("q", "k", "v", "do"),
        ),
        q,
        k,
        v,
        do,
    )
    dq2, dk2, dv2, ddo = bwd_vjp((ddq, ddk, ddv))
    return dq2, dk2, dv2, ddo


def compare_bwdbwd_and_bwdbwdstats(
    forward_fn, backward_fn_non_stats, backward_fn_stats, inp, **settings
):
    random_input = random_like_tree(inp, jrandom.PRNGKey(0))

    forward_out, auto_vjp = jax.vjp(partial(forward_fn, **settings), *random_input)
    random_dout = random_like_tree(forward_out, jrandom.PRNGKey(1))

    q, k, v, do = inp
    ddq, ddk, ddv = random_dout

    import torch

    S = (q @ k.T) * scale
    P = softmax(S)
    _O = P @ v
    L = jax.nn.logsumexp(S, axis=-1)

    # torch.save(
    #     torch.from_numpy(np.asarray(q, dtype=np.float32)), input_tensors_path / "q.pt"
    # )
    # torch.save(
    #     torch.from_numpy(np.asarray(k, dtype=np.float32)), input_tensors_path / "k.pt"
    # )
    # torch.save(
    #     torch.from_numpy(np.asarray(v, dtype=np.float32)), input_tensors_path / "v.pt"
    # )
    # torch.save(
    #     torch.from_numpy(np.asarray(_O, dtype=np.float32)), input_tensors_path / "o.pt"
    # )
    # torch.save(
    #     torch.from_numpy(np.asarray(do, dtype=np.float32)), input_tensors_path / "do.pt"
    # )
    # torch.save(
    #     torch.from_numpy(np.asarray(ddq, dtype=np.float32)),
    #     input_tensors_path / "ddq.pt",
    # )
    # torch.save(
    #     torch.from_numpy(np.asarray(ddk, dtype=np.float32)),
    #     input_tensors_path / "ddk.pt",
    # )
    # torch.save(
    #     torch.from_numpy(np.asarray(ddv, dtype=np.float32)),
    #     input_tensors_path / "ddv.pt",
    # )
    # torch.save(
    #     torch.from_numpy(np.asarray(L, dtype=np.float32)),
    #     input_tensors_path / "L.pt",
    # )

    auto_din = auto_vjp(random_dout)
    if not isinstance(random_dout, tuple):
        random_dout = (random_dout,)
    # bwd_din = backward_fn(*random_input, *random_dout, **settings)
    dq2, dk2, dv2, ddo = backward_fn_non_stats(
        q=q,
        k=k,
        v=v,
        # o=_O,
        do=do,
        ddq=ddq,
        ddk=ddk,
        ddv=ddv,
        scale=scale,
        is_causal=False,
    )
    dq2_stats, dk2_stats, dv2_stats, ddo_stats = bwdbwd_from_vjp(
        stats=L,
        q=q,
        k=k,
        v=v,
        o=_O,
        do=do,
        ddq=ddq,
        ddk=ddk,
        ddv=ddv,
        scale=scale,
        is_causal=False,
    )
    print(f"{jnp.std(dq2-dq2_stats).item()=} {jnp.max(dq2-dq2_stats).item()=}")
    # print(jnp.std(dk2 - dk2_stats))
    # print(jnp.std(dv2 - dv2_stats))
    # print(jnp.std(ddo - ddo_stats))
    print(f"{jnp.std(dk2 - dk2_stats).item()=} {jnp.max(dk2 - dk2_stats).item()=}")
    print(f"{jnp.std(dv2 - dv2_stats).item()=} {jnp.max(dv2 - dv2_stats).item()=}")
    print(f"{jnp.std(ddo - ddo_stats).item()=} {jnp.max(ddo - ddo_stats).item()=}")


if __name__ == "__main__":
    # print(get_mask_bottom_right(10, 20, 5).astype(jnp.int32))

    test_data_folder = Path("test_data")
    (input_tensors_path := test_data_folder / "input_items").mkdir(
        parents=True, exist_ok=True
    )
    (output_tensors_path := test_data_folder / "output_items").mkdir(
        parents=True, exist_ok=True
    )

    key1, key2, key3 = jrandom.split(jrandom.PRNGKey(42), 3)
    nq = 128
    nkv = 256
    d_in = 64
    d_out = 64
    dtype = jnp.bfloat16
    scale = jnp.array(1.0 / jnp.sqrt(d_in), dtype=jnp.float32)
    q = jrandom.normal(key1, (nq, d_in), dtype=dtype)
    k = jrandom.normal(key2, (nkv, d_in), dtype=dtype)
    v = jrandom.normal(key3, (nkv, d_out), dtype=dtype)
    do = jrandom.normal(key1, (nq, d_out), dtype=dtype)

    out, fwd_vjp = jax.vjp(
        jax.remat(
            partial(attn_fwd, scale=scale),
            policy=jax.checkpoint_policies.save_only_these_names(
                "q", "k", "v", "denom"
            ),
        ),
        q,
        k,
        v,
    )
    # jaxpr_graph(fwd_vjp, do).render(filename='auto_bwd')
    # jaxpr_graph(partial(attn_bwd, scale=scale), q, k, v, do).render(filename='custom_bwd')

    jnp.set_printoptions(linewidth=250, precision=32, floatmode="fixed")

    out, bwd_vjp = jax.vjp(
        jax.remat(
            partial(attn_bwd, scale=scale),
            policy=jax.checkpoint_policies.save_only_these_names("q", "k", "v", "do"),
        ),
        q,
        k,
        v,
        do,
    )

    # print(out)
    # jaxpr_graph(bwd_vjp, (q, k, v)).render(filename='auto_bwd_bwd')

    # check_backward_matched(attn_fwd, attn_bwd, (q, k, v), scale=scale, is_causal=True)
    # check_backward_matched(
    #     attn_bwd, attn_bwd_bwd, (q, k, v, do), scale=scale, is_causal=False
    # )
    # jaxpr_graph(attn_bwd_bwd, q, k, v, do, q, k, v).render(filename='custom_bwd_bwd')

    scuffed_save_bwd_bwd(
        # attn_bwd, attn_bwd_bwd, (q, k, v, do), scale=scale, is_causal=False
        attn_bwd,
        attn_bwd_bwd_stats,
        (q, k, v, do),
        scale=scale,
        is_causal=False,
    )
    # compare_bwdbwd_and_bwdbwdstats(
    #     # attn_bwd, attn_bwd_bwd, (q, k, v, do), scale=scale, is_causal=False
    #     attn_bwd,
    #     attn_bwd_bwd,
    #     attn_bwd_bwd_stats,
    #     (q, k, v, do),
    #     scale=scale,
    #     is_causal=False,
    # )

    # attn_bwd_bwd(
    #     q,
    #     k,
    #     v,
    #     do,
    #     ddq,
    #     ddk,
    #     ddv,
    # )

    # def _softmax_jvp(axis, primals, tangents):
    #   (x, where, initial), (x_dot, _, _) = primals, tangents
    #   y = _softmax(x, axis, where, initial)
    #   x_dot = checkpoint_name(x_dot, 'x_dot')
    #   return y, y * (x_dot - jnp.sum((y * x_dot).astype(jnp.float32), axis, where=where, keepdims=True, dtype=jnp.float32)).astype(x_dot.dtype)

    # from ttt.model.grad_viz import jaxpr_graph

    # s = q @ k.T
    # ds = q @ k.T
    # jaxpr_graph(
    #     partial(_softmax_jvp, -1), (s, None, -jnp.inf), (ds, None, None)
    # ).render(filename="softmax_jvp")
    # _out, softmax_vjp_fun = jax.vjp(
    #     partial(_softmax, axis=-1, where=None, initial=-jnp.inf), s
    # )
    # jaxpr_graph(softmax_vjp_fun, ds).render(filename="softmax_vjp")
