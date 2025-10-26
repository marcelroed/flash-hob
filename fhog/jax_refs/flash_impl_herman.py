import torch
import triton
import triton.language as tl
from triton.testing import do_bench
from triton.tools.tensor_descriptor import TensorDescriptor

#  REP_MS, WARMUP_MS = 3000, 500
REP_MS, WARMUP_MS = 10_000, 1_000
#  REP_MS, WARMUP_MS = 30_000, 5_000

#  VERIFY_DATA_PATH = "/data/brunborg/data.pt"
#  VERIFY_DATA_PATH = "/data/brunborg/datav2.pt"
VERIFY_DATA_PATH = "/data/brunborg/datav3.pt"

N_HEAD: tl.constexpr = 16
SEQ_LEN: tl.constexpr = 16384
HEAD_DIM: tl.constexpr = 64


@triton.jit
def flash_fwd_kernel(
    L,
    Q_tdesc,
    K_tdesc,
    V_tdesc,
    O_tdesc,
    n_head: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    dtype = tl.bfloat16
    block_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    batch_idx = batch_head_idx // n_head
    head_idx = batch_head_idx % n_head
    start_row = batch_idx + head_idx * seq_len
    Q_offset = start_row + block_idx * block_size
    row_offset = block_idx * block_size + tl.arange(0, block_size)
    col_offset = tl.arange(0, block_size)
    l_max = tl.zeros([block_size], dtype=tl.float32) - float("inf")
    l_sum = tl.zeros([block_size], dtype=tl.float32) + 1.0
    output_accumulator = tl.zeros([block_size, head_dim], dtype=tl.float32)
    logit_scale: tl.constexpr = 0.18033688011112042
    Q = Q_tdesc.load([Q_offset, 0])
    # causal mask
    col_start, col_end = 0, block_idx * block_size
    row_ptr = start_row + col_start
    for col in range(col_start, col_end, block_size):
        col = tl.multiple_of(col, block_size)
        k_blk = K_tdesc.load([row_ptr, 0]).T
        qk = tl.dot(Q, k_blk)
        l_max_next = tl.maximum(l_max, tl.max(qk, 1) * logit_scale)
        qk = qk * logit_scale - l_max_next[:, None]
        qk_exp = tl.math.exp2(qk)
        qk_exp_sum = tl.sum(qk_exp, 1)
        alpha = tl.math.exp2(l_max - l_max_next)
        l_sum = l_sum * alpha + qk_exp_sum
        output_accumulator = output_accumulator * alpha[:, None]
        v_blk = V_tdesc.load([row_ptr, 0])
        qk_exp = qk_exp.to(dtype)
        output_accumulator = tl.dot(qk_exp, v_blk, output_accumulator)
        l_max = l_max_next
        row_ptr += block_size
    # no mask
    col_start, col_end = block_idx * block_size, (block_idx + 1) * block_size
    row_ptr = start_row + col_start
    for col in range(col_start, col_end, block_size):
        col = tl.multiple_of(col, block_size)
        k_blk = K_tdesc.load([row_ptr, 0]).T
        qk = tl.dot(Q, k_blk)
        mask = row_offset[:, None] >= (col + col_offset[None, :])
        qk = qk * logit_scale + tl.where(mask, 0, -1.0e6)
        l_max_next = tl.maximum(l_max, tl.max(qk, 1))
        qk -= l_max_next[:, None]
        qk_exp = tl.math.exp2(qk)
        qk_exp_sum = tl.sum(qk_exp, 1)
        alpha = tl.math.exp2(l_max - l_max_next)
        l_sum = l_sum * alpha + qk_exp_sum
        output_accumulator = output_accumulator * alpha[:, None]
        v_blk = V_tdesc.load([row_ptr, 0])
        qk_exp = qk_exp.to(dtype)
        output_accumulator = tl.dot(qk_exp, v_blk, output_accumulator)
        l_max = l_max_next
        row_ptr += block_size
    l_max += tl.math.log2(l_sum)
    output_accumulator = output_accumulator / l_sum[:, None]
    scale_factor_ptr = L + batch_head_idx * seq_len + row_offset
    tl.store(scale_factor_ptr, l_max)
    O_tdesc.store([Q_offset, 0], output_accumulator.to(dtype))


@triton.jit
def flash_bwd_delta_kernel(
    O, dO, D, seq_len, block_size: tl.constexpr, head_dim: tl.constexpr
):
    seq_idx = tl.program_id(0) * block_size + tl.arange(0, block_size)
    batch_head_idx = tl.program_id(1)
    oemb_idx = tl.arange(0, head_dim)
    O = tl.load(
        O
        + batch_head_idx * head_dim * seq_len
        + seq_idx[:, None] * head_dim
        + oemb_idx[None, :]
    )
    dO = tl.load(
        dO
        + batch_head_idx * head_dim * seq_len
        + seq_idx[:, None] * head_dim
        + oemb_idx[None, :]
    ).to(tl.float32)
    delta_vals = tl.sum(O * dO, axis=1)
    tl.store(D + batch_head_idx * seq_len + seq_idx, delta_vals)


@triton.jit
def flash_bwd_kv_grad(
    dK,
    dV,
    dO,
    Q,
    K,
    V,
    L,
    D,
    stride_tok,
    stride_feat,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    start_seq,
    start_block,
    num_steps,
    causal: tl.constexpr,
):
    block_cols: tl.constexpr = 32 if causal else 64
    block_idx = start_block + tl.arange(0, block_cols)
    seq_idx = start_seq + tl.arange(0, block_size)
    feat_idx = tl.arange(0, head_dim)
    Q_ptr = Q + block_idx[None, :] * stride_tok + feat_idx[:, None] * stride_feat
    dO_ptr = dO + block_idx[:, None] * stride_tok + feat_idx[None, :] * stride_feat
    curr_block = start_block
    block_step = block_cols
    for _ in range(num_steps):
        Q_blk = tl.load(Q_ptr)
        block_idx = curr_block + tl.arange(0, block_cols)
        log_scale = tl.load(L + block_idx)
        attn_logits = tl.dot(K, Q_blk)
        attn_weights = tl.math.exp2(attn_logits - log_scale[None, :])
        if causal:
            mask = block_idx[None, :] >= seq_idx[:, None]
            attn_weights = tl.where(mask, attn_weights, 0.0)
        dO_blk = tl.load(dO_ptr)
        ttn_weights_bf16 = attn_weights.to(tl.bfloat16)
        dV += tl.dot(ttn_weights_bf16, dO_blk)
        D_vals = tl.load(D + block_idx)
        dLdQT = tl.dot(V, tl.trans(dO_blk)).to(tl.float32)
        d = (attn_weights * (dLdQT - D_vals[None, :])).to(tl.bfloat16)
        dK += tl.dot(d, tl.trans(Q_blk))
        curr_block += block_step
        Q_ptr += block_step * stride_tok
        dO_ptr += block_step * stride_tok
    return dK, dV


@triton.jit
def flash_bwd_q_grad_kernel(
    dQ,
    Q,
    K,
    V,
    dO,
    L,
    D,
    stride_tok,
    stride_d,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    row_idx,
    col_idx,
    num_steps,
    causal: tl.constexpr,
):
    Q_row_idx = row_idx + tl.arange(0, BLOCK_M2)
    D_vals = tl.load(D + Q_row_idx)
    last_blk_col_idx = col_idx + (num_steps - 1) * BLOCK_N2
    cur_blk_col_idx = last_blk_col_idx
    blk_col_step = -BLOCK_N2
    feat_idx = tl.arange(0, HEAD_DIM)
    blk_col_idx = cur_blk_col_idx + tl.arange(0, BLOCK_N2)
    K_blk_ptr = K + blk_col_idx[None, :] * stride_tok + feat_idx[:, None] * stride_d
    V_blk_ptr = V + blk_col_idx[None, :] * stride_tok + feat_idx[:, None] * stride_d
    for _ in range(num_steps):
        K_blk = tl.load(K_blk_ptr)
        V_blk = tl.load(V_blk_ptr)
        qk = tl.dot(Q, K_blk)
        attn_weights = tl.math.exp2(qk - L)
        if causal:
            mask_idx = cur_blk_col_idx + tl.arange(0, BLOCK_N2)
            causal_mask = Q_row_idx[:, None] >= mask_idx[None, :]
            attn_weights = tl.where(causal_mask, attn_weights, 0.0)
        V_dot_grad = tl.dot(dO, V_blk).to(tl.float32)
        score_grads = attn_weights * (V_dot_grad - D_vals[:, None])
        score_grads = score_grads.to(tl.bfloat16)
        dQ += tl.dot(score_grads, tl.trans(K_blk))
        cur_blk_col_idx += blk_col_step
        K_blk_ptr += blk_col_step * stride_tok
        V_blk_ptr += blk_col_step * stride_tok
    return dQ


@triton.jit
def flash_bwd_kernel(
    Q,
    K,
    V,
    L,
    D,
    dO,
    dQ,
    dK,
    dV,
    n_head: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
):
    head_idx = tl.program_id(2)
    logit_offset = (head_idx * seq_len).to(tl.int64)
    memory_offset = (
        stride_h * (head_idx % n_head) + stride_z * (head_idx // n_head)
    ).to(tl.int64)
    blk_tile_idx = tl.program_id(0)

    Q += memory_offset
    K += memory_offset
    V += memory_offset
    dO += memory_offset
    dQ += memory_offset
    dK += memory_offset
    dV += memory_offset
    L += logit_offset
    D += logit_offset

    feat_idx = tl.arange(0, head_dim)

    start_seq_idx = blk_tile_idx * block_size
    start_blk_idx = start_seq_idx

    seq_idx = start_seq_idx + tl.arange(0, block_size)

    dv = tl.zeros([block_size, head_dim], dtype=tl.float32)
    dk = tl.zeros([block_size, head_dim], dtype=tl.float32)

    k = tl.load(K + seq_idx[:, None] * stride_tok + feat_idx[None, :] * stride_d)
    v = tl.load(V + seq_idx[:, None] * stride_tok + feat_idx[None, :] * stride_d)

    non_causal_kv_steps: tl.constexpr = 2

    dk, dv = flash_bwd_kv_grad(
        dK=dk,
        dV=dv,
        Q=Q,
        K=k,
        V=v,
        dO=dO,
        L=L,
        D=D,
        stride_tok=stride_tok,
        stride_feat=stride_d,
        block_size=block_size,
        head_dim=head_dim,
        start_seq=start_seq_idx,
        start_block=start_blk_idx,
        num_steps=non_causal_kv_steps,
        causal=True,
    )

    start_blk_idx += non_causal_kv_steps * 32
    non_causal_kv_steps = (seq_len - start_blk_idx) // block_size

    dk, dv = flash_bwd_kv_grad(
        dK=dk,
        dV=dv,
        Q=Q,
        K=k,
        V=v,
        dO=dO,
        L=L,
        D=D,
        stride_tok=stride_tok,
        stride_feat=stride_d,
        block_size=block_size,
        head_dim=head_dim,
        start_seq=start_seq_idx,
        start_block=start_blk_idx,
        num_steps=non_causal_kv_steps,
        causal=False,
    )

    grad_v_ptr = dV + seq_idx[:, None] * stride_tok + feat_idx[None, :] * stride_d
    tl.store(grad_v_ptr, dv)

    dk *= 0.125
    dk_ptr = dK + seq_idx[:, None] * stride_tok + feat_idx[None, :] * stride_d
    tl.store(dk_ptr, dk)

    start_blk_idx = blk_tile_idx * block_size
    end_seq_idx = start_blk_idx + block_size

    mask_block_cols: tl.constexpr = block_size // 2
    seq_row_idx = start_blk_idx + tl.arange(0, block_size)

    q_blk = tl.load(
        Q + seq_row_idx[:, None] * stride_tok + feat_idx[None, :] * stride_d
    )
    dq = tl.zeros([block_size, head_dim], dtype=tl.float32)
    do = tl.load(dO + seq_row_idx[:, None] * stride_tok + feat_idx[None, :] * stride_d)

    log_scale = tl.load(L + seq_row_idx)
    log_scale = log_scale[:, None]

    non_causal_kv_steps = block_size // mask_block_cols
    dq = flash_bwd_q_grad_kernel(
        dQ=dq,
        Q=q_blk,
        K=K,
        V=V,
        dO=do,
        L=log_scale,
        D=D,
        stride_tok=stride_tok,
        stride_d=stride_d,
        BLOCK_M2=block_size,
        BLOCK_N2=mask_block_cols,
        HEAD_DIM=head_dim,
        row_idx=start_blk_idx,
        col_idx=end_seq_idx - non_causal_kv_steps * mask_block_cols,
        num_steps=non_causal_kv_steps,
        causal=True,  #
    )
    end_seq_idx -= non_causal_kv_steps * mask_block_cols

    non_causal_kv_steps = end_seq_idx // block_size
    dq = flash_bwd_q_grad_kernel(
        dQ=dq,
        Q=q_blk,
        K=K,
        V=V,
        dO=do,
        L=log_scale,
        D=D,
        stride_tok=stride_tok,
        stride_d=stride_d,
        BLOCK_M2=block_size,
        BLOCK_N2=block_size,
        HEAD_DIM=head_dim,
        row_idx=start_blk_idx,
        col_idx=end_seq_idx - non_causal_kv_steps * block_size,
        num_steps=non_causal_kv_steps,
        causal=False,
    )

    dq_ptr = dQ + seq_row_idx[:, None] * stride_tok + feat_idx[None, :] * stride_d
    tl.store(dq_ptr, dq * 0.6931471824645996)


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal):
        O = torch.empty_like(Q)

        L = torch.empty(
            (Q.shape[0], Q.shape[1], Q.shape[2]), device=Q.device, dtype=torch.float32
        )

        Y_DIM = N_HEAD * SEQ_LEN
        block_size = 64

        Q_tdesc = TensorDescriptor(
            Q,
            shape=[Y_DIM, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[block_size, HEAD_DIM],
        )
        V_tdesc = TensorDescriptor(
            V,
            shape=[Y_DIM, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[block_size, HEAD_DIM],
        )
        K_tdesc = TensorDescriptor(
            K,
            shape=[Y_DIM, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[block_size, HEAD_DIM],
        )
        O_tdesc = TensorDescriptor(
            O,
            shape=[Y_DIM, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[block_size, HEAD_DIM],
        )

        flash_fwd_kernel[Q.shape[2] // block_size, Q.shape[0] * Q.shape[1], 1](
            L=L,
            Q_tdesc=Q_tdesc,
            K_tdesc=K_tdesc,
            V_tdesc=V_tdesc,
            O_tdesc=O_tdesc,
            n_head=N_HEAD,
            seq_len=SEQ_LEN,
            head_dim=HEAD_DIM,
            block_size=block_size,
            num_warps=4,
            num_stages=2,
            num_ctas=1,
        )

        ctx.save_for_backward(Q, K, V, O, L)

        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, o, L = ctx.saved_tensors

        dO = dO.contiguous()
        dQ = torch.empty_like(q)
        dK = torch.empty_like(k)
        dV = torch.empty_like(v)

        D = torch.empty_like(L)

        pregame_size = 256
        flash_bwd_delta_kernel[SEQ_LEN // pregame_size, N_HEAD](
            O=o,
            dO=dO,
            D=D,
            seq_len=SEQ_LEN,
            head_dim=HEAD_DIM,
            block_size=pregame_size,
        )

        flash_bwd_kernel[SEQ_LEN // 64, 1, N_HEAD](
            Q=q,
            K=k * 0.18033688011112042,
            V=v,
            L=L,
            D=D,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            n_head=N_HEAD,
            seq_len=SEQ_LEN,
            head_dim=HEAD_DIM,
            stride_z=q.stride(0),
            stride_h=q.stride(1),
            stride_tok=q.stride(2),
            stride_d=q.stride(3),
            block_size=64,
            num_ctas=1,
            num_stages=5,
            num_warps=4,
        )

        return dQ, dK, dV, None, None


flash = FlashAttention2.apply  # torch.compile makes it around 5/1000 slower


def get_data(seed=10410111410997110):
    torch.manual_seed(seed)
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    q = torch.randn(
        1,
        n_heads,
        sequence_length,
        d_head,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    return q, k, v


def verify_similarity_to_torch_sdpa():
    q, k, v = get_data()

    o_triton = flash(q, k, v, True)
    o_triton.sum().backward()
    dq_triton, dk_triton, dv_triton = q.grad, k.grad, v.grad

    q, k, v = get_data()
    o_torch = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
    )
    o_torch.sum().backward()
    dq_torch, dk_torch, dv_torch = q.grad, k.grad, v.grad

    o_diff = (o_triton - o_torch).abs().mean().item()
    dq_diff = (dq_triton - dq_torch).abs().mean().item()
    dk_diff = (dk_triton - dk_torch).abs().mean().item()
    dv_diff = (dv_triton - dv_torch).abs().mean().item()
    assert o_diff < 5e-5
    assert dq_diff < 8e-5
    assert dk_diff < 2e-4
    assert dv_diff < 3e-4
    print(f"{o_diff=}")
    print(f"{dq_diff=}")
    print(f"{dk_diff=}")
    print(f"{dv_diff=}")
    torch.testing.assert_close(o_triton, o_torch, atol=4e-3, rtol=1e-3)

    print(
        "✅ Verification successful: Triton kernel and torch sdpa same output and gradients"
    )


def store():
    q, k, v = get_data()
    o = flash(q, k, v, True)
    o.sum().backward()
    data = {
        "q": q.detach().cpu(),
        "k": k.detach().cpu(),
        "v": v.detach().cpu(),
        "o": o.detach().cpu(),
        "dq": q.grad.detach().cpu(),
        "dk": k.grad.detach().cpu(),
        "dv": v.grad.detach().cpu(),
    }
    torch.save(data, VERIFY_DATA_PATH)


def verify():
    data = torch.load(VERIFY_DATA_PATH)
    q, k, v = get_data()
    o = flash(q, k, v, True)
    o.sum().backward()

    def assert_same(current, key):
        assert torch.equal(current.detach().cpu(), data[key]), f"Mismatch {key}"

    assert_same(q, "q")
    assert_same(k, "k")
    assert_same(v, "v")
    assert_same(o, "o")

    def assert_close(current, key):
        assert torch.allclose(
            current.detach().cpu(), data[key], atol=1e-3, rtol=1e-4
        ), f"Mismatch {key}"

    assert_close(q.grad, "dq")
    assert_close(k.grad, "dk")
    assert_close(v.grad, "dv")
    #  assert_same(q.grad, "dq")
    #  assert_same(k.grad, "dk")
    #  assert_same(v.grad, "dv")
    print(
        "✅ Verification successful: Triton kernel and torch sdpa same output and gradients"
    )


def benchh():
    q, k, v = get_data()

    def run():
        o = flash(q, k, v, True)
        o.sum().backward()

    results = do_bench(run, rep=REP_MS, warmup=WARMUP_MS)
    print(results)


if __name__ == "__main__":
    #  store()
    #  verify_similarity_to_torch_sdpa()
    #  verify()
    benchh()
