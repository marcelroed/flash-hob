import torch
import triton.language as tl
from einops import einsum
from torch import autograd
import triton


def cdiv(a, b):
    return (a + b - 1) // b


@triton.jit
def flash_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    K_BLOCK_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Get the batch index
    query_block_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_block_index * Q_BLOCK_SIZE, 0),
        block_shape=(Q_BLOCK_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_BLOCK_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_BLOCK_SIZE, D),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr)

    o = tl.zeros((Q_BLOCK_SIZE, D), tl.float32)
    l = tl.zeros((Q_BLOCK_SIZE,), tl.float32)
    m = tl.full((Q_BLOCK_SIZE,), -float("inf"), dtype=tl.float32)

    T_k = tl.cdiv(N_KEYS, K_BLOCK_SIZE)

    if is_causal:
        q_idx = query_block_index * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE)

    for j in range(T_k):
        k = tl.load(K_block_ptr)

        if is_causal:
            k_idx = j * K_BLOCK_SIZE + tl.arange(0, K_BLOCK_SIZE)
            S = tl.where(
                q_idx[:, None] >= k_idx[None, :],
                scale * tl.dot(q, k.T),
                -1e6,
            )
        else:
            S = scale * tl.dot(q, k.T)

        m_new = tl.maximum(m, tl.max(S, axis=1))
        P_tilde = tl.exp(S - m_new[:, None])
        alpha = tl.exp(m - m_new)
        l = alpha * l + tl.sum(P_tilde, axis=1)

        v = tl.load(V_block_ptr)
        o = o * alpha[:, None]
        o = tl.dot(P_tilde.to(v.dtype), v, acc=o)

        m = m_new

        K_block_ptr = K_block_ptr.advance((K_BLOCK_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_BLOCK_SIZE, 0))

    # Normalize outputs
    o = o / l[:, None]

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_block_index * Q_BLOCK_SIZE, 0),
        block_shape=(Q_BLOCK_SIZE, D),
        order=(1, 0),
    )
    tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty))

    l = m + tl.log(l)

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_block_index * Q_BLOCK_SIZE,),
        block_shape=(Q_BLOCK_SIZE,),
        order=(0,),
    )
    tl.store(L_block_ptr, l)


class FlashAttention2(autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        # Get the batch size and sequence length
        N_BATCH, N_QUERIES, D = q.shape
        _, N_KEYS, _ = k.shape

        # Allocate the output tensor
        output = torch.empty((N_BATCH, N_QUERIES, D), device=q.device, dtype=q.dtype)
        L = torch.empty((N_BATCH, N_QUERIES), device=q.device, dtype=torch.float)

        scale = D**-0.5

        Q_BLOCK_SIZE = 128
        K_BLOCK_SIZE = 64

        T_q = cdiv(N_QUERIES, Q_BLOCK_SIZE)

        # Call the Triton kernel
        flash_kernel[(T_q, N_BATCH)](
            q,
            k,
            v,
            output,
            L,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            L.stride(0),
            L.stride(1),
            N_QUERIES,
            N_KEYS,
            scale,
            D=D,
            Q_BLOCK_SIZE=Q_BLOCK_SIZE,
            K_BLOCK_SIZE=K_BLOCK_SIZE,
            is_causal=is_causal,
        )

        # Save the input tensors for the backward pass
        ctx.save_for_backward(q, k, v, output, L)
        ctx.is_causal = is_causal

        return output

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        # Call PyTorch backward
        dQ, dK, dV = flash_backward_torch(q, k, v, do, o, L, is_causal=is_causal)

        return dQ, dK, dV, None


@torch.compile(fullgraph=True, dynamic=False)
def flash_backward_torch(q, k, v, do, o, L, is_causal: bool):
    scale = q.shape[-1] ** -0.5
    D = einsum(o, do, "batch query d, batch query d -> batch query")
    S = einsum(q, k, "batch query d, batch key d -> batch query key") * scale
    if is_causal:
        S = torch.where(
            torch.arange(q.shape[-2], device=q.device)[None, :, None]
            >= torch.arange(k.shape[-2], device=q.device)[None, None, :],
            S,
            -1e6,
        )

    P = torch.exp(S - L[:, :, None])
    dV = einsum(P, do, "batch query key, batch query d -> batch key d")
    dP = einsum(do, v, "batch query d, batch key d -> batch query key")
    dS = P * (dP - D[:, :, None])
    dQ = scale * einsum(dS, k, "batch query key, batch key d -> batch query d")
    dK = scale * einsum(dS, q, "batch query key, batch query d -> batch key d")
    return dQ, dK, dV


def run_regular_attention(q, k, v, is_causal=False):
    O = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.mean(O).backward()

    dQ, dK, dV = q.grad, k.grad, v.grad

    return O, dQ, dK, dV


def test_flash():
    batch_size = 8
    n_queries = 128
    n_keys = 128
    d = 64
    q = torch.randn(batch_size, n_queries, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch_size, n_keys, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch_size, n_keys, d, device="cuda", dtype=torch.bfloat16)
    is_causal = False

    output = FlashAttention2.apply(q, k, v, is_causal=is_causal)

    loss = torch.mean(output)
    loss.backward()

    dQ = q.grad
    dK = k.grad
    dV = v.grad



