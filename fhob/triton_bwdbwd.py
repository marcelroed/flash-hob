import torch
import triton
import triton.language as tl


@triton.jit
def bwdbwd_kernel_stage1(
    # Inputs
    Q_ptr,
    K_ptr,
    V_ptr,
    D_ptr,
    dO_ptr,
    ddQ_ptr,
    ddK_ptr,
    ddV_ptr,
    L_ptr,
    # Outputs
    dQ2_ptr,
    ddO_ptr,
    dD_ptr,
    B_ptr,
    # Input strides
    stride_Qb, stride_Qi, stride_Qd,
    stride_Kb, stride_Kj, stride_Kd,
    stride_Vb, stride_Vj, stride_Vd,
    stride_Db, stride_Di,
    stride_dOb, stride_dOi, stride_dOd,
    stride_ddQb, stride_ddQi, stride_ddQd,
    stride_ddKb, stride_ddKj, stride_ddKd,
    stride_ddVb, stride_ddVj, stride_ddVd,
    stride_Lb, stride_Li,
    # Output Strides
    stride_dQ2b, stride_dQ2i, stride_dQ2d,
    stride_ddOb, stride_ddOi, stride_ddOd,
    stride_dDb, stride_dDi,
    stride_Bb, stride_Bi,
    # Sizes
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    K_BLOCK_SIZE: tl.constexpr,
):  # fmt: off
    query_block_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Compute block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_Qb,
        shape=(N_QUERIES, D),
        strides=(stride_Qi, stride_Qd),
        offset=(query_block_index * Q_BLOCK_SIZE, 0),
        block_shape=(Q_BLOCK_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_Kb,
        shape=(N_KEYS, D),
        strides=(stride_Kj, stride_Kd),
        block_shape=(K_BLOCK_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_Vb,
        shape=(N_KEYS, D),
        strides=(stride_Vj, stride_Vd),
        block_shape=(K_BLOCK_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_Lb,
        shape=(N_QUERIES,),
        strides=(stride_Li,),
        block_shape=(Q_BLOCK_SIZE,),
        offset=(query_block_index * Q_BLOCK_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + batch_index * stride_Db,
        shape=(N_QUERIES,),
        strides=(stride_Di,),
        block_shape=(Q_BLOCK_SIZE,),
        offset=(query_block_index * Q_BLOCK_SIZE,),
        order=(0,),
    )

    ddQ_block_ptr = tl.make_block_ptr(
        base=ddQ_ptr + batch_index * stride_ddQb,
        shape=(N_QUERIES, D),
        strides=(stride_ddQi, stride_ddQd),
        offset=(query_block_index * Q_BLOCK_SIZE, 0),
        block_shape=(Q_BLOCK_SIZE, D),
        order=(1, 0),
    )

    ddK_block_ptr = tl.make_block_ptr(
        base=ddK_ptr + batch_index * stride_ddKb,
        shape=(N_KEYS, D),
        strides=(stride_ddKj, stride_ddKd),
        block_shape=(K_BLOCK_SIZE, D),
        order=(1, 0),
    )

    ddV_block_ptr = tl.make_block_ptr(
        base=ddV_ptr + batch_index * stride_ddVb,
        shape=(N_KEYS, D),
        strides=(stride_ddVj, stride_ddVd),
        block_shape=(K_BLOCK_SIZE, D),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_index * stride_dOb,
        shape=(N_QUERIES, D),
        strides=(stride_dOi, stride_dOd),
        block_shape=(Q_BLOCK_SIZE, D),
        offset=(query_block_index * Q_BLOCK_SIZE, 0),
        order=(1, 0),
    )

    dD_block_ptr = tl.make_block_ptr(
        base=dD_ptr + batch_index * stride_dDb,
        shape=(N_QUERIES,),
        strides=(stride_dDi,),
        block_shape=(Q_BLOCK_SIZE,),
        offset=(query_block_index * Q_BLOCK_SIZE,),
        order=(0,),
    )
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr + batch_index * stride_Bb,
        shape=(N_QUERIES,),
        strides=(stride_Bi,),
        block_shape=(Q_BLOCK_SIZE,),
        offset=(query_block_index * Q_BLOCK_SIZE,),
        order=(0,),
    )

    T_k = tl.cdiv(N_KEYS, K_BLOCK_SIZE)
    Q_i = tl.load(Q_block_ptr)
    L_i = tl.load(L_block_ptr)
    dO_i = tl.load(dO_block_ptr)
    ddQ_i = tl.load(ddQ_block_ptr)

    # Loop over K blocks to compute the reduced dD
    dD_i_acc = tl.zeros((Q_BLOCK_SIZE,), dtype=tl.float32)

    for j in range(T_k):  # COMPUTING dD_i
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)
        ddK_j = tl.load(ddK_block_ptr)
        ddV_j = tl.load(ddV_block_ptr)

        # Compute attention scores
        S_ij = tl.dot(Q_i, K_j.T) * scale
        P_ij = tl.exp(S_ij - L_i[:, None])

        dP_ij = tl.dot(dO_i, V_j)

        ddS_ij = (tl.dot(ddQ_i, K_j.T) + tl.dot(Q_i, ddK_j.T)) * scale

        # TODO: Try tl.dot((b, 1, k), (b, k, 1)) with acc
        dD_i = tl.sum(ddS_ij * P_ij, axis=1)
        dD_i_acc += dD_i
        K_block_ptr = K_block_ptr.advance((K_BLOCK_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_BLOCK_SIZE, 0))
        ddK_block_ptr = ddK_block_ptr.advance((K_BLOCK_SIZE, 0))
        ddV_block_ptr = ddV_block_ptr.advance((K_BLOCK_SIZE, 0))

    tl.store(dD_block_ptr, dD_i_acc)  # Finished computing dD_i
    K_block_ptr = K_block_ptr.advance((-T_k * K_BLOCK_SIZE, 0))
    V_block_ptr = V_block_ptr.advance((-T_k * K_BLOCK_SIZE, 0))
    ddK_block_ptr = ddK_block_ptr.advance((-T_k * K_BLOCK_SIZE, 0))
    ddV_block_ptr = ddV_block_ptr.advance((-T_k * K_BLOCK_SIZE, 0))

    D_i = tl.load(D_block_ptr)

    # dP2_ij = tl.dot(dO_i, ddV_j.T) - dP_ij * dD_i[:, None] - ddS * d[:, None] + dP * ddS
    B_i_acc = tl.zeros((Q_BLOCK_SIZE,), dtype=tl.float32)

    for j in range(T_k):  # COMPUTING B_i
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)
        ddK_j = tl.load(ddK_block_ptr)
        ddV_j = tl.load(ddV_block_ptr)

        # Compute attention scores
        S_ij = tl.dot(Q_i, K_j.T) * scale
        P_ij = tl.exp(S_ij - L_i[:, None])

        dP_ij = tl.dot(dO_i, V_j)

        ddS_ij = (tl.dot(ddQ_i, K_j.T) + tl.dot(Q_i, ddK_j.T)) * scale

        dP2_ij = (
            tl.dot(dO_i, ddV_j.T)
            - dP_ij * dD_i_acc[:, None]
            - ddS_ij * D_i[:, None]
            + dP_ij * ddS_ij
        )
        B_i = tl.sum(dP2_ij * P_ij, axis=1)
        B_i_acc += B_i

        # TODO: Try tl.dot((b, 1, k), (b, k, 1)) with acc
        K_block_ptr = K_block_ptr.advance((K_BLOCK_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_BLOCK_SIZE, 0))
        ddK_block_ptr = ddK_block_ptr.advance((K_BLOCK_SIZE, 0))
        ddV_block_ptr = ddV_block_ptr.advance((K_BLOCK_SIZE, 0))

    tl.store(B_block_ptr, B_i_acc)
    K_block_ptr = K_block_ptr.advance((-T_k * K_BLOCK_SIZE, 0))
    V_block_ptr = V_block_ptr.advance((-T_k * K_BLOCK_SIZE, 0))
    ddK_block_ptr = ddK_block_ptr.advance((-T_k * K_BLOCK_SIZE, 0))
    ddV_block_ptr = ddV_block_ptr.advance((-T_k * K_BLOCK_SIZE, 0))

    dQ2_i_acc = tl.zeros((Q_BLOCK_SIZE, D), dtype=tl.float32)
    ddO_i_acc = tl.zeros((Q_BLOCK_SIZE, D), dtype=tl.float32)
    for j in range(T_k):  # COMPUTING B_i
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)
        ddK_j = tl.load(ddK_block_ptr)
        ddV_j = tl.load(ddV_block_ptr)

        # Compute attention scores
        S_ij = tl.dot(Q_i, K_j.T) * scale
        P_ij = tl.exp(S_ij - L_i[:, None])

        dP_ij = tl.dot(dO_i, V_j)

        ddS_ij = (tl.dot(ddQ_i, K_j.T) + tl.dot(Q_i, ddK_j.T)) * scale

        dP2_ij = (
            tl.dot(dO_i, ddV_j.T)
            - dP_ij * dD_i_acc[:, None]
            - ddS_ij * D_i[:, None]
            + dP_ij * ddS_ij
        )

        dS2_ij = P_ij * (dP2_ij - B_i_acc[:, None]) * scale

        dS_ij = scale * P_ij * (dP_ij - D_i[:, None])
        dQ2_i_acc = tl.dot(dS_ij, ddK_j, acc=dQ2_i_acc)
        dQ2_i_acc = tl.dot(dS2_ij, K_j, acc=dQ2_i_acc)

        ddP_ij = P_ij * (ddS_ij - dD_i_acc[:, None])
        ddO_i_acc = tl.dot(ddP_ij, V_j, acc=ddO_i_acc)
        ddO_i_acc = tl.dot(P_ij, ddV_j, acc=ddO_i_acc)

        # TODO: Try tl.dot((b, 1, k), (b, k, 1)) with acc
        K_block_ptr = K_block_ptr.advance((K_BLOCK_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_BLOCK_SIZE, 0))
        ddK_block_ptr = ddK_block_ptr.advance((K_BLOCK_SIZE, 0))
        ddV_block_ptr = ddV_block_ptr.advance((K_BLOCK_SIZE, 0))

    dQ2_block_ptr = tl.make_block_ptr(
        base=dQ2_ptr + batch_index * stride_dQ2b,
        shape=(N_QUERIES, D),
        strides=(stride_dQ2i, stride_dQ2d),
        block_shape=(Q_BLOCK_SIZE, D),
        offset=(query_block_index * Q_BLOCK_SIZE, 0),
        order=(1, 0),
    )
    ddO_block_ptr = tl.make_block_ptr(
        base=ddO_ptr + batch_index * stride_ddOb,
        shape=(N_QUERIES, D),
        strides=(stride_ddOi, stride_ddOd),
        block_shape=(Q_BLOCK_SIZE, D),
        offset=(query_block_index * Q_BLOCK_SIZE, 0),
        order=(1, 0),
    )
    tl.store(dQ2_block_ptr, dQ2_i_acc)
    tl.store(ddO_block_ptr, ddO_i_acc)


@triton.jit
def bwdbwd_kernel_stage2(
    # Inputs
    Q_ptr,
    K_ptr,
    V_ptr,
    D_ptr,
    dO_ptr,
    ddQ_ptr,
    ddK_ptr,
    ddV_ptr,
    L_ptr,
    dD_ptr,
    B_ptr,
    # Outputs
    dK2_ptr,
    dV2_ptr,
    # Input strides
    stride_Qb, stride_Qi, stride_Qd,
    stride_Kb, stride_Kj, stride_Kd,
    stride_Vb, stride_Vj, stride_Vd,
    stride_Db, stride_Di,
    stride_dOb, stride_dOi, stride_dOd,
    stride_ddQb, stride_ddQi, stride_ddQd,
    stride_ddKb, stride_ddKj, stride_ddKd,
    stride_ddVb, stride_ddVj, stride_ddVd,
    stride_Lb, stride_Li,
    stride_dDb, stride_dDi,
    stride_Bb, stride_Bi,
    # Output Strides
    stride_dK2b, stride_dK2j, stride_dK2d,
    stride_dV2b, stride_dV2j, stride_dV2d,
    # Sizes
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    K_BLOCK_SIZE: tl.constexpr,
):  # fmt: off
    key_block_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Compute block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_Qb,
        shape=(N_QUERIES, D),
        strides=(stride_Qi, stride_Qd),
        offset=(0, 0),
        block_shape=(Q_BLOCK_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_Kb,
        shape=(N_KEYS, D),
        strides=(stride_Kj, stride_Kd),
        offsets=(key_block_index * K_BLOCK_SIZE, 0),
        block_shape=(K_BLOCK_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_Vb,
        shape=(N_KEYS, D),
        strides=(stride_Vj, stride_Vd),
        offsets=(key_block_index * K_BLOCK_SIZE, 0),
        block_shape=(K_BLOCK_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_Lb,
        shape=(N_QUERIES,),
        strides=(stride_Li,),
        block_shape=(Q_BLOCK_SIZE,),
        offsets=(0,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + batch_index * stride_Db,
        shape=(N_QUERIES,),
        strides=(stride_Di,),
        block_shape=(Q_BLOCK_SIZE,),
        offsets=(0,),
        order=(0,),
    )

    ddQ_block_ptr = tl.make_block_ptr(
        base=ddQ_ptr + batch_index * stride_ddQb,
        shape=(N_QUERIES, D),
        strides=(stride_ddQi, stride_ddQd),
        offsets=(0, 0),
        block_shape=(Q_BLOCK_SIZE, D),
        order=(1, 0),
    )

    ddK_block_ptr = tl.make_block_ptr(
        base=ddK_ptr + batch_index * stride_ddKb,
        shape=(N_KEYS, D),
        strides=(stride_ddKj, stride_ddKd),
        offsets=(key_block_index * K_BLOCK_SIZE, 0),
        block_shape=(K_BLOCK_SIZE, D),
        order=(1, 0),
    )

    ddV_block_ptr = tl.make_block_ptr(
        base=ddV_ptr + batch_index * stride_ddVb,
        shape=(N_KEYS, D),
        strides=(stride_ddVj, stride_ddVd),
        offsets=(key_block_index * K_BLOCK_SIZE, 0),
        block_shape=(K_BLOCK_SIZE, D),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_index * stride_dOb,
        shape=(N_QUERIES, D),
        strides=(stride_dOi, stride_dOd),
        block_shape=(Q_BLOCK_SIZE, D),
        order=(1, 0),
    )

    dD_block_ptr = tl.make_block_ptr(
        base=dD_ptr + batch_index * stride_dDb,
        shape=(N_QUERIES,),
        strides=(stride_dDi,),
        offsets=(0,),
        block_shape=(Q_BLOCK_SIZE,),
        order=(0,),
    )
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr + batch_index * stride_Bb,
        shape=(N_QUERIES,),
        strides=(stride_Bi,),
        block_shape=(Q_BLOCK_SIZE,),
        offsets=(0,),
        order=(0,),
    )

    T_q = tl.cdiv(N_QUERIES, Q_BLOCK_SIZE)
    Q_i = tl.load(Q_block_ptr)
    L_i = tl.load(L_block_ptr)
    dO_i = tl.load(dO_block_ptr)
    ddQ_i = tl.load(ddQ_block_ptr)

    dK2_j_acc = tl.zeros((K_BLOCK_SIZE, D), dtype=tl.float32)
    dV2_j_acc = tl.zeros((K_BLOCK_SIZE, D), dtype=tl.float32)

    K_j = tl.load(K_block_ptr)
    V_j = tl.load(V_block_ptr)
    ddK_j = tl.load(ddK_block_ptr)
    ddV_j = tl.load(ddV_block_ptr)
    for i in range(T_q):  # COMPUTING dK2_j and dV2_j
        Q_i = tl.load(Q_block_ptr)
        L_i = tl.load(L_block_ptr)
        dO_i = tl.load(dO_block_ptr)
        ddQ_i = tl.load(ddQ_block_ptr)
        dD_i = tl.load(dD_block_ptr)
        B_i = tl.load(B_block_ptr)
        D_i = tl.load(D_block_ptr)
        dD_i = tl.load(dD_block_ptr)

        # Compute attention scores
        S_ij = tl.dot(Q_i, K_j.T) * scale
        P_ij = tl.exp(S_ij - L_i[:, None])

        dP_ij = tl.dot(dO_i, V_j)

        ddS_ij = (tl.dot(ddQ_i, K_j.T) + tl.dot(Q_i, ddK_j.T)) * scale

        dP2_ij = (
            tl.dot(dO_i, ddV_j.T)
            - dP_ij * dD_i[:, None]
            - ddS_ij * D_i[:, None]
            + dP_ij * ddS_ij
        )

        dS2_ij = P_ij * (dP2_ij - B_i[:, None]) * scale

        dS_ij = scale * P_ij * (dP_ij - D_i[:, None])

        ddP_ij = P_ij * (ddS_ij - dD_i[:, None])
        dV2_j_acc = tl.dot(ddP_ij.T, dO_i, acc=dV2_j_acc)

        dK2_j_acc = tl.dot(dS_ij.T, ddQ_i, acc=dK2_j_acc)
        dK2_j_acc = tl.dot(dS2_ij.T, Q_i, acc=dK2_j_acc)

        # TODO: Try tl.dot((b, 1, k), (b, k, 1)) with acc
        Q_block_ptr = Q_block_ptr.advance((Q_BLOCK_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_BLOCK_SIZE,))
        dO_block_ptr = dO_block_ptr.advance((Q_BLOCK_SIZE, 0))
        ddQ_block_ptr = ddQ_block_ptr.advance((Q_BLOCK_SIZE, 0))
        dD_block_ptr = dD_block_ptr.advance((Q_BLOCK_SIZE,))
        B_block_ptr = B_block_ptr.advance((Q_BLOCK_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_BLOCK_SIZE,))
        dD_block_ptr = dD_block_ptr.advance((Q_BLOCK_SIZE,))

    dK2_block_ptr = tl.make_block_ptr(
        base=dK2_ptr + batch_index * stride_dK2b,
        shape=(N_KEYS, D),
        strides=(stride_dK2j, stride_dK2d),
        block_shape=(K_BLOCK_SIZE, D),
        offsets=(key_block_index * K_BLOCK_SIZE, 0),
        order=(1, 0),
    )
    dV2_block_ptr = tl.make_block_ptr(
        base=dV2_ptr + batch_index * stride_dV2b,
        shape=(N_KEYS, D),
        strides=(stride_dV2j, stride_dV2d),
        block_shape=(K_BLOCK_SIZE, D),
        offsets=(key_block_index * K_BLOCK_SIZE, 0),
        order=(1, 0),
    )
    tl.store(dK2_block_ptr, dK2_j_acc)
    tl.store(dV2_block_ptr, dV2_j_acc)






class Bwd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q, K, V, dO, L, scale,
    ):
        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[1]

        T_q = 

        