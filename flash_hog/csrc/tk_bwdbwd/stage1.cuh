// ThunderKittens kernel: stage 1 of the causal-attention double-backward (SM90 only).

#pragma once

#include "kittens.cuh"

namespace flash_hog_tk {
namespace stage1 {

constexpr int HEAD_DIM = 64;

template<int D> struct bwd_over_bwd_attend_ker_tile_dims {};
template<> struct bwd_over_bwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64);    // head_dim 
    constexpr static int qo_height  = (4*16);  // wgmma M = 64 (4 warps x 16 rows)
    constexpr static int kv_height  = (2*16);  // 32: 4 streamed tensors + width-32 fp32
                                               // register tiles fit the 224-reg budget
    constexpr static int stages     = (4);     // ring: 4 stages x 4 tensors x 4KB = 64KB
};

template<int D> struct bwd_over_bwd_globals {
    using dims   = bwd_over_bwd_attend_ker_tile_dims<D>;
    using q_tile = kittens::st_bf<dims::qo_height, dims::tile_width>;
    using k_tile = kittens::st_bf<dims::kv_height, dims::tile_width>;
    using per_example_vec = kittens::sv_fl<dims::qo_height>;
    using q_gl = kittens::gl<kittens::bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = kittens::gl<kittens::bf16, -1, -1, -1, -1, k_tile>;
};

constexpr int CONSUMER_WARPGROUPS = (2);       // block = 128 queries (2 WGs x 64 rows)
constexpr int PRODUCER_WARPGROUPS = (1);
constexpr int NUM_WARPGROUPS = (CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS);
constexpr int NUM_WORKERS = (NUM_WARPGROUPS * kittens::WARPGROUP_WARPS);

constexpr int TK_TQ = bwd_over_bwd_attend_ker_tile_dims<HEAD_DIM>::qo_height;
constexpr int TK_TK = bwd_over_bwd_attend_ker_tile_dims<HEAD_DIM>::kv_height;
constexpr int KV_STAGES = bwd_over_bwd_attend_ker_tile_dims<HEAD_DIM>::stages;
constexpr int TK_CWG = CONSUMER_WARPGROUPS;
constexpr int TK_WORKERS = NUM_WORKERS;
constexpr int SMEM_BYTES = 160 * 1024;

using tk_qtile = bwd_over_bwd_globals<HEAD_DIM>::q_tile;
using tk_ktile = bwd_over_bwd_globals<HEAD_DIM>::k_tile;
using tk_qgl   = bwd_over_bwd_globals<HEAD_DIM>::q_gl;
using tk_kgl   = bwd_over_bwd_globals<HEAD_DIM>::k_gl;

struct tk_globals {                // field order is load-bearing (aggregate init)
    tk_qgl Q, dO, ddQ, dQ2, ddO;   // query-side (64-row tiles)
    tk_kgl K, V, ddK, ddV;         // key-side (32-row tiles)
    const float* Lp;               // (B,H,T) logsumexp (natural log)
    const float* Dp;               // (B,H,T) D = rowsum(dO*O)
    float* dDo;                    // (B,H,T) out: dD   (stage2 input)
    float* Bo;                     // (B,H,T) out: B    (stage2 input)
    int N;
    float scale;
};

// helper for causal mask
__device__ static inline void tk_cmask(kittens::rt_fl<16, TK_TK>& S, int q16, int kt) {
    #pragma unroll
    for (int j = 0; j < TK_TK / 16; j++) {
        int k16 = kt * (TK_TK / 16) + j;
        auto& sub = reinterpret_cast<kittens::rt_fl<16, 16>&>(S.tiles[0][j]);
        if (k16 > q16) kittens::warp::neg_infty(sub);
        else if (k16 == q16)
            kittens::warp::make_causal(sub, sub, kittens::base_types::constants<float>::neg_infty());
    }
}

__global__ __launch_bounds__(TK_WORKERS * 32, 1)
void tk_stage1(const __grid_constant__ tk_globals g) {
    using namespace kittens;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    const int wid = warpid(), wgid = wid / 4;   // wg 0,1 = consumers; wg 2 = producer

    using G = bwd_over_bwd_globals<HEAD_DIM>;
    using q_tile = G::q_tile;
    using k_tile = G::k_tile;
    using per_example_vec = G::per_example_vec;

    const int qblk = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    constexpr float LOG2E = 1.44269504089f;

    q_tile (&q_s)[CONSUMER_WARPGROUPS]  = al.allocate<G::q_tile, CONSUMER_WARPGROUPS>();
    q_tile (&do_s)[CONSUMER_WARPGROUPS] = al.allocate<G::q_tile, CONSUMER_WARPGROUPS>();
    q_tile (&dq_s)[CONSUMER_WARPGROUPS] = al.allocate<G::q_tile, CONSUMER_WARPGROUPS>();  // ddQ
    k_tile (&k_s)[KV_STAGES]  = al.allocate<G::k_tile, KV_STAGES>();
    k_tile (&v_s)[KV_STAGES]  = al.allocate<G::k_tile, KV_STAGES>();
    k_tile (&dk_s)[KV_STAGES] = al.allocate<G::k_tile, KV_STAGES>();   // ddK
    k_tile (&dv_s)[KV_STAGES] = al.allocate<G::k_tile, KV_STAGES>();   // ddV
    per_example_vec (&l_s)[CONSUMER_WARPGROUPS] = al.allocate<G::per_example_vec, CONSUMER_WARPGROUPS>();
    per_example_vec (&d_s)[CONSUMER_WARPGROUPS] = al.allocate<G::per_example_vec, CONSUMER_WARPGROUPS>();

    const int k_tiles = 4 * (qblk + 1);          // causal: key tiles this block needs
    const int total_k_tiles = 2 * k_tiles;       // the key sequence is streamed twice

    __shared__ kittens::semaphore q_semaphore, k_arrived[KV_STAGES], compute_done[KV_STAGES];

    if (threadIdx.x == 0) {
        init_semaphore(q_semaphore, 0, 1);
        for (int s = 0; s < KV_STAGES; s++) {
            init_semaphore(k_arrived[s], 0, 1);
            init_semaphore(compute_done[s], TK_CWG, 0);
        }

        tma::expect_bytes(q_semaphore, 3 * CONSUMER_WARPGROUPS * sizeof(q_tile));
        for (int i = 0; i < CONSUMER_WARPGROUPS; i++) {
            coord<q_tile> qidx = {batch, head, qblk * CONSUMER_WARPGROUPS + i, 0};
            tma::load_async(q_s[i],  g.Q,   qidx, q_semaphore);
            tma::load_async(do_s[i], g.dO,  qidx, q_semaphore);
            tma::load_async(dq_s[i], g.ddQ, qidx, q_semaphore);
        }

        for (int k_iter = 0; k_iter < KV_STAGES - 1 && k_iter < total_k_tiles; k_iter++) {
            coord<k_tile> kv_idx = {batch, head, k_iter, 0};
            tma::expect_bytes(k_arrived[k_iter], 4 * sizeof(k_tile));
            tma::load_async(k_s[k_iter],  g.K,   kv_idx, k_arrived[k_iter]);
            tma::load_async(v_s[k_iter],  g.V,   kv_idx, k_arrived[k_iter]);
            tma::load_async(dk_s[k_iter], g.ddK, kv_idx, k_arrived[k_iter]);
            tma::load_async(dv_s[k_iter], g.ddV, kv_idx, k_arrived[k_iter]);
        }
    }

    const int vbase = ((batch * gridDim.y + head) * g.N) + qblk * CONSUMER_WARPGROUPS * TK_TQ;
    for (int i = threadIdx.x; i < TK_CWG * TK_TQ; i += TK_WORKERS * 32) {
        l_s[i / TK_TQ][i % TK_TQ] = g.Lp[vbase + i] * LOG2E;
        d_s[i / TK_TQ][i % TK_TQ] = g.Dp[vbase + i];
    }

    __syncthreads();

    if (wgid == NUM_WARPGROUPS - 1) {
        // producer wg
        warpgroup::decrease_registers<32>();
        if (wid == CONSUMER_WARPGROUPS * 4) {
            for (int kv_idx = KV_STAGES - 2; kv_idx <= total_k_tiles - 2; kv_idx++) {
                int it = kv_idx + 1, s = it % KV_STAGES;
                int kt = (it < k_tiles) ? it : it - k_tiles;
                coord<k_tile> kv_tile_idx = {batch, head, kt, 0};
                warp::tma::expect_bytes(k_arrived[s], 4 * sizeof(k_tile));
                warp::tma::load_async(k_s[s],  g.K,   kv_tile_idx, k_arrived[s]);
                warp::tma::load_async(v_s[s],  g.V,   kv_tile_idx, k_arrived[s]);
                warp::tma::load_async(dk_s[s], g.ddK, kv_tile_idx, k_arrived[s]);
                warp::tma::load_async(dv_s[s], g.ddV, kv_tile_idx, k_arrived[s]);
                wait(compute_done[kv_idx % KV_STAGES], (kv_idx / KV_STAGES) % 2);
            }
        }
    } else {
        // consumer wgs
        warpgroup::increase_registers<224>();

        wait(q_semaphore, 0);

        col_vec<rt_fl<16, TK_TK>> lv, dvv, dDv, r1, r2, r3, Bv, tv;
        warp::zero(dDv); warp::zero(r1); warp::zero(r2); warp::zero(r3);
        rt_fl<16, HEAD_DIM> accQ, accO;
        warp::zero(accQ); warp::zero(accO);
        const float scale2 = g.scale * LOG2E;

        warpgroup::load(lv,  l_s[wgid]);
        warpgroup::load(dvv, d_s[wgid]);

        const int q16 = qblk * (TK_CWG * (TK_TQ/16)) + wgid * (TK_TQ/16) + (wid % 4);
        for (int kv_idx = 0; kv_idx < total_k_tiles; kv_idx++) {
            const int kt = kv_idx % k_tiles;             // position in the key sequence
            const int ring_idx = kv_idx % KV_STAGES;

            wait(k_arrived[ring_idx], (kv_idx / KV_STAGES) % 2);

            rt_fl<16, TK_TK> S, ddS, dP, dPa, intermediate;

            warpgroup::mm_ABt(S,   q_s[wgid],  k_s[ring_idx]);    // S   = Q K^T
            warpgroup::mm_ABt(ddS, dq_s[wgid], k_s[ring_idx]);    // ddS = ddQ K^T ...
            warpgroup::mma_ABt(ddS, q_s[wgid], dk_s[ring_idx]);   //     ... + Q ddK^T
            warpgroup::mm_ABt(dP,  do_s[wgid], v_s[ring_idx]);    // dP  = dO V^T
            warpgroup::mm_ABt(dPa, do_s[wgid], dv_s[ring_idx]);   // dPa = dO ddV^T
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            warp::mul(S, S, scale2);
            warp::mul(ddS, ddS, g.scale);
            tk_cmask(S, q16, kt);
            warp::sub_row(S, S, lv);
            warp::exp2(S, S); // S -> P vioa 

            if (kv_idx < k_tiles) {
                warp::mul(intermediate, ddS, S);
                warp::row_sum(dDv, intermediate, dDv);            // dD += sum ddS*P
                warp::mul(intermediate, intermediate, dP);
                warp::row_sum(r3, intermediate, r3);              // r3 += sum dP*ddS*P
                warp::mul(intermediate, dPa, S);
                warp::row_sum(r1, intermediate, r1);              // r1 += sum (dO.ddV)*P
                warp::mul(intermediate, dP, S);
                warp::row_sum(r2, intermediate, r2);              // r2 += sum dP*P
            } else {
                // second sweep
                warp::mul_row(intermediate, dP, dDv);
                warp::sub(dPa, dPa, intermediate);
                warp::mul_row(intermediate, ddS, dvv);
                warp::sub(dPa, dPa, intermediate);
                warp::mul(intermediate, dP, ddS);
                warp::add(dPa, dPa, intermediate);

                warp::sub_row(dP, dP, dvv);                       // dS  (into dP)
                warp::mul(dP, dP, S);
                warp::mul(dP, dP, g.scale);
                warp::sub_row(dPa, dPa, Bv);                      // dS2 (into dPa)
                warp::mul(dPa, dPa, S);
                warp::mul(dPa, dPa, g.scale);
                warp::sub_row(ddS, ddS, dDv);                     // ddP (into ddS)
                warp::mul(ddS, ddS, S);

                rt_bf<16, TK_TK> mb0, mb1, mb2, mb3;
                warp::copy(mb0, dP);
                warp::copy(mb1, dPa);
                warp::copy(mb2, ddS);
                warp::copy(mb3, S);
                warpgroup::mma_AB(accQ, mb0, dk_s[ring_idx]);     // dQ2 += dS  @ ddK
                warpgroup::mma_AB(accQ, mb1, k_s[ring_idx]);      // dQ2 += dS2 @ K
                warpgroup::mma_AB(accO, mb2, v_s[ring_idx]);      // ddO += ddP @ V
                warpgroup::mma_AB(accO, mb3, dv_s[ring_idx]);     // ddO += P   @ ddV
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();
            }

            if (warpgroup::laneid() == 0) {
                arrive(compute_done[ring_idx], 1);
            }

            if (kv_idx == k_tiles - 1) {
                warp::mul(tv, dDv, r2);
                warp::sub(Bv, r1, tv);
                warp::mul(tv, dvv, dDv);
                warp::sub(Bv, Bv, tv);
                warp::add(Bv, Bv, r3);
            }
        }

        // write out
        warpgroup::store(q_s[wgid],  accQ);
        warpgroup::store(do_s[wgid], accO);
        warpgroup::store(l_s[wgid],  dDv);
        warpgroup::store(d_s[wgid],  Bv);
        group<4>::sync(wgid + 4);
        if (wid % 4 == 0) {
            coord<tk_qtile> idx = {batch, head, qblk * CONSUMER_WARPGROUPS + wgid, 0};
            warp::tma::store_async(g.dQ2, q_s[wgid], idx);
            warp::tma::store_async(g.ddO, do_s[wgid], idx);
        }
        for (int i = warpgroup::laneid(); i < TK_TQ; i += 128) {
            g.dDo[vbase + wgid * TK_TQ + i] = l_s[wgid][i];
            g.Bo[vbase + wgid * TK_TQ + i]  = d_s[wgid][i];
        }
        warp::tma::store_async_wait();
    }
}

}  // namespace stage1
}  // namespace flash_hog_tk