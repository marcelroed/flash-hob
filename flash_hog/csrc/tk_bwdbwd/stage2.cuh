// ThunderKittens kernel: stage 2 of the causal-attention double-backward (SM90 only).

#pragma once

#include "kittens.cuh"

namespace flash_hog_tk {
namespace stage2 {

constexpr int HEAD_DIM = 64;

template<int D> struct bwd_over_bwd_stage2_tile_dims {};
template<> struct bwd_over_bwd_stage2_tile_dims<64> {
    constexpr static int tile_width = (64);    // head_dim
    constexpr static int kv_height  = (4*16);  // keys per consumer wg
    constexpr static int qo_height  = (2*16);  // 32 streamed queries per ring stage
    constexpr static int stages     = (4);
};

template<int D> struct bwd_over_bwd_stage2_globals {
    using dims   = bwd_over_bwd_stage2_tile_dims<D>;
    using k_tile = kittens::st_bf<dims::kv_height, dims::tile_width>;
    using q_tile = kittens::st_bf<dims::qo_height, dims::tile_width>; // stream over q tiles
    using per_query_vec = kittens::sv_fl<dims::qo_height>;
    using k_gl = kittens::gl<kittens::bf16, -1, -1, -1, -1, k_tile>;
    using q_gl = kittens::gl<kittens::bf16, -1, -1, -1, -1, q_tile>;
    using v_gl = kittens::gl<float, -1, -1, -1, -1, per_query_vec>;    // L/D/dD/B
};

constexpr int CONSUMER_WARPGROUPS = (2);
constexpr int PRODUCER_WARPGROUPS = (1);
constexpr int NUM_WARPGROUPS = (CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS);
constexpr int NUM_WORKERS = (NUM_WARPGROUPS * kittens::WARPGROUP_WARPS);

constexpr int TK_TKEY = bwd_over_bwd_stage2_tile_dims<HEAD_DIM>::kv_height;
constexpr int TK_TQ = bwd_over_bwd_stage2_tile_dims<HEAD_DIM>::qo_height;
constexpr int Q_STAGES = bwd_over_bwd_stage2_tile_dims<HEAD_DIM>::stages;
constexpr int TK_CWG = CONSUMER_WARPGROUPS;
constexpr int TK_WORKERS = NUM_WORKERS;
constexpr int SMEM_BYTES = 160 * 1024;

using tk_ktile = bwd_over_bwd_stage2_globals<HEAD_DIM>::k_tile;
using tk_qtile = bwd_over_bwd_stage2_globals<HEAD_DIM>::q_tile;
using tk_kgl   = bwd_over_bwd_stage2_globals<HEAD_DIM>::k_gl;
using tk_qgl   = bwd_over_bwd_stage2_globals<HEAD_DIM>::q_gl;
using tk_vec   = bwd_over_bwd_stage2_globals<HEAD_DIM>::per_query_vec;
using tk_vgl   = bwd_over_bwd_stage2_globals<HEAD_DIM>::v_gl;

struct tk_globals {                // field order is load-bearing (aggregate init)
    tk_qgl Q, dO, ddQ;             // query-side (streamed, 32-row tiles)
    tk_kgl K, V, ddK, ddV;         // key-side (resident, 64-row tiles)
    tk_kgl dK2, dV2;               // outputs (64-row tiles)
    tk_vgl L, D, dD, Bv;           // per-query scalars, gl dims (B,H,1,T)
    int N;
    float scale;
};


__device__ static inline void tk_cmask_t(kittens::rt_fl<16, TK_TQ>& S, int k16, int qt) {
    #pragma unroll
    for (int j = 0; j < TK_TQ / 16; j++) {
        int q16 = qt * (TK_TQ / 16) + j;
        auto& sub = reinterpret_cast<kittens::rt_fl<16, 16>&>(S.tiles[0][j]);
        if (q16 < k16) kittens::warp::neg_infty(sub);
        else if (q16 == k16)
            kittens::warp::make_causal_t(sub, sub, kittens::base_types::constants<float>::neg_infty());
    }
}

__global__ __launch_bounds__(TK_WORKERS * 32, 1)
void tk_stage2(const __grid_constant__ tk_globals g) {
    using namespace kittens;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    const int wid = warpid(), wgid = wid / 4;   // wg 0,1 = consumers; wg 2 = producer
    const int kblk = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    constexpr float LOG2E = 1.44269504089f;

    using G = bwd_over_bwd_stage2_globals<HEAD_DIM>;
    using q_tile = G::q_tile;
    using k_tile = G::k_tile;
    using per_query_vec = G::per_query_vec;

    k_tile (&k_s)[CONSUMER_WARPGROUPS]  = al.allocate<G::k_tile, CONSUMER_WARPGROUPS>();
    k_tile (&v_s)[CONSUMER_WARPGROUPS]  = al.allocate<G::k_tile, CONSUMER_WARPGROUPS>();
    k_tile (&dk_s)[CONSUMER_WARPGROUPS] = al.allocate<G::k_tile, CONSUMER_WARPGROUPS>();  // ddK
    k_tile (&dv_s)[CONSUMER_WARPGROUPS] = al.allocate<G::k_tile, CONSUMER_WARPGROUPS>();  // ddV
    q_tile (&q_s)[Q_STAGES]  = al.allocate<G::q_tile, Q_STAGES>();
    q_tile (&dq_s)[Q_STAGES] = al.allocate<G::q_tile, Q_STAGES>();    // ddQ
    q_tile (&do_s)[Q_STAGES] = al.allocate<G::q_tile, Q_STAGES>();
    per_query_vec (&l_s)[Q_STAGES]  = al.allocate<G::per_query_vec, Q_STAGES>();
    per_query_vec (&d_s)[Q_STAGES]  = al.allocate<G::per_query_vec, Q_STAGES>();
    per_query_vec (&dd_s)[Q_STAGES] = al.allocate<G::per_query_vec, Q_STAGES>();
    per_query_vec (&b_s)[Q_STAGES]  = al.allocate<G::per_query_vec, Q_STAGES>();

    const int q_start = (TK_CWG * TK_TKEY / TK_TQ) * kblk;   // first q tile (causal)
    const int TOTAL = g.N / TK_TQ - q_start;                  // q tiles to stream

    __shared__ kittens::semaphore k_semaphore, q_arrived[Q_STAGES], compute_done[Q_STAGES];

    if (threadIdx.x == 0) {
        init_semaphore(k_semaphore, 0, 1);
        for (int s = 0; s < Q_STAGES; s++) {
            init_semaphore(q_arrived[s], 0, 1);
            init_semaphore(compute_done[s], TK_CWG, 0);
        }

        tma::expect_bytes(k_semaphore, 4 * CONSUMER_WARPGROUPS * sizeof(k_tile));
        for (int w = 0; w < CONSUMER_WARPGROUPS; w++) {
            coord<k_tile> kidx = {batch, head, kblk * CONSUMER_WARPGROUPS + w, 0};
            tma::load_async(k_s[w],  g.K,   kidx, k_semaphore);
            tma::load_async(v_s[w],  g.V,   kidx, k_semaphore);
            tma::load_async(dk_s[w], g.ddK, kidx, k_semaphore);
            tma::load_async(dv_s[w], g.ddV, kidx, k_semaphore);
        }

        for (int j = 0; j < Q_STAGES - 1 && j < TOTAL; j++) {
            tma::expect_bytes(q_arrived[j], 3 * sizeof(q_tile) + 4 * sizeof(per_query_vec));
            int qt = q_start + j;
            coord<q_tile> qidx = {batch, head, qt, 0};
            coord<per_query_vec> vidx = {batch, head, 0, qt};
            tma::load_async(q_s[j],  g.Q,   qidx, q_arrived[j]);
            tma::load_async(do_s[j], g.dO,  qidx, q_arrived[j]);
            tma::load_async(dq_s[j], g.ddQ, qidx, q_arrived[j]);
            tma::load_async(l_s[j],  g.L,   vidx, q_arrived[j]);
            tma::load_async(d_s[j],  g.D,   vidx, q_arrived[j]);
            tma::load_async(dd_s[j], g.dD,  vidx, q_arrived[j]);
            tma::load_async(b_s[j],  g.Bv,  vidx, q_arrived[j]);
        }
    }

    __syncthreads();

    if (wgid == NUM_WARPGROUPS - 1) {
        // producer wg
        warpgroup::decrease_registers<32>();
        if (wid == CONSUMER_WARPGROUPS * 4) {        // one warp issues the TMA loads
            for (int j = Q_STAGES - 2; j <= TOTAL - 2; j++) {
                int it = j + 1, s = it % Q_STAGES;
                int qt = it + q_start;
                coord<q_tile> qidx = {batch, head, qt, 0};
                coord<per_query_vec> vidx = {batch, head, 0, qt};
                warp::tma::expect_bytes(q_arrived[s], 3 * sizeof(q_tile) + 4 * sizeof(per_query_vec));
                warp::tma::load_async(q_s[s],  g.Q,   qidx, q_arrived[s]);
                warp::tma::load_async(do_s[s], g.dO,  qidx, q_arrived[s]);
                warp::tma::load_async(dq_s[s], g.ddQ, qidx, q_arrived[s]);
                warp::tma::load_async(l_s[s],  g.L,   vidx, q_arrived[s]);
                warp::tma::load_async(d_s[s],  g.D,   vidx, q_arrived[s]);
                warp::tma::load_async(dd_s[s], g.dD,  vidx, q_arrived[s]);
                warp::tma::load_async(b_s[s],  g.Bv,  vidx, q_arrived[s]);
                wait(compute_done[(it - 1) % Q_STAGES], ((it - 1) / Q_STAGES) % 2);
            }
        }
    } else {
        // consumer wgs
        warpgroup::increase_registers<224>();
        const int k16 = kblk * (TK_CWG * 4) + wgid * 4 + (wid % 4);  // this warp's key rows
        const float scale2 = g.scale * LOG2E;

        rt_fl<16, HEAD_DIM> accK, accV;
        warp::zero(accK);
        warp::zero(accV);
        wait(k_semaphore, 0);

        for (int q_idx = 0; q_idx < TOTAL; q_idx++) {
            const int ring_idx = q_idx % Q_STAGES;

            wait(q_arrived[ring_idx], (q_idx / Q_STAGES) % 2);

            rt_fl<16, TK_TQ> S, ddS, dP, dPa, intermediate;

            warpgroup::mm_ABt(S,   k_s[wgid],  q_s[ring_idx]);    // S^T   = K Q^T
            warpgroup::mm_ABt(ddS, k_s[wgid],  dq_s[ring_idx]);   // ddS^T = K ddQ^T ...
            warpgroup::mma_ABt(ddS, dk_s[wgid], q_s[ring_idx]);   //       ... + ddK Q^T
            warpgroup::mm_ABt(dP,  v_s[wgid],  do_s[ring_idx]);   // dP^T  = V dO^T
            warpgroup::mm_ABt(dPa, dv_s[wgid], do_s[ring_idx]);   // dPa^T = ddV dO^T
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            // per-query vectors broadcast along COLUMNS
            row_vec<rt_fl<16, TK_TQ>> lrv, drv, ddrv, brv;
            warp::load(lrv,  l_s[ring_idx]);
            warp::load(drv,  d_s[ring_idx]);
            warp::load(ddrv, dd_s[ring_idx]);
            warp::load(brv,  b_s[ring_idx]);
            warp::mul(lrv, lrv, LOG2E);                           // exp2 path

            warp::mul(S, S, scale2);
            warp::mul(ddS, ddS, g.scale);
            tk_cmask_t(S, k16, q_idx + q_start);
            warp::sub_col(S, S, lrv);
            warp::exp2(S, S);                                     // S := P

            // dP2 (into dPa) = dPa - dP*dD - ddS*D + dP*ddS
            warp::mul_col(intermediate, dP, ddrv);
            warp::sub(dPa, dPa, intermediate);
            warp::mul_col(intermediate, ddS, drv);
            warp::sub(dPa, dPa, intermediate);
            warp::mul(intermediate, dP, ddS);
            warp::add(dPa, dPa, intermediate);

            warp::sub_col(dP, dP, drv);                           // dS  (into dP)
            warp::mul(dP, dP, S);
            warp::mul(dP, dP, g.scale);
            warp::sub_col(dPa, dPa, brv);                         // dS2 (into dPa)
            warp::mul(dPa, dPa, S);
            warp::mul(dPa, dPa, g.scale);
            warp::sub_col(ddS, ddS, ddrv);                        // ddP (into ddS)
            warp::mul(ddS, ddS, S);

            rt_bf<16, TK_TQ> mb0, mb1, mb2;                       // own temp per async mma
            warp::copy(mb0, dP);
            warp::copy(mb1, dPa);
            warp::copy(mb2, ddS);
            warpgroup::mma_AB(accK, mb0, dq_s[ring_idx]);         // dK2 += dS  @ ddQ
            warpgroup::mma_AB(accK, mb1, q_s[ring_idx]);          // dK2 += dS2 @ Q
            warpgroup::mma_AB(accV, mb2, do_s[ring_idx]);         // dV2 += ddP @ dO
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            if (warpgroup::laneid() == 0) arrive(compute_done[ring_idx], 1);
        }

        // write out
        warpgroup::store(k_s[wgid], accK);
        warpgroup::store(v_s[wgid], accV);
        group<4>::sync(wgid + 4);
        if (wid % 4 == 0) {
            coord<tk_ktile> idx = {batch, head, kblk * TK_CWG + wgid, 0};
            warp::tma::store_async(g.dK2, k_s[wgid], idx);
            warp::tma::store_async(g.dV2, v_s[wgid], idx);
        }
        warp::tma::store_async_wait();
    }
}

}  // namespace stage2
}  // namespace flash_hog_tk