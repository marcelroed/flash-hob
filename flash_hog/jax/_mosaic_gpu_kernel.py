"""
Mosaic GPU Hopper SM90 bwd-of-bwd implementation.
"""



from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu

LOG2E = 1.4426950408889634

_SWIZZLE = 128
_ELEMS = _SWIZZLE // 2
_TR = (plgpu.TilingTransform((8, _ELEMS)), plgpu.SwizzleTransform(_SWIZZLE))


def _mm(a, b, m, n):
    # wgmma into an accumulator
    def body(acc):
        plgpu.wgmma(acc, a, b)
        return acc[...]
    return pl.run_scoped(body, plgpu.ACC((m, n), jnp.float32))

# Stage 1 kernel
BQ = 64
BK = 64
K_PIPELINE_DEPTH = 3


def _stage1(Q, K, V, dO, ddQ, ddK, ddV, L, D, scale):
    B, H, T, hd = Q.shape
    scale = float(scale)
    scale2 = scale * LOG2E
    num_q_tiles = T // (2 * BQ)
    rrow = plgpu.Layout.WGMMA.reduce(1)

    def kernel(Q_g, dO_g, ddQ_g, L_g, D_g,
               K_g, V_g, ddK_g, ddV_g,
               dQ2_g, ddO_g, dD_g, B_g,
               scoped):
        qblk = pl.program_id(0)
        b = pl.program_id(1)
        h = pl.program_id(2)
        wg = jax.lax.axis_index("wg")
        
        (qo_s, do_s, dq_s, k_s, v_s, dk_s, dv_s,
        a0_s, a1_s, a2_s, a3_s, sc_s), (kbar, k_done), qbar = scoped

        @pl.when(wg < 2)
        def _compute():
            plgpu.set_max_registers(232, action="increase")
            q_base = qblk * (2 * BQ) + wg * BQ
            qo, do, dq = qo_s.at[wg], do_s.at[wg], dq_s.at[wg]
            a0, a1, a2, a3 = a0_s.at[wg], a1_s.at[wg], a2_s.at[wg], a3_s.at[wg]
            sl, sd, sb = sc_s.at[wg, 0], sc_s.at[wg, 1], sc_s.at[wg, 2]

            plgpu.copy_gmem_to_smem(Q_g.at[b, h, pl.ds(q_base, BQ)], qo, qbar.at[wg])
            plgpu.copy_gmem_to_smem(dO_g.at[b, h, pl.ds(q_base, BQ)], do, qbar.at[wg])
            plgpu.copy_gmem_to_smem(ddQ_g.at[b, h, pl.ds(q_base, BQ)], dq, qbar.at[wg])
            plgpu.copy_gmem_to_smem(L_g.at[b, h, pl.ds(q_base, BQ)], sl, qbar.at[wg])
            plgpu.copy_gmem_to_smem(D_g.at[b, h, pl.ds(q_base, BQ)], sd, qbar.at[wg])
            plgpu.barrier_wait(qbar.at[wg])
            
            Lv = plgpu.load(sl, (), layout=rrow).astype(jnp.float32) * LOG2E
            Dv = plgpu.load(sd, (), layout=rrow).astype(jnp.float32)
            
            num_k_tiles = (qblk * (2 * BQ) + 2 * BQ - 1) // BK + 1
            q_row = q_base + plgpu.layout_cast(jax.lax.broadcasted_iota(jnp.int32, (BQ, BK), 0), plgpu.Layout.WGMMA) # needs to be in wgmma layout
            
            def scores(kt, slot):
                ks, vs, dks, dvs = k_s.at[slot], v_s.at[slot], dk_s.at[slot], dv_s.at[slot]
                S = _mm(qo, plgpu.transpose_ref(ks, (1, 0)), BQ, BK)
                ddS = _mm(dq, plgpu.transpose_ref(ks, (1, 0)), BQ, BK)
                ddS = ddS + _mm(qo, plgpu.transpose_ref(dks, (1, 0)), BQ, BK)
                dP = _mm(do, plgpu.transpose_ref(vs, (1, 0)), BQ, BK)
                dPa = _mm(do, plgpu.transpose_ref(dvs, (1, 0)), BQ, BK)
                k_col = kt * BK + plgpu.layout_cast(jax.lax.broadcasted_iota(jnp.int32, (BQ, BK), 1), plgpu.Layout.WGMMA)
                S = S * scale2 - jax.lax.broadcast_in_dim(Lv, (BQ, BK), (0,))
                S = jnp.where(k_col <= q_row, S, -jnp.inf)
                P = jnp.exp2(S)
                ddS = ddS * scale
                return P, ddS, dP, dPa
            
            @pl.when(num_k_tiles > 0)
            def _():
                plgpu.barrier_wait(kbar.at[0]) # wait for first tile to laod to smem

            z = plgpu.layout_cast(jnp.zeros((BQ,), jnp.float32), rrow)

            def sweepA(kt, carry):
                dD, r13 = carry
                slot = jax.lax.rem(kt, jnp.int32(K_PIPELINE_DEPTH))
                P, ddS, dP, dPa = scores(kt, slot)
                dD = dD + jnp.sum(ddS * P, axis=1)
                r13 = r13 + jnp.sum((dPa + dP * ddS) * P, axis=1)
                plgpu.barrier_arrive(k_done.at[slot])
                wait_step = kt + 1
                wait_slot = jax.lax.rem(wait_step, jnp.int32(K_PIPELINE_DEPTH))
                
                #wait for next tile to load to smem
                @pl.when(wait_step < 2 * num_k_tiles)
                def _():
                    plgpu.barrier_wait(kbar.at[wait_slot])
                return dD, r13

            dD, r13 = jax.lax.fori_loop(0, num_k_tiles, sweepA, (z, z))
            Bv = r13 - 2 * Dv * dD

            def sweepB(accQ_ref, accO_ref):
                def body(s, _):
                    kt = s - num_k_tiles
                    slot = jax.lax.rem(s, jnp.int32(K_PIPELINE_DEPTH))
                    P, ddS, dP, dPa = scores(kt, slot)
                    dDb = jax.lax.broadcast_in_dim(dD, (BQ, BK), (0,))
                    Dvb = jax.lax.broadcast_in_dim(Dv, (BQ, BK), (0,))
                    dP2 = dPa - dP * dDb - ddS * Dvb + dP * ddS
                    dS = scale * P * (dP - Dvb)
                    dS2 = scale * P * (dP2 - jax.lax.broadcast_in_dim(Bv, (BQ, BK), (0,)))
                    ddP = P * (ddS - dDb)
                    a0[...] = dS.astype(jnp.bfloat16)
                    a1[...] = dS2.astype(jnp.bfloat16)
                    a2[...] = ddP.astype(jnp.bfloat16)
                    a3[...] = P.astype(jnp.bfloat16)
                    plgpu.commit_smem()
                    plgpu.wgmma(accQ_ref, a0, dk_s.at[slot])
                    plgpu.wgmma(accQ_ref, a1, k_s.at[slot])
                    plgpu.wgmma(accO_ref, a2, v_s.at[slot])
                    plgpu.wgmma(accO_ref, a3, dv_s.at[slot])
                    plgpu.wgmma_wait(1)
                    plgpu.barrier_arrive(k_done.at[slot])
                    wait_step = s + 1
                    wait_slot = jax.lax.rem(wait_step, jnp.int32(K_PIPELINE_DEPTH))

                    @pl.when(wait_step < 2 * num_k_tiles)
                    def _():
                        plgpu.barrier_wait(kbar.at[wait_slot])
                    return ()
                jax.lax.fori_loop(num_k_tiles, 2 * num_k_tiles, body, ())
                qo[...] = accQ_ref[...].astype(jnp.bfloat16)
                do[...] = accO_ref[...].astype(jnp.bfloat16)
                sb[...] = dD
                sd[...] = Bv
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(qo, dQ2_g.at[b, h, pl.ds(q_base, BQ)])
                plgpu.copy_smem_to_gmem(do, ddO_g.at[b, h, pl.ds(q_base, BQ)])
                plgpu.copy_smem_to_gmem(sb, dD_g.at[b, h, pl.ds(q_base, BQ)])
                plgpu.copy_smem_to_gmem(sd, B_g.at[b, h, pl.ds(q_base, BQ)])
                plgpu.wait_smem_to_gmem(0)
                return ()

            pl.run_scoped(sweepB, plgpu.ACC((BQ, hd), jnp.float32),
                          plgpu.ACC((BQ, hd), jnp.float32))

        @pl.when(wg == 2)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")
            num_k_tiles = (qblk * (2 * BQ) + 2 * BQ - 1) // BK + 1
            total = 2 * num_k_tiles

            def load(kt, slot):
                s = (b, h, pl.ds(kt * BK, BK))
                plgpu.copy_gmem_to_smem(K_g.at[s], k_s.at[slot], kbar.at[slot])
                plgpu.copy_gmem_to_smem(V_g.at[s], v_s.at[slot], kbar.at[slot])
                plgpu.copy_gmem_to_smem(ddK_g.at[s], dk_s.at[slot], kbar.at[slot])
                plgpu.copy_gmem_to_smem(ddV_g.at[s], dv_s.at[slot], kbar.at[slot])

            @pl.loop(0, K_PIPELINE_DEPTH)
            def _load_initial(it):
                @pl.when(it < total)
                def _():
                    load(jax.lax.rem(it, jnp.int32(num_k_tiles)), jax.lax.rem(it, jnp.int32(K_PIPELINE_DEPTH)))

            @pl.loop(0, total - K_PIPELINE_DEPTH)
            def _stream_tiles(it):
                step = it + K_PIPELINE_DEPTH
                slot = jax.lax.rem(step, jnp.int32(K_PIPELINE_DEPTH))
                plgpu.barrier_wait(k_done.at[slot])
                load(jax.lax.rem(step, jnp.int32(num_k_tiles)), slot)

    out_shapes = (
        jax.ShapeDtypeStruct((B, H, T, hd), jnp.bfloat16),   # dQ2
        jax.ShapeDtypeStruct((B, H, T, hd), jnp.bfloat16),   # ddO
        jax.ShapeDtypeStruct((B, H, T), jnp.float32),        # dD
        jax.ShapeDtypeStruct((B, H, T), jnp.float32),        # B
    )

    def entry(Q_g, dO_g, ddQ_g, L_g, D_g, K_g, V_g, ddK_g, ddV_g,
              dQ2_g, ddO_g, dD_g, B_g):
        qsh = plgpu.SMEM((2, BQ, hd), jnp.bfloat16, transforms=_TR)
        ksh = plgpu.SMEM((K_PIPELINE_DEPTH, BK, hd), jnp.bfloat16, transforms=_TR)
        ash = plgpu.SMEM((2, BQ, BK), jnp.bfloat16, transforms=_TR)
        scsh = plgpu.SMEM((2, 3, BQ), jnp.float32)
        pl.run_scoped(
            lambda *sc: kernel(Q_g, dO_g, ddQ_g, L_g, D_g, K_g, V_g, ddK_g, ddV_g,
                               dQ2_g, ddO_g, dD_g, B_g, sc),
            (qsh, qsh, qsh, ksh, ksh, ksh, ksh, ash, ash, ash, ash, scsh),
            (plgpu.Barrier(num_arrivals=4, num_barriers=K_PIPELINE_DEPTH),
             plgpu.Barrier(num_arrivals=2, num_barriers=K_PIPELINE_DEPTH)),
            plgpu.Barrier(num_arrivals=5, num_barriers=2),
            collective_axes="wg",
        )

    return plgpu.kernel(
        entry,
        out_shape=out_shapes,
        grid=(num_q_tiles, B, H),
        grid_names=("q", "b", "h"),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )(Q, dO, ddQ, L, D, K, V, ddK, ddV)


# Stage 2 kernel
BK2 = 64
BQ2 = 64
Q_PIPELINE_DEPTH = 4


def _stage2(Q, K, V, dO, ddQ, ddK, ddV, L, D, dD, B, scale):
    Bsz, H, T, hd = Q.shape
    scale = float(scale)
    scale2 = scale * LOG2E
    num_k_tiles = T // (2 * BK2)
    nq = T // BQ2
    rcol = plgpu.Layout.WGMMA.reduce(0)

    def kernel(K_g, V_g, ddK_g, ddV_g,
               Q_g, dO_g, ddQ_g,
               L_g, D_g, dD_g, B_g,
               dK2_g, dV2_g,
               scoped):
        kblk = pl.program_id(0)
        b = pl.program_id(1)
        h = pl.program_id(2)
        wg = jax.lax.axis_index("wg")
        (k_s, v_s, dk_s, dv_s, q_s, do_s, dq_s, l_s, d_s, dd_s, bb_s,
         a0_s, a1_s, a2_s), (qbar, q_done), kbar = scoped
        
        @pl.when(wg < 2)
        def _compute():
            plgpu.set_max_registers(232, action="increase")
            k_base = kblk * (2 * BK2) + wg * BK2
            k2, v2, dk2, dv2 = k_s.at[wg], v_s.at[wg], dk_s.at[wg], dv_s.at[wg]
            a0, a1, a2 = a0_s.at[wg], a1_s.at[wg], a2_s.at[wg]
            ko = dk_s.at[wg] # reuse the ddK buffer for dK2

            plgpu.copy_gmem_to_smem(K_g.at[b, h, pl.ds(k_base, BK2)], k2, kbar.at[wg])
            plgpu.copy_gmem_to_smem(V_g.at[b, h, pl.ds(k_base, BK2)], v2, kbar.at[wg])
            plgpu.copy_gmem_to_smem(ddK_g.at[b, h, pl.ds(k_base, BK2)], dk2, kbar.at[wg])
            plgpu.copy_gmem_to_smem(ddV_g.at[b, h, pl.ds(k_base, BK2)], dv2, kbar.at[wg])
            plgpu.barrier_wait(kbar.at[wg])

            q_start = (kblk * (2 * BK2)) // BQ2
            num_q_tiles = nq - q_start
            k_row = k_base + plgpu.layout_cast(
                jax.lax.broadcasted_iota(jnp.int32, (BK2, BQ2), 0), plgpu.Layout.WGMMA)

            def scores(qt, slot):
                qs, dos, dqs = q_s.at[slot], do_s.at[slot], dq_s.at[slot]
                S = _mm(k2, plgpu.transpose_ref(qs, (1, 0)), BK2, BQ2)
                ddS = _mm(k2, plgpu.transpose_ref(dqs, (1, 0)), BK2, BQ2)
                ddS = ddS + _mm(dk2, plgpu.transpose_ref(qs, (1, 0)), BK2, BQ2)
                dP = _mm(v2, plgpu.transpose_ref(dos, (1, 0)), BK2, BQ2)
                dPa = _mm(dv2, plgpu.transpose_ref(dos, (1, 0)), BK2, BQ2)
                Lv = plgpu.load(l_s.at[slot], (), layout=rcol).astype(jnp.float32) * LOG2E
                Dv = plgpu.load(d_s.at[slot], (), layout=rcol).astype(jnp.float32)
                dDv = plgpu.load(dd_s.at[slot], (), layout=rcol).astype(jnp.float32)
                Bv = plgpu.load(bb_s.at[slot], (), layout=rcol).astype(jnp.float32)
                q_col = qt * BQ2 + plgpu.layout_cast(
                    jax.lax.broadcasted_iota(jnp.int32, (BK2, BQ2), 1), plgpu.Layout.WGMMA)
                S = S * scale2 - jax.lax.broadcast_in_dim(Lv, (BK2, BQ2), (1,))
                S = jnp.where(q_col >= k_row, S, -jnp.inf)
                P = jnp.exp2(S)
                ddS = ddS * scale
                return P, ddS, dP, dPa, Dv, dDv, Bv

            @pl.when(num_q_tiles > 0)
            def _():
                plgpu.barrier_wait(qbar.at[0])

            def run(accK_ref, accV_ref):
                def body(t, _):
                    qt = q_start + t
                    slot = jax.lax.rem(t, jnp.int32(Q_PIPELINE_DEPTH))
                    P, ddS, dP, dPa, Dv, dDv, Bv = scores(qt, slot)
                    dDvb = jax.lax.broadcast_in_dim(dDv, (BK2, BQ2), (1,))
                    Dvb = jax.lax.broadcast_in_dim(Dv, (BK2, BQ2), (1,))
                    dP2 = dPa - dP * dDvb - ddS * Dvb + dP * ddS
                    dS = scale * P * (dP - Dvb)
                    dS2 = scale * P * (dP2 - jax.lax.broadcast_in_dim(Bv, (BK2, BQ2), (1,)))
                    ddP = P * (ddS - dDvb)
                    a0[...] = ddP.astype(jnp.bfloat16)
                    a1[...] = dS.astype(jnp.bfloat16)
                    a2[...] = dS2.astype(jnp.bfloat16)
                    plgpu.commit_smem()
                    plgpu.wgmma(accV_ref, a0, do_s.at[slot])
                    plgpu.wgmma(accK_ref, a1, dq_s.at[slot])
                    plgpu.wgmma(accK_ref, a2, q_s.at[slot])
                    plgpu.wgmma_wait(1)
                    plgpu.barrier_arrive(q_done.at[slot])
                    wait_step = t + 1
                    wait_slot = jax.lax.rem(wait_step, jnp.int32(Q_PIPELINE_DEPTH))

                    @pl.when(wait_step < num_q_tiles)
                    def _():
                        plgpu.barrier_wait(qbar.at[wait_slot])
                    return ()
                jax.lax.fori_loop(0, num_q_tiles, body, ())
                ko[...] = accK_ref[...].astype(jnp.bfloat16)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(ko, dK2_g.at[b, h, pl.ds(k_base, BK2)])
                k2[...] = accV_ref[...].astype(jnp.bfloat16)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(k2, dV2_g.at[b, h, pl.ds(k_base, BK2)])
                plgpu.wait_smem_to_gmem(0)
                return ()

            pl.run_scoped(run, plgpu.ACC((BK2, hd), jnp.float32), plgpu.ACC((BK2, hd), jnp.float32))

        @pl.when(wg == 2)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")
            q_start = (kblk * (2 * BK2)) // BQ2
            total = nq - q_start

            def load(qt, slot):
                qd = (b, h, pl.ds(qt * BQ2, BQ2))
                plgpu.copy_gmem_to_smem(Q_g.at[qd], q_s.at[slot], qbar.at[slot])
                plgpu.copy_gmem_to_smem(dO_g.at[qd], do_s.at[slot], qbar.at[slot])
                plgpu.copy_gmem_to_smem(ddQ_g.at[qd], dq_s.at[slot], qbar.at[slot])
                plgpu.copy_gmem_to_smem(L_g.at[qd], l_s.at[slot], qbar.at[slot])
                plgpu.copy_gmem_to_smem(D_g.at[qd], d_s.at[slot], qbar.at[slot])
                plgpu.copy_gmem_to_smem(dD_g.at[qd], dd_s.at[slot], qbar.at[slot])
                plgpu.copy_gmem_to_smem(B_g.at[qd], bb_s.at[slot], qbar.at[slot])

            @pl.loop(0, Q_PIPELINE_DEPTH)
            def _load_initial(it):
                @pl.when(it < total)
                def _():
                    load(q_start + it, jax.lax.rem(it, jnp.int32(Q_PIPELINE_DEPTH)))

            @pl.loop(0, total - Q_PIPELINE_DEPTH)
            def _stream_tiles(it):
                step = it + Q_PIPELINE_DEPTH
                slot = jax.lax.rem(step, jnp.int32(Q_PIPELINE_DEPTH))
                plgpu.barrier_wait(q_done.at[slot])
                load(q_start + step, slot)

    out_shapes = (
        jax.ShapeDtypeStruct((Bsz, H, T, hd), jnp.bfloat16),   # dK2
        jax.ShapeDtypeStruct((Bsz, H, T, hd), jnp.bfloat16),   # dV2
    )

    def entry(K_g, V_g, ddK_g, ddV_g, Q_g, dO_g, ddQ_g, L_g, D_g, dD_g, B_g,
              dK2_g, dV2_g):
        ksh = plgpu.SMEM((2, BK2, hd), jnp.bfloat16, transforms=_TR)
        qsh = plgpu.SMEM((Q_PIPELINE_DEPTH, BQ2, hd), jnp.bfloat16, transforms=_TR)
        vsh = plgpu.SMEM((Q_PIPELINE_DEPTH, BQ2), jnp.float32)
        ash = plgpu.SMEM((2, BK2, BQ2), jnp.bfloat16, transforms=_TR)
        pl.run_scoped(
            lambda *sc: kernel(K_g, V_g, ddK_g, ddV_g, Q_g, dO_g, ddQ_g,
                               L_g, D_g, dD_g, B_g, dK2_g, dV2_g, sc),
            (ksh, ksh, ksh, ksh, qsh, qsh, qsh, vsh, vsh, vsh, vsh,
             ash, ash, ash),
            (plgpu.Barrier(num_arrivals=7, num_barriers=Q_PIPELINE_DEPTH),
             plgpu.Barrier(num_arrivals=2, num_barriers=Q_PIPELINE_DEPTH)),
            plgpu.Barrier(num_arrivals=4, num_barriers=2),
            collective_axes="wg",
        )

    return plgpu.kernel(
        entry,
        out_shape=out_shapes,
        grid=(num_k_tiles, Bsz, H),
        grid_names=("k", "b", "h"),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )(K, V, ddK, ddV, Q, dO, ddQ, L, D, dD, B)


def flash_bwdbwd_mosaic(Q, K, V, O, dO, ddQ, ddK, ddV, L, *, scale):
    D = jnp.sum(dO.astype(jnp.float32) * O.astype(jnp.float32), axis=-1)
    dQ2, ddO, dD, B = _stage1(Q, K, V, dO, ddQ, ddK, ddV, L, D, scale)
    dK2, dV2 = _stage2(Q, K, V, dO, ddQ, ddK, ddV, L, D, dD, B, scale)
    return dQ2, dK2, dV2, ddO
