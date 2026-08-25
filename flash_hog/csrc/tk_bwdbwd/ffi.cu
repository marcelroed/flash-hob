// XLA FFI plugin: ThunderKittens double-backward (stage1 + stage2) for causal attention.
//
// "TkBwdBwd": launches tk_stage1 then tk_stage2 on the XLA stream (stage2
// reads the dD/B vectors stage1 writes; same-stream ordering suffices).
//
//   inputs : Q, K, V, dO, ddQ, ddK, ddV   bf16 (B, H, T, hd)   [BHTD, contiguous]
//            L, D                         f32  (B, H, T)
//   attr   : scale (f32)
//   outputs: dQ2, ddO, dK2, dV2           bf16 (B, H, T, hd)
//            dD, B                        f32  (B, H, T)       [stage1->stage2 scratch]
//
// Constraints: SM90 (Hopper), head_dim == 64, T % 128 == 0, causal, q_heads == kv_heads.
// Built at runtime by flash_hog/jax/_tk_build.py (pip CUDA tools, cached).

#include "stage1.cuh"
#include "stage2.cuh"

#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

namespace s1 = flash_hog_tk::stage1;
namespace s2 = flash_hog_tk::stage2;

static constexpr int HEAD_DIM = 64;

static ffi::Error TkBwdBwdImpl(
    cudaStream_t stream,
    ffi::AnyBuffer q, ffi::AnyBuffer k, ffi::AnyBuffer v, ffi::AnyBuffer dO,
    ffi::AnyBuffer ddq, ffi::AnyBuffer ddk, ffi::AnyBuffer ddv,
    ffi::AnyBuffer l, ffi::AnyBuffer d,
    float scale,
    ffi::Result<ffi::AnyBuffer> dq2, ffi::Result<ffi::AnyBuffer> ddo,
    ffi::Result<ffi::AnyBuffer> dk2, ffi::Result<ffi::AnyBuffer> dv2,
    ffi::Result<ffi::AnyBuffer> dd, ffi::Result<ffi::AnyBuffer> b) {
    auto dims = q.dimensions();
    if (dims.size() != 4) return ffi::Error::InvalidArgument("Q must be (B,H,T,hd)");
    const unsigned B = dims[0], H = dims[1], T = dims[2], hd = dims[3];
    if (hd != HEAD_DIM) return ffi::Error::InvalidArgument("head_dim must be 64");
    if (T % 128 != 0) return ffi::Error::InvalidArgument("T must be divisible by 128");

    static bool attrs_set = false;
    if (!attrs_set) {
        cudaFuncSetAttribute(s1::tk_stage1, cudaFuncAttributeMaxDynamicSharedMemorySize, s1::SMEM_BYTES);
        cudaFuncSetAttribute(s2::tk_stage2, cudaFuncAttributeMaxDynamicSharedMemorySize, s2::SMEM_BYTES);
        attrs_set = true;
    }

    using kittens::bf16;
    auto bfp = [](ffi::AnyBuffer& x) { return reinterpret_cast<bf16*>(x.untyped_data()); };
    auto bfr = [](ffi::Result<ffi::AnyBuffer>& x) { return reinterpret_cast<bf16*>(x->untyped_data()); };
    auto flp = [](ffi::AnyBuffer& x) { return reinterpret_cast<float*>(x.untyped_data()); };
    auto flr = [](ffi::Result<ffi::AnyBuffer>& x) { return reinterpret_cast<float*>(x->untyped_data()); };

    s1::tk_globals G1{
        s1::tk_qgl{bfp(q), B, H, T, HEAD_DIM},   s1::tk_qgl{bfp(dO), B, H, T, HEAD_DIM},
        s1::tk_qgl{bfp(ddq), B, H, T, HEAD_DIM}, s1::tk_qgl{bfr(dq2), B, H, T, HEAD_DIM},
        s1::tk_qgl{bfr(ddo), B, H, T, HEAD_DIM},
        s1::tk_kgl{bfp(k), B, H, T, HEAD_DIM},   s1::tk_kgl{bfp(v), B, H, T, HEAD_DIM},
        s1::tk_kgl{bfp(ddk), B, H, T, HEAD_DIM}, s1::tk_kgl{bfp(ddv), B, H, T, HEAD_DIM},
        flp(l), flp(d), flr(dd), flr(b), (int)T, scale};
    s2::tk_globals G2{
        s2::tk_qgl{bfp(q), B, H, T, HEAD_DIM},   s2::tk_qgl{bfp(dO), B, H, T, HEAD_DIM},
        s2::tk_qgl{bfp(ddq), B, H, T, HEAD_DIM},
        s2::tk_kgl{bfp(k), B, H, T, HEAD_DIM},   s2::tk_kgl{bfp(v), B, H, T, HEAD_DIM},
        s2::tk_kgl{bfp(ddk), B, H, T, HEAD_DIM}, s2::tk_kgl{bfp(ddv), B, H, T, HEAD_DIM},
        s2::tk_kgl{bfr(dk2), B, H, T, HEAD_DIM}, s2::tk_kgl{bfr(dv2), B, H, T, HEAD_DIM},
        s2::tk_vgl{flp(l), B, H, 1, T},  s2::tk_vgl{flp(d), B, H, 1, T},
        s2::tk_vgl{flr(dd), B, H, 1, T}, s2::tk_vgl{flr(b), B, H, 1, T},
        (int)T, scale};

    dim3 grid(T / 128, H, B);
    s1::tk_stage1<<<grid, s1::TK_WORKERS * 32, s1::SMEM_BYTES, stream>>>(G1);
    s2::tk_stage2<<<grid, s2::TK_WORKERS * 32, s2::SMEM_BYTES, stream>>>(G2);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return ffi::Error::Internal(cudaGetErrorString(e));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TkBwdBwd, TkBwdBwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()   // Q
        .Arg<ffi::AnyBuffer>()   // K
        .Arg<ffi::AnyBuffer>()   // V
        .Arg<ffi::AnyBuffer>()   // dO
        .Arg<ffi::AnyBuffer>()   // ddQ
        .Arg<ffi::AnyBuffer>()   // ddK
        .Arg<ffi::AnyBuffer>()   // ddV
        .Arg<ffi::AnyBuffer>()   // L
        .Arg<ffi::AnyBuffer>()   // D
        .Attr<float>("scale")
        .Ret<ffi::AnyBuffer>()   // dQ2
        .Ret<ffi::AnyBuffer>()   // ddO
        .Ret<ffi::AnyBuffer>()   // dK2
        .Ret<ffi::AnyBuffer>()   // dV2
        .Ret<ffi::AnyBuffer>()   // dD
        .Ret<ffi::AnyBuffer>());  // B