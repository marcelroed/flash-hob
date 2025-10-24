import cutlass
import cutlass.cute as cute


@cute.kernel
def kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("Hello, world!")


@cute.jit
def host():
    cutlass.cuda.initialize_cuda_context()
    kernel().launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


host()
