"""Load FlashAttention-4 CuTe-DSL kernels for use from JAX, without torch.

The FA4 cute kernels (and quack-kernels) reference a handful of torch symbols at
*import* time only (dtype sentinels, ``torch.library.custom_op`` decorators that
register torch ops we never call, a couple of fake-tensor base classes). We never
touch torch on the data path -- inputs/outputs are JAX arrays handed to the kernel
through ``cutlass.jax.cutlass_call`` -- so a minimal import-time shim is enough.

Import order matters: cutlass/tvm_ffi must be imported with NO torch in
``sys.modules`` so tvm_ffi cleanly skips its optional torch integration. Only then
do we install the shim and import the kernels.
"""
import sys
import types
import os


def _install_torch_shim():
    if "torch" in sys.modules:
        return sys.modules["torch"]

    t = types.ModuleType("torch")

    class _DT:
        def __init__(self, n):
            self.n = n

        def __repr__(self):
            return f"torch.{self.n}"

    for n in [
        "float16", "bfloat16", "float32", "float64",
        "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "bool", "complex64", "complex128", "half", "float", "double",
    ]:
        setattr(t, n, _DT(n))

    class Tensor:  # noqa: D401
        pass

    class device:
        pass

    class SymInt:
        pass

    t.Tensor = Tensor
    t.device = device
    t.SymInt = SymInt
    t.dtype = _DT  # the dtype *class* (instances are the sentinels above)
    t.Size = tuple
    # Use a PEP 440-parseable version: torch_c_dlpack_ext (pulled by quack's
    # from_dlpack import) runs packaging.Version(torch.__version__) at import.
    # We never call into torch_c_dlpack_ext on the JAX data path, but its module
    # import must not raise. The assert in load() still gates real-torch via the
    # local marker below.
    t.__version__ = "2.99.0+shim"
    t._is_fa4_loader_shim = True

    # Permissive fallback: any import-time attribute we did not explicitly model
    # (almost always a bare type annotation) resolves to a harmless placeholder
    # class. We never execute torch on the data path, so this only affects
    # annotations/decorator no-ops, never real computation.
    _placeholders = {}

    def _torch_getattr(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name not in _placeholders:
            _placeholders[name] = type(f"_torch_{name}", (), {})
        return _placeholders[name]

    t.__getattr__ = _torch_getattr

    # torch.version
    ver = types.ModuleType("torch.version")
    ver.cuda = "13.0"
    t.version = ver

    # torch.cuda
    cu = types.ModuleType("torch.cuda")
    cu.get_device_capability = lambda *a, **k: (9, 0)
    cu.is_available = lambda *a, **k: False
    cu.current_stream = lambda *a, **k: None
    t.cuda = cu

    # torch.library: custom_op registers a torch op we never invoke.
    class _FakeOp:
        def __init__(self, fn=None):
            self._fn = fn

        def register_fake(self, f=None):
            return f if f is not None else (lambda g: g)

        def register_kernel(self, *a, **k):
            return lambda f: f

        def register_autograd(self, *a, **k):
            return lambda f: f

        def __call__(self, *a, **k):
            if self._fn is not None:
                return self._fn(*a, **k)

    def custom_op(name, fn=None, **kwargs):
        if fn is None:
            return lambda g: _FakeOp(g)
        return _FakeOp(fn)

    lib = types.ModuleType("torch.library")
    lib.custom_op = custom_op
    lib.register_fake = lambda *a, **k: (lambda f: f)
    lib.register_kernel = lambda *a, **k: (lambda f: f)
    lib.triton_op = custom_op
    lib.Library = type("Library", (), {"__init__": lambda self, *a, **k: None,
                                       "define": lambda self, *a, **k: None,
                                       "impl": lambda self, *a, **k: None})
    t.library = lib

    # torch._subclasses.fake_tensor.FakeTensorMode
    subcl = types.ModuleType("torch._subclasses")
    faketensor = types.ModuleType("torch._subclasses.fake_tensor")
    faketensor.FakeTensorMode = type("FakeTensorMode", (), {
        "__init__": lambda self, *a, **k: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: False,
    })
    subcl.fake_tensor = faketensor
    t._subclasses = subcl

    # torch.utils._python_dispatch.TorchDispatchMode
    utils = types.ModuleType("torch.utils")
    pydisp = types.ModuleType("torch.utils._python_dispatch")
    pydisp.TorchDispatchMode = type("TorchDispatchMode", (), {
        "__init__": lambda self, *a, **k: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: False,
    })
    utils._python_dispatch = pydisp
    t.utils = utils

    # torch.compiler.is_compiling / disable (used as bare or called decorators)
    comp = types.ModuleType("torch.compiler")
    comp.is_compiling = lambda *a, **k: False

    def _passthrough(fn=None, *a, **k):
        if callable(fn):
            return fn
        return lambda f: f

    comp.disable = _passthrough
    t.compiler = comp
    t.no_grad = _passthrough
    t.inference_mode = _passthrough
    t._dynamo = types.ModuleType("torch._dynamo")
    t._dynamo.disable = _passthrough
    sys.modules["torch._dynamo"] = t._dynamo

    for modname, mod in [
        ("torch", t),
        ("torch.version", ver),
        ("torch.cuda", cu),
        ("torch.library", lib),
        ("torch._subclasses", subcl),
        ("torch._subclasses.fake_tensor", faketensor),
        ("torch.utils", utils),
        ("torch.utils._python_dispatch", pydisp),
        ("torch.compiler", comp),
    ]:
        sys.modules[modname] = mod
    return t


def _bypass_pkg_init(pkg_name):
    """Register ``pkg_name`` as a bare namespace package pointing at its real
    directory, so its (heavy) __init__ does not execute. Submodules are still
    importable directly. Returns the directory."""
    import importlib.util
    spec = importlib.util.find_spec(pkg_name)
    # find_spec runs parent __init__ only for dotted names; for top-level it does
    # not import the module, but it may for packages. Guard via locations instead.
    locs = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else None
    if not locs:
        raise RuntimeError(f"could not locate package dir for {pkg_name}")
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = locs
    sys.modules[pkg_name] = pkg
    return locs[0]


def _install_min_quack_activation():
    """quack 0.5.2 targets cutlass-dsl 4.6.0.dev0, whose quack.activation imports
    cutlass._mlir_helpers.math (absent in 4.5.2). The FA4 SM90 kernels only use
    quack.activation.sub_packed_f32x2, so provide a minimal module that avoids the
    incompatible import."""
    from functools import partial
    import cutlass.cute as cute
    from cutlass._mlir.dialects import nvvm
    act = types.ModuleType("quack.activation")
    act.sub_packed_f32x2 = partial(
        cute.arch.calc_packed_f32x2_op,
        src_c=None,
        calc_func=nvvm.sub_packed_f32x2,
    )
    sys.modules["quack.activation"] = act


def _bypass_flash_attn_init():
    """Register flash_attn.cute as a bare namespace package so its __init__
    (which imports the torch-heavy interface module) does not run."""
    import flash_attn  # namespace package from flash-attn-4 install
    cute_dir = None
    for base in flash_attn.__path__:
        cand = os.path.join(base, "cute")
        if os.path.isdir(cand):
            cute_dir = cand
            break
    if cute_dir is None:
        raise RuntimeError("could not locate flash_attn/cute directory")
    pkg = types.ModuleType("flash_attn.cute")
    pkg.__path__ = [cute_dir]
    sys.modules["flash_attn.cute"] = pkg


def load():
    """Return a module namespace with the FA4 SM90 kernel classes."""
    assert "torch" not in sys.modules or getattr(sys.modules["torch"], "_is_fa4_loader_shim", False), (
        "real torch already imported; import cutlass first, then call load()"
    )
    # NOTE: with nvidia-cutlass-dsl[cu13] the DSL backend is the CUDA-13 build,
    # so cutlass.CUDA_VERSION is natively (13, 3) and FA4's utils.fmax picks the
    # correct new nvvm.fmax binding -- no CUDA-version correction needed.
    _install_torch_shim()
    # Skip quack's heavy __init__ (rmsnorm/autotuner/triton); the kernels only
    # need a few pure-CuTe submodules (copy_utils, layout_utils, sm90_utils,
    # cute_dsl_utils, activation).
    _bypass_pkg_init("quack")
    _install_min_quack_activation()
    _bypass_flash_attn_init()
    import flash_attn.cute.flash_fwd_sm90 as fwd
    import flash_attn.cute.flash_bwd_sm90 as bwd
    import flash_attn.cute.flash_bwd_preprocess as pre
    import flash_attn.cute.flash_bwd_postprocess as post
    _patch_quack_bulk_copy_gmem()
    return types.SimpleNamespace(fwd=fwd, bwd=bwd, pre=pre, post=post)


def _as_gmem_tensor(t):
    """Rebuild a cute.Tensor with its iterator's address space forced to ``gmem``.

    Tensors handed to the kernel by ``cutlass.jax.cutlass_call`` arrive with a
    *generic* address-space pointer (cutlass 4.6.0.dev0 no longer tags JAX/XLA
    buffers as gmem at the FFI boundary). TMA paths (mQ/mK/mV/mdO) don't care --
    they build TMA descriptors -- but the backward loads LSE/dPsum with a
    ``cp.async.bulk`` G2S copy, whose verifier in 4.6.0.dev0 *requires* a gmem
    source memref ("bulk copy src expects gmem based memref, but got generic").
    These buffers are genuinely global device memory, so re-tagging the pointer
    gmem is sound. Layout (shape/stride) is preserved."""
    import cutlass.cute as cute
    from cutlass.cute import AddressSpace
    if t is None:
        return t
    it = t.iterator
    if it.memspace == AddressSpace.gmem:
        return t
    gptr = cute.make_ptr(
        t.element_type, it.toint(), AddressSpace.gmem, assumed_align=it.alignment
    )
    return cute.make_tensor(gptr, t.layout)


def _patch_quack_bulk_copy_gmem():
    """Wrap ``quack.copy_utils.cpasync_bulk_get_copy_fn`` so its ``src_tensor`` is
    re-tagged gmem (see ``_as_gmem_tensor``). cutlass 4.6.0.dev0's bulk-copy
    verifier rejects the generic-address-space source that cutlass_call hands the
    kernel; the underlying buffer is global memory, so this is a pure re-tag.
    Only the FA4 backward exercises this (LSE/dPsum cp.async.bulk loads); the
    forward TMA path is unaffected."""
    import quack.copy_utils as cu
    orig = cu.cpasync_bulk_get_copy_fn

    def cpasync_bulk_get_copy_fn(src_tensor, dst_tensor, single_stage=False, **kwargs):
        return orig(_as_gmem_tensor(src_tensor), dst_tensor, single_stage=single_stage, **kwargs)

    cu.cpasync_bulk_get_copy_fn = cpasync_bulk_get_copy_fn
