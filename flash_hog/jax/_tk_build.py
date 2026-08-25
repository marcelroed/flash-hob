"""JIT build of the ThunderKittens double-backward plugin.

Compiles csrc/tk_bwdbwd/ffi.cu with pip-provided CUDA tools (`flash-hog[tk]`);
a host C++ compiler is the only non-pip requirement. ThunderKittens (header-only,
pinned commit) is fetched once as a tarball, or set THUNDERKITTENS_PATH.
Outputs are cached under ~/.cache/flash_hog/ keyed on sources + TK commit + arch.
The plugin links with `-cudart none`; driver/cudart symbols resolve at load time
(_tk_gpu preloads them RTLD_GLOBAL).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

TK_COMMIT = "34b15f7e7012de25ae162c8d9dc85296dd342676"
_TK_TARBALL = f"https://github.com/HazyResearch/ThunderKittens/archive/{TK_COMMIT}.tar.gz"

_SRC_DIR = Path(__file__).resolve().parent.parent / "csrc" / "tk_bwdbwd"
_SOURCES = ("stage1.cuh", "stage2.cuh", "ffi.cu")
_ARCH = "sm_90a"

# CUDA pip wheels: consolidated cu13-era layout (nvidia/cu13/...) or per-component
# cu12-era layout (nvidia/cuda_nvcc/..., nvidia/cuda_runtime/...)
_WHEEL_ROOTS = {
    "nvcc_bin": ("cu13/bin", "cu12/bin", "cuda_nvcc/bin"),
    "include": ("cu13/include", "cu12/include", "cuda_runtime/include"),
    "lib": ("cu13/lib", "cu12/lib", "cuda_runtime/lib"),
}


def _cache_root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "flash_hog"


def nvidia_wheel_paths(kind: str) -> list[Path]:
    out = []
    for base in sys.path:
        for sub in _WHEEL_ROOTS[kind]:
            p = Path(base) / "nvidia" / sub
            if p.exists() and p not in out:
                out.append(p)
    return out


def _find_nvcc() -> str | None:
    for d in nvidia_wheel_paths("nvcc_bin"):
        if (d / "nvcc").exists():
            return str(d / "nvcc")
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and (Path(cuda_home) / "bin" / "nvcc").exists():
        return str(Path(cuda_home) / "bin" / "nvcc")
    return shutil.which("nvcc")


def _download(url: str, dest: Path) -> None:
    import ssl

    try:
        with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return
    except urllib.error.URLError as e:
        if not isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            raise
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        raise RuntimeError(
            f"downloading ThunderKittens failed ({url}): SSL verification failed and "
            "certifi is unavailable. Install ca-certificates or set THUNDERKITTENS_PATH."
        ) from None
    with urllib.request.urlopen(url, context=ctx) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _thunderkittens() -> Path:
    env = os.environ.get("THUNDERKITTENS_PATH")
    if env:
        return Path(env)
    dst = _cache_root() / f"ThunderKittens-{TK_COMMIT}"
    if (dst / "include").exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst.parent) as td:
        tar_path = Path(td) / "tk.tar.gz"
        _download(_TK_TARBALL, tar_path)
        with tarfile.open(tar_path) as tf:
            tf.extractall(td)
        extracted = Path(td) / f"ThunderKittens-{TK_COMMIT}"
        if not (extracted / "include").exists():
            raise RuntimeError(f"unexpected ThunderKittens tarball layout under {extracted}")
        os.replace(extracted, dst)  # atomic against concurrent builders
    return dst


def _source_hash() -> str:
    h = hashlib.sha256()
    for name in _SOURCES:
        h.update((_SRC_DIR / name).read_bytes())
    h.update(TK_COMMIT.encode())
    h.update(_ARCH.encode())
    return h.hexdigest()[:16]


def cached_so() -> Path | None:
    so = _cache_root() / "tk_bwdbwd" / _source_hash() / "libtk_bwdbwd.so"
    return so if so.exists() else None


def build(force: bool = False, verbose: bool = True) -> Path:
    """Build (or reuse) the plugin; returns the path to libtk_bwdbwd.so."""
    out_dir = _cache_root() / "tk_bwdbwd" / _source_hash()
    so = out_dir / "libtk_bwdbwd.so"
    if so.exists() and not force:
        return so

    nvcc = _find_nvcc()
    if nvcc is None:
        raise RuntimeError("nvcc not found — `pip install 'flash-hog[tk]'` (or set CUDA_HOME).")
    if not (shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")):
        raise RuntimeError("no host C++ compiler found (g++/clang++); nvcc needs one.")

    import jax  # XLA FFI headers

    tk = _thunderkittens()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "libtk_bwdbwd.so.tmp"

    cmd = [
        nvcc, "-shared", "-Xcompiler", "-fPIC", "-std=c++20", "-O3", "--use_fast_math",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-forward-unknown-to-host-compiler", "-Xcompiler=-fno-strict-aliasing",
        "-Xcompiler=-Wno-psabi", "-DNDEBUG", "-DKITTENS_SM90",
        "-gencode", f"arch=compute_90a,code={_ARCH}",
        f"-I{tk / 'include'}", f"-I{tk / 'prototype'}", f"-I{jax.ffi.include_dir()}",
        "-cudart", "none",
    ]
    cmd += [f"-I{inc}" for inc in nvidia_wheel_paths("include")]
    cmd += [str(_SRC_DIR / "ffi.cu"), "-o", str(tmp)]

    if verbose:
        print(f"[flash-hog] building TK plugin (one-time, ~10-60 s): {so}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"TK plugin build failed (rc={proc.returncode}):\n"
            f"{' '.join(cmd)}\n{proc.stdout[-2000:]}\n{proc.stderr[-6000:]}"
        )
    os.replace(tmp, so)
    return so
