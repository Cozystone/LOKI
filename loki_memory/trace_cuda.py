"""Launch small storage kernels on PyTorch's current CUDA stream through NVRTC."""

import ctypes as ct
from ctypes.util import find_library
from functools import lru_cache
from pathlib import Path
import os
import sys
import torch


def _lib(names):
    for name in names:
        if name:
            try:
                return ct.CDLL(str(name))
            except OSError:
                pass
    raise RuntimeError("install NVRTC matching the PyTorch CUDA runtime")


@lru_cache(None)
def _module(device):
    root = Path(torch.__file__).parent / "lib"
    paths = (
        list(root.glob("nvrtc64_*.dll"))
        + [
            p
            for base in sys.path
            for p in Path(base).glob("nvidia/cuda_nvrtc/lib/libnvrtc.so*")
        ]
        if os.name != "nt"
        else sorted(root.glob("nvrtc64_*.dll"))
    )
    paths = [p for p in paths if ".alt" not in p.name]
    nv = _lib([*paths, find_library("nvrtc")])
    cu = _lib(["nvcuda.dll" if os.name == "nt" else "libcuda.so.1"])
    prog = ct.c_void_p()
    source = Path(__file__).with_name("trace_kernels.cu").read_bytes()

    def checked(rc):
        if rc:
            raise RuntimeError(f"CUDA compiler/driver error {rc}")

    checked(nv.nvrtcCreateProgram(ct.byref(prog), source, b"trace.cu", 0, None, None))
    major, minor = torch.cuda.get_device_capability(device)
    args = (ct.c_char_p * 1)(f"--gpu-architecture=compute_{major}{minor}".encode())
    try:
        rc = nv.nvrtcCompileProgram(prog, 1, args)
        size = ct.c_size_t()
        if rc:
            nv.nvrtcGetProgramLogSize(prog, ct.byref(size))
            buf = ct.create_string_buffer(size.value)
            nv.nvrtcGetProgramLog(prog, buf)
            raise RuntimeError(buf.value.decode())
        checked(nv.nvrtcGetPTXSize(prog, ct.byref(size)))
        buf = ct.create_string_buffer(size.value)
        checked(nv.nvrtcGetPTX(prog, buf))
    finally:
        nv.nvrtcDestroyProgram(ct.byref(prog))
    mod = ct.c_void_p()
    checked(cu.cuModuleLoadData(ct.byref(mod), buf))
    funcs = {}
    for name in (
        "scores",
        "finish",
        "finite_inputs",
        "pairs",
        "assign",
        "update",
        "edges",
    ):
        fn = ct.c_void_p()
        checked(cu.cuModuleGetFunction(ct.byref(fn), mod, name.encode()))
        funcs[name] = fn
    return cu, mod, funcs


def launch(name, device, grid, threads, *args):
    with torch.cuda.device(device):
        cu, _, funcs = _module(torch.cuda.current_device())
        vals = [
            ct.c_void_p(x.data_ptr()) if isinstance(x, torch.Tensor) else ct.c_int(x)
            for x in args
        ]
        params = (ct.c_void_p * len(vals))(*(ct.addressof(x) for x in vals))
        rc = cu.cuLaunchKernel(
            funcs[name],
            grid,
            1,
            1,
            threads,
            1,
            1,
            0,
            ct.c_void_p(torch.cuda.current_stream().cuda_stream),
            params,
            None,
        )
        if rc:
            raise RuntimeError(f"{name}: CUDA launch {rc}")
