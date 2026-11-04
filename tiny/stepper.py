"""The `step` backend: the gpu-only checkpoint's layers run by tiny/stepper.c on a CPU core.

Same weights and the same per-token computation as the `merged` backend (transformers on MPS),
with the zero weights left out (tiny/sparse_pack.py's build/gpu-only/sparse.npz). The model
writes every chip token of a launch; the host feeds what it fed before and reads memory from the
recurrent state, as qgpu.runtime.Device does. With check=True each launch's tokens and memory
must equal kernel.run_blocks' exactly.
"""

import ctypes
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from gpu import Result
from qgpu.circuit import kernel
from qgpu.circuit.kernel import EXEC
from qgpu.runtime import host_tokens

HERE = Path(__file__).resolve().parent
LINEARS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
           "gate_proj", "up_proj", "down_proj")
DIMS = {"in_proj_qkv": (3200, 6144), "in_proj_z": (3200, 2048), "in_proj_b": (3200, 16),
        "in_proj_a": (3200, 16), "out_proj": (2048, 3200), "gate_proj": (3200, 6144),
        "up_proj": (3200, 6144), "down_proj": (6144, 3200)}


def build() -> Path:
    src, lib = HERE / "stepper.c", HERE / "build" / "libstepper.dylib"
    if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
        lib.parent.mkdir(exist_ok=True)
        subprocess.run(["cc", "-O3", "-mcpu=native", "-ffp-contract=off", "-shared", "-fPIC",
                        str(src), "-o", str(lib)], check=True)
    return lib


class _State:
    """What qgpu.runtime.Device.read takes from a transformers cache: a layer's recurrent state."""

    def __init__(self, model):
        self.model = model

    def get_recurrent_state(self, layer, state_idx=0):
        import torch
        p = self.model.lib.st_state(self.model.m, layer)
        a = np.ctypeslib.as_array(ctypes.cast(p, ctypes.POINTER(ctypes.c_float)), (16, 128, 128))
        return torch.from_numpy(a.copy())[None]


class Step:
    name = "step"

    def __init__(self, merged: str | Path, check: bool = False):
        from qgpu.runtime import Device

        merged = Path(merged)
        self.meta = json.loads((merged / "qgpu.json").read_text())
        assert self.meta.get("gpu_only"), "the stepper runs tiny/gpu_only.py's checkpoint"
        cfg = json.loads((merged / "config.json").read_text())
        self.gpu = Device(self.meta, cfg)
        self.check = check
        self.seconds = 0.0
        lib = self.lib = ctypes.CDLL(str(build()))
        P = ctypes.c_void_p
        lib.st_new.restype = P
        lib.st_new.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.st_mat.argtypes = [P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, P, P, P]
        lib.st_layer.argtypes = [P] * 8
        lib.st_layer.argtypes[1] = ctypes.c_int
        lib.st_norm.argtypes = [P, P]
        lib.st_reset.argtypes = [P]
        lib.st_state.restype = P
        lib.st_state.argtypes = [P, ctypes.c_int]
        lib.st_step.argtypes = [P, ctypes.c_int, ctypes.c_int]
        lib.st_run.argtypes = [P, P, ctypes.c_int, P, ctypes.c_int, ctypes.c_int, P]
        pack = np.load(merged / "sparse.npz")
        self.keep = {k: np.ascontiguousarray(pack[k]) for k in pack.files}  # the C side's memory
        k = self.keep
        nl, vocab = int(pack["layers"]), cfg["vocab_size"]
        self.m = lib.st_new(nl, vocab)
        ptr = lambda a: a.ctypes.data
        for i in range(nl):
            for j, lin in enumerate(LINEARS):
                nin, nout = DIMS[lin]
                lib.st_mat(self.m, i, j, nin, nout, ptr(k[f"{i}.{lin}.colptr"]),
                           ptr(k[f"{i}.{lin}.rows"]), ptr(k[f"{i}.{lin}.vals"]))
            lib.st_layer(self.m, i, ptr(k[f"{i}.conv"]), ptr(k[f"{i}.A_log"]),
                         ptr(k[f"{i}.dt_bias"]), ptr(k[f"{i}.norm.weight"]),
                         ptr(k[f"{i}.input_layernorm"]), ptr(k[f"{i}.post_attention_layernorm"]))
        lib.st_mat(self.m, -1, 0, vocab, 3200, ptr(k["emb.colptr"]), ptr(k["emb.rows"]),
                   ptr(k["emb.vals"]))
        lib.st_mat(self.m, -1, 1, 3200, vocab, ptr(k["head.colptr"]), ptr(k["head.rows"]),
                   ptr(k["head.vals"]))
        lib.st_norm(self.m, ptr(k["norm"]))
        self.tokens = 0  # every token stepped: host-fed and generated

    def run(self, L, ln) -> Result:
        lw = ln.lw
        assert L.lanes == self.meta["lanes"] and L.stages == self.meta["stages"]
        host = np.array(host_tokens(lw), dtype=np.int32)
        stop = np.array([EXEC(0, lw.warps, ph) for ph in (False, True)], dtype=np.int32)
        out = np.zeros(200000, dtype=np.int32)
        t0 = time.time()
        self.lib.st_reset(self.m)
        n = self.lib.st_run(self.m, host.ctypes.data, len(host), stop.ctypes.data, len(stop),
                            len(out), out.ctypes.data)
        self.seconds += time.time() - t0
        self.tokens += len(host) + n - 1  # the end token is not run
        assert n < len(out), "the launch did not end"
        new = out[:n].tolist()
        mem = self.gpu.read(_State(self))
        res = Result(L.outputs(ln, mem), len(host), len(new),
                     sum(1 for t in new if kernel.kind(t) == "exec") - 1)
        if self.check:
            tr = kernel.run_blocks(lw.blocks, lw.mem, lw.warps, lanes=L.lanes)
            want = tr.tokens[len(host):]
            res.verified = new == want and res.bufs == L.outputs(ln, tr.mem)
            if not res.verified:
                bad = next((j for j, (a, b) in enumerate(zip(new, want)) if a != b), None)
                raise AssertionError(f"{ln.name}: the stepper's launch differs from the reference "
                                     f"(first different token {bad} of {len(want)}, "
                                     f"{len(new)} written)")
        return res
