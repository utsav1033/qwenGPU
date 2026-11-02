"""Launching tiny/tiny.cu's kernels on qgpu: the driver's side (cudaMalloc, cudaMemcpy, launch).

`Launcher.prepare` lays out a launch's buffers in global memory and lowers the PTX for it
(qgpu.lower, specialized on the scalars); a backend runs it and returns every buffer's words:
- Sim: the circuit's reference interpreter (qgpu.circuit.kernel.run_blocks).
- Merged: the merged Qwen3.5 checkpoint, greedy-decoding the launch's chip tokens; with
  check=True each launch's tokens and memory must equal run_blocks' exactly (bridge's --check).
"""

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from qgpu import ptx
from qgpu.circuit import kernel
from qgpu.circuit.kernel import EXEC
from qgpu.lower import lower
from qgpu.merge import STAGES
from qgpu.runtime import host_tokens

HERE = Path(__file__).resolve().parent
CLANG = "/opt/homebrew/opt/llvm/bin/clang"
M32 = 0xFFFFFFFF


def build() -> dict[str, ptx.Kernel]:
    src, out = HERE / "tiny.cu", HERE / "tiny.ptx"
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        subprocess.run([CLANG, "-x", "cuda", "--cuda-device-only", "-nocudainc", "-nocudalib",
                        "--cuda-gpu-arch=sm_70", "-Xclang", "-target-feature", "-Xclang",
                        "+ptx70", "-O2", "-S", str(src), "-o", str(out)], check=True)
    return ptx.load(out)


@dataclass
class Launch:
    name: str
    lw: object  # qgpu.lower.Launch
    sizes: list[int]  # each buffer's length, in parameter order


@dataclass
class Result:
    bufs: list[list[int]]  # every buffer's words after the launch (uint32)
    host: int  # chip tokens the host fed (program image, memory, the first EXEC)
    gen: int  # chip tokens the GPU wrote
    records: int  # EXEC records: machine blocks run
    verified: bool | None = None


class Launcher:
    def __init__(self, lanes: int = 4, stages: int = 4):
        self.kernels = build()
        self.lanes, self.stages = lanes, stages
        mul, logic, atom = STAGES[stages]
        self.cfg = {"lanes": lanes, "stages": stages, "mul_stages": mul, "logic_stages": logic,
                    "atom_stages": atom, "write_through": True}
        self.cache = {}

    def prepare(self, name: str, bufs: list[list[int]], scalars: list[int], threads: int) -> Launch:
        """bufs: the kernel's pointer parameters in order (initial words), scalars: its int
        parameters in order; one thread per item, rounded up to whole warps."""
        k = self.kernels[name]
        args, data, it_b, it_s = [], {}, iter(bufs), iter(scalars)
        for p in k.params:
            if p.type in (".u64", ".b64", ".s64"):
                bid = len(data)
                data[bid] = [int(v) & M32 for v in next(it_b)]
                args.append(bid)
            else:
                args.append(int(next(it_s)) & M32)
        assert sum(len(b) for b in data.values()) < kernel.SINK, f"{name}: too much memory"
        total = max(1 << (threads - 1).bit_length(), self.lanes)
        block = min(total, 32)
        key = (name, tuple(args), tuple(len(b) for b in data.values()), total)
        if key not in self.cache:
            self.cache[key] = lower(k, (total // block,), (block,), args, data, **self.cfg)
        lw = self.cache[key]
        mem = {}
        for bid, words in data.items():
            base = lw.buffers[bid]
            for j, v in enumerate(words):
                mem[base + j] = v
        lw = type(lw)(lw.blocks, mem, lw.warps, lw.buffers, lw.dead)
        return Launch(name, lw, [len(b) for b in data.values()])

    def outputs(self, ln: Launch, mem: dict[int, int]) -> list[list[int]]:
        return [[mem.get(ln.lw.buffers[bid] + j, 0) for j in range(n)]
                for bid, n in enumerate(ln.sizes)]


class Sim:
    """The reference interpreter of the circuit: the token stream the model must write."""
    name = "sim"

    def run(self, L: Launcher, ln: Launch) -> Result:
        lw = ln.lw
        tr = kernel.run_blocks(lw.blocks, lw.mem, lw.warps, lanes=L.lanes)
        host = len(host_tokens(lw))
        recs = sum(1 for t in tr.tokens[host:] if kernel.kind(t) == "exec") - 1
        return Result(L.outputs(ln, tr.mem), host, len(tr.tokens) - host, recs)


class Merged:
    """The merged Qwen3.5 checkpoint: the host feeds the program image, the memory and the
    first EXEC token; the model greedy-decodes the rest of the launch; the host reads global
    memory from the model's recurrent state (qgpu.runtime.Device). Every launch runs in a fresh
    context, so memory starts at zero."""
    name = "merged"

    def __init__(self, merged: str | Path, device: str | None = None, check: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        from qgpu.runtime import Device

        merged = Path(merged)
        self.meta = json.loads((merged / "qgpu.json").read_text())
        # tiny/gpu_only.py's checkpoint: the circuit's layers alone, a text-only model
        auto = AutoModelForCausalLM if self.meta.get("gpu_only") else AutoModelForImageTextToText
        self.v0 = self.meta["chip_token_offset"]
        self.dev = device or ("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
        if self.dev == "cuda":  # exact bf16
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_tf32 = False
        self.torch = torch
        self.model = auto.from_pretrained(merged, dtype=torch.bfloat16).to(self.dev).eval()
        cfg = self.model.config
        self.gpu = Device(self.meta, (cfg.text_config if hasattr(cfg, "text_config") else cfg).to_dict())
        self.check = check
        self.seconds = 0.0

    def run(self, L: Launcher, ln: Launch) -> Result:
        torch, lw = self.torch, ln.lw
        assert L.lanes == self.meta["lanes"] and L.stages == self.meta["stages"]
        host = host_tokens(lw)
        end = {self.v0 + EXEC(0, lw.warps, ph) for ph in (False, True)}
        t0 = time.time()
        cache, ids, new, pos = None, [self.v0 + t for t in host], [], 0
        with torch.no_grad():
            while True:
                # positions given: a cache of DeltaNet layers alone cannot count its tokens
                p = torch.arange(pos, pos + len(ids), device=self.dev)[None]
                # with DeltaNet layers alone there is no full-attention mask to build
                mask = {"linear_attention": None} if self.meta.get("gpu_only") else None
                out = self.model(input_ids=torch.tensor([ids], device=self.dev), position_ids=p,
                                 attention_mask=mask, past_key_values=cache, use_cache=True,
                                 logits_to_keep=1)
                pos += len(ids)
                cache = out.past_key_values
                t = int(out.logits[0, -1].argmax())
                new.append(t)
                ids = [t]
                if t in end or not self.v0 <= t < self.v0 + kernel.CHIP_VOCAB:
                    break
                assert len(new) < 200000, "the launch did not end"
            # the end token is not run: memory is complete before it
        self.seconds += time.time() - t0
        mem = self.gpu.read(cache)
        res = Result(L.outputs(ln, mem), len(host), len(new),
                     sum(1 for t in new if kernel.kind(t - self.v0) == "exec") - 1)
        if self.check:
            tr = kernel.run_blocks(lw.blocks, lw.mem, lw.warps, lanes=L.lanes)
            want = [t + self.v0 for t in tr.tokens[len(host):]]
            res.verified = new == want and res.bufs == L.outputs(ln, tr.mem)
            if not res.verified:
                bad = next((j for j, (a, b) in enumerate(zip(new, want)) if a != b), None)
                raise AssertionError(f"{ln.name}: the model's launch differs from the reference "
                                     f"(first different token {bad} of {len(want)})")
        return res
