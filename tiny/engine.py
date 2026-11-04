"""The host engine: the quantized char transformer (tiny/qmodel.json) as qgpu kernel launches.

The GPU has 126 words of memory, so the host keeps the weights, the K / V caches and the
activations, and streams them in per launch: each launch gets one weight tile (at most 64
words: 16 rows x 16 inputs, or 8 x 32) and the activations it reads, and the host reads back
what it wrote, like cudaMemcpy around every kernel. The host only moves words: every value
(embedding sums, norms, matmuls, attention, SiLU, the sampled token) is computed by a kernel.
Sampling is Gumbel-max: the host draws the noise from a seeded RNG (like a curand seed) and the
`sample` kernel takes argmax(logits + noise).

One position (33 launches): embed; per layer rmsnorm, q, k, v, scores, softmax, av, o (+=
residual), rmsnorm, up x 2, silu, down x 2 (+= residual); then rmsnorm, head x 2, sample. The
kernels that write the residual stream (embed, mm_res) also leave sum(x^2) for the next norm.

Backends: ref (tiny/ref.py, numpy), sim (kernel.run_blocks), merged (the merged Qwen3.5
checkpoint; --check compares each launch with run_blocks token for token), step (the gpu-only
checkpoint's layers on a CPU core, zero weights left out: tiny/stepper.c; --check too).

Run:
  uv run python tiny/engine.py --backend sim --names 5 --verify     # sim, and ref beside it
  uv run python tiny/engine.py --backend merged --merged build/merged-2b --check --names 1
  uv run python tiny/engine.py --backend step --merged build/gpu-only --check --names 1
  uv run python tiny/engine.py --eval 200                           # NLL: float vs integer
"""

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref  # noqa: E402

HERE = Path(__file__).resolve().parent
TILE = 64  # words of one weight tile


@dataclass
class Stats:
    launches: int = 0
    host: int = 0  # chip tokens the host fed
    gen: int = 0  # chip tokens the GPU wrote
    records: int = 0
    verified: int = 0
    by_kernel: Counter = field(default_factory=Counter)
    gen_by_kernel: Counter = field(default_factory=Counter)

    def add(self, name, r):
        self.launches += 1
        self.by_kernel[name] += 1
        if r is not None:
            self.host, self.gen, self.records = self.host + r.host, self.gen + r.gen, self.records + r.records
            self.gen_by_kernel[name] += r.gen
            self.verified += bool(r.verified)


class GPU:
    """A qgpu backend (sim or merged) behind the reference's launch(): prepare (lower), run,
    read back; with verify, every launch is also run on the numpy reference and must match."""

    def __init__(self, backend, verify: bool = False):
        from gpu import Launcher
        self.L, self.backend, self.verify = Launcher(), backend, verify
        self.name = backend.name
        self.last = None

    def launch(self, name, bufs, scalars, threads):
        ln = self.L.prepare(name, bufs, scalars, threads)
        self.last = r = self.backend.run(self.L, ln)
        if self.verify:
            want = [ref.words(b) for b in ref.KERNELS[name](*[list(b) for b in bufs], *scalars)]
            assert r.bufs == want, f"{name} {scalars}: the GPU's buffers differ from the reference"
        return r.bufs


class Engine:
    def __init__(self, qm: dict, dev):
        self.qm, self.dev = qm, dev
        self.stats = Stats()
        self.trace = []  # (kernel, output words) of every launch, for comparing backends

    def run(self, name, bufs, scalars, threads, out=-1):
        """One launch; its output buffer `out` (an index, or a tuple of them)."""
        bufs = [ref.words(b) for b in bufs]
        res = self.dev.launch(name, bufs, scalars, threads)
        self.stats.add(name, getattr(self.dev, "last", None))
        got = tuple(res[o] for o in out) if isinstance(out, tuple) else res[out]
        self.trace.append((name, got))
        return got

    # --- the model's pieces, each one or more launches ------------------------------------

    def matvec(self, kernel, W, x, rows, kw, MS, out):
        """W (rows x kw words) times x, in tiles of TILE words; out: the output buffer's
        initial words (for mm_res, the residual stream: then (x, sum of x^2) comes back)."""
        per = TILE // kw  # rows per tile
        res, ss = list(out), [0]
        for r0 in range(0, rows, per):
            n = min(per, rows - r0)
            tile = W[r0 * kw:(r0 + n) * kw]
            if kernel == "mm_res":
                res, ss = self.run(kernel, [tile, x, res, ss], [n, kw, n, *MS, r0], n, (2, 3))
            elif kernel == "mm_raw":
                res[r0:r0 + n] = self.run(kernel, [tile, x, [0] * n], [n, kw, n, *MS], n)
            else:
                assert r0 == 0 and n == rows
                res = self.run(kernel, [tile, x, out], [n, kw, n, *MS], n)
        return (res, ss) if kernel == "mm_res" else res

    def norm(self, x, ss, p):
        return self.run("rmsnorm", [x, ss, ref.words(p["G"]), [0] * 8], [p["th"], p["S"]], 4)

    def step(self, c: int, t: int, cache: list, noise: list[int] | None):
        """Position t, token c: the next token (with noise) or the Q8 logits (noise None)."""
        qm, D = self.qm, self.qm["D"]
        e = qm["embed"]
        x, ss = self.run("embed", [e["tok"][c], e["pos"][t], [0] * D, [0]],
                         [D, e["Mt"], e["Mp"], e["S"]], D, (2, 3))
        T = t + 1
        for li, lay in enumerate(qm["layers"]):
            kc, vc = cache[li]
            h = self.norm(x, ss, lay["n1"])
            q = self.matvec("mm_pair", lay["q"]["W"], h, 16, 4, lay["q"]["MS"], [0] * 8)
            kc.append(self.matvec("mm_byte", lay["k"]["W"], h, 16, 4, lay["k"]["MS"], [0] * 4))
            vc.append(self.matvec("mm_byte", lay["v"]["W"], h, 16, 4, lay["v"]["MS"], [0] * 4))
            # K rows (head, position), 2 words each; each head's rows read its own q half
            K = [w for hd in range(2) for j in range(T) for w in kc[j][2 * hd:2 * hd + 2]]
            sc = self.run("mm_raw", [K, q, [0] * 2 * T], [2 * T, 2, T, *lay["score"]], 2 * T)
            ex = self.run("softmax", [sc, [0] * 2 * T], [2 * T, T], 2 * T)
            a = self.run("av", [ex, [w for j in range(T) for w in vc[j]], [0] * 8], [16, T], 16)
            x, ss = self.matvec("mm_res", lay["o"]["W"], a, 16, 4, lay["o"]["MS"], x)
            h = self.norm(x, ss, lay["n2"])
            u = self.matvec("mm_raw", lay["up"]["W"], h, 32, 4, lay["up"]["MS"], [0] * 32)
            s = self.run("silu", [u, [0] * 16], [32, *lay["silu"]], 32)
            x, ss = self.matvec("mm_res", lay["down"]["W"], s, 16, 8, lay["down"]["MS"], x)
        h = self.norm(x, ss, qm["nf"])
        V = len(qm["chars"])
        lg = self.matvec("mm_raw", qm["head"]["W"], h, V, 4, qm["head"]["MS"], [0] * V)
        if noise is None:
            return lg
        return self.run("sample", [lg, noise, [0]], [V], 4)[0]

    def generate(self, rng: np.random.Generator, max_len: int | None = None) -> str:
        """One name, sampled: Gumbel noise in Q8 from the host's RNG (at most max_len
        characters generated, end token included)."""
        cache = [([], []) for _ in self.qm["layers"]]
        c, out = 0, []
        for t in range(min(self.qm["T"], max_len or self.qm["T"])):
            g = -np.log(-np.log(rng.uniform(1e-12, 1.0, len(self.qm["chars"]))))
            c = self.step(c, t, cache, ref.words(np.round(g * 256).astype(np.int64)))
            if c == 0:
                break
            out.append(self.qm["chars"][c])
        return "".join(out)


def backend(name, args):
    if name == "ref":
        return ref.Ref()
    from gpu import Merged, Sim
    if name == "step":
        from stepper import Step
        return GPU(Step(args.merged, check=args.check), verify=args.verify)
    be = Sim() if name == "sim" else Merged(args.merged, args.device, check=args.check)
    return GPU(be, verify=args.verify)


def evaluate(qm, n: int):
    """Mean NLL per character on validation names: the float model vs the integer reference."""
    import torch
    import torch.nn.functional as F

    from train import Tiny, dataset
    _, (Xva, Yva) = dataset()
    Xva, Yva = Xva[:n], Yva[:n]
    model = Tiny()
    model.load_state_dict(torch.load(HERE / "model.pt"))
    with torch.no_grad():
        fl = F.cross_entropy(model(Xva).view(-1, 27), Yva.view(-1), ignore_index=-1).item()
    eng = Engine(qm, ref.Ref())
    nll, cnt = 0.0, 0
    for x, y in zip(Xva.tolist(), Yva.tolist()):
        cache = [([], []) for _ in qm["layers"]]
        for t in range(qm["T"]):
            if y[t] < 0:
                break
            lg = ref.us(eng.step(x[t], t, cache, None)) / 256.0
            nll -= lg[y[t]] - np.log(np.exp(lg - lg.max()).sum()) - lg.max()
            cnt += 1
    print(f"validation NLL on {n} names ({cnt} characters): float {fl:.4f}, "
          f"integer {nll / cnt:.4f} nats/char")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("ref", "sim", "merged", "step"), default="sim")
    ap.add_argument("--names", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="check every launch's buffers against the numpy reference")
    ap.add_argument("--check", action="store_true",
                    help="merged, step: every launch token for token against kernel.run_blocks")
    ap.add_argument("--merged", default="build/merged-2b")
    ap.add_argument("--device")
    ap.add_argument("--max-len", type=int, help="stop each name after this many characters")
    ap.add_argument("--eval", type=int, metavar="N", help="NLL of float vs integer on N names")
    args = ap.parse_args()
    qm = json.loads((HERE / "qmodel.json").read_text())
    if args.eval:
        return evaluate(qm, args.eval)

    eng = Engine(qm, backend(args.backend, args))
    refe = Engine(qm, ref.Ref())  # the numpy integer reference, same noise
    rng, rrng = np.random.default_rng(args.seed), np.random.default_rng(args.seed)
    t0, chars, same = time.time(), 0, True
    for k in range(args.names):
        t1, before = time.time(), eng.stats.launches
        name = eng.generate(rng, args.max_len)
        want = refe.generate(rrng, args.max_len)
        ok = name == want and [w for _, w in eng.trace] == [w for _, w in refe.trace]
        same &= ok
        eng.trace.clear(), refe.trace.clear()
        chars += min(len(name) + 1, args.max_len or 99)  # the end '.' is generated too
        print(f"{k + 1}: {name!r:18} reference {want!r:18} {'match' if ok else 'DIFFERENT'} "
              f"({eng.stats.launches - before} launches, {time.time() - t1:.1f}s)", flush=True)
    st = eng.stats
    print(f"\n{args.names} names, {chars} generated characters (end tokens included) on "
          f"{eng.dev.name}: {'token for token identical to' if same else 'DIFFERENT from'} "
          f"the numpy integer reference, every launch's output too")
    print(f"launches: {st.launches} ({st.launches / chars:.1f} per character)")
    if st.gen:
        print(f"chip tokens: {st.gen} written by the GPU ({st.gen / chars:.0f} per character), "
              f"{st.host} fed by the host ({st.host / chars:.0f} per character); "
              f"{st.records} machine-block records ({st.records / chars:.0f} per character)")
        if args.check or args.verify:
            print(f"verified launches: {st.verified if args.check else st.launches}/{st.launches}")
        print("generated tokens by kernel per character:",
              {k: round(v / chars) for k, v in st.gen_by_kernel.most_common()})
    be = getattr(eng.dev, "backend", None)
    if getattr(be, "seconds", 0):
        print(f"the model's own time: {be.seconds:.0f}s"
              + (f", {be.tokens} tokens stepped, {be.tokens / be.seconds:.0f} tokens/s"
                 if getattr(be, "tokens", 0) else ""))
    print(f"{time.time() - t0:.0f}s")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
