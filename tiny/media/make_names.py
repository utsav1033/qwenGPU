"""Data for the playground page (names.js): names sampled from the integer model (only the
samples that are popular real names), the model's
next-letter probabilities at every step, and the exact tokens the GPU writes per position.

Token counts come from the qgpu simulator, which writes the same tokens as the model
(checked token for token on one character). They depend only on the position, since every
kernel's control flow is fixed by its launch scalars; this script checks that on two inputs.

Run: uv run python tiny/media/make_names.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import engine  # noqa: E402
import gpu  # noqa: E402
import ref  # noqa: E402

qm = json.loads((HERE.parent / "qmodel.json").read_text())
chars, T = qm["chars"], qm["T"]


def tokens_per_position(c):
    eng = engine.Engine(qm, engine.GPU(gpu.Sim()))
    cache, out = [([], []) for _ in qm["layers"]], []
    for t in range(T):
        before = eng.stats.gen
        eng.step(c, t, cache, None)
        out.append(eng.stats.gen - before)
    return out


per_pos = tokens_per_position(0)
assert per_pos == tokens_per_position(5), "token counts depend on the data"
print("tokens per position:", per_pos)

def unpair(words):
    """Pair format (2 words per 4 values: x2 + x0 2^16, x3 + x1 2^16) back to int8 values."""
    out = []
    for g in range(0, len(words), 2):
        quad = [0] * 4
        for j, slot in ((0, (0, 2)), (1, (1, 3))):
            w = int(ref.us(words[g + j]))
            lo = ((w & 0xFFFF) ^ 0x8000) - 0x8000
            quad[slot[0]], quad[slot[1]] = (w - lo) >> 16, lo
        out += quad
    return out


def inside_of(trace, T_):
    """The tiny model's values at one position, from the launches' outputs: the embedded input
    (residual, 16), per layer each head's attention over the positions so far and the MLP's 32
    SiLU outputs (int8)."""
    k = [name for name, _ in trace]
    emb = [int(v) for v in ref.us(trace[0][1][0])]
    att, mlp = [], []
    for li in range(2):
        base = 1 + 14 * li
        assert k[base + 5] == "softmax" and k[base + 11] == "silu", k
        e = [int(v) for v in trace[base + 5][1]]
        heads = []
        for h in range(2):
            row = e[h * T_:(h + 1) * T_]
            tot = sum(row) or 1
            heads.append([round(v / tot, 3) for v in row])
        att.append(heads)
        mlp.append(unpair(trace[base + 11][1]))
    return {"emb": emb, "att": att, "mlp": mlp}


# keep the samples that are real, well-known names (names.txt is ordered by popularity); seed 0
# stays first: its first letter is the one run inside Qwen
rank = {n: i for i, n in enumerate((HERE.parent / "names.txt").read_text().split())}
eng = engine.Engine(qm, ref.Ref())
names = []
for seed in range(20000):
    if len(names) >= 40:
        break
    rng = np.random.default_rng(seed)
    cache, c, steps = [([], []) for _ in qm["layers"]], 0, []
    for t in range(T):
        eng.trace.clear()
        lg = ref.us(eng.step(c, t, cache, None)).astype(np.int64)
        inside = inside_of(eng.trace, t + 1)
        g = np.round(-np.log(-np.log(rng.uniform(1e-12, 1.0, len(chars)))) * 256).astype(np.int64)
        c = int(np.argmax(lg + g))
        p = np.exp((lg - lg.max()) / 256.0)
        p /= p.sum()
        top = np.argsort(-p)[:5]
        steps.append({**inside, "c": chars[c], "p": round(float(p[c]), 4),
                      "top": [[chars[k], round(float(p[k]), 4)] for k in top]})
        if c == 0:
            break
    name = "".join(s["c"] for s in steps).rstrip(".")
    if seed == 0 or (len(name) >= 4 and rank.get(name, 10**9) < 2000):
        names.append(steps)
        print(seed, name, rank.get(name))

name_of = lambda st: "".join(s["c"] for s in st).rstrip(".")
names = names[:1] + sorted(names[1:], key=lambda st: rank[name_of(st)])  # most popular first
(HERE / "names.js").write_text("window.NAMES=" + json.dumps({"perPos": per_pos, "names": names}) + ";")
