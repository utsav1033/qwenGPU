"""Per-site error of the integer engine against the float model, on validation names: each
traced launch output decoded to real values and compared with the float activation."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref
from engine import Engine
from train import Tiny, dataset, D, H, HERE

qm = json.loads((HERE / "qmodel.json").read_text())
model = Tiny(); model.load_state_dict(torch.load(HERE / "model.pt")); model.eval()
_, (Xva, Yva) = dataset()

def pair_vals(w):
    w = ref.us(w); out = []
    for k in range(0, len(w), 2):
        ex, ox = int(w[k]), int(w[k + 1])
        x0 = (ex + 32768) >> 16; x2 = ex - (x0 << 16)
        x1 = (ox + 32768) >> 16; x3 = ox - (x1 << 16)
        out += [x0, x1, x2, x3]
    return np.array(out, float)

def byte_vals(w):
    return np.array([((int(v) >> 8 * b) & 255) - 128 for v in w for b in range(4)], float)

errs = {}
def rel(key, a, b):
    errs.setdefault(key, []).append(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))

@torch.no_grad()
def float_sites(x_ids):
    t = len(x_ids); idx = torch.tensor([x_ids])
    x = model.tok(idx) + model.pos(torch.arange(t)); S = {"x0": x}
    for li, b in enumerate(model.blocks):
        h = b.n1(x); S[f"n1{li}"] = h
        q, k, v = b.qkv(h).split(D, -1); S[f"q{li}"], S[f"k{li}"], S[f"v{li}"] = q, k, v
        qh, kh, vh = (z.view(1, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        att = (qh @ kh.transpose(-1, -2)) / np.sqrt(D // H)
        S[f"sc{li}"] = att
        a = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True).transpose(1, 2).reshape(1, t, D)
        S[f"a{li}"] = a
        x = x + b.o(a); S[f"xo{li}"] = x
        h = b.n2(x); S[f"n2{li}"] = h
        u = b.up(h); S[f"u{li}"] = u
        S[f"s{li}"] = F.silu(u)
        x = x + b.down(F.silu(u)); S[f"xd{li}"] = x
    S["nf"] = model.nf(x); S["lg"] = model.head(model.nf(x))
    return {k: v[0] for k, v in S.items()}

c = qm["calib"]; s_x = qm["s_x"]
for x, y in list(zip(Xva.tolist(), Yva.tolist()))[:60]:
    n = sum(1 for v in y if v >= 0); ids = x[:n]
    fs = float_sites(ids)
    eng = Engine(qm, ref.Ref()); cache = [([], []) for _ in qm["layers"]]
    for t in range(n):
        eng.trace.clear(); eng.step(ids[t], t, cache, None)
        tr = [w for _, w in eng.trace]; names = [k for k, _ in eng.trace]
        it = iter(range(len(tr)))
        rel("x0", ref.us(tr[0]) * s_x, fs["x0"][t].numpy())
        p = 1
        for li in range(2):
            sc = lambda key: c[key] / 127
            p += 1; rel(f"n1{li}", pair_vals(tr[p]) * sc(f"n{li}a"), fs[f"n1{li}"][t].numpy()); p += 1
            rel(f"q{li}", pair_vals(tr[p]) * sc(f"q{li}"), fs[f"q{li}"][t].numpy()); p += 1
            rel(f"k{li}", byte_vals(tr[p]) * sc(f"k{li}"), fs[f"k{li}"][t].numpy()); p += 1
            rel(f"v{li}", byte_vals(tr[p]) * sc(f"v{li}"), fs[f"v{li}"][t].numpy()); p += 1
            T = t + 1
            sco = ref.us(tr[p]) / 256; rel(f"sc{li}", sco, fs[f"sc{li}"][:, t, :T].numpy().ravel()); p += 1
            p += 1  # softmax
            rel(f"a{li}", pair_vals(tr[p]) * sc(f"v{li}"), fs[f"a{li}"][t].numpy()); p += 1
            rel(f"xo{li}", ref.us(tr[p]) * s_x, fs[f"xo{li}"][t].numpy()); p += 1
            p += 1; rel(f"n2{li}", pair_vals(tr[p]) * sc(f"n{li}b"), fs[f"n2{li}"][t].numpy()); p += 1
            u = np.concatenate([ref.us(tr[p]), ref.us(tr[p + 1])]) / 256; rel(f"u{li}", u, fs[f"u{li}"][t].numpy()); p += 2
            rel(f"s{li}", pair_vals(tr[p]) * sc(f"s{li}"), fs[f"s{li}"][t].numpy()); p += 1
            p += 1; rel(f"xd{li}", ref.us(tr[p]) * s_x, fs[f"xd{li}"][t].numpy()); p += 1
        p += 1; rel("nf", pair_vals(tr[p]) * c["nf"] / 127, fs["nf"][t].numpy()); p += 1
        lg = np.concatenate([ref.us(tr[p]), ref.us(tr[p + 1])]) / 256; rel("lg", lg, fs["lg"][t].numpy())
for k, v in errs.items():
    print(f"{k:5s} rel err mean {np.mean(v):.4f} max {np.max(v):.4f}")
