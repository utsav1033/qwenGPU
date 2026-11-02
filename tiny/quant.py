"""Quantize tiny/model.pt for the integer engine (tiny/engine.py): int8 weights in byte format,
activation scales calibrated on training names, and every rescale as an integer (M, S) pair.

Scales: the residual stream is int32 with a power-of-two scale (|x| < 2^11); every matmul input
is int8 (pair format) with a per-site scale; scores, MLP pre-activations and logits are Q8.
Writes tiny/qmodel.json.

Run: uv run python tiny/quant.py
"""

import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from train import CHARS, D, EPS, FF, H, HERE, L, T, Tiny, dataset


def ms(rho: float, top: int = 1 << 11) -> tuple[int, int]:
    """rho ~ M / 2^S with top / 2 <= M < top (S >= 1)."""
    S = max(1, math.floor(math.log2(top / rho)))
    M = round(rho * 2**S)
    while M >= top:
        S -= 1
        M = round(rho * 2**S)
    assert S >= 1 and 0 < M < top, (rho, M, S)
    return M, S


def q8(w: np.ndarray) -> tuple[np.ndarray, float]:
    sc = float(np.abs(w).max()) / 127
    return np.clip(np.round(w / sc), -127, 127).astype(np.int64), sc


def byte_words(q: np.ndarray) -> list[int]:
    """Rows of int8 (length a multiple of 4) as byte-format words, row by row."""
    v = (q.reshape(-1, 4) + 128).astype(np.int64)
    return [int(a | b << 8 | c << 16 | d << 24) for a, b, c, d in v]


@torch.no_grad()
def calibrate(model: Tiny, X: torch.Tensor) -> dict:
    """Max |activation| at every quantized site, and the smallest rms at each norm's input."""
    st = {}

    def see(key, t):
        st[key] = max(st.get(key, 0.0), float(t.abs().max()))

    def rms_min(key, x, mask):
        r = x.pow(2).mean(-1).sqrt()[mask]
        st[key] = min(st.get(key, 1e9), float(r.min()))

    mask = torch.ones(X.shape, dtype=torch.bool)
    x = model.tok(X) + model.pos(torch.arange(T))
    see("x", x)
    for li, b in enumerate(model.blocks):
        rms_min(f"rms{li}a", x, mask)
        h = b.n1(x)
        see(f"n{li}a", h)
        q, k, v = b.qkv(h).split(D, -1)
        see(f"q{li}", q), see(f"k{li}", k), see(f"v{li}", v)
        qh, kh, vh = (z.view(*z.shape[:2], H, D // H).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        x = x + b.o(a.transpose(1, 2).reshape(*x.shape))
        see("x", x)
        rms_min(f"rms{li}b", x, mask)
        h = b.n2(x)
        see(f"n{li}b", h)
        uu = b.up(h)
        see(f"u{li}", uu)
        sl = F.silu(uu)
        see(f"s{li}", sl)
        x = x + b.down(sl)
        see("x", x)
    rms_min("rmsf", x, mask)
    see("nf", model.nf(x))
    return st


def norm_params(g: np.ndarray, s_n: float, rms_min_int: float) -> dict:
    """rmsnorm's threshold th, gains G and shift S: x * r * G / 2^S = x / rms * g / s_n,
    with r ~ sqrt(th) 2^14 / rms < 2^15 and |x r G| < 2^30 (|x| <= 4 rms for 16 values)."""
    th = 4
    while th < 1024 and math.sqrt(th * 4) * 2**14 / rms_min_int < 2**14:
        th *= 4
    S = math.floor(math.log2(2**27 * s_n / float(np.abs(g).max())))
    G = np.round(g / s_n * 2**S / (math.sqrt(th) * 2**14)).astype(np.int64)
    assert np.abs(G).max() * math.sqrt(th) * 2**16 < 2**30
    return {"th": th, "G": [int(v) for v in G], "S": S}


def main():
    model = Tiny()
    model.load_state_dict(torch.load(HERE / "model.pt"))
    model.eval()
    (Xtr, _), _ = dataset()
    st = calibrate(model, Xtr[:4000])
    sd = {k: v.numpy().astype(np.float64) for k, v in model.state_dict().items()}

    xmax = st["x"]
    s_x = 2.0 ** math.ceil(math.log2(xmax * 1.25 / 2**11))
    qm = {"chars": CHARS, "T": T, "D": D, "H": H, "FF": FF, "L": L, "s_x": s_x, "calib": st}

    tq, s_te = q8(sd["tok.weight"])
    pq, s_pe = q8(sd["pos.weight"])
    Mt, St = ms(s_te / s_x)
    Mp, Sp = ms(s_pe / s_x)
    S = min(St, Sp)
    qm["embed"] = {"tok": [byte_words(r) for r in tq], "pos": [byte_words(r) for r in pq],
                   "Mt": round(s_te / s_x * 2**S), "Mp": round(s_pe / s_x * 2**S), "S": S}

    act = lambda key: st[key] * 1.0 / 127  # noqa: E731  int8 scale of a site
    layers = []
    for li in range(L):
        p = f"blocks.{li}."
        s_na, s_nb = act(f"n{li}a"), act(f"n{li}b")
        s_q, s_k, s_v, s_s = act(f"q{li}"), act(f"k{li}"), act(f"v{li}"), act(f"s{li}")
        Wq, Wk, Wv = np.split(sd[p + "qkv.weight"], 3)
        lay = {"n1": norm_params(sd[p + "n1.g"], s_na, st[f"rms{li}a"] / s_x),
               "n2": norm_params(sd[p + "n2.g"], s_nb, st[f"rms{li}b"] / s_x)}
        for name, W, s_out in (("q", Wq, s_q), ("k", Wk, s_k), ("v", Wv, s_v)):
            wq, sw = q8(W)
            lay[name] = {"W": byte_words(wq), "MS": ms(sw * s_na / s_out)}
        lay["score"] = ms(s_q * s_k / math.sqrt(D // H) * 256)
        wq, sw = q8(sd[p + "o.weight"])
        lay["o"] = {"W": byte_words(wq), "MS": ms(sw * s_v / s_x)}
        wq, sw = q8(sd[p + "up.weight"])
        lay["up"] = {"W": byte_words(wq), "MS": ms(sw * s_nb * 256)}
        lay["silu"] = ms(1 / (256 * s_s))
        wq, sw = q8(sd[p + "down.weight"])
        lay["down"] = {"W": byte_words(wq), "MS": ms(sw * s_s / s_x)}
        layers.append(lay)
    qm["layers"] = layers
    s_nf = act("nf")
    qm["nf"] = norm_params(sd["nf.g"], s_nf, st["rmsf"] / s_x)
    wq, sw = q8(sd["head.weight"])
    qm["head"] = {"W": byte_words(wq), "MS": ms(sw * s_nf * 256)}
    (HERE / "qmodel.json").write_text(json.dumps(qm))
    print(f"residual scale 2^{int(math.log2(s_x))}, max |x| {xmax:.2f} -> {xmax / s_x:.0f}; "
          f"norm thresholds {[l_['n1']['th'] for l_ in layers]} {[l_['n2']['th'] for l_ in layers]} "
          f"{qm['nf']['th']}; {sum(v.size for v in sd.values())} parameters -> int8")
    print("calibration:", {k: round(v, 3) for k, v in st.items()})


if __name__ == "__main__":
    main()
