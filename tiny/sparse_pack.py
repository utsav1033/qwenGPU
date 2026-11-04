"""The gpu-only checkpoint's nonzero weights, for tiny/stepper.c (the sparse CPU stepper).

The circuit's layers are 0.13% nonzero: of each weight matrix the pack keeps, column by column
(input by input), the outputs a nonzero weight connects, with the bf16 weight as an exact fp32.
Everything else the layers hold (norm weights, conv taps, A_log, dt_bias) is kept whole. The
tied embedding / lm_head is kept both ways: by token (embedding rows) and by hidden dim (the
lm_head's columns).

Run: uv run python tiny/sparse_pack.py build/gpu-only     # writes build/gpu-only/sparse.npz
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

LINEARS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
           "gate_proj", "up_proj", "down_proj")


def csc(w: torch.Tensor):
    """w (out, in) -> colptr (in + 1), rows, vals: for each input, its nonzero outputs ascending."""
    out, inp = w.shape
    nz = w.t().nonzero()  # (in, out) pairs, sorted by input then output
    cols, rows = nz[:, 0], nz[:, 1]
    vals = w.float()[rows, cols]
    colptr = torch.zeros(inp + 1, dtype=torch.int64)
    colptr[1:] = torch.bincount(cols, minlength=inp).cumsum(0)
    return colptr.int().numpy(), rows.int().numpy(), vals.numpy()


def main(src: str):
    src = Path(src)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((src / "config.json").read_text())
    files = {}

    def get(k):
        name = index[k]
        if name not in files:
            files[name] = safe_open(src / name, "pt")
        return files[name].get_tensor(k)

    out = {"layers": np.int32(cfg["num_hidden_layers"])}
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for lin in LINEARS:
            mod = "mlp." if lin in ("gate_proj", "up_proj", "down_proj") else "linear_attn."
            cp, r, v = csc(get(p + mod + lin + ".weight"))
            out[f"{i}.{lin}.colptr"], out[f"{i}.{lin}.rows"], out[f"{i}.{lin}.vals"] = cp, r, v
        out[f"{i}.conv"] = get(p + "linear_attn.conv1d.weight").float()[:, 0, :].numpy()
        for k in ("A_log", "dt_bias", "norm.weight"):
            out[f"{i}.{k}"] = get(p + "linear_attn." + k).float().numpy()
        for k in ("input_layernorm", "post_attention_layernorm"):
            out[f"{i}.{k}"] = get(p + k + ".weight").float().numpy()
        print(f"layer {i}", end="\r", flush=True)
    out["norm"] = get("model.norm.weight").float().numpy()
    emb = get("model.embed_tokens.weight")  # (vocab, hidden)
    out["emb.colptr"], out["emb.rows"], out["emb.vals"] = csc(emb.t().contiguous())  # by token
    out["head.colptr"], out["head.rows"], out["head.vals"] = csc(emb)  # by hidden dim
    np.savez(src / "sparse.npz", **out)
    n = sum(v.size for k, v in out.items() if k.endswith(".vals"))
    print(f"{src / 'sparse.npz'}: {n:,} nonzero weights")


if __name__ == "__main__":
    main(sys.argv[1])
