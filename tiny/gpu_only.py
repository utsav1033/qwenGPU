"""The GPU's part of a merged checkpoint, as a text-only Qwen3.5 model that fits 16 GB.

Keeps, untouched (bf16, bit for bit): the circuit's layers (the first chip_layers layers), the
final norm, and the embedding rows of the chip tokens (tied with the lm_head). Drops Qwen's own
layers (muted on chip tokens by the merge), the text vocabulary and the vision tower. Chip token
t becomes token id t (chip_token_offset 0). The circuit's layers are all Gated DeltaNet, so the
model has no attention cache that grows with the launch.

Run: uv run python tiny/gpu_only.py build/merged-2b build/gpu-only
"""

import json
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

PRE = "model.language_model."


def main(src: str, dst: str):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "qgpu.json").read_text())
    cfg = json.loads((src / "config.json").read_text())
    n, v0, nv = meta["chip_layers"], meta["chip_token_offset"], meta["chip_vocab"]
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]

    def keep(k):
        if k.startswith(PRE + "layers."):
            return int(k[len(PRE + "layers."):].split(".")[0]) < n
        return k in (PRE + "norm.weight", PRE + "embed_tokens.weight")

    shards, shard, size, total = [], {}, 0, 0
    for name in sorted(set(index.values())):
        with safe_open(src / name, "pt") as f:
            for k in f.keys():
                if not keep(k):
                    continue
                t = f.get_tensor(k)
                if k.endswith("embed_tokens.weight"):
                    t = t[v0:v0 + nv].clone()
                shard["model." + k[len(PRE):]] = t.contiguous()
                size += t.numel() * t.element_size()
                if size > 2**31:
                    shards.append(shard)
                    total += size
                    shard, size = {}, 0
    shards.append(shard)
    total += size
    wm = {}
    for i, sh in enumerate(shards):
        fn = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, str(dst / fn), metadata={"format": "pt"})
        wm.update({k: fn for k in sh})
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": wm}, indent=2))
    tc = dict(cfg["text_config"])
    tc.update(num_hidden_layers=n, layer_types=tc["layer_types"][:n], vocab_size=nv,
              architectures=["Qwen3_5ForCausalLM"], tie_word_embeddings=True)
    tc.pop("eos_token_id", None)
    (dst / "config.json").write_text(json.dumps(tc, indent=2))
    (dst / "qgpu.json").write_text(json.dumps(
        {**meta, "chip_token_offset": 0, "gpu_only": True, "from": str(src)}, indent=2))
    print(f"{dst}: {n} layers, {nv} chip tokens, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main(*sys.argv[1:3])
