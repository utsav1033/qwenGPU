# tiny: a transformer running on the GPU inside Qwen

[qwenGPU](https://github.com/vixhal-baraiya/qwenGPU) compiles a tiny GPU into Qwen3.5-2B's weights:
every token the model writes is one step of the machine. This folder runs a small transformer on it.

The model is a 5,296-parameter name generator, trained on Karpathy's makemore names and quantized
to int8. It runs as 10 CUDA kernels written for qgpu, which has 4 cores, 126 words of memory and
integers only.

**Result:** on my MacBook, Qwen wrote **55,497 tokens over 1 h 43 min to produce one letter**, with
zero errors. Every token matched qgpu's simulator, and the output matched a plain numpy version of the
model ([log](logs/gpu-only-one-character.log)).

▶ [Video](media/post.mp4): names being generated, with the model's attention and neurons as Qwen
computes them. It's a replay with real values. [Playground](media/playground.html).

## How it works

- **Matmul:** two int8 products fit in one 32-bit multiply, because the result's top 16 bits hold
  `a0·x0 + a2·x2` exactly. The inner loop is hand-ordered PTX, so each 4-stage block is full.
- **exp:** `(1 − x/1024)^1024`, which is ten squarings in fixed point.
- **Division:** Newton's method, three steps.
- **RMSNorm:** 1/√ found one bit at a time.
- **Memory:** 126 words can't hold the model, so the host streams in one 64-word weight tile per
  launch. One letter takes 33 launches.

Also found a [qgpu compiler bug](repro/argmax.py) along the way: an argmax always returned 0.

## Run it

```bash
uv run python tiny/test_kernels.py 20                           # each kernel vs numpy
uv run python tiny/engine.py --backend sim --names 20 --verify  # whole names on the simulator
```

On the real Qwen weights (needs Qwen3.5-2B, ~9 GB of free RAM, ~1h 45m per letter):

```bash
uv run python -m qgpu.merge models/Qwen3.5-2B build/merged-2b
uv run python tiny/gpu_only.py build/merged-2b build/gpu-only   # just the GPU layers, 8.7 GB
uv run python tiny/engine.py --backend merged --merged build/gpu-only --check --verify --names 1 --max-len 1
```

The full merged model (15.3 GB) doesn't fit in 16 GB, so `gpu_only.py` keeps only the 45 GPU layers,
unchanged. Qwen's language layers aren't used.

## Files

`train.py` model + training · `quant.py` int8 · `tiny.cu` kernels · `ref.py` numpy reference ·
`engine.py` host engine · `gpu.py` launch / simulator / Qwen backends · `logs/` raw output

Names from [karpathy/makemore](https://github.com/karpathy/makemore). GPU by
[vixhal-baraiya/qwenGPU](https://github.com/vixhal-baraiya/qwenGPU).
