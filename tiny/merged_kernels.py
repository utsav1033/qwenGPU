"""One launch of every kernel on the merged checkpoint, each compared token for token with
kernel.run_blocks (the --check of qgpu.bridge) and its buffers with the numpy reference.

Run: uv run python tiny/merged_kernels.py build/merged-2b [kernel ...]
"""
import sys
import time

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
import ref  # noqa: E402
from gpu import Launcher, Merged  # noqa: E402
from test_kernels import cases  # noqa: E402


def main():
    merged, only = sys.argv[1], set(sys.argv[2:])
    L, gpu = Launcher(), Merged(merged, check=True)
    seen = set()
    for name, bufs, sc, th in cases(np.random.default_rng(0)):
        if name in seen or (only and name not in only):
            continue
        seen.add(name)
        t0 = time.time()
        r = gpu.run(L, L.prepare(name, bufs, sc, th))
        want = [ref.words(b) for b in ref.KERNELS[name](*[list(b) for b in bufs], *sc)]
        dt = time.time() - t0
        print(f"{name:9s} {r.gen:5d} tokens written by the model in {dt:5.0f}s "
              f"({r.gen / dt:.1f}/s): run_blocks {'identical' if r.verified else 'DIFFERENT'}, "
              f"numpy reference {'identical' if r.bufs == want else 'DIFFERENT'}", flush=True)


if __name__ == "__main__":
    main()
