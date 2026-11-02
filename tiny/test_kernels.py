"""Every kernel on qgpu's simulator (kernel.run_blocks) against the numpy reference, on random
inputs in each kernel's range: every buffer must be identical, word for word.

Run: uv run python tiny/test_kernels.py [trials]
"""

import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from gpu import Launcher, Sim  # noqa: E402
import ref  # noqa: E402


def byte_words(v):
    v = np.asarray(v).reshape(-1, 4) + 128
    return [int(a | b << 8 | c << 16 | d << 24) for a, b, c, d in v]


def pair_words(v):
    v = np.asarray(v).reshape(-1, 4)
    return ref.words([w for a, b, c, d in v for w in (c + a * 65536, d + b * 65536)])


def cases(rng):
    i8 = lambda n: rng.integers(-127, 128, n)  # noqa: E731
    z = lambda n: [0] * n  # noqa: E731
    yield "mm_pair", [byte_words(i8(256)), pair_words(i8(16)), z(8)], [16, 4, 16, 1500, 18], 16
    yield "mm_byte", [byte_words(i8(256)), pair_words(i8(16)), z(4)], [16, 4, 16, 1500, 18], 16
    yield "mm_raw", [byte_words(i8(256)), pair_words(i8(16)), z(16)], [16, 4, 16, 1100, 12], 16
    T = int(rng.integers(1, 17))
    yield "mm_raw", [byte_words(i8(16 * T)), pair_words(i8(16)), z(2 * T)], [2 * T, 2, T, 1100, 12], 2 * T
    yield "mm_raw", [byte_words(i8(176)), pair_words(i8(16)), z(11)], [11, 4, 11, 1100, 12], 11
    x = ref.words(rng.integers(-3000, 3000, 16))
    yield "mm_res", [byte_words(i8(256)), pair_words(i8(32)), x, ref.words(rng.integers(0, 2**24, 1))], [8, 8, 8, 1200, 14, 8], 8
    yield "embed", [byte_words(i8(16)), byte_words(i8(16)), z(16), z(1)], [16, 1300, 900, 9], 16
    xr = rng.integers(-2000, 2000, 16)
    th = int(rng.choice([4, 64, 1024]))
    yield "rmsnorm", [ref.words(xr), ref.words([int((xr * xr).sum())]),
                      ref.words(rng.integers(-1000, 1000, 16) // th), z(8)], [th, 19], 4
    sc = ref.words(rng.integers(-2000, 2000, 2 * T))
    yield "softmax", [sc, z(2 * T)], [2 * T, T], 2 * T
    e = ref.softmax(sc, z(2 * T), 2 * T, T)[1]
    yield "av", [e, byte_words(i8(16 * T)), z(8)], [16, T], 16
    yield "silu", [ref.words(rng.integers(-4000, 4000, 32)), z(16)], [32, 1500, 14], 32
    yield "sample", [ref.words(rng.integers(-3000, 3000, 27)), ref.words(rng.integers(-500, 2000, 27)), z(1)], [27], 4


def main(trials=5):
    L, sim = Launcher(), Sim()
    rng = np.random.default_rng(0)
    seen = {}
    for t in range(trials):
        for name, bufs, sc, th in cases(rng):
            want = ref.KERNELS[name](*[list(b) for b in bufs], *sc)
            ln = L.prepare(name, bufs, sc, th)
            got = sim.run(L, ln)
            assert got.bufs == [ref.words(b) for b in want], f"{name} {sc}: differs\n{got.bufs}\n{want}"
            seen.setdefault(name, []).append((len(ln.lw.blocks), got.records, got.host, got.gen))
    for name, runs in seen.items():
        b, rec, host, gen = runs[-1]
        print(f"{name:9s} ok x{len(runs):2d}  {b:2d} blocks, last launch {rec:4d} records, "
              f"{host:4d} host + {gen:5d} generated tokens")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
