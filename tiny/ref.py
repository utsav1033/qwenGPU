"""The numpy integer reference: every kernel of tiny/tiny.cu, bit for bit, vectorized over threads.

Each function takes the kernel's buffers (lists of uint32 words, in parameter order) and its int
scalars, and returns every buffer after the launch, as the GPU backends do. Unsigned C
arithmetic is uint64 masked to 32 bits; signed is int64 wrapped to int32; C's >> on int is
numpy's (arithmetic) >>.
"""

import numpy as np

M32 = 0xFFFFFFFF


def u(x):
    return np.asarray(x, dtype=np.uint64) & np.uint64(M32)


def s(x):
    x = np.asarray(x, dtype=np.int64)
    return ((x + (1 << 31)) & M32) - (1 << 31)


def us(x):  # uint32 bits as int32
    return s(np.asarray(x, dtype=np.uint64).astype(np.int64))


def umulhi(a, b):
    return (u(a) * u(b)) >> np.uint64(32)


def words(x):
    return [int(v) & M32 for v in np.asarray(x).ravel()]


def clamp8(y):
    return np.clip(y, -127, 127)


def rq(acc, M, S):
    return s(s(acc * M) + (1 << (S - 1))) >> S


def put_pair(out, i, y):
    out = u(out)
    mul = np.where(i & 2, 1, 65536)
    np.add.at(out, 2 * (i >> 2) + (i & 1), u(s(y * mul)))
    return words(out)


def put_byte(out, i, y):
    out = u(out)
    np.add.at(out, i >> 2, u((y + 128) << (8 * (i & 3))))
    return words(out)


def top(w, i):  # byte (i & 3) of word w, as a signed value (byte - 128)
    return ((u(w) >> u(8 * (i & 3))) & np.uint64(255)).astype(np.int64) - 128


def recip(D):
    D = u(D)
    d2 = u(D + D)
    y = u(np.uint64(1515870810) - umulhi(D, 2021161081))
    for _ in range(3):
        y2 = u(y + y)
        t = umulhi(d2, y2)
        y = umulhi(y2, u(np.uint64(1 << 32) - u(t + t)))  # 2 (2^31 - t), mod 2^32
    return y


def expq(z):
    q = u(np.uint64(1 << 30) - u(u(z) * np.uint64(4096)))
    for _ in range(10):
        q2 = u(q + q)
        q = umulhi(q2, q2)
    return q


def dot4(w, ex, ox):
    w = u(w)
    ew = u((w & np.uint64(0x00FF00FF)) - np.uint64(0x00800080))
    ow = u(((w >> np.uint64(8)) & np.uint64(0x00FF00FF)) - np.uint64(0x00800080))
    l1 = u(ew * u(ex) + np.uint64(0x80008000))
    l2 = u(ow * u(ox) + np.uint64(0x80008000))
    return us((l1 >> np.uint64(16)) + (l2 >> np.uint64(16)))


def mm_acc(W, x, n, kw, g):
    W, x = u(W), u(x)
    i = np.arange(n)
    acc = np.zeros(n, dtype=np.int64)
    for k in range(kw):
        acc = s(acc + dot4(W[i * kw + k], x[(i // g) * 2 * kw + 2 * k],
                           x[(i // g) * 2 * kw + 2 * k + 1]))
    return s(acc - (kw << 16)), i


def mm_pair(W, x, out, n, kw, g, M, S):
    acc, i = mm_acc(W, x, n, kw, g)
    return [W, x, put_pair(out, i, clamp8(rq(acc, M, S)))]


def mm_byte(W, x, out, n, kw, g, M, S):
    acc, i = mm_acc(W, x, n, kw, g)
    return [W, x, put_byte(out, i, clamp8(rq(acc, M, S)))]


def mm_raw(W, x, out, n, kw, g, M, S):
    acc, i = mm_acc(W, x, n, kw, g)
    out = list(out)
    for j, v in zip(i, rq(acc, M, S)):
        out[j] = int(v) & M32
    return [W, x, out]


def mm_res(W, x, out, ss, n, kw, g, M, S, r0):
    acc, i = mm_acc(W, x, n, kw, g)
    o = us(out)
    o[r0 + i] = s(o[r0 + i] + rq(acc, M, S))
    return [W, x, words(o), words([int(u(us(ss)[0] + np.sum(s(o[r0 + i] * o[r0 + i]))))])]


def embed(te, pe, out, ss, n, Mt, Mp, S):
    i = np.arange(n)
    a, b = top(u(te)[i >> 2], i), top(u(pe)[i >> 2], i)
    v = s(s(a * Mt + b * Mp) + (1 << (S - 1))) >> S
    out = list(out)
    for j, val in zip(i, v):
        out[j] = int(val) & M32
    return [te, pe, out, words([int(u(us(ss)[0] + np.sum(s(v * v))))])]


def rmsnorm(x, ss, G, out, th, S):
    q = int(u(ss)[0])
    r = 0
    bit = 1 << 14
    while bit:
        c = r | bit
        if ((c * c & M32) * q) >> 32 < th:
            r = c
        bit >>= 1
    i = np.arange(16)
    y = clamp8(rq(s(us(x)[i] * r), us(G)[i], S))
    return [x, ss, G, put_pair(out, i, y)]


def softmax(sc, e, n, T):
    i = np.arange(n)
    v = us(sc)
    base = np.where(i >= T, T, 0)
    m = np.array([v[b:b + T].max() for b in base])
    d = np.minimum(s(m - v[i]), 1 << 14)
    e = list(e)
    for j, val in zip(i, expq(d) >> np.uint64(15)):
        e[j] = int(val)
    return [sc, e]


def av(e, V, out, n, T):
    c = np.arange(n)
    ev, V = us(e), u(V)
    acc = np.zeros(n, dtype=np.int64)
    tot = np.zeros(n, dtype=np.int64)
    for j in range(T):
        ej = ev[(c >> 3) * T + j]
        acc = s(acc + s(ej * top(V[4 * j + (c >> 2)], c)))
        tot = s(tot + ej)
    su = u(tot)
    f = (16 - 8 * (su >= 1 << 16) - 4 * (su >= 1 << 17) - 2 * (su >= 1 << 18)
         - (su >= 1 << 19)).astype(np.uint64)
    r = recip(u(su * u(f << np.uint64(11))))
    a = u(np.abs(acc))
    q = (u(umulhi(a, r) * f + np.uint64(1 << 16)) >> np.uint64(17)).astype(np.int64)
    return [e, V, put_pair(out, c, clamp8(np.where(acc < 0, -q, q)))]


def silu(h, out, n, M, S):
    i = np.arange(n)
    v = us(h)[:n]
    z = np.minimum(u(np.abs(v)), np.uint64(1 << 14))
    q = np.minimum(expq(z), np.uint64((1 << 30) - 1))
    r = recip(np.uint64(1 << 30) + q)
    sg = np.where(v < 0, np.uint64(1 << 30) - r, r)
    mag = us(umulhi(u(z + z), u(sg + sg)))
    return [h, put_pair(out, i, clamp8(rq(np.where(v < 0, -mag, mag), M, S)))]


def sample(lg, g, out, n):
    v = s(us(lg)[:n] + us(g)[:n])
    out = list(out)
    out[0] = int(np.argmax(v))  # the first maximum, as the loop's strict >
    return [lg, g, out]


KERNELS = {f.__name__: f for f in (mm_pair, mm_byte, mm_raw, mm_res, embed, rmsnorm,
                                   softmax, av, silu, sample)}


class Ref:
    """The numpy reference as a backend: the same launches, no GPU."""
    name = "ref"

    def launch(self, name, bufs, scalars, threads):
        return KERNELS[name](*[list(b) for b in bufs], *scalars)
