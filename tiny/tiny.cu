#include "../kernels/qcuda.h"
// The tiny char transformer's kernels, in qgpu's CUDA subset: 32-bit integers only, shifts by
// launch constants, division only by a parameter, at most 126 words of memory per launch.
// Every scalar parameter is known at launch (the driver specializes the code on it).
//
// Formats (tiny/ref.py is the bit-exact numpy reference of every kernel here):
// - byte: 4 int8 per word, byte b = v_b + 128 (weights, the K and V caches).
// - pair: an activation vector read by a matmul, 2 words per 4 values:
//   ex = x2 + x0 * 2^16, ox = x3 + x1 * 2^16 (integer sums, the values in [-127, 127]).
// - Q8: a real value times 256 (scores, logits, MLP pre-activations).
// - Q30 / Q15: fixed-point fractions (exp, reciprocals).

typedef unsigned U;

// A constant clang cannot see through: a select between powers of two must stay a select
// (clang would make it a shift by a per-thread amount or a lookup table, which qgpu lacks).
static __device__ __forceinline__ U opaque(U v) {
  U r;
  asm("mov.b32 %0, %1;" : "=r"(r) : "r"(v));
  return r;
}

// x >> k for an index (clang's i & ~3 byte offsets are not addresses qgpu's driver reads)
static __device__ __forceinline__ int shr_idx(U x, int k) { return (int)__umulhi(x, 1u << (32 - k)); }

// a0 x0 + a1 x1 + a2 x2 + a3 x3 + 2^16, w in byte format, (ex, ox) in pair format. One
// multiply puts a0 x0 + a2 x2 in bits 16..31: (a0 + a2 2^16)(x2 + x0 2^16) = a0 x2 +
// (a0 x0 + a2 x2) 2^16 + a2 x0 2^32, exact mod 2^32 because |a x| <= 127^2 < 2^15.
static __device__ __forceinline__ int dot4(U w, int ex, int ox) {
  U ew = (w & 0x00FF00FFu) - 0x00800080u;
  U ow = ((w >> 8) & 0x00FF00FFu) - 0x00800080u;
  U l1 = ew * (U)ex + 0x80008000u;
  U l2 = ow * (U)ox + 0x80008000u;
  return (int)((l1 >> 16) + (l2 >> 16));
}

static __device__ __forceinline__ int clamp8(int y) { return y > 127 ? 127 : (y < -127 ? -127 : y); }

// round(acc * M / 2^S), S >= 1
static __device__ __forceinline__ int rq(int acc, int M, int S) {
  return (acc * M + (1 << (S - 1))) >> S;
}

// value i of a vector into pair format (out zeroed by the host)
static __device__ __forceinline__ void put_pair(int* out, U i, int y) {
  int m = 65536 - (int)(opaque(65535) * ((i >> 1) & 1));
  atomicAdd(&out[2 * shr_idx(i, 2) + (int)(i & 1)], y * m);
}

// the multiplier that moves byte (i & 3) of a word to the top byte
static __device__ __forceinline__ U top_mul(U i) {
  return (256 - opaque(255) * (i & 1)) * (65536 - opaque(65535) * ((i >> 1) & 1));
}

// value i of a vector into byte format (out zeroed by the host)
static __device__ __forceinline__ void put_byte(int* out, U i, int y) {
  U m = (1 + opaque(255) * (i & 1)) * (1 + opaque(65535) * ((i >> 1) & 1));
  atomicAdd(&out[shr_idx(i, 2)], (int)((U)(y + 128) * m));
}

// ~2^60 / D for D in [2^30, 2^31): Newton's method from 24/17 - 8/17 d (d = D / 2^30 in
// [1, 2), the error at most 1/17), three steps, in Q30: y = y (2 - d y), with 2 (2 - d y) =
// -2 mulhi(2 D, 2 y) mod 2^32. The step is inline PTX with the loop counter first, so a step
// packs into two blocks with the loop's setp in the second.
static __device__ __forceinline__ U recip(U D) {
  U d2 = D + D;
  U y = 1515870810u - __umulhi(D, 2021161081u);
  int k = 3;
  do {
    U y2, t, t2, e2;
    asm volatile(
        "add.s32 %0, %0, -1;\n\t"
        "add.s32 %1, %5, %5;\n\t"
        "mul.hi.u32 %2, %6, %1;\n\t"
        "add.s32 %3, %2, %2;\n\t"
        "neg.s32 %4, %3;\n\t"
        "mul.hi.u32 %5, %1, %4;"
        : "+r"(k), "=r"(y2), "=r"(t), "=r"(t2), "=r"(e2), "+r"(y)
        : "r"(d2));
  } while (k);
  return y;
}

// exp(-z / 256) in Q30 for z in [0, 2^14]: (1 - z / 2^18)^1024, ten squarings, one block each
static __device__ __forceinline__ U expq(U z) {
  U q = (1u << 30) - z * 4096u;
  int k = 10;
  do {
    U q2;
    asm volatile(
        "add.s32 %0, %0, -1;\n\t"
        "add.s32 %1, %2, %2;\n\t"
        "mul.hi.u32 %2, %1, %1;"
        : "+r"(k), "=r"(q2), "+r"(q));
  } while (k);
  return q;
}

// acc + dot4(*wp, xp[0], xp[1]), hand-scheduled for qgpu's 4-stage blocks (one multiply, in
// stage 2, per block; and only in stage 1). The driver packs instructions in program order, so
// this order fills five blocks: ld | and | mul.hi | sub, ld | and | mul | add,
// ld | sub | mul | add, - | - | mul.hi | add, - | - | mul.hi | add (clang's order took ten).
static __device__ __forceinline__ int dot4_step(const U* wp, const int* xp, int acc) {
  U w, e, t, ew, ex, o, m1, l1, ox, ow, m2, l2, h1, h2;
  int a1, a2;
  asm volatile(
      "ld.global.b32 %0, [%16];\n\t"
      "and.b32 %1, %0, 16711935;\n\t"      // e = w & 0x00FF00FF
      "mul.hi.u32 %2, %0, 16777216;\n\t"   // t = w >> 8
      "sub.s32 %3, %1, 8388736;\n\t"       // ew = e - 0x00800080
      "ld.global.b32 %4, [%17];\n\t"
      "and.b32 %5, %2, 16711935;\n\t"      // o = t & 0x00FF00FF
      "mul.lo.s32 %6, %3, %4;\n\t"
      "add.s32 %7, %6, -2147450880;\n\t"   // l1 = ew * ex + 0x80008000
      "ld.global.b32 %8, [%17+4];\n\t"
      "sub.s32 %9, %5, 8388736;\n\t"       // ow
      "mul.lo.s32 %10, %9, %8;\n\t"
      "add.s32 %11, %10, -2147450880;\n\t" // l2
      "mul.hi.u32 %12, %7, 65536;\n\t"     // l1 >> 16
      "add.s32 %14, %12, %18;\n\t"
      "mul.hi.u32 %13, %11, 65536;\n\t"    // l2 >> 16
      "add.s32 %15, %13, %14;"
      : "=r"(w), "=r"(e), "=r"(t), "=r"(ew), "=r"(ex), "=r"(o), "=r"(m1), "=r"(l1), "=r"(ox),
        "=r"(ow), "=r"(m2), "=r"(l2), "=r"(h1), "=r"(h2), "=r"(a1), "=r"(a2)
      : "l"((const __attribute__((address_space(1))) U*)wp),
        "l"((const __attribute__((address_space(1))) int*)xp), "r"(acc));
  return a2;
}

// row i of a matmul tile: W (byte format, kw words a row) times x (pair format, 2 kw words;
// rows i of group i / g read the group's own x)
#define MM_ACC                                                                   \
  const U* wr = W + i * kw;                                                      \
  const int* xr = x + (i / g) * 2 * kw;                                          \
  int acc = 0;                                                                   \
  _Pragma("unroll 1") for (int k = 0; k < kw; k++, wr++, xr += 2)               \
    acc = dot4_step(wr, xr, acc);                                                \
  acc -= kw << 16;

// int8 out, pair format (the next matmul's input)
extern "C" __global__ void mm_pair(const U* W, const int* x, int* out, int n, int kw, int g,
                                   int M, int S) {
  U i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (U)n) {
    MM_ACC
    put_pair(out, i, clamp8(rq(acc, M, S)));
  }
}

// int8 out, byte format (a K or V cache row)
extern "C" __global__ void mm_byte(const U* W, const int* x, int* out, int n, int kw, int g,
                                   int M, int S) {
  U i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (U)n) {
    MM_ACC
    put_byte(out, i, clamp8(rq(acc, M, S)));
  }
}

// int32 out (Q8 scores, logits, pre-activations)
extern "C" __global__ void mm_raw(const U* W, const int* x, int* out, int n, int kw, int g,
                                  int M, int S) {
  U i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (U)n) {
    MM_ACC
    out[i] = rq(acc, M, S);
  }
}

// the residual stream += W x, rows r0 .. r0 + n - 1; ss += the new values squared (for the
// next RMSNorm)
extern "C" __global__ void mm_res(const U* W, const int* x, int* out, int* ss, int n, int kw,
                                  int g, int M, int S, int r0) {
  U i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (U)n) {
    MM_ACC
    int v = out[r0 + i] + rq(acc, M, S);
    out[r0 + i] = v;
    atomicAdd(&ss[0], v * v);
  }
}

// x = token row + position row (byte format), into the residual stream's scale; ss = sum x^2
extern "C" __global__ void embed(const U* te, const U* pe, int* out, int* ss, int n, int Mt,
                                 int Mp, int S) {
  U i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (U)n) {
    U m = top_mul(i);
    int a = (int)((te[shr_idx(i, 2)] * m) >> 24) - 128;
    int b = (int)((pe[shr_idx(i, 2)] * m) >> 24) - 128;
    int v = (a * Mt + b * Mp + (1 << (S - 1))) >> S;
    out[i] = v;
    atomicAdd(&ss[0], v * v);
  }
}

// RMSNorm of 16 values, one warp: ss = sum x^2 (from the kernel that wrote x), r = the largest
// r < 2^15 with r^2 ss < th 2^32 (r ~ sqrt(th) 2^14 / rms(x)), then y = x r G / 2^S (G: the
// gain over the output scale), pair format. Lane l does values l, l + 4, l + 8, l + 12.
extern "C" __global__ void rmsnorm(const int* x, const int* ss, const int* G, int* out, int th,
                                   int S) {
  U l = threadIdx.x;
  U q = ss[0], r = 0;
#pragma unroll 1
  for (U bit = 1u << 14; bit; bit >>= 1) {
    U c = r | bit;
    r = __umulhi(c * c, q) < (U)th ? c : r;
  }
  int m = 65536 - (int)(opaque(65535) * ((l >> 1) & 1));  // put_pair's, for i = 4 k + l
#pragma unroll
  for (int k = 0; k < 4; k++) {
    int i = 4 * k + (int)l;
    atomicAdd(&out[2 * k + (int)(l & 1)], clamp8(rq(x[i] * (int)r, G[i], S)) * m);
  }
}

// attention weights: e = exp(s - max of its head's row) in Q15, two rows of T scores (Q8)
extern "C" __global__ void softmax(const int* s, int* e, int n, int T) {
  U i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (U)n) {
    int base = i >= (U)T ? T : 0;  // two heads
    int m = s[base];
#pragma unroll 1
    for (int k = 1; k < T; k++) {
      int v = s[base + k];
      m = v > m ? v : m;
    }
    int d = m - s[i];
    d = d < (1 << 14) ? d : (1 << 14);
    e[i] = expq((U)d) >> 15;
  }
}

// attention output: channel c = sum_j e_j v_j[c] / sum_j e_j, V in byte format (T rows of 4
// words), e from softmax (rows of T per head, 8 channels a head), pair format out
extern "C" __global__ void av(const int* e, const U* V, int* out, int n, int T) {
  U c = blockIdx.x * blockDim.x + threadIdx.x;  // n = 16 threads, no guard
  {
    const int* eh = e + shr_idx(c, 3) * T;
    U hi = 65536 - opaque(65535) * ((c >> 1) & 1);  // also put_pair's multiplier
    U m = (256 - opaque(255) * (c & 1)) * hi;  // top_mul(c)
    int w = shr_idx(c, 2);
    int acc = 0, sum = 0;
#pragma unroll 1
    for (int j = 0; j < T; j++) {
      int ej = eh[j];
      acc += ej * ((int)((V[4 * j + w] * m) >> 24) - 128);
      sum += ej;
    }
    // 1 / sum: sum = 2^k * [2^15, 2^16), D = sum * 2^(15 - k) in [2^30, 2^31)
    U su = sum;
    U f = 16 - (su >= (1u << 16) ? 8 : 0) - (su >= (1u << 17) ? 4 : 0)
          - (su >= (1u << 18) ? 2 : 0) - (su >= (1u << 19) ? 1 : 0);  // 2^(4 - k)
    U r = recip(su * (f << 11));
    U neg = opaque(0) - acc;
    U a = acc < 0 ? neg : (U)acc;
    U q = (__umulhi(a, r) * f + (1u << 16)) >> 17;  // |acc| / sum
    atomicAdd(&out[2 * shr_idx(c, 2) + (int)(c & 1)], clamp8(acc < 0 ? -(int)q : (int)q) * (int)hi);
  }
}

// SiLU of Q8 values: v * sigmoid(v), sigmoid(|v|) = 1 / (1 + exp(-|v|)); int8 out in pair
// format, round(silu * M / 2^S)
extern "C" __global__ void silu(const int* h, int* out, int n, int M, int S) {
  U i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (U)n) {
    int v = h[i];
    U neg = opaque(0) - v;
    U z = v < 0 ? neg : (U)v;
    z = z < (1u << 14) ? z : (1u << 14);
    U q = expq(z);
    q = q < (1u << 30) - 1 ? q : (1u << 30) - 1;
    U r = recip((1u << 30) + q);
    U sg = v < 0 ? (1u << 30) - r : r;
    int mag = (int)__umulhi(z + z, sg + sg);
    put_pair(out, i, clamp8(rq(v < 0 ? -mag : mag, M, S)));
  }
}

// next token: argmax of logits + Gumbel noise (both Q8), the first index on ties (one warp).
// The best value is reloaded rather than kept: with best = max(v, best) next to the select,
// qgpu's driver recomputes the select's setp after the max has overwritten best.
extern "C" __global__ void sample(const int* l, const int* g, int* out, int n) {
  int i = threadIdx.x;
  int idx = 0;
#pragma unroll 1
  for (int k = 1; k < n; k++) idx = l[k] + g[k] > l[idx] + g[idx] ? k : idx;
  if (i == 0) out[0] = idx;
}
