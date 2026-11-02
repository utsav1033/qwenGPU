#include "../../kernels/qcuda.h"
// qgpu driver repro: the select's predicate is recomputed after `max` overwrote its operand
extern "C" __global__ void argmax(const int* a, int* out, int n) {
  int best = a[0], idx = 0;
#pragma unroll 1
  for (int k = 1; k < n; k++) {
    int v = a[k];
    bool b = v > best;
    best = b ? v : best;
    idx = b ? k : idx;
  }
  if (threadIdx.x == 0) out[0] = idx;
}
