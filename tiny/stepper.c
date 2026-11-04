// The gpu-only checkpoint's 45 Gated DeltaNet layers, one token at a time, on a CPU core.
//
// The same computation as transformers' Qwen3.5 decode step (Qwen3_5DecoderLayer with
// torch_recurrent_gated_delta_rule and causal_conv1d_update, in bf16 with fp32 inside), with the
// weights that are zero left out: a product walks only the inputs that are nonzero and, for each,
// only the outputs its nonzero weights reach (tiny/sparse_pack.py). A zero weight or a zero input
// adds 0 to a sum, so skipping it changes nothing. Values are rounded to bf16 where torch's
// tensors are bf16. DeltaNet heads whose key is zero are not updated (the update adds k * delta,
// which is 0), and those whose query is zero output 0.
//
// Built by tiny/stepper.py with the Mac's clang; driven through ctypes.

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define D 3200     // hidden
#define F 6144     // MLP units, and qkv channels
#define H 16       // DeltaNet heads (key = value heads)
#define K 128      // head dim
#define QK (H * K) // 2048
#define CONV 4
#define NT 4       // threads for the lm_head: each takes a quarter of the vocabulary

static inline float bf(float x) {  // round to bf16, nearest even (torch's .to(bfloat16))
    uint32_t u;
    memcpy(&u, &x, 4);
    if ((u & 0x7fffffff) > 0x7f800000) return x;  // NaN
    u += 0x7fff + ((u >> 16) & 1);
    u &= 0xffff0000u;
    memcpy(&x, &u, 4);
    return x;
}
static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

typedef struct {  // a weight matrix by input: for input j, outputs rows[colptr[j]..colptr[j+1])
    int nin, nout;
    const int32_t *colptr, *rows;
    const float *vals;
    int nlive;
    int32_t *live;  // the outputs any weight reaches, ascending
} Mat;

enum { QKV, Z, B, A, O, GATE, UP, DOWN, NMAT };

typedef struct {
    Mat m[NMAT];
    const float *conv;  // (F, CONV)
    const float *A_log, *dt_bias, *gnorm, *in_norm, *post_norm;
    float *S;            // (H, K, K) recurrent state, fp32: S[h][key][value]
    float *cs;           // (F, CONV - 1) conv state, bf16 values
    // this layer's activations: entries outside the live outputs are never written, so stay 0
    float *qkv, *z, *bb, *aa, *cv, *core, *y, *o, *g, *u, *mm, *dn;
    char act[H];  // heads whose output was nonzero last token
} Layer;

typedef struct {
    int nl, vocab;
    Layer *L;
    const float *norm;
    Mat emb, head;  // by token (rows: hidden dims), by hidden dim (rows: tokens)
    float *acc, *logits;
    // the lm_head in NT vocabulary ranges: split[j * (NT + 1) + t] is where range t starts in
    // column j (rows ascend), so each range's logits are summed in the same order as one pass
    int32_t *split;
    float x[D];
    int32_t xi[D];
    int nx;
    _Atomic int gen, done;
    float bv[NT];
    int best[NT];
    pthread_t th[NT];
} Model;

static void mat_init(Mat *m, int nin, int nout, const int32_t *cp, const int32_t *rows,
                     const float *vals) {
    m->nin = nin, m->nout = nout, m->colptr = cp, m->rows = rows, m->vals = vals;
    char *seen = calloc(nout, 1);
    for (int i = 0; i < cp[nin]; i++) seen[rows[i]] = 1;
    m->nlive = 0;
    for (int r = 0; r < nout; r++) m->nlive += seen[r];
    m->live = malloc(sizeof(int32_t) * (m->nlive + 1));
    for (int r = 0, n = 0; r < nout; r++)
        if (seen[r]) m->live[n++] = r;
    free(seen);
}

// y = bf16(W x) on W's live outputs; x has nx nonzero entries at xi. y must be zero elsewhere
// (it is: the caller's buffers are cleared on exactly the live outputs they last had).
static void matvec(Model *M, const Mat *m, const float *x, const int32_t *xi, int nx, float *y) {
    float *acc = M->acc;
    for (int r = 0; r < m->nlive; r++) acc[m->live[r]] = 0.0f;
    for (int t = 0; t < nx; t++) {
        int j = xi[t];
        float xv = x[j];
        for (int p = m->colptr[j]; p < m->colptr[j + 1]; p++) acc[m->rows[p]] += m->vals[p] * xv;
    }
    for (int r = 0; r < m->nlive; r++) y[m->live[r]] = bf(acc[m->live[r]]);
}

// The residual's nonzero dims, as a bitmap: the norms walk only those, in ascending order, so
// their sums are the full sums (the rest add +0).
#define NW ((D + 63) / 64)
static uint64_t hz[NW];
static inline void hz_set(int i, float v) {
    if (v != 0.0f) hz[i >> 6] |= 1ull << (i & 63); else hz[i >> 6] &= ~(1ull << (i & 63));
}

// Qwen3_5RMSNorm: fp32 mean of squares, x * rsqrt(var + eps) * (1 + w), to bf16. Writes x on
// the residual's nonzero dims and lists x's nonzero entries in xi; returns how many.
static int rmsnorm(const float *h, const float *w, float *x, int32_t *xi) {
    int32_t hi[D];
    int n = 0;
    for (int k = 0; k < NW; k++)
        for (uint64_t b = hz[k]; b; b &= b - 1) hi[n++] = k * 64 + __builtin_ctzll(b);
    float ss = 0.0f;
    for (int t = 0; t < n; t++) ss += h[hi[t]] * h[hi[t]];
    float r = 1.0f / sqrtf(ss / D + 1e-6f);
    int nx = 0;
    for (int t = 0; t < n; t++) {
        int i = hi[t];
        x[i] = bf((h[i] * r) * (1.0f + w[i]));
        if (x[i] != 0.0f) xi[nx++] = i;
    }
    return nx;
}

Model *st_new(int nl, int vocab) {
    Model *M = calloc(1, sizeof(Model));
    M->nl = nl, M->vocab = vocab;
    M->L = calloc(nl, sizeof(Layer));
    M->acc = calloc(vocab > F ? vocab : F, sizeof(float));
    M->logits = calloc(vocab, sizeof(float));
    return M;
}

void st_mat(Model *M, int layer, int which, int nin, int nout, const int32_t *cp,
            const int32_t *rows, const float *vals) {
    Mat *m = layer < 0 ? (which ? &M->head : &M->emb) : &M->L[layer].m[which];
    mat_init(m, nin, nout, cp, rows, vals);
}

void st_layer(Model *M, int layer, const float *conv, const float *A_log, const float *dt_bias,
              const float *gnorm, const float *in_norm, const float *post_norm) {
    Layer *L = &M->L[layer];
    L->conv = conv, L->A_log = A_log, L->dt_bias = dt_bias, L->gnorm = gnorm;
    L->in_norm = in_norm, L->post_norm = post_norm;
    L->S = calloc(H * K * K, sizeof(float));
    L->cs = calloc(F * (CONV - 1), sizeof(float));
    float *b = calloc(5 * F + 3 * QK + 2 * D + 2 * H, sizeof(float));
    L->qkv = b, b += F, L->z = b, b += QK, L->bb = b, b += H, L->aa = b, b += H;
    L->cv = b, b += F, L->core = b, b += QK, L->y = b, b += QK, L->o = b, b += D;
    L->g = b, b += F, L->u = b, b += F, L->mm = b, b += F, L->dn = b;
}

// range t of the lm_head: logits of tokens [t * vocab / NT, (t + 1) * vocab / NT), argmax of
// their bf16 values (the first, on ties)
static void head_range(Model *M, int t) {
    int v0 = (int)((long)t * M->vocab / NT), v1 = (int)((long)(t + 1) * M->vocab / NT);
    float *lg = M->logits;
    const Mat *hm = &M->head;
    memset(lg + v0, 0, sizeof(float) * (v1 - v0));
    for (int i = 0; i < M->nx; i++) {
        int j = M->xi[i];
        float xv = M->x[j];
        const int32_t *sp = M->split + j * (NT + 1);
        for (int p = sp[t]; p < sp[t + 1]; p++) lg[hm->rows[p]] += hm->vals[p] * xv;
    }
    int best = v0;
    float bv = bf(lg[v0]);
    for (int v = v0 + 1; v < v1; v++) {
        float b = bf(lg[v]);
        if (b > bv) bv = b, best = v;
    }
    M->bv[t] = bv, M->best[t] = best;
}

typedef struct { Model *M; int t; } Arg;

static void *worker(void *p) {  // spins between tokens: a wake-up would cost more than the work
    Arg a = *(Arg *)p;
    free(p);
    Model *M = a.M;
    int seen = 0;
    for (;;) {
        int g;
        while ((g = atomic_load_explicit(&M->gen, memory_order_acquire)) == seen)
            __asm__ volatile("yield");
        if (g < 0) return NULL;
        seen = g;
        head_range(M, a.t);
        atomic_fetch_add_explicit(&M->done, 1, memory_order_release);
    }
}

void st_norm(Model *M, const float *norm) {
    M->norm = norm;
    const Mat *hm = &M->head;
    M->split = malloc(sizeof(int32_t) * D * (NT + 1));
    for (int j = 0; j < D; j++)
        for (int t = 0; t <= NT; t++) {
            int v = (int)((long)t * M->vocab / NT), lo = hm->colptr[j], hi = hm->colptr[j + 1];
            while (lo < hi) {  // the first entry of column j at token v or later
                int mid = (lo + hi) / 2;
                if (hm->rows[mid] < v) lo = mid + 1; else hi = mid;
            }
            M->split[j * (NT + 1) + t] = lo;
        }
    for (int t = 1; t < NT; t++) {
        Arg *a = malloc(sizeof(Arg));
        a->M = M, a->t = t;
        pthread_create(&M->th[t], NULL, worker, a);
    }
}

void st_reset(Model *M) {  // a fresh context: every state zero
    for (int l = 0; l < M->nl; l++) {
        memset(M->L[l].S, 0, sizeof(float) * H * K * K);
        memset(M->L[l].cs, 0, sizeof(float) * F * (CONV - 1));
    }
}

float *st_state(Model *M, int layer) { return M->L[layer].S; }

// One token through every layer; returns the argmax of the bf16 logits (the first, on ties).
int st_step(Model *M, int tok, int head) {
    static float h[D], x[D];
    static int32_t xi[F];
    memset(h, 0, sizeof h), memset(hz, 0, sizeof hz);
    const Mat *e = &M->emb;
    for (int p = e->colptr[tok]; p < e->colptr[tok + 1]; p++)
        h[e->rows[p]] = e->vals[p], hz_set(e->rows[p], e->vals[p]);

    for (int l = 0; l < M->nl; l++) {
        Layer *L = &M->L[l];
        float *qkv = L->qkv, *z = L->z, *bb = L->bb, *aa = L->aa, *cv = L->cv, *core = L->core;
        float *y = L->y, *o = L->o, *g = L->g, *u = L->u, *mm = L->mm, *dn = L->dn;
        // --- token mixer: Gated DeltaNet
        int nx = rmsnorm(h, L->in_norm, x, xi);
        matvec(M, &L->m[QKV], x, xi, nx, qkv);
        matvec(M, &L->m[Z], x, xi, nx, z);
        matvec(M, &L->m[B], x, xi, nx, bb);
        matvec(M, &L->m[A], x, xi, nx, aa);
        // causal_conv1d_update: the last 3 inputs and this one, in bf16; silu, in bf16
        for (int r = 0; r < L->m[QKV].nlive; r++) {
            int c = L->m[QKV].live[r];
            float *s = L->cs + c * (CONV - 1);
            const float *w = L->conv + c * CONV;
            float acc = s[0] * w[0] + s[1] * w[1] + s[2] * w[2] + qkv[c] * w[3];
            cv[c] = bf(silu(bf(acc)));
            s[0] = s[1], s[1] = s[2], s[2] = qkv[c];
        }
        // the delta rule, per head, fp32; then Qwen3_5RMSNormGated on the heads that output
        int32_t yi[QK];
        int ny = 0;
        for (int hd = 0; hd < H; hd++) {
            float *q = cv + hd * K, *k = cv + QK + hd * K, *v = cv + 2 * QK + hd * K;
            float beta = bf(sigm(bb[hd]));
            float gg = -expf(L->A_log[hd]) * softplus(aa[hd] + L->dt_bias[hd]);
            float decay = expf(gg);
            float *S = L->S + hd * K * K;
            if (decay != 1.0f)
                for (int i = 0; i < K * K; i++) S[i] *= decay;
            int32_t ki[K], qi[K];
            int nk = 0, nq = 0;
            float kss = 0.0f, qss = 0.0f;
            for (int i = 0; i < K; i++) {
                kss += k[i] * k[i], qss += q[i] * q[i];
                if (k[i] != 0.0f) ki[nk++] = i;
                if (q[i] != 0.0f) qi[nq++] = i;
            }
            if (nk) {
                float kn = 1.0f / sqrtf(kss + 1e-6f), kf[K], kv[K];
                for (int t = 0; t < nk; t++) kf[ki[t]] = k[ki[t]] * kn;
                for (int c = 0; c < K; c++) kv[c] = 0.0f;
                for (int t = 0; t < nk; t++) {
                    const float *row = S + ki[t] * K;
                    float kk = kf[ki[t]];
                    for (int c = 0; c < K; c++) kv[c] += row[c] * kk;
                }
                float delta[K];
                for (int c = 0; c < K; c++) delta[c] = (v[c] - kv[c]) * beta;
                for (int t = 0; t < nk; t++) {
                    float *row = S + ki[t] * K;
                    float kk = kf[ki[t]];
                    for (int c = 0; c < K; c++) row[c] = row[c] + kk * delta[c];
                }
            }
            float *c = core + hd * K, *yy = y + hd * K;
            if (!nq) {  // a zero query reads 0, and the gated norm of 0 is 0
                if (L->act[hd]) memset(yy, 0, sizeof(float) * K), L->act[hd] = 0;
                continue;
            }
            L->act[hd] = 1;
            float qn = 1.0f / sqrtf(qss + 1e-6f), out[K];
            for (int i = 0; i < K; i++) out[i] = 0.0f;
            for (int t = 0; t < nq; t++) {
                const float *row = S + qi[t] * K;
                float qq = (q[qi[t]] * qn) / 11.313708498984761f;
                for (int i = 0; i < K; i++) out[i] += row[i] * qq;
            }
            float ss = 0.0f;
            for (int i = 0; i < K; i++) c[i] = bf(out[i]), ss += c[i] * c[i];
            float r = 1.0f / sqrtf(ss / K + 1e-6f), *zz = z + hd * K;
            for (int i = 0; i < K; i++) {
                float t = bf(c[i] * r);
                t = bf(L->gnorm[i] * t);
                yy[i] = t == 0.0f ? 0.0f : bf(t * silu(zz[i]));
                if (yy[i] != 0.0f) yi[ny++] = hd * K + i;
            }
        }
        matvec(M, &L->m[O], y, yi, ny, o);
        for (int r = 0; r < L->m[O].nlive; r++) {
            int i = L->m[O].live[r];
            h[i] = bf(h[i] + o[i]), hz_set(i, h[i]);
        }
        // --- MLP
        nx = rmsnorm(h, L->post_norm, x, xi);
        matvec(M, &L->m[GATE], x, xi, nx, g);
        matvec(M, &L->m[UP], x, xi, nx, u);
        int nm = 0;
        for (int r = 0; r < L->m[GATE].nlive; r++) {
            int i = L->m[GATE].live[r];
            mm[i] = u[i] == 0.0f ? 0.0f : bf(bf(silu(g[i])) * u[i]);
            if (mm[i] != 0.0f) xi[nm++] = i;
        }
        matvec(M, &L->m[DOWN], mm, xi, nm, dn);
        for (int r = 0; r < L->m[DOWN].nlive; r++) {
            int i = L->m[DOWN].live[r];
            h[i] = bf(h[i] + dn[i]), hz_set(i, h[i]);
        }
    }
    if (!head) return -1;  // a token the host feeds: its prediction is not read
    // final norm, lm_head (the embedding, tied), argmax
    M->nx = rmsnorm(h, M->norm, M->x, M->xi);
    atomic_store_explicit(&M->done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&M->gen, 1, memory_order_release);
    head_range(M, 0);
    while (atomic_load_explicit(&M->done, memory_order_acquire) < NT - 1) __asm__ volatile("yield");
    int best = M->best[0];
    float bv = M->bv[0];
    for (int t = 1; t < NT; t++)
        if (M->bv[t] > bv) bv = M->bv[t], best = M->best[t];
    return best;
}

// Feed n tokens (the host's); then, if gen, decode greedily until a token in stop[0..ns) or
// one at or past `vocab_end`, at most max_new. Returns how many tokens were generated (into out).
int st_run(Model *M, const int32_t *toks, int n, const int32_t *stop, int ns, int max_new,
           int32_t *out) {
    int t = 0;
    for (int i = 0; i < n; i++) t = st_step(M, toks[i], i == n - 1);
    int g = 0;
    while (g < max_new) {
        out[g++] = t;
        for (int s = 0; s < ns; s++)
            if (t == stop[s]) return g;
        t = st_step(M, t, 1);
    }
    return g;
}
