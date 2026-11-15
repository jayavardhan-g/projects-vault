---
type: interview-guide
day: 6
date: 2026-11-15
tags: [interview, cuda, streams, checkpointing, flashattention, mock]
---
# Day 6, interview edition: streams, gradient checkpointing and FlashAttention (explain, defend, survive the follow-ups)

Part of [[Interview plan (Nov 2026)]]. Built from [[Day 06 - CUDA II]] (the study note) and [[CUDA kernels - prep]] (decisions 9–25, stories S1–S6, numbers sheet). Repo `sysml`, commit `16870fb`: `mlp/mlp_bench.cu` (PA3) and `flashattention/flash_lib.cu` + `bench_attention.py` (PA4). Line numbers are checked against that commit. The matmul half is [[Day 05 - CUDA I (interview)]]; this note reuses its execution model, coalescing, banks and roofline.

> [!important] What this note is for
> Today's three pieces are the ones interviewers push hardest on, because each has a tempting wrong story: "streams made it 1.73× faster" (probably locality, not concurrency), "13.3% less activation memory" (it's 32% of activations, 13.3% of the total), "FlashAttention is faster" (yours is slower than PyTorch; its win is memory). This note gives you the right story, the code to back it, and the questions that test it.

**How to use it (about 5 hours):**

| pass | what | time |
|---|---|---|
| 1 | §0 one-page summary, then §6 (say it out loud) | 40 min |
| 2 | §1–§2 streams, §3 checkpointing, §4 FlashAttention: card → animation → code walk → question bank | 3 h |
| 3 | §5 performance analysis | 30 min |
| 4 | §7 mock interview, out loud | 1 h+ |

**Labels** (same as Day 5): **[IMPL]** in the code · **[MEAS]** in `results/pa3_mlp.csv` / `pa4_attention.csv` · **[THEORY]** predicted, not measured · **[RECON]** your own reading of code or binaries, not documented by the project · **[FUTURE]** not done.

**Animations:** click **▶ Step through it** under each figure (← / → step, space plays). New ones for this note are `d06i-*` (and `d05i-occupancy` for the FlashAttention occupancy), made by `Days/figures/_gen/cudafigs.py`.

---

## 0. Part 2 of the project on one page

### 0.1 The resume bullet, and the wording you should actually use

> Resume (both versions): "Accelerated MLP inference **1.73×** using column-wise CUDA stream parallelism at large dimensions, while reducing peak activation memory **13.3%** through gradient checkpointing."

| problem in the bullet | what the repo supports |
|---|---|
| "peak activation memory 13.3%" | 13.3% is the **whole training footprint** (962 → 834 MiB). **Activation memory fell 32%** (400 → 272 MiB). |
| "at large dimensions" | **one** width: 1.73× at N = 32768; 1.02× at 16384; slower below. |
| "stream parallelism" | the mechanism is **unproven**; the baseline already fills the GPU at that size, so the evidence points to **cache locality**. |
| "MLP inference" + "checkpointing" in one sentence | different workloads: a 2-layer batch-32 **forward pass** (1.73×) and a 4-layer **training step** (13.3%). |

> [!quote] Say this version
> "I sped up a batch-32 MLP forward pass 1.73× at width 32K by partitioning the output columns across four CUDA streams (splitting the batch never helped), and I implemented gradient checkpointing for a 4-layer MLP that cut activation memory 32%, 13% of total training memory, for a 26% slower backward pass."

### 0.2 The three pieces, measured **[MEAS]**

| piece | setup | result | the honest one-liner |
|---|---|---|---|
| **Streams** (PA3 part 1) | 2-layer forward, batch 32, width N = 32 … 32768, custom `mm_thin`, 3 warm-up + 20 timed | baseline 44.43 ms → column split **25.65 ms (1.73×)** at N = 32768; batch split 52.28 ms (slower everywhere) | "a decomposition win, most likely L2 locality; I couldn't prove it's concurrency" |
| **Checkpointing** (PA3 part 2) | 4-layer MLP training step, config N4096 (batch 4096, widths 1024 → 4096 → 4096 → 4096 → 1024), 100 iterations | activations 400 → 272 MiB (−32%); whole step 962 → 834 MiB (−13.3%); backward 136.15 → 171.69 ms (+26.1%) | "the recompute cost is exactly the recomputed forward FLOPs; the first version saved 0 MB" |
| **FlashAttention** (PA4) | forward only, B = 8, 16 heads, D = 64, FP32, N up to 8192, 3 warm-up + 10 timed | 3,358 → 2,377 → 339 → **160 ms** (21× over the first version); PyTorch explicit 105 ms, fused SDPA 63 ms; max error 6.7e-7 | "a one-line padding fix was 7×; the kernel is 1.5× slower than PyTorch, and its value is memory" |

### 0.3 The ten questions, answered once

| question | streams | checkpointing | FlashAttention |
|---|---|---|---|
| **Problem** | make a small-batch MLP forward pass faster | train with less activation memory | attention without the N × N score matrix |
| **Why GPU** | the forward is two 32 × N × N GEMMs: 6.9 × 10¹⁰ FMAs at N = 32K | training is GEMMs | 2.2 × 10¹² FLOPs at N = 8192 |
| **Baseline** | two `mm_thin` launches on the default stream | keep all 7 activations | quiz layout: grid (B, NH), 128 blocks of one warp, 3,358 ms |
| **Bottleneck** | W (4.29 GB) re-read per wave; GPU already full at 32K | 400 MiB of activations alive at the loss | too few blocks, then a 32-way bank conflict, then the shared-memory roof |
| **Optimisation** | split output columns over 4 streams | keep A2, Z4; recompute the rest into one aliased arena | grid (Tr, NH, B); pad K tile to D + 1; 4 query rows per warp |
| **Why it should help** | each kernel's W slice is read by blocks resident together | the two recomputed segments are never alive together | more blocks; conflict-free banks; one K read feeds 4 FMAs |
| **Hardware level** | L2 hits instead of HBM re-reads (hypothesis) | fewer live `cudaMalloc` bytes | shared-memory bank parallelism; registers |
| **Trade-off** | host barrier between layers; 4× the launches | +26% backward time; a hand-enforced ordering rule | padding breaks 16-byte alignment; 50% occupancy |
| **Alternatives** | batch split (lost); one wider kernel with a better block order | drop Z (use A for the ReLU mask): more saving, zero recompute | swizzled layout; transposed K; tensor cores |
| **Limitations** | one width; no correctness check in Part 1; mechanism unproven | checksum of one tensor as the equality check; redundant segment-B recompute | forward only; FP32; slower than PyTorch; `o_acc` in local memory |

### 0.4 Implemented vs measured vs predicted vs not done

| | |
|---|---|
| **[IMPL]** | `mm_thin`; baseline, batch-split and column-split forward passes with 4 streams; a 4-layer MLP training step with forward, backward (`mm_tiled`, `mm_tiled_AT`, `mm_tiled_BT`) and SGD; checkpointed mode with a shared recompute arena; peak memory by `cudaMemGetInfo`; a weight-checksum equality check; a FlashAttention forward kernel with online softmax, padded K tile, warp-shuffle reductions, 4 rows per warp; a ctypes C ABI; a Python benchmark against PyTorch explicit and fused attention. |
| **[MEAS]** | times and memory above; FlashAttention max abs difference ≤ 7.2e-7 vs PyTorch explicit. |
| **[THEORY]/[RECON]** | waves and the L2 locality explanation of 1.73×; the 0.9 × forward recompute prediction (which matched); bank-conflict degrees; the shared-memory roofs 8.4 and 13.4 TFLOP/s for FlashAttention; theoretical occupancy (50%); `o_acc` in local memory (SASS). |
| **[FUTURE]** | the one-stream experiment that would settle 1.73×; a PyTorch MLP baseline; per-tensor gradient checks; FlashAttention backward, causal mask, FP16/BF16 tensor cores, `cp.async`, FlashAttention-2 work split; Nsight profiles. |

> [!warning] The history caveat
> Git has one commit. The FlashAttention steps 3,358 → 2,377 → 339 ms and the "first checkpointing version saved 0 MB" are documented in `RESULTS.md` and code comments, but only the **final** versions are in the repo. Say "from my notes" for the history; only 160 ms and the final memory numbers are reproducible.

---

## 1. CUDA streams, and the kernel every variant uses (`mm_thin`)

### 1.1 What a stream is **[THEORY]**

- A **stream** is an in-order queue of GPU work. Within one stream, kernel 2 starts after kernel 1 finishes. Across streams there is no ordering unless you add events or syncs.
- Kernels in different streams **may** run at the same time, **only if the GPU has free resources** (SMs with room for their blocks). Copies and kernels in different streams can overlap if the host memory is pinned and copy engines are free.
- The launch's 4th parameter picks the stream: `kernel<<<grid, block, shared_bytes, stream>>>`. Without it, work goes to the default stream. Streams made with `cudaStreamCreate` are "blocking" streams: they synchronise with the legacy default stream.
- **Streams help** when one kernel leaves SMs idle (small grids) or to overlap copies with compute. **They don't help** when a kernel already fills every SM: the second kernel's blocks just queue behind the first's.

![[d6-5-streams.svg]] [[d6-5-streams.html|▶ Step through it]]

### 1.2 Component card: `mm_thin<256, 32>` (`mlp_bench.cu:46–80`)

| | |
|---|---|
| **Problem** | C[M, N] = A[M, K]·B[K, N] with tiny M (batch 32) and huge K = N (up to 32768) |
| **Design [IMPL]** | one block = (one batch row `m`, a chunk of 256 output columns); 256 threads, one column each; the row's K-slice of A is staged 32 floats at a time in shared memory and reused by all 256 threads |
| **Why not square tiles [IMPL, documented]** | after a 4-way batch split each kernel has only 8 rows: a 32-row square tile would waste 24 of every 32 rows, which would make the stream comparison meaningless (comment at `:42–44`) |
| **Hardware level** | `B[(k0 + kk)·N + col]`: adjacent threads, adjacent columns → coalesced; `sA[kk]`: same address for all → broadcast |
| **Cost [RECON]** | each of the 32 row-blocks re-reads the same W column slice from global memory: reuse of W across the batch depends entirely on L1/L2 hits. About 0.5 FLOP per byte at the cache level (one 4-byte W load per FMA). |

### 1.3 The code, line by line

```cpp
46  template<int BN, int BK>
47  __global__ void mm_thin(const float* __restrict__ A,
48                          const float* __restrict__ B,
49                          const float* __restrict__ bias,
50                          float* __restrict__ C,
51                          int M, int K, int N, int relu)
52  {
53      __shared__ float sA[BK];
54
55      const int m   = blockIdx.y;
56      const int col = blockIdx.x * BN + threadIdx.x;
57      if (m >= M) return;
58
59      float acc = 0.0f;
60      for (int k0 = 0; k0 < K; k0 += BK) {
61          // Stage this row-slice of A cooperatively.
62          for (int t = threadIdx.x; t < BK; t += BN) {
63              int k = k0 + t;
64              sA[t] = (k < K) ? A[(size_t)m * K + k] : 0.0f;
65          }
66          __syncthreads();
67
68          if (col < N) {
69              int kmax = (K - k0 < BK) ? (K - k0) : BK;
70              for (int kk = 0; kk < kmax; ++kk)
71                  acc = fmaf(sA[kk], B[(size_t)(k0 + kk) * N + col], acc);
72          }
73          __syncthreads();
74      }
75
76      if (col < N) {
77          float v = acc + (bias ? bias[col] : 0.0f);
78          C[(size_t)m * N + col] = relu ? (v > 0.0f ? v : 0.0f) : v;
79      }
80  }
```

| line | what | why | what breaks if changed |
|---|---|---|---|
| 47–50 | `__restrict__`: promise that the pointers don't alias | lets the compiler cache loads and use the read-only path | if a caller passed overlapping A and C, results would be undefined |
| 53 | 32 floats of shared memory per block | one K-slice of row `m` | |
| 55 | `m = blockIdx.y`: the batch row | grid.y = batch rows (32, or 8 in the batch split) | |
| 56 | `col` from `threadIdx.x` | coalesced B loads and C stores | |
| 57 | early `return` | **safe here** although the kernel has barriers: `m` is the same for every thread of the block, so the whole block returns together and no barrier is left waiting | if the condition depended on `threadIdx`, it would be the divergent-barrier bug |
| 62–65 | the first BK = 32 threads copy A[m][k0 … k0+31] into `sA` | BK < BN, so the loop runs once for threads 0–31 and not at all for 32–255 | |
| 66 | barrier: `sA` complete | every thread reads all 32 entries | remove → reads of unwritten `sA` cells |
| 68–72 | each thread: 32 FMAs, `sA[kk]` (broadcast) × one W element (coalesced across lanes) | the guard is around the compute, not the barriers | |
| 69 | `kmax` handles a ragged last K-slice | | |
| 73 | barrier: everyone done with `sA` before it is overwritten | | remove → race on `sA` |
| 76–78 | bias + optional ReLU, one store | fused epilogue, no extra kernel | |

**Traced with real numbers (baseline, N = 32768):** grid `((32768 + 255)/256, 32)` = (128, 32) = 4,096 blocks. Block (bx = 5, by = 7), thread 10: `m = 7`, `col = 5·256 + 10 = 1,290`. It computes `A1[7][1290] = ReLU(Σₖ X[7][k]·W1[k][1290] + b1[1290])` over 32,768 values of k, in 1,024 slices of 32. Its warp (threads 0–31) reads `W1[k][1280 … 1311]` at each k: one 128-byte line.

#### How it is called, on an example

**Real call** (baseline layer 1, `mlp_bench.cu:203–255`) **[IMPL]**:
```cpp
constexpr int BN = 256, BK = 32;                             // :203
dim3 blk(BN);                                                // :250  256 threads, 1-D block
dim3 grd((N + BN - 1) / BN, B);                              // :251  (column chunks, batch rows)
mm_thin<BN,BK><<<grd, blk>>>(X, W1, b1, A1, B, N, N, 1);     // :255  A1 = ReLU(X·W1 + b1)
```

| kernel parameter | host argument | shape / value |
|---|---|---|
| `A` | `X` | [M × K] = [32 × N], the input batch |
| `B` | `W1` | [K × N] = [N × N], the weights |
| `bias` | `b1` | [N] |
| `C` | `A1` | [32 × N], the output activations |
| `M, K, N` | `B, N, N` | 32, N, N. **Name clash:** the host's `B` is the batch size (32); the kernel's `B` is the weight matrix. |
| `relu` | `1` | apply ReLU in the epilogue (0 for layer 2) |

**The maths:** C[m][col] = act(Σₖ A[m][k]·B[k][col] + bias[col]). Block (bx, by) owns **one row** m = by and **256 columns** 256·bx … 256·bx + 255; thread `tx` owns one column. That's a matrix-vector product per block: one row of X times a 256-column slice of W.

**Hand-sized call [THEORY]:** `mm_thin<2,2><<<dim3(2, 2), dim3(2)>>>(X, W, b, Y, 2, 4, 4, 1)`: batch M = 2, K = N = 4, blocks of 2 columns, K staged 2 at a time. Inputs (X = rows 0–1 of Day 5's toy A, W = Day 5's toy B):

$$X = \begin{bmatrix}1&2&0&1\\0&1&2&1\end{bmatrix},\ W = \begin{bmatrix}1&0&2&1\\2&1&0&0\\0&1&1&2\\1&2&0&1\end{bmatrix},\ b = [-4, 0, -3, 0]$$

X·W = [[6, 4, 2, 2], [3, 5, 2, 5]]; + b = [[2, 4, −1, 2], [−1, 5, −1, 5]]; ReLU → **Y = [[2, 4, 0, 2], [0, 5, 0, 5]]**.

Grid (2, 2): block (bx, by) computes Y[by][2bx] and Y[by][2bx + 1]. Block (1, 0), i.e. row m = 0, columns 2–3:

| k0 | `sA` = X[0][k0 … k0+1] | thread 0 (col 2) `acc` | thread 1 (col 3) `acc` |
|---|---|---|---|
| 0 | [1, 2] | 1·W[0][2] + 2·W[1][2] = 1·2 + 2·0 = 2 | 1·W[0][3] + 2·W[1][3] = 1·1 + 2·0 = 1 |
| 2 | [0, 1] | 2 + 0·1 + 1·0 = 2 | 1 + 0·2 + 1·1 = 2 |
| epilogue `:77–78` | | 2 + (−3) = −1 → ReLU → **0** | 2 + 0 = **2** |

Y[0][2] = 0, Y[0][3] = 2 ✓. The same `sA` pair is reused by both threads (in the real kernel by all 256): that is the only reuse `mm_thin` designs in.

### 1.4 Question bank: streams and `mm_thin`

**L1. What is a CUDA stream?**
> [!success]- Answer
> An ordered queue of GPU operations. Operations in one stream execute in issue order; operations in different streams have no implied order and may overlap if resources allow. In my code, `cudaStreamCreate(&st[i])` makes four streams (`:206–207`) and each launch picks one with the 4th launch argument: `mm_thin<<<g, blk, 0, st[s]>>>`.

**L1. When do two streams actually run at the same time?**
> [!success]- Answer
> When the kernels in them can be resident together: the first kernel must leave SMs (or SM resources) free for the second's blocks. Small grids that fill only part of the GPU overlap well; a kernel whose grid fills all 132 SMs for several waves leaves nothing for the other stream until its last wave drains. Copy/compute overlap needs pinned host memory and `cudaMemcpyAsync`. [THEORY]

**L2. Why is the early `return` on line 57 safe even though the kernel uses `__syncthreads()`?**
> [!success]- Answer
> Its condition, `m >= M` with `m = blockIdx.y`, is uniform across the block: either every thread returns or none does. The danger with barriers is *divergent* exit, where some threads of a block leave and others wait at a barrier for them. The thread-dependent guards (`col < N`) are written around the compute and the store, not around the barriers, which is the correct pattern.

**L2. Why one block per (row, 256 columns) rather than square tiles?**
> [!success]- Answer
> With batch 32, and 8 rows per kernel after a 4-way batch split, a square 32 × 32 tile would leave most of its rows empty: 24 of every 32 rows wasted, which would make the stream comparison measure tile waste instead of streams. A thin block keeps every thread busy on a real output. The cost is that W reuse across the 32 batch rows is left to the caches, which turns out to be the key to the 1.73× story. [IMPL, documented reason]

**L3. "`mm_thin` reads W once per FMA. Isn't it hopelessly memory-bound?"**
> [!success]- Answer
> Per FMA it issues one 4-byte global load of W, so about 0.5 FLOP/B at the L1/L2 level. Across the forward pass that's ~275 GB of W requests (32 rows × N² × 4 B × 2 layers), which at 25.65 ms is ~10.7 TB/s, more than twice HBM bandwidth, so most requests must hit L1/L2. The baseline manages only ~6.2 TB/s of requests. Either way it's memory-hierarchy-bound, at about 3–5 TFLOP/s, far from the 66.9 peak. A cuBLAS GEMM would do better; the brief's PyTorch baseline was never implemented. [THEORY from MEAS times]

---

## 2. Baseline vs batch split vs column split (1.73×, and why it's probably not concurrency)

### 2.1 Component card

| | |
|---|---|
| **Problem** | make the 2-layer forward (Y = ReLU(X·W1 + b1)·W2 + b2, X is 32 × N) faster with 4 streams |
| **Baseline [IMPL]** | 2 launches per iteration on the default stream, grid (N/256, 32), no host sync inside the loop (`:254–257`) |
| **Batch split [IMPL]** | stream s takes rows 8s … 8s+7, but **all of W1 and W2** (`:262–273`) |
| **Column split [IMPL, beyond the brief]** | stream s takes output columns c0 … c0 + N/4 − 1 and all 32 rows; a host barrier between the layers because layer 2 needs all of A1 (`:280–299`) |
| **Measured [MEAS]** | N = 32768: 44.43 / 52.28 / **25.65 ms**; N = 16384: 6.29 / 6.85 / 6.19 (1.02×); N = 1024: 0.099 / 0.120 / 0.125; N = 32: 0.0069 / 0.0278 / 0.0339 |
| **Why batch split lost [THEORY]** | W is N² floats against 32·N for X, so time ≈ how many times W is read; each batch-split stream re-reads all of W for only 8 rows, plus 4× launches and 4 host syncs per iteration |
| **Why column split won [RECON hypothesis]** | each column-split kernel is 1,024 blocks, **one wave**, so all 32 batch rows of a W chunk run together and share it through L2; the baseline (4,096 blocks, ~3.9 waves) sweeps all of W in every wave, re-reading it from HBM ~4× per layer |
| **Limitations** | one width; no output check in Part 1; column split wrong for N ≤ 512; stream variants include host syncs in the timed loop; mechanism unproven |

![[d06i-split.svg]] [[d06i-split.html|▶ Step through it]]

### 2.2 The three variants, line by line

```cpp
// (a) baseline :254–257
auto baseline = [&] {
    mm_thin<BN,BK><<<grd, blk>>>(X,  W1, b1, A1, B, N, N, 1);   // layer 1, ReLU
    mm_thin<BN,BK><<<grd, blk>>>(A1, W2, b2, Y,  B, N, N, 0);   // layer 2
};
```
Two launches in the default stream: layer 2 starts after layer 1 ends (same stream = in order), so no barrier is needed. `grd = ((N + 255)/256, B)`, `blk = 256`.

```cpp
// (b) batch split :261–273
const int chunk = (B + NS - 1) / NS;                        // (32 + 3)/4 = 8
auto streamed = [&] {
    for (int s = 0; s < NS; ++s) {
        int r0 = s * chunk;                                 // 0, 8, 16, 24
        int rows = (r0 + chunk <= B) ? chunk : (B - r0);
        if (rows <= 0) continue;
        dim3 g((N + BN - 1) / BN, rows);                    // (128, 8) at N = 32768
        mm_thin<BN,BK><<<g, blk, 0, st[s]>>>(X  + (size_t)r0 * N, W1, b1,
                                             A1 + (size_t)r0 * N, rows, N, N, 1);
        mm_thin<BN,BK><<<g, blk, 0, st[s]>>>(A1 + (size_t)r0 * N, W2, b2,
                                             Y  + (size_t)r0 * N, rows, N, N, 0);
    }
    for (int s = 0; s < NS; ++s) cudaStreamSynchronize(st[s]);
};
```

| piece | what | consequence |
|---|---|---|
| `X + r0·N`, `A1 + r0·N` | pointer offsets: rows are contiguous in row-major, so "rows 16–23" is just an offset | each stream's layer 2 needs only its own rows of A1, so no cross-stream barrier |
| `W1`, `W2` passed **unchanged** | every stream multiplies its 8 rows by the whole weight matrix | 4 kernels each sweep 4.29 GB of W per layer |
| grid `(128, 8)` | 1,024 blocks per kernel | |
| `cudaStreamSynchronize` × 4 | the **host** waits for each stream, every iteration, inside the timed loop | at small N, host round-trips are part of the "4× slower" |

```cpp
// (c) column split :279–299
const int cchunk = (N + NS - 1) / NS;                       // 8,192 at N = 32768
auto streamed_col = [&] {
    for (int s = 0; s < NS; ++s) {
        int c0 = s * cchunk;                                // 0, 8192, 16384, 24576
        int cols = (c0 + cchunk <= N) ? cchunk : (N - c0);
        if (cols <= 0) continue;
        dim3 g((cols + BN - 1) / BN, B);                    // (32, 32) = 1,024 blocks
        mm_thin<BN,BK><<<g, blk, 0, st[s]>>>(X, W1 + c0, b1 + c0,
                                             A1 + c0, B, N, N, 1);
    }
    for (int s = 0; s < NS; ++s) cudaStreamSynchronize(st[s]);   // layer 2 needs ALL of A1
    for (int s = 0; s < NS; ++s) {
        ... same with (A1, W2 + c0, b2 + c0, Y + c0) ...
    }
    for (int s = 0; s < NS; ++s) cudaStreamSynchronize(st[s]);
};
```

| piece | what | consequence |
|---|---|---|
| `W1 + c0` | a **column** offset into a row-major matrix: the kernel still uses N as the row stride (leading dimension), but its threads only index columns `c0 …` | each stream reads a disjoint quarter of W1; the comment at `:313–317` explains why passing N as the width is right |
| `A1 + c0` | the output columns this stream owns | |
| grid `(cols/256, B)` | 32 column chunks × 32 rows = 1,024 blocks | ≤ 1,056: one wave (§2.3) |
| barrier between layers | every layer-2 output needs every A1 column, which all 4 streams produce | the two layers can't be pipelined per stream |
| **bug for N ≤ 512 [RECON]** | the guard inside `mm_thin` is `col < N` with the full width, but `cchunk < 256`, so a 256-thread block covers more than its chunk: threads write into other streams' columns and the last stream writes past the end of `A1`/`Y` | the N ≥ 1024 results, including 1.73×, are unaffected (chunks are multiples of 256); fix: pass `cols` as a separate bound |

#### The three variants called on the toy: the maths of each split

Same toy as §1.3 (X is 2 × 4, W is 4 × 4, Y = ReLU(X·W + b)), with **2 streams instead of 4** and `BN = BK = 2` **[THEORY]**:

| variant | the launches | the maths | each launch reads |
|---|---|---|---|
| baseline | `mm_thin<<<dim3(2, 2), 2>>>(X, W, b, Y, 2, 4, 4, 1)` | Y = X·W | all of X, all of W |
| batch split (chunk = 1 row) | stream s: `mm_thin<<<dim3(2, 1), 2, 0, st[s]>>>(X + 4s, W, b, Y + 4s, 1, 4, 4, 1)` | $\begin{bmatrix}X_0\\X_1\end{bmatrix} W = \begin{bmatrix}X_0 W\\X_1 W\end{bmatrix}$: stream s computes row s | 1 row of X, **all of W** |
| column split (chunk = 2 cols) | stream s: `mm_thin<<<dim3(1, 2), 2, 0, st[s]>>>(X, W + 2s, b + 2s, Y + 2s, 2, 4, 4, 1)` | $X\,[\,W_L \mid W_R\,] = [\,X W_L \mid X W_R\,]$: stream s computes columns 2s, 2s + 1 | all of X, **half of W** |

All three produce Y = [[2, 4, 0, 2], [0, 5, 0, 5]] (checked with a CPU port of `mm_thin`).

**How a pointer offset selects a sub-matrix.** Row offsets are easy: rows are contiguous, so `X + 4s` is row s. Column offsets work because the **full N is still passed as the row stride**: the kernel reads `B[(k0 + kk)·N + col]` with N = 4 and a local `col` ∈ {0, 1}. With `B = W + 2` that is `W[k·4 + 2 + col]` = **W[k][2 + col]**, columns 2–3. The store `C[m·N + col]` with `C = Y + 2` lands in Y[m][2 + col] the same way. This is BLAS's "leading dimension": a pointer to the top-left element plus the stride of the full matrix is a view of a sub-matrix, with no copy. (It's also why the N ≤ 512 bug exists: the guard `col < N` uses the stride, not the chunk width.)

**Why layer 2 needs the barrier, in the same notation:** layer 2's output column j is A1 · W2[:, j], which needs **every** column of A1, and in the column split those columns were produced by different streams. In the batch split, row s of layer 2 needs only row s of A1, made earlier in the same stream.

At the real size: batch split offsets `X + r0·N` with r0 = 0, 8, 16, 24; column split `W1 + c0` with c0 = 0, 8,192, 16,384, 24,576.

### 2.3 Waves, L2, and the locality explanation

![[d06i-waves.svg]] [[d06i-waves.html|▶ Step through it]]

![[d06m-stream-splits.mp4]]
*Manim video: baseline waves vs batch split vs column split, and how many times W comes from HBM (1 min). Pause and scrub with the video controls.*

**Count the waves [THEORY]:** `mm_thin` has 256 threads per block and an SM holds up to 2,048 threads, so up to 8 blocks per SM → **1,056 blocks per wave** on 132 SMs (ignoring register limits, which aren't documented for this kernel).

| | grid | blocks | waves | idle SMs for another stream? |
|---|---|---|---|---|
| baseline, N = 1024 | 4 × 32 | 128 | 0.12 | yes, most of the GPU |
| baseline, N = 32768 | 128 × 32 | 4,096 | ~3.9 | **no** |
| one column-split kernel, N = 32768 | 32 × 32 | 1,024 | ~1 | no (it fills one wave by itself) |

**The argument, step by step:**
1. At N = 32768 the baseline already fills the GPU for ~4 waves, so there are no idle SMs for streams to use. **Concurrency has little to offer.**
2. Blocks are typically dispatched `blockIdx.x` first. A baseline wave (~1,056 blocks) covers all 128 column chunks for only ~8 batch rows, so **every wave sweeps all of W1** (4.29 GB, ~85× the ~50 MB L2). By the time wave 2 needs chunk 0 again, it has been evicted: W is streamed from HBM about 4 times per layer.
3. A column-split kernel is 32 chunks wide and 32 rows tall: **all 32 batch rows of each chunk are resident together**, so each W line is fetched from HBM about once and then served from L2 to the other rows.
4. The data fits: the baseline's time grows **7.1×** from N = 16K to 32K (6.29 → 44.43 ms) for 4× the work, a large-N pathology; the column split grows **4.1×** (6.19 → 25.65 ms), linear. The gain appears only where the baseline needs several waves.

**The experiment that would settle it [FUTURE]** (prep pack exercise 2): replace `st[s]` with `st[0]` in the column split, so the four chunks run one after another on one stream. If it stays near 25–27 ms, the win is locality; if it returns to ~44 ms, overlap mattered. An even cleaner test: one kernel with `blockIdx.x` as the batch row and `blockIdx.y` as the column chunk.

### 2.4 How Part 1 is timed (`:301–311`)

```cpp
auto measure = [&](auto fn) {
    for (int i = 0; i < 3; ++i) fn();            // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < iters; ++i) fn();        // 20 iterations
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    return (double)ms / iters;
};
```
- The events are recorded on the default stream; the four streams are blocking streams, so the default stream's `end` marker can't pass work still running in them, and the stream variants also sync on the host every iteration. The time is valid for all three. **[RECON]**
- **Asymmetry [RECON]:** `baseline` queues 40 launches back to back with no host sync; `streamed` and `streamed_col` call `cudaStreamSynchronize` 4 or 8 times **per iteration**, so their time includes host round-trips. At N = 32 (baseline 6.9 µs per iteration) that's most of the "4× slower"; at N = 32768 (tens of ms) it's negligible.
- One sweep, no repeats, no error bars. **Part 1 has no correctness check** of A1 or Y.

### 2.5 Question bank: the stream experiments

**L2. Why did the batch split never help?**
> [!success]- Answer
> Three reasons. (1) Traffic: W is N² floats against 32·N for X, so cost ≈ how often W is read, and each of the four 8-row kernels reads all of W1 and W2: more passes over the dominant data, with only 8 rows of reuse per pass. (2) At large N the baseline already fills the GPU (4,096 blocks, ~3.9 waves), so there are no idle SMs to run streams side by side. (3) At small N it's 8 launches + 4 host syncs per iteration vs 2 launches, and the baseline iteration is 7 µs. Result: slower at every width, 52.28 vs 44.43 ms at 32K and 4.1× slower at N = 32.

**L2. Why is there a host barrier between the layers in the column split?**
> [!success]- Answer
> Layer 2's output column j is Σ over **all** of A1's columns times W2's column j, and layer 1 was split by column, so A1's columns come from four different streams. Every stream must finish layer 1 before any layer-2 kernel starts. The batch split doesn't need it because each stream's layer 2 uses only its own rows of A1, which its own stream produced in order.

**L3. "You just said the GPU was already full. Where does the 1.73× come from?"** (the key question)
> [!success]- Answer
> Honestly, not from added parallelism. The evidence: the baseline's time grows 7.1× from 16K to 32K for 4× the work, while the column split grows 4.1×, so the baseline hits a large-N pathology that the column split avoids. My leading hypothesis is L2 locality: a column-split kernel's 1,024 blocks fit in one wave, so all 32 batch rows of each W chunk are resident together and share its lines through L2; the baseline's waves each sweep all 4.29 GB of W, so W is re-read from HBM about 4 times per layer. TLB reach and DRAM page locality are other candidates. I couldn't separate them without ncu. The test is to run the four chunks on one stream: if it's still ~26 ms, it's locality.

**L3. "Is the column split even correct?"**
> [!success]- Answer
> At N = 32768, yes by construction: each stream computes its own columns from the full A1 after the barrier, with chunks that are multiples of 256. But Part 1 has no output check, and re-reading the code I found the split is wrong for N ≤ 512: the kernel's guard uses the full width N, while the chunk is narrower than one 256-thread block, so threads write outside their chunk and the last stream writes past the end of A1/Y. The small-N column-split timings come from buggy code. The fix is to pass `cols` as the bound. I'd say this up front.

**L3. "Are the small-N comparisons fair?"**
> [!success]- Answer
> Not fully. The stream variants call `cudaStreamSynchronize` inside the timed loop every iteration; the baseline just queues launches. So at N = 32 the 4× includes host round-trips, not only launch overhead. A fair version would sync once at the end, or capture each variant in a CUDA graph. The large-N result isn't affected, because there each iteration takes tens of milliseconds.

**L3. "When would streams clearly help?"**
> [!success]- Answer
> When something is idle: overlapping host↔device copies with compute (`cudaMemcpyAsync` from pinned memory on one stream while a kernel runs on another, chunked like a pipeline), or running several small independent kernels that each fill only part of the GPU (for example per-head or per-expert work at small batch). In my project neither condition held at large N, which is the lesson I took from it.

---

## 3. Gradient checkpointing: one arena, two segments

### 3.1 Component card

| | |
|---|---|
| **Problem** | backward needs every layer's Z (for the ReLU mask) and its input A_prev (for dW = A_prevᵀ·dZ), and runs last layer first, so the forward pass must keep them all: 400 MiB of activations at N4096 |
| **Optimisation [IMPL]** | keep only two checkpoints, **A2** and **Z4** (assignment-prescribed); recompute segment B (Z3, A3) from A2 and segment A (Z1, A1, Z2) from X during backward, both into **one arena** of `max(2·n3, 2·n1 + n2)` floats (`:453–462`) |
| **Why it saves memory** | backward consumes segment B completely (layers 4, 3) before it needs segment A (layers 2, 1), so the two are never alive together and can share storage |
| **Measured [MEAS]** | activations 400 → 272 MiB (−32%); whole training step 962 → 834 MiB (−13.3%); backward 136.15 → 171.69 ms (+35.54 ms, +26.1%); checksums equal to 10 digits |
| **Cost, predicted [THEORY]** | recomputed FLOPs = 0.9 of a forward; 0.9 × 39.5 ms = 35.5 ms; measured +35.54 ms |
| **Trade-offs** | +26% backward time; a strict ordering rule enforced only by a comment (`:513`) |
| **Alternatives** | drop Z entirely and take the ReLU mask from A (A > 0 ⇔ Z > 0): saves 192 MiB with no recompute [RECON, prep Q27]; a 1-bit mask; ping-ponged dZ buffers; √n checkpoint spacing for deep nets |
| **Limitations** | 4 layers, so the √n theory barely applies; equality check is one checksum; recompute of segment B is redundant in this schedule (§3.6) |

### 3.2 What backward needs, and the trade

One layer: `Z = A_prev·W + b`, `A = ReLU(Z)`. Its backward needs **Z** (`k_relu_deriv(z, dZ)` zeroes the gradient where Z ≤ 0) and **A_prev** (`mm_tiled_AT(a_prev, dZ, dW)`). Backward runs layer 4 → 1, so without checkpointing everything made in forward is kept until its last use.

Toy (16 layers, 1 unit each): keep all = 16 units; checkpoint every 4th layer = 4 checkpoints + one 4-layer segment rebuilt at a time = **8 units**, for about one extra forward. In general ~2√n instead of n (Chen et al., 2016). **[THEORY]**

![[d6-6-ckpt.svg]] [[d6-6-ckpt.html|▶ Step through it]]

### 3.3 What is stored, discarded and recomputed, step by step

![[d06i-ckpt-life.svg]] [[d06i-ckpt-life.html|▶ Step through it]]

![[d06m-ckpt-arena.mp4]]
*Manim video: keep-all vs checkpointed memory side by side, the arena being overwritten, and the 0 MB first version (1 min). Pause and scrub with the video controls.*

```
Z1 = X·W1 + b1,  A1 = ReLU(Z1)      ← segment A (arena)
Z2 = A1·W2 + b2, A2 = ReLU(Z2)      ← A2: checkpoint (own buffer)
Z3 = A2·W3 + b3, A3 = ReLU(Z3)      ← segment B (same arena, overwrites segment A)
Z4 = A3·W4 + b4                     ← Z4: checkpoint (the MSE loss needs it)
```

Sizes at N4096: a 4096 × 4096 float tensor = **64 MiB**; Z4 (4096 × 1024) = **16 MiB**.
- Baseline: Z1, A1, Z2, A2, Z3, A3 (6 × 64) + Z4 (16) = **400 MiB**.
- Checkpointed: A2 (64) + Z4 (16) + arena max(Z3 + A3 = 128, Z1 + A1 + Z2 = 192) = **272 MiB**.
- Whole step (`cudaMemGetInfo`): weights + gradients 320 MiB, dZ buffers 208 MiB, X/Y 32 MiB are the same in both modes, so the same 128 MiB saving is 13.3% of 962 MiB. **[MEAS]** (sizes are MiB, 2²⁰, though the CSV labels them "MB")

### 3.4 The code, line by line

**Allocation (`:411–420`)**
```cpp
float *Z1=nullptr,*A1=nullptr,*Z2=nullptr,*A2=nullptr,*Z3=nullptr,*A3=nullptr,*Z4=nullptr;
CUDA_CHECK(cudaMalloc(&A2, n2 * sizeof(float)));   // checkpoint
CUDA_CHECK(cudaMalloc(&Z4, n4 * sizeof(float)));   // checkpoint
if (!checkpointed) {
    CUDA_CHECK(cudaMalloc(&Z1, n1 * sizeof(float)));
    ... A1, Z2, Z3, A3 ...
}
```
Both modes keep A2 and Z4. Only the baseline allocates the other five. That is the whole memory difference, apart from the arena.

**The arena (`:453–462`)**
```cpp
float *arena = nullptr;
float *rZ1=nullptr,*rA1=nullptr,*rZ2=nullptr,*rZ3=nullptr,*rA3=nullptr;
if (checkpointed) {
    size_t segA = 2 * n1 + n2;                       // Z1, A1, Z2  = 192 MiB
    size_t segB = 2 * n3;                            // Z3, A3      = 128 MiB
    size_t arena_floats = segA > segB ? segA : segB;
    CUDA_CHECK(cudaMalloc(&arena, arena_floats * sizeof(float)));
    rZ1 = arena;  rA1 = rZ1 + n1;  rZ2 = rA1 + n1;   // segment A layout
    rZ3 = arena;  rA3 = rZ3 + n3;                    // segment B, aliased
}
```

| line | what | why | what breaks if changed |
|---|---|---|---|
| `segA`, `segB` | float counts of the two segments | the arena must hold the larger | |
| `max(segA, segB)` | one allocation for both | they are never alive together | allocate `segA + segB` → 320 MiB, the five dropped buffers again: **0 MB saved** (the documented first-version bug, prep exercise 3) |
| `rZ1 = arena` and `rZ3 = arena` | **the two layouts alias**: Z3 and Z1 start at the same address | that is the saving | any read of z3 after segment A is rebuilt sees Z1's numbers: silently wrong gradients |

**Forward (`:464–485`)** writes z1, a1, z2 into the arena, A2 into its own buffer, then z3, a3 over the start of the arena, and Z4 into its own buffer: `z1 = checkpointed ? rZ1 : Z1` etc. at `:465–469`.

**Backward, segment B (`:493–501`)**
```cpp
if (checkpointed) {
    // segment B, from the A2 checkpoint
    mm_tiled<<<g2(Nb,H3), dim3(TS,TS)>>>(A2, W3, rZ3, Nb, H2, H3);
    k_add_bias_relu<<<g1(n3),256>>>(rZ3, b3, Nb, H3, 0);
    k_relu_into<<<g1(n3),256>>>(rZ3, rA3, n3);
    z3 = rZ3; a3 = rA3;
} else { z3 = Z3; a3 = A3; }
```
Then layer 4 (`k_mse_grad(Z4, …)`, `mm_tiled_AT(a3, dZ4, dW4)`) and layer 3 (`mm_tiled_BT(dZ4, W4, dZ3)`, `k_relu_deriv(z3, dZ3)`, `mm_tiled_AT(A2, dZ3, dW3)`) at `:503–511`.

**The ordering rule (`:513–523`)**
```cpp
// z3 / a3 are dead from here on: rebuild segment A over the same arena.
if (checkpointed) {
    mm_tiled<<<g2(Nb,H1), dim3(TS,TS)>>>(X, W1, rZ1, Nb, IN, H1);
    k_add_bias_relu<<<g1(n1),256>>>(rZ1, b1, Nb, H1, 0);
    k_relu_into<<<g1(n1),256>>>(rZ1, rA1, n1);
    mm_tiled<<<g2(Nb,H2), dim3(TS,TS)>>>(rA1, W2, rZ2, Nb, H1, H2);
    k_add_bias_relu<<<g1(n2),256>>>(rZ2, b2, Nb, H2, 0);
    z1 = rZ1; a1 = rA1; z2 = rZ2;
}
```
All these kernels run on the default stream, so "every use of z3/a3 before this line" is guaranteed by stream order on the GPU, but only because the **source order** is right. Nothing checks it. Then layer 2 and 1 gradients and SGD (`:525–543`).

`mm_tiled` here (`:90–107`) is the same 32 × 32 1-to-1 tiled kernel as Day 5's `mm_tiled_1to1<32>`. Custom GEMMs are used everywhere, not cuBLAS, because cuBLAS keeps an internal workspace that would pollute `cudaMemGetInfo` (`:17–18`, documented).

#### How the training-step kernels are called, on an example

Every launch in the training step uses one of two shape helpers (`mlp_bench.cu:183–184`) **[IMPL]**:
```cpp
static inline int  g1(size_t n, int tpb = 256) { return (int)((n + tpb - 1) / tpb); }   // ceil(n/256) blocks
static inline dim3 g2(int M, int N) { return dim3((N + TS - 1) / TS, (M + TS - 1) / TS); }  // TS = 32
```
- `g2(M, N)` takes C's shape as (rows, columns) but returns `dim3(columns, rows)`: grid.x walks C's columns so that `tx → col` and loads coalesce (Day 5 §3). Used with `dim3(TS, TS)` = 32 × 32 blocks: **one thread per entry of C**, Day 5's 1-to-1 tiling.
- `g1(n)` with 256-thread blocks: **one thread per element** of a flat array of n floats, thread i = `blockIdx.x·256 + threadIdx.x`.

**Every kernel, its call for layer 3 / 4, and its maths** (buffers are row-major [rows × cols]):

| kernel | a real call (line) | maths |
|---|---|---|
| `mm_tiled(A, B, C, M, K, N)` | `<<<g2(Nb,H3), dim3(TS,TS)>>>(A2, W3, rZ3, Nb, H2, H3)` :495 | C = A·B: **Z3 = A2·W3** |
| `mm_tiled_AT(A, B, C, M, K, N)`, A stored [K × M] | `<<<g2(H2,H3), …>>>(A2, dZ3, dW3, H2, Nb, H3)` :510 | *meant to be* C = Aᵀ·B: **dW3 = A2ᵀ·dZ3**. ⚠️ It isn't: see the warning below the toy. |
| `mm_tiled_BT(A, B, C, M, K, N)`, B stored [N × K] | `<<<g2(Nb,H3), …>>>(dZ4, W4, dZ3, Nb, OUT, H3)` :508 | C = A·Bᵀ: **dZ3 ← dZ4·W4ᵀ** (the gradient reaching A3) |
| `k_add_bias_relu(Z, b, M, D, relu)` | `<<<g1(n3),256>>>(rZ3, b3, Nb, H3, 0)` :496 | Z[i] += b[i mod D] (relu = 0 at every call in the training step, so bias only) |
| `k_relu_into(Z, A, n)` | `<<<g1(n3),256>>>(rZ3, rA3, n3)` :497 | A[i] = max(Z[i], 0) |
| `k_mse_grad(Z, Y, dZ, n)` | `<<<g1(n4),256>>>(Z4, Yt, dZ4, n4)` :504 | dZ[i] = 2(Z[i] − Y[i]) / n, n = Nb·OUT (gradient of the mean squared error) |
| `k_relu_deriv(Z, dZ, n)` | `<<<g1(n3),256>>>(z3, dZ3, n3)` :509 | dZ[i] ← dZ[i] if Z[i] > 0, else 0 (in place) |
| `k_bias_grad(dZ, db, M, D)` | `<<<g1(OUT),256>>>(dZ4, db4, Nb, OUT)` :506 | db[j] = Σᵢ dZ[i][j]: one thread per **column**, looping over M rows |
| `k_sgd(p, g, lr, n)` | `<<<g1((size_t)IN*H1),256>>>(W1, dW1, lr, …)` :536 | p[i] −= lr·g[i] |

At N4096, `mm_tiled(A2, W3, rZ3, 4096, 4096, 4096)` launches g2 = (128, 128) blocks of 32 × 32: 16.8 M threads, one per entry of Z3.

**One layer, end to end, by hand [THEORY]** (batch Nb = 2, 2 inputs, 2 outputs; checked with a CPU port of the kernels). Map it onto layer 3: A_prev ↔ A2, W ↔ W3, Z ↔ Z3, A ↔ A3.

$$A_\text{prev} = \begin{bmatrix}1&2\\3&0\end{bmatrix},\quad W = \begin{bmatrix}1&-1\\0&2\end{bmatrix},\quad b = [0, -1]$$

| # | call | what the threads do | result |
|---|---|---|---|
| 1 | `mm_tiled<<<g2(2,2), dim3(32,32)>>>(A_prev, W, Z, 2, 2, 2)` | grid (1, 1); threads with tx, ty < 2 own Z[ty][tx]; the other 1,020 load zeros and skip the store | Z = A_prev·W = [[1, 3], [3, −3]] |
| 2 | `k_add_bias_relu<<<g1(4),256>>>(Z, b, 2, 2, 0)` | threads i = 0…3: Z[i] += b[i mod 2] | Z = [[1, 2], [3, −4]] |
| 3 | `k_relu_into<<<1,256>>>(Z, A, 4)` | | A = [[1, 2], [3, 0]] |
| | *backward; the gradient arriving at A is G = [[1, 1], [1, 1]] (in the code, the next layer's `mm_tiled_BT` output, already in the `dZ` buffer)* | | |
| 4 | `k_relu_deriv<<<1,256>>>(Z, dZ, 4)` | zero the entries where Z ≤ 0 (only Z[1][1] = −4) | dZ = [[1, 1], [1, 0]] |
| 5 | `mm_tiled_AT<<<g2(2,2), …>>>(A_prev, dZ, dW, 2, 2, 2)` | meant: C = Aᵀ·B | should be dW = [[1, 3], [2, 0]]·[[1, 1], [1, 0]] = [[4, 1], [2, 2]]; **the kernel as written returns [[3, 1], [3, 3]] = A_prev·dZ** (warning below) |
| 6 | `k_bias_grad<<<1,256>>>(dZ, db, 2, 2)` | thread j sums column j | db = [2, 1] |
| 7 | `mm_tiled_BT<<<g2(2,2), …>>>(dZ, W, dA_prev, 2, 2, 2)` | C = A·Bᵀ | dA_prev = [[1, 1], [1, 0]]·[[1, 0], [−1, 2]] = [[0, 2], [1, 0]]: the G for the layer below |
| 8 | `k_sgd<<<1,256>>>(W, dW, 0.1, 4)` | | W = [[0.6, −1.1], [−0.2, 1.8]] (b likewise) |

**How the transposes happen without a transpose.** `mm_tiled_BT`: thread (tx, ty) stores `sB[ty][tx] = B[col·K + t·TS + ty]` = W[col][k] = (Wᵀ)[k][col], so `sB` holds the tile of Wᵀ and the inner loop is the ordinary tiled product. Nothing is copied; the cost is that this load is no longer lane-adjacent (lanes `tx` are K floats apart: §3.6, "why backward is 3.4× the forward").

> [!danger] `mm_tiled_AT` does not compute Aᵀ·B (found while writing this example) **[RECON, CPU port; not run on the GPU]**
> `mlp_bench.cu:117–118`: thread (tx, ty) stores `sA[tx][ty] = A[(t·TS + tx)·M + row]` with `row = by·TS + ty`, so `sA[i][j]` = A[t·TS + i][by·TS + j]: the **untransposed** tile of A. The inner loop then reads `sA[ty][k]` as if it held Aᵀ. Two transposes cancel inside each 32 × 32 tile, while the tile *positions* are transposed. Result: within one tile it computes A·B (the toy above: [[3, 1], [3, 3]] instead of [[4, 1], [2, 2]]); at real sizes it computes a block-scrambled product. A line-for-line Python port gives the wrong answer on every random shape tried; `mm_tiled` and `mm_tiled_BT` ported the same way are correct.
> - **Consequence:** every dW (dW1 … dW4) in the training step is wrong. **Not affected:** the memory numbers, the timings (same FLOPs and loads), and the baseline-vs-checkpointed checksum equality (both modes use the same wrong kernel, which is exactly why that check couldn't catch it, as §3.5 warns).
> - **Fix (verified on the same port for TS = 2, 4, 32 and non-square shapes):** load with the indices swapped, `aR = t·TS + ty`, column `by·TS + tx`: `sA[tx][ty] = (aR < K && by*TS + tx < M) ? A[(size_t)aR * M + by*TS + tx] : 0.f;`. As a bonus, lanes `tx` then read adjacent floats, so the load becomes coalesced.
> - **In an interview:** "A per-tensor gradient check against PyTorch would have caught it; that's the first thing I'd add."

A loss example for step 0 of backward: Z4 = [[1, 2], [3, −4]], Y = [[0, 2], [1, 0]], n = 4 → `k_mse_grad` gives dZ4 = 2(Z4 − Y)/4 = [[0.5, 0], [1, −2]].

### 3.5 Measuring memory and correctness

- **Peak memory (`:358–360`, `:546–550`)**: `cudaMemGetInfo` before allocating anything and again after **every** buffer, arena included, is allocated. Every buffer is a `cudaMalloc` made up front and nothing is allocated inside the loop, so footprint = peak, deterministically. Caveat: it's device-wide, so another process allocating on the same GPU would contaminate it. **[RECON]**
- **Correctness (`:570–579`)**: both modes run 3 warm-up + 100 timed steps (103 SGD updates) from the same seed; the sum of W4 matches to 10 digits (−379.7093289 at N4096; −84.09, −259.73 in the other configs). Both modes run identical kernels in identical order, so bitwise equality is expected; it's a modest test (one tensor). A per-tensor gradient comparison after one step would be stronger. **[MEAS] [RECON]**
- **Weights scaled by 1/√fan_in (`:396–404`, story S4)**: with U(−1, 1) weights on 4096-wide layers the training loop went to inf/NaN in a few steps, and NaN ≠ NaN made the checksum comparison meaningless.

### 3.6 The cost, derived, and one thing found while preparing

**Cost [THEORY → MEAS]:** in units of a 4096³ GEMM, the forward layers are 0.25 + 1 + 1 + 0.25 = 2.5; backward recomputes layer 3 (1) + layers 1–2 (0.25 + 1) = 2.25. 2.25 / 2.5 = 0.9 × 39.5 ms forward = **35.5 ms** predicted; **35.54 ms** measured (+26.1%). Configs 1 and 2: +23.8% and +24.2%.

**Why backward is 3.4× the forward, not ~2× [RECON]:** backward does two GEMMs per layer (~2× the FLOPs), but its transposed-operand kernels `mm_tiled_AT` and `mm_tiled_BT` load one operand uncoalesced.

**Found while preparing: the segment-B recompute is redundant in this schedule [RECON, not measured].** In checkpointed mode the forward pass writes z3, a3 into `rZ3`, `rA3` (`:468–469`, `:479–481`), and nothing writes the arena between the end of forward and the start of backward (Z4 has its own buffer). So at `:495–497` the arena **already holds** valid z3, a3, and recomputing them repeats the layer-3 GEMM. Skipping it would save about 1 of the 2.25 recompute units: ≈ 0.4 × 39.5 ≈ **16 ms** of the 35.5 ms overhead, with the same 272 MiB. This holds because forward and backward run back to back in this harness; a framework with other work between them (or a deeper net with more segments) would need the recompute. It's a good "what would you improve" answer, and a good sign you read your own code critically. Don't present it as measured.

### 3.7 Question bank: checkpointing

**L1. What is gradient checkpointing?**
> [!success]- Answer
> Trading compute for memory in training: during the forward pass keep only some activations (checkpoints) and discard the rest; during backward, recompute the discarded ones from the nearest checkpoint just before they're needed. Memory falls (to O(√n) for n layers with √n spacing); compute rises by about one extra forward pass. In my 4-layer MLP the checkpoints were A2 and Z4, and the cost was +26% on the backward pass.

**L2. Why A2 and Z4?**
> [!success]- Answer
> The assignment prescribed them, and they make sense: Z4 is the network output, which the MSE gradient needs first; A2 splits the network into two 2-layer segments and is also the input layer 3's weight gradient needs (dW3 = A2ᵀ·dZ3). Everything else (Z1, A1, Z2, Z3, A3) is recomputed.

**L2. Why did the first version save 0 MB?**
> [!success]- Answer
> It gave each recomputed tensor its own permanent buffer. Five 64 MiB buffers = 320 MiB, plus A2 and Z4 = 400 MiB: exactly what the baseline holds. Recomputation only saves memory if the recomputed tensors **share** storage, which works here because segment B is fully consumed before segment A is rebuilt. One arena of max(128, 192) = 192 MiB serves both. That's story S3, and the lesson I'd generalise: memory is saved by *lifetimes not overlapping*, not by recomputing.

**L2. Explain the +26% from first principles.**
> [!success]- Answer
> Count GEMM work. Forward: layer 1 is 4096 × 1024 × 4096 (0.25 of a 4096³ GEMM), layers 2 and 3 are 1 each, layer 4 is 0.25: 2.5 units, 39.5 ms. Backward recomputes layers 3 (1 unit) and 1–2 (1.25 units): 2.25 units = 0.9 × 39.5 = 35.5 ms. Measured +35.54 ms on a 136.15 ms backward. The overhead is exactly the recomputed forward work, nothing else.

**L3. "Is 13.3% 'peak activation memory'?"**
> [!success]- Answer
> No, and my resume wording is wrong there. 13.3% is the reduction of the whole training-step footprint, 962 → 834 MiB, which includes 320 MiB of weights and gradients, 208 MiB of dZ buffers and 32 MiB of X/Y that checkpointing can't touch. Activation memory alone fell 400 → 272 MiB, 32%. I'd say "32% of activation memory, 13.3% of total".

**L3. "Could you have saved more memory without any recomputation?"**
> [!success]- Answer
> Yes. ReLU's derivative only needs the sign of Z, and A = ReLU(Z) is positive exactly where Z is, so the backward can use A instead of Z for the mask. That lets you drop Z1, Z2 and Z3 entirely: 192 MiB at zero time cost, more than checkpointing saved here. A 1-bit mask would be smaller still, and ping-ponging the dZ buffers would save ~80 MiB more. I followed the assignment's design but I'd point this out. [RECON, prep Q27]

**L3. "Why can checkpointing reduce memory but increase runtime?"**
> [!success]- Answer
> Because it buys memory with compute: activations that would have been read back from memory in backward are recomputed by running forward layers again. The memory saving is the size of the dropped tensors (if their lifetimes don't overlap with the recompute buffers); the time cost is the FLOPs of the recomputed layers, 0.9 of a forward here (+35.5 ms). It's worth it when memory is the binding constraint (bigger batch, longer sequence, deeper model) and the recompute is cheaper than the alternative (smaller batch, offloading).

**L3. "How do you know the recomputed activations are correct?"**
> [!success]- Answer
> The two modes run 103 SGD steps from the same seed, and W4's checksum matches to 10 digits in all three configs. W4's update depends on the recomputed A3, and over many steps on the whole chain, so a wrong recompute would diverge. It's a modest check: one tensor, and identical kernels in identical order make bitwise equality expected. A stronger test is comparing every gradient tensor after a single step.

---

## 4. FlashAttention forward (PA4)

### 4.1 Component card

| | |
|---|---|
| **Problem** | O = softmax(QKᵀ/√D)·V. The score matrix is N × N per (batch, head): 8 · 16 · 8192² · 4 B = **34.4 GB** at the benchmark size, so the PyTorch explicit path must be batch-chunked (chunk 2 under a 16 GB budget) |
| **Optimisation [IMPL]** | tile over keys; per query row keep a running max m, sum l and output o; rescale when a larger max appears (online softmax); scores live only in registers and a 4 KB shared tile; never written to HBM |
| **Kernel layout [IMPL]** | grid (Tr = 256, NH = 16, B = 8) = 32,768 blocks; block (32, 8) = 8 warps; warp w owns query rows 4w … 4w+3; 32-key tiles; shared Qi 32×64 · Kj 32×65 · Vj 32×64 · S 32×32 = **28,800 B** |
| **The measured story [MEAS, history from notes]** | quiz layout 3,358 ms → grid over query tiles 2,377 (1.4×) → **pad Kj stride to D + 1: 339 (7.0×)** → 4 rows per warp: **160 ms** (2.1×) = 21× total |
| **vs PyTorch [MEAS]** | 160.05 ms vs explicit 105.08 (1.5× slower) and fused SDPA 62.62 (2.6× slower); faster only at N ≤ 128 (launch-bound regime); max abs error ≤ 7.2e-7 |
| **Bottleneck now [THEORY]** | shared-memory bandwidth: both GEMMs read operands from shared memory (~1.25 loads per FMA); measured 13.75 TFLOP/s vs a ~13.4 model roof |
| **Trade-offs** | padding costs 128 B per block and breaks 16-byte row alignment (no `LDS.128` on Kj); 62 registers → 50% occupancy; `o_acc` in local memory |
| **Limitations** | forward only, no causal mask, FP32 CUDA cores, D ≤ 128, ctypes ABI with no checks, default stream only |

### 4.2 Online softmax, derived

**Safe softmax:** softmax(s)ᵢ = e^(sᵢ − m) / Σⱼ e^(sⱼ − m) with m = max s, so every exponent ≤ 0. Needs the row max first. **Online softmax** removes that need. Keep for each row, after the tiles seen so far, with m = the max so far:

$$l = \sum_{\text{seen } j} e^{s_j - m}, \qquad o = \sum_{\text{seen } j} e^{s_j - m}\, v_j$$

For a new tile with max $m_t$, sum $l_t = \sum e^{s - m_t}$ and $pv_t = \sum e^{s - m_t} v$:

$$m' = \max(m, m_t),\quad l' = e^{m - m'}\,l + e^{m_t - m'}\,l_t,\quad o' = e^{m - m'}\,o + e^{m_t - m'}\,pv_t$$

Multiplying the old sums by $e^{m - m'}$ turns every $e^{s_j - m}$ into $e^{s_j - m'}$; the tile's terms get $e^{m_t - m'}$. All terms are now relative to the same max, so by induction **o / l at the end is exactly softmax(s)·v**. In the code, `e_run` = $e^{m - m'}$ and `e_tile` = $e^{m_t - m'}$.

![[d6-7-online-softmax.svg]] [[d6-7-online-softmax.html|▶ Step through it]]

![[d06m-online-softmax.mp4]]
*Manim video: the §4.2 toy row; tile 1's weights shrink by e^(2 − 3) when the bigger max arrives (45 s). Pause and scrub with the video controls.*

**The same toy inside the real code** (one row, two 2-key tiles, s = [2, 1 | 3, 0], v = [1, 3 | 2, 0]; answer 1.786):

![[d06i-flash-dbg.svg]] [[d06i-flash-dbg.html|▶ Step through it]]

### 4.3 One block, start to finish

![[d06i-flash-tiles.svg]] [[d06i-flash-tiles.html|▶ Step through it]]

### 4.4 The host launch, line by line (`flash_lib.cu:189–214`)

```cpp
189  int flash_attention_forward(const float* Q, const float* K, const float* V,
190                              float* O, int B, int NH, int N, int D)
191  {
192      if (D > 128) return -1;      // o_acc holds ceil(D/32) <= 4 feature slots
194      const int Tr = (N + BR - 1) / BR;
195      const int Tc = (N + BC - 1) / BC;
196      const float scale = 1.0f / sqrtf((float)D);
198      size_t shared = ((size_t)BR * D + (size_t)BC * (D + 1) + (size_t)BC * D
199                       + (size_t)BR * BC) * sizeof(float);
201      static int configured = 0;
202      if (shared > 48 * 1024 && !configured) {
203          cudaError_t e = cudaFuncSetAttribute(flash_forward_kernel,
205              cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
206          if (e != cudaSuccess) return (int)e;
207          configured = 1;
208      }
210      dim3 grid(Tr, NH, B);
211      dim3 block(WARP, WPB);
212      flash_forward_kernel<<<grid, block, shared>>>(Q, K, V, O, B, NH, N, D, Tc, scale);
213      return (int)cudaGetLastError();
214  }
```

| line | what | value at N = 8192, D = 64 |
|---|---|---|
| 189 | `extern "C"` entry point: raw device pointers, called from Python by ctypes | no dtype / contiguity / device checks [RECON] |
| 192 | reject D > 128 | each lane holds at most 4 output features (x, x + 32, x + 64, x + 96) |
| 194–195 | query tiles and key tiles | Tr = Tc = 256 |
| 196 | 1/√D | 0.125 |
| 198–199 | dynamic shared size: Qi + Kj (padded) + Vj + S | (2,048 + 2,080 + 2,048 + 1,024) × 4 = **28,800 B** |
| 201–208 | opt in above 48 KB, once | never fires at D = 64 |
| 210–211 | grid (256, 16, 8), block (32, 8) | 32,768 blocks × 256 threads |
| 212 | launch on the **default stream** with dynamic shared memory | correct only because PyTorch's current stream is also the default stream here [RECON] |
| 213 | returns launch errors to Python | (Day 5's matmul harness lacked this) |

#### How it is called, on an example

**The real call chain** (Python → C → kernel) **[IMPL]**:
```python
q = torch.randn(B, NH, N, D, device=dev, dtype=torch.float32)    # bench_attention.py:135  (8, 16, 8192, 64)
got = custom_flash(lib, q, k, v)                                  # :147

def custom_flash(lib, q, k, v):                                   # :46–55
    B, NH, N, D = q.shape
    o = torch.empty_like(q)
    rc = lib.flash_attention_forward(
        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(o.data_ptr()),
        B, NH, N, D)
```
`data_ptr()` is the GPU address of the tensor's first float. ctypes hands the four addresses and four ints to the C function above, which computes the launch shape and launches. The tensors must be contiguous float32 [B, NH, N, D] on the GPU, so element (b, h, n, x) is at flat index `((b·NH + h)·N + n)·D + x`; nothing checks it.

**The maths:** for each (b, h) independently, O[b, h] = softmax(Q[b, h]·K[b, h]ᵀ / √D)·V[b, h], each an N × D matrix. Block (qt, h, b) computes **rows 32·qt … 32·qt + 31** of O[b, h]; warp w of the block computes 4 of those rows (4w … 4w + 3).

**Hand-sized call [THEORY]:** tensors of shape (1, 2, 4, 2), so `flash_attention_forward(q, k, v, o, 1, 2, 4, 2)`: B = 1, NH = 2, N = 4, D = 2.

| host line | value |
|---|---|
| `Tr = Tc = (4 + 31) / 32` | 1 query tile, 1 key tile |
| `scale = 1 / sqrtf(2)` | 0.7071 |
| `shared = (32·2 + 32·3 + 32·2 + 32·32)·4` | 4,992 B |
| `grid(Tr, NH, B)` | (1, 2, 1): block (0, 0, 0) is head 0, block (0, 1, 0) is head 1 |
| `block(WARP, WPB)` | (32, 8) = 256 threads, as always |

Head 1's block starts at `base = (0·2 + 1)·4·2 = 8`: floats 8–15 of each flat tensor. Follow head 0 (base = 0):

$$Q = \begin{bmatrix}1&0\\0&1\\1&1\\0&0\end{bmatrix},\quad K = \begin{bmatrix}1&0\\0&1\\1&1\\-1&0\end{bmatrix},\quad V = \begin{bmatrix}1&0\\0&1\\2&2\\0&-1\end{bmatrix}$$

**Who does what in block (0, 0, 0):**
- **Warp 0** owns query rows 0–3, all live. **Warps 1–7** own rows 4–31, all ≥ N: `live[r]` is false, they stage zeros, compute on them, and skip the final store. At N = 4, 7 of 8 warps are wasted: the ragged-tile cost at tiny N.
- **Score phase:** lane c < `kv_len` = 4 owns **key c**. Lane 2, for example, computes the scores of key 2 = [1, 1] against all 4 rows of warp 0: q·k = 1, 1, 2, 0 → × 0.7071 = 0.707, 0.707, 1.414, 0 (column 2 of the table below). Lanes 4–31 get s = −∞, so p = 0.
- **PV phase:** lane x < D = 2 owns **output feature x**; lanes 2–31 skip the loop (`x = lane ≥ D`).

**The values** (rounded; each row: s = q·kᶜ / √2 for keys c = 0…3, then m = max, p = e^(s − m), l = Σp, pv = Σ p·vᶜ, O = pv / l):

| row | s (keys 0–3) | m | p | l | pv (features 0, 1) | **O** |
|---|---|---|---|---|---|---|
| 0 | 0.707, 0, 0.707, −0.707 | 0.707 | 1, 0.493, 1, 0.243 | 2.736 | 3.000, 2.250 | **1.096, 0.822** |
| 1 | 0, 0.707, 0.707, 0 | 0.707 | 0.493, 1, 1, 0.493 | 2.986 | 2.493, 2.507 | **0.835, 0.840** |
| 2 | 0.707, 0.707, 1.414, −0.707 | 1.414 | 0.493, 0.493, 1, 0.120 | 2.106 | 2.493, 2.373 | **1.184, 1.127** |
| 3 | 0, 0, 0, 0 | 0 | 1, 1, 1, 1 | 4 | 3, 2 | **0.75, 0.50** |

- The **m** column is the 5-step `__shfl_xor_sync` max across lanes 0–31; **l** is the shuffle sum. The **p** row is what warp 0 writes into its 4 rows of `S`; the PV phase reads it back.
- With a single key tile the online-softmax bookkeeping is trivial: `m_run` starts at −∞, so `e_run` = e^(−∞) = 0 and `e_tile` = 1, giving `o_acc` = pv and `l_run` = l. With N > 32 there are more key tiles and the rescaling of §4.2 starts to matter (the d06i-flash-dbg animation traces a two-tile row).
- Sanity check: row 3 is a zero query, so all scores are equal and O is the plain mean of V's rows, [3/4, 2/4] ✓.

At the real size, `flash_attention_forward(…, 8, 16, 8192, 64)` gives Tr = Tc = 256, grid (256, 16, 8) = 32,768 blocks, and the same per-block roles, with 256 key tiles per block instead of 1.

### 4.5 The kernel, part by part (`flash_lib.cu:47–184`)

**Setup (`:55–78`)**
```cpp
const int KS = D + 1;             // padded Kj row stride, see header
extern __shared__ float smem[];
float* Qi = smem;                 // BR * D
float* Kj = Qi + BR * D;          // BC * KS
float* Vj = Kj + BC * KS;         // BC * D
float* S  = Vj + BC * D;          // BR * BC
const int lane = threadIdx.x;     // 0..31
const int w    = threadIdx.y;     // warp index within block, 0..7
const int tid  = w * WARP + lane;
const int qt = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
const size_t base = ((size_t)b * NH + h) * N * D;
const int r0 = w * RPW;           // first query row of this warp
const int qbase = qt * BR;
bool live[RPW]; for (r) live[r] = (qbase + r0 + r) < N;   // ragged last tile
```
With block (qt = 10, h = 3, b = 0), warp 2, lane 7: `base = 3 · 8192 · 64 = 1,572,864` (where head 3's Q, K, V start), `r0 = 8`, `qbase = 320`, so this warp owns query rows **328–331**. One dynamic buffer is carved into four arrays by pointer offsets.

**Stage Q and initialise (`:81–96`)**: all 256 threads copy the 32 × 64 Q tile (`idx = tid; idx < 2048; idx += 256`, 8 elements each, consecutive threads → consecutive addresses: coalesced). Then per row: `o_acc[r][0..3] = 0`, `m_run[r] = −INFINITY`, `l_run[r] = 0`.

**Loop over key tiles (`:98–110`)**
```cpp
for (int j = 0; j < Tc; ++j) {
    const int kv_base = j * BC;
    int kv_len = N - kv_base;  if (kv_len > BC) kv_len = BC;     // ragged last tile
    __syncthreads();                                            // nobody still reads the old tile
    for (int idx = tid; idx < kv_len * D; idx += NTHREADS) {
        int r = idx / D, x = idx - r * D;
        const size_t g = base + (size_t)(kv_base + r) * D + x;
        Kj[r * KS + x] = K[g];                                  // padded stride 65
        Vj[r * D  + x] = V[g];
    }
    __syncthreads();                                            // tile complete
```
The two barriers have the same two jobs as in the matmul tile loop (Day 5 §4.5). A warp's 32 consecutive `idx` share r and have consecutive x: consecutive global addresses (coalesced) and consecutive shared words (no store conflict, even with stride 65).

**Scores (`:114–126`)**: lane c owns **key c**; x is the outer loop so one Kj load feeds all 4 rows:
```cpp
float a[RPW] = {0};
if (lane < kv_len) {
    const float* kptr = Kj + lane * KS;
    for (int x = 0; x < D; ++x) {
        const float kv = kptr[x];                                   // Kj[c·65 + x]
        for (int r = 0; r < RPW; ++r)
            a[r] = fmaf(Qi[(r0 + r) * D + x], kv, a[r]);            // Qi: same address for all lanes
    }
}
```
**Online softmax (`:130–152`)**, per row r: `s = a[r]·scale` (or −∞ past `kv_len`); tile max by 5 `__shfl_xor_sync` steps; `p = __expf(s − m_tile)`; tile sum by 5 more shuffles; `S[(r0 + r)·32 + lane] = p`; then `m_new`, `e_run`, `e_tile`, `l_run`, `m_run` exactly as in §4.2.

![[d6-8-shuffle.svg]] [[d6-8-shuffle.html|▶ Step through it]]

`__shfl_xor_sync(0xffffffff, v, off)` returns `v` from lane `lane XOR off`, register to register. With offsets 16, 8, 4, 2, 1 every lane ends with the warp's max (or sum) in 5 steps: no shared memory, no barrier. `0xffffffff` = all 32 lanes participate (required: lanes past `kv_len` still run the shuffle with −∞ / 0).

**`__syncwarp()` (`:153`)**: each warp writes and later reads only **its own 4 rows** of S, so only its 32 lanes need to agree; a block-wide `__syncthreads()` would make warps wait for each other for nothing.

**PV (`:157–172`)**: lane x now owns **features** x and x + 32:
```cpp
int jj = 0;
for (int x = lane; x < D; x += WARP, ++jj) {
    float pv[RPW] = {0};
    for (int c = 0; c < kv_len; ++c) {
        const float vv = Vj[c * D + x];                             // adjacent x → adjacent banks
        for (int r = 0; r < RPW; ++r)
            pv[r] = fmaf(S[(r0 + r) * BC + c], vv, pv[r]);           // S: broadcast
    }
    for (int r = 0; r < RPW; ++r)
        o_acc[r][jj] = o_acc[r][jj] * e_run[r] + pv[r] * e_tile[r];
}
```
**Finish (`:175–183`)**: for each live row, `O[base + qrow·D + x] = o_acc[r][jj] · (1 / l_run[r])`: divide once, at the end.

**The role switch** is the design's key idea: in the score phase lanes own keys (so the 32 scores of a tile row are spread across lanes, ready for a warp reduction); in the PV phase lanes own output features (so each lane accumulates its own output columns with no reduction). S in shared memory is the hand-off between the two roles.

### 4.6 The bank conflict, and why D + 1 fixes it (7.0×)

![[d6-9-banks.svg]] [[d6-9-banks.html|▶ Step through it]]

![[d06m-bank-conflict.mp4]]
*Manim video: Kj words slide from stride 64 to stride 65, and the 32-request pile-up on bank 5 flattens (50 s). Pause and scrub with the video controls.*

In the score loop, at a fixed feature x, lane c reads `Kj[c·stride + x]`.
- **Stride 64 (= D):** bank = (64c + x) mod 32 = **x mod 32** for every c. 32 different addresses in one bank → **32-way conflict**, on every one of the 64 feature steps of the hottest loop.
- **Stride 65 (= D + 1):** bank = (65c + x) mod 32 = **(c + x) mod 32**: all 32 different → conflict-free. Lanes 0, 1, 31 at x = 5: banks 5, 6, 4.

Measured **2,377 → 339 ms (7.0×)** from that one line (S1). Not 32×: the conflict was in the hottest loop but not the only cost.

**Every other shared access, checked [RECON]:**

| access | across the warp | conflict? |
|---|---|---|
| `Kj[r·65 + x]` store | r fixed, x consecutive | none |
| `Kj[c·65 + x]` score read | (c + x) mod 32 all different | none (was 32-way) |
| `Qi[(r0 + r)·64 + x]` | same address | broadcast |
| `S[(r0 + r)·32 + lane]` store | consecutive | none |
| `Vj[c·64 + x]`, lane owns x | consecutive x | none → V needs no padding |
| `S[(r0 + r)·32 + c]` read | same address | broadcast |

Correctness is unchanged because the store and the load use the **same** stride. Alternatives: an XOR **swizzle** (`col ^ f(row)`) removes conflicts without wasting a column and keeps alignment for vector loads (CUTLASS does this); storing K transposed moves the conflict to the store, which happens once per tile instead of 64 times.

### 4.7 After the fix: the shared-memory roof, and 4 rows per warp (2.1×)

At one query row per warp, each score FMA needed 2 shared loads (a Kj value and a Qi broadcast). By the Day 5 model, that caps the SM near **8.4 TFLOP/s**; the kernel measured **6.5 TFLOP/s** (S2). With 4 rows per warp, one Kj load feeds 4 FMAs (4 Qi broadcasts + 1 Kj = 5 loads per 4 FMAs): the model roof rises to ~**13.4**, and the kernel measured **13.75 TFLOP/s** (339 → 160 ms). It is on the shared roof again. The same idea as Day 5's register blocking, applied over query rows. **[THEORY] [MEAS]**

### 4.8 Resources: registers, occupancy, local memory

![[d05i-occupancy.svg]] [[d05i-occupancy.html|▶ Step through it]]

62 registers (rounded to 64) × 256 threads = 16,384 per block → 4 blocks per SM → **50% theoretical occupancy** (occupancy API; `cuobjdump` agrees). Shared memory (28,800 B per block) would allow ~7. **Registers limit it.** `__launch_bounds__(256)` on the kernel tells the compiler the block size so it can budget registers.

**`o_acc` lives in local memory [RECON, SASS]:** `cuobjdump` shows `STACK:64` and `LDL`/`STL`. `o_acc[4][4]` is exactly 64 bytes, and it's indexed by `jj`, which depends on the runtime D, so the compiler can't map it to registers. The traffic is once per KV tile, not in the inner loop, so the cost is probably small. Fix: make D a template parameter. **[FUTURE]**

### 4.9 Why it's still slower than PyTorch

- PyTorch's explicit path does two big GEMMs in cuBLAS, heavily register-blocked and not bound by shared bandwidth; your kernel reads both operands of both matmuls from shared memory (~1.25 loads per FMA). It's the same ~2× gap as Day 5's 48% of cuBLAS.
- No tensor cores, FP32. FlashAttention's speed win in the paper comes when attention is HBM-bound (FP16 tensor cores, long sequences); an FP32 CUDA-core kernel without deeper register tiling never reaches that regime.
- The "crossover" at N ≤ 128 is PyTorch's launch overhead (a ~0.11 ms floor from several kernel launches), not your kernel being better.
- **Its real value: memory.** No 34.4 GB tensor; one call instead of a Python loop over batch chunks.

### 4.10 Question bank: FlashAttention

**L1. Why does FlashAttention avoid materialising the full attention matrix?**
> [!success]- Answer
> Because it is O(N²) memory per head and would have to be written to and read back from HBM. At B = 8, 16 heads, N = 8192 in FP32 it's 34.4 GB, more than the 16 GB budget my PyTorch baseline had, so the explicit path had to run in batch chunks of 2. Online softmax makes the full matrix unnecessary: each key tile's scores are used immediately (max, sum, P·V) and discarded, and only O(N) state (m, l, o per row) persists. Memory becomes linear in N, and the result is still exact.

**L1. What is a warp shuffle, and why use it here?**
> [!success]- Answer
> `__shfl_xor_sync(mask, v, off)` lets each lane read a register of another lane in its warp directly. A butterfly with offsets 16, 8, 4, 2, 1 reduces 32 values to their max (or sum) in 5 steps, leaving the result in every lane. I use it for the per-row tile max and sum because the 32 scores of a tile row live in 32 lanes: no shared memory traffic and no barrier.

**L2. Why `__syncwarp()` and not `__syncthreads()` after writing S?**
> [!success]- Answer
> Each warp writes p into its own 4 rows of S and only that warp reads them in the PV phase. The only threads that must agree are the 32 lanes of that warp, so a warp-level barrier is enough. A block barrier would make all 8 warps wait for the slowest for no reason. The block-wide barriers are still needed around the Kj/Vj tile loads, which all warps share.

**L2. Why 4 query rows per warp?**
> [!success]- Answer
> At 1 row per warp each FMA needed 2 shared loads (Kj and Qi), which pinned the kernel to the shared-memory roof (~8.4 TFLOP/s by my model; measured 6.5). With 4 rows per warp and x as the outer loop, one Kj load feeds 4 FMAs: 5 loads per 4 FMAs, a ~13.4 roof, and I measured 13.75 (339 → 160 ms). More rows per warp would raise reuse again but costs registers (`a[RPW]`, `o_acc[RPW][4]`, softmax state) and already sits at 62 registers / 50% occupancy.

**L2. Why did the grid change (128 → 32,768 blocks) give only 1.4×?**
> [!success]- Answer
> Because the bank conflict was the real limit. With 32-way serialised Kj reads in the hottest loop, more blocks just meant more warps queued on the same serialised shared memory. Going from 128 blocks (fewer than the 132 SMs) to 32,768 fixed under-utilisation, but each SM was then bottlenecked on conflicts. The 7.0× came only when the padding removed them.

**L3. "Tell me about the bank conflict."** (story S1)
> [!success]- Answer
> After moving to a grid over query tiles I expected a big win and got 1.4×. Without a profiler (ncu needed root), I wrote out the shared addresses of the hot loop: lane c reads `Kj[c·64 + x]` at feature x, so the bank is (64c + x) mod 32 = x mod 32, the same for all 32 lanes: a 32-way conflict on every feature step. I changed the row stride to 65; the bank becomes (c + x) mod 32, all different. One line, 2,377 → 339 ms, 7.0×. Correctness unchanged because the store and load use the same stride. Then I checked every other shared access in the kernel for the same problem; V didn't need padding because there lanes own features.

**L3. "Is 21× the bank-conflict fix?"**
> [!success]- Answer
> No. 21× is cumulative from my first (quiz-layout) version over three changes: 1.4× from the grid, 7.0× from padding, 2.1× from 4 rows per warp. It's against my own first version, not against PyTorch, and the intermediate versions aren't in git (single commit); only the final 160 ms is reproducible.

**L3. "Your FlashAttention is slower than PyTorch. So what did you achieve?"**
> [!success]- Answer
> An exact attention kernel whose memory is linear in sequence length: at N = 8192 it needs no 34.4 GB score tensor and runs in one call, while the explicit path has to loop over batch chunks. On speed it's 1.5× slower than PyTorch's cuBLAS-based explicit path and 2.6× slower than fused SDPA, for a reason I can quantify: it's on the shared-memory roof (13.75 TFLOP/s against a ~13.4 model), the same register-blocking gap as my matmul's 48% of cuBLAS. The next steps are tensor cores and FlashAttention-2's work partitioning.

**L3. "What happens if N isn't a multiple of 32?"**
> [!success]- Answer
> Both directions are handled. Keys: `kv_len` = min(32, N − j·32) for the last tile; only `kv_len·D` elements are loaded; lanes ≥ `kv_len` get s = −∞ (so p = e^(−∞ − m) = 0) and skip the score loop. Queries: `live[r]` marks rows past N; the Q staging loads zeros for them and the final store skips them. The Day 6 rebuild exercise (NumPy online-softmax attention) uses N = 1000 for exactly this reason. [IMPL]

**L3. "What would you do next?"**
> [!success]- Answer
> [FUTURE] (1) Make D a template parameter so `o_acc` moves to registers. (2) Use tensor cores (FP16/BF16 `mma.sync`) for QKᵀ and PV, with FP32 accumulation and softmax. (3) FlashAttention-2's changes: split work across warps by query rows without shared partial results (already partly done), and cut non-matmul work. (4) Prefetch the next K/V tile with `cp.async` (double buffering). (5) Replace padding with a swizzle so Kj can be read with vector loads. (6) Add the causal mask and the backward pass (recompute scores from Q, K, V and the saved logsumexp).

---

## 5. Performance analysis for Part 2

### 5.1 How each benchmark is timed

| benchmark | warm-up | timed | how | caveats **[RECON]** |
|---|---|---|---|---|
| PA3 part 1 (streams) | 3 iterations + `cudaDeviceSynchronize` | 20 iterations between one event pair, mean | `measure` (`:301–311`) | stream variants sync the host inside the loop; baseline doesn't |
| PA3 part 2 (checkpointing) | 3 forward + backward | 100 iterations, events e0/e1/e2 around forward and backward **per iteration** (`:557–568`) | separate forward and backward means | host sync every iteration (fine: each step is ~200 ms) |
| PA4 (attention) | 3 | 10 per path, CUDA events | `timeit` in `bench_attention.py:81` | the explicit path loops over batch chunks in Python at 8K, so it includes some Python overhead |

All: one sweep, no repeats, no error bars, clocks not locked.

### 5.2 Latency vs throughput, in Part 2

- **Launch latency dominates small problems.** At N = 32 the whole 2-layer forward takes 6.9 µs; adding 6 launches and 4 host syncs makes it 4× slower. PyTorch attention has a ~0.11 ms floor below N = 256, which is why your kernel "wins" at N ≤ 128. Neither says anything about throughput.
- **Throughput dominates large problems.** At N = 32768 the forward pass moves ~275 GB of W requests; what matters is how many of them reach HBM. At N = 8192 attention is 2.2 TFLOP of FMAs; what matters is FLOPs per shared-memory load.
- **Checkpointing is a throughput-for-memory trade:** +35.5 ms per step for −128 MiB.

### 5.3 Arithmetic intensity and roofs for each piece

| piece | intensity [THEORY] | roof [THEORY] | measured [MEAS] | binding limit |
|---|---|---|---|---|
| `mm_thin` (streams) | ~0.5 FLOP/B at L1/L2 (one W load per FMA) | cache bandwidth | 3.1 (baseline) and 5.4 TFLOP/s (column split), i.e. ~6.2 and ~10.7 TB/s of W requests | memory hierarchy; how often W reaches HBM |
| `mm_tiled` (training) | 8 FLOP/B HBM, 2 shared reads per FMA | ~8.4 TFLOP/s shared roof (Day 5) | forward 39.5 ms for 2.5 units of 4096³ GEMMs ≈ 8.7 TFLOP/s | shared memory, same as Day 5's 1-to-1 |
| FlashAttention, 1 row/warp | 2 shared loads per FMA | ~8.4 | 6.5 TFLOP/s | shared bandwidth (after the conflict fix) |
| FlashAttention, 4 rows/warp | 5 loads per 4 FMAs | ~13.4 | 13.75 TFLOP/s | shared bandwidth again |

(`mm_tiled` row: 2.5 × 2 × 4096³ = 3.4 × 10¹¹ FLOPs in 39.5 ms ≈ 8.7 TFLOP/s, consistent with Day 5's 8.64 for the same kernel shape. This is a derived number, not in the CSV.)

### 5.4 Memory–time trade-offs: how to reason about them

- **Checkpointing:** memory saved = bytes of dropped tensors whose lifetimes no longer overlap; time paid = FLOPs recomputed. Here 128 MiB for 0.9 forward. Worth it when memory caps batch size or model size.
- **FlashAttention:** memory saved = the N² score tensor (34.4 GB); time = recomputation is zero in forward, but the kernel's own efficiency decides speed (yours is slower than cuBLAS-based attention). In backward, FlashAttention recomputes scores from Q, K, V instead of storing them: checkpointing again.
- **Streams:** no memory trade; the column split's win came from better reuse of the same data, not from doing work in parallel.

### 5.5 Measured vs theoretical, Part 2

| claim | theory said | measured | verdict |
|---|---|---|---|
| batch split helps | streams overlap → faster | slower at every N | theory's precondition (idle SMs) false at large N |
| checkpoint cost | 0.9 × 39.5 = 35.5 ms | +35.54 ms | exact |
| checkpoint memory | 400 → 272 MiB activations | 962 → 834 MiB total (same 128 MiB) | exact |
| padding fix | 32-way conflict removed → up to ~32× on that loop | 7.0× overall | other costs remain |
| 4 rows per warp | roof 8.4 → 13.4 | 6.5 → 13.75 TFLOP/s | on the new roof |
| un-padding the final kernel (prep exercise 1) | ~4× slower, ~500–650 ms | not run | prediction only |

### 5.6 Question bank: performance analysis, Part 2

**L2. How would you make the stream timings fair?**
> [!success]- Answer
> Remove the per-iteration host syncs from the stream variants: record one event at the end of each stream's work and have the default stream (or one stream) wait on those events with `cudaStreamWaitEvent`, then sync once after all iterations, the same as the baseline. Or capture each variant's iteration in a CUDA graph and replay it, which also removes most launch overhead. Then compare at small N again; I expect the "4× slower" to shrink a lot. [FUTURE]

**L2. Why is the checkpointing overhead reported on the backward pass, not the whole step?**
> [!success]- Answer
> Because that's where the recompute happens; forward is identical in both modes (39.5 ms). +35.5 ms is +26.1% of backward and about +20% of the whole step (175.7 → 211.2 ms). Both are honest; I report backward because it isolates the cost. [derived from MEAS]

**L3. "How do you know your FlashAttention is shared-memory-bound and not compute-bound?"**
> [!success]- Answer
> It runs at 13.75 TFLOP/s, about 20% of the 66.9 FP32 peak, so the FMA pipes are mostly idle. The model of shared loads per FMA predicted the ceiling at each stage (8.4 at 1 row per warp, 13.4 at 4) and the measurements landed at or just above it twice, and raising FMAs per shared load was the change that moved the time. HBM isn't it either: each block reads its Q tile once and its head's K and V once (about 4 MB), so 32,768 blocks request at most ~137 GB even with zero L2 reuse, against the ~768 GB that 160 ms at 4.8 TB/s could deliver. With ncu I'd confirm with shared-memory pipe utilisation.

**L3. "Why might a faster-looking design (more streams, more blocks) not help?"**
> [!success]- Answer
> Because parallelism only helps if a resource is idle. In PA3 the GPU was already full at large N, so four streams added launches and host syncs; in PA4, going from 128 to 32,768 blocks gave only 1.4× because the new blocks queued on a 32-way serialised shared-memory loop. In both cases the fix was to find the binding resource first (L2 reuse of W; shared-memory banks) and change the data layout or decomposition, not the amount of parallelism.

---

## 6. Say it out loud

### 6.1 The 30-second introduction (Part 2)

> "The second half of my CUDA project was three ML-systems pieces on the H200. For a batch-32 MLP forward pass I tried CUDA streams: splitting the batch never helped, splitting the output columns was 1.73× faster at width 32K, and I believe that's cache locality rather than concurrency. I implemented gradient checkpointing for a 4-layer MLP that cut activation memory by a third for a 26% slower backward pass. And I wrote a FlashAttention forward kernel, where one padding change that removed a 32-way bank conflict gave 7×."

### 6.2 The 2-minute version

> "**Streams.** The MLP has batch 32, so the weight matrix (4.3 GB at width 32K) dominates the traffic. Splitting the batch across four streams makes each stream re-read all of it, so it was slower at every width. Splitting the output columns gives each stream a quarter of the weights: 44 ms down to 26 at 32K. But the single-stream kernel already fills the GPU for about four waves there, so I don't claim concurrency. My explanation is L2 locality: each column-split kernel fits in one wave, so all 32 rows reuse each weight chunk while it's in L2. The experiment that would settle it is running the four chunks on one stream.
>
> **Checkpointing.** I keep A2 and Z4 and recompute the rest in backward. My first version saved zero bytes, because I gave every recomputed tensor its own buffer. The fix: backward finishes with layer 3's activations before it needs layers 1–2, so both segments can share one arena. Activations went 400 → 272 MiB; the backward pass got 35.5 ms slower, exactly the recomputed forward FLOPs.
>
> **FlashAttention.** Tiles of 32 keys, an online softmax with a running max, sum and output per row, and the N × N matrix never exists: at 8K that's a 34 GB tensor avoided. The key bug was a bank conflict: the K tile had a 64-float row stride, so all 32 lanes hit one bank. Padding to 65 gave 7×; four query rows per warp gave another 2×. It's still 1.5× slower than PyTorch, because both of its matmuls read from shared memory, the same gap as my matmul's 48% of cuBLAS."

### 6.3 The 5-minute walkthrough (structure)

1. **Frame (20 s).** Same H200, FP32, custom kernels, CUDA events; three experiments with three different lessons.
2. **Streams (90 s).** Draw X (32 × N) next to W (N × N). Batch split: four colours over the whole W. Column split: four vertical stripes. The table: 44.43 / 52.28 / 25.65 ms. Then waves: 4,096 blocks vs 1,056 per wave, and the locality argument and the one-stream test.
3. **Checkpointing (90 s).** The 4-layer chain with A2 and Z4 boxed; the arena drawn as one box that holds segment A or B; the timeline (forward → rebuild B → grads 4, 3 → rebuild A → grads 2, 1); the 0 MB bug; 400 → 272 MiB and 962 → 834; +35.5 ms = 0.9 forward. Mention the A-instead-of-Z alternative and the redundant segment-B recompute.
4. **FlashAttention (100 s).** Online softmax on the toy (two tiles, rescale by e^(2−3)); the block layout (32 rows, 8 warps × 4 rows, lanes as keys then as features, shuffles); the bank table for stride 64 vs 65; the 3,358 → 2,377 → 339 → 160 ladder; why it's slower than PyTorch and what it saves.
5. **Honesty and next steps (20 s).** One width for 1.73×; Part 1 unchecked; history not in git; next: tensor cores, `cp.async`, FA-2 partitioning.

### 6.4 Whiteboard: the FlashAttention block

```
grid (Tr=256, NH=16, B=8)          one block (qt, h, b): 32 query rows, 8 warps × 4 rows
                                   shared: Qi[32×64] | Kj[32×65] | Vj[32×64] | S[32×32] = 28,800 B
for j in 0..255 (key tiles of 32):
   __syncthreads; load Kj (stride 65), Vj; __syncthreads
   scores   lane c = key c:   a[r] = Σx Qi[r][x]·Kj[c][x]          (1 Kj read → 4 FMAs)
   softmax  per row r:        m_t = warpmax(s), p = e^(s−m_t), l_t = warpsum(p), S[r][c] = p
                              m' = max(m, m_t); e_run = e^(m−m'); e_tile = e^(m_t−m'); l = e_run·l + e_tile·l_t
   __syncwarp
   PV       lane x = feature: o[r][x] = o[r][x]·e_run + (Σc S[r][c]·Vj[c][x])·e_tile
O[r][x] = o[r][x] / l[r]
```
And next to it, the bank line: `stride 64: bank = x mod 32 (all lanes) → 32-way;  stride 65: bank = (c + x) mod 32 → none`.

### 6.5 Likely follow-ups

| follow-up | where |
|---|---|
| "When do streams overlap?" / "Why did the batch split lose?" | §1.4, §2.5 |
| "Prove the 1.73× is concurrency." | §2.3, §2.5 |
| "Is the column split correct?" | §2.5 |
| "Why A2 and Z4? Why 0 MB at first?" | §3.7 |
| "Is 13.3% activation memory?" | §3.7 |
| "Could you save more memory?" | §3.7 |
| "Derive online softmax. Why exact?" | §4.2 |
| "Tell me about the bank conflict." | §4.6, §4.10 |
| "Why slower than PyTorch?" | §4.9 |
| "Why __syncwarp?" / "What are shuffles?" | §4.5, §4.10 |
| "Why ctypes?" | §7.2 I27 |
| "You never profiled. How do you know?" | §7.3 D20 |

### 6.6 Claims to avoid unless your data supports them

| don't say | why | say instead |
|---|---|---|
| "streams gave 1.73×" / "stream parallelism" | mechanism unproven; GPU already full | "partitioning output columns across 4 streams gave 1.73× at width 32K; most likely L2 locality" |
| "at large dimensions" | one width: 1.02× at 16K, slower below | "at width 32K" |
| "MLP inference" as if a real model | synthetic weights, my own kernel as baseline, no PyTorch baseline | "a batch-32, 2-layer MLP forward pass with my own GEMM" |
| "13.3% less activation memory" | it's 13.3% of the total; activations −32% | "32% of activation memory (13.3% of total training memory)" |
| "checkpointing is free" / "small overhead" | +26% backward | "+26% backward, equal to the recomputed forward FLOPs" |
| "my FlashAttention is faster than PyTorch" | 1.5× slower (explicit), 2.6× (fused) | "linear memory, exact; 1.5× slower than PyTorch's explicit path" |
| "faster at short sequences" | only N ≤ 128, PyTorch launch overhead | "only in the launch-bound regime, which isn't a real win" |
| "21× from fixing a bank conflict" | padding alone is 7.0× | "7× from padding; 21× cumulative over three changes vs my first version" |
| "verified correct" for Part 1 | no output check; column split wrong for N ≤ 512 | "the 32K result is correct by construction; Part 1 has no output check" |
| "Nsight showed…", "50% achieved occupancy" | ncu needed root; occupancy is theoretical | "by address arithmetic; theoretical occupancy from the occupancy API" |
| any PA1 (KV cache) speedup | never run, no numbers | "I built it but never ran the benchmark" |
| "registers hold the output accumulator" (FlashAttention) | `o_acc` is in local memory | "the softmax statistics are in registers; `o_acc` spilled to local memory because D is a runtime argument" |

---

## 7. Mock interview: 100 questions with model answers

**Drill:** answer out loud first, then open. Mark strong / OK / weak in [[#My notes and fumbles]]. First person, as you would say it.

### 7.1 Fundamentals (30)

**F1. What is the default stream, and why does it matter?**
> [!success]- Answer
> The stream a launch goes to when none is given. The legacy default stream synchronises with all blocking streams: work in it waits for earlier work in those streams and vice versa, so mixing it with other streams can silently serialise them. My baseline runs on the default stream; my four streams come from `cudaStreamCreate` (blocking), and my timing events are recorded on the default stream, which is what makes `end` wait for the stream work. Per-thread default streams (`--default-stream per-thread`) or `cudaStreamNonBlocking` avoid the implicit sync.

**F2. `cudaStreamSynchronize` vs `cudaDeviceSynchronize` vs `cudaEventSynchronize`?**
> [!success]- Answer
> The host waits for, respectively: everything queued in one stream; everything on the device in all streams; the GPU reaching one recorded event. I use stream syncs in the stream variants (one per stream per phase), a device sync after warm-ups, and an event sync before reading elapsed time. Each blocks the CPU, so inside a timed loop they add host round-trips: the asymmetry in my Part 1 timing.

**F3. What do `cudaMemcpyAsync` and pinned memory have to do with streams?**
> [!success]- Answer
> `cudaMemcpyAsync` enqueues a copy in a stream and returns, so it can overlap with kernels in other streams, but only if the host buffer is pinned (page-locked, `cudaMallocHost`); from pageable memory the driver stages through a pinned buffer and the copy isn't truly asynchronous. Overlapping copies with compute is the classic use of streams. My project overlapped kernels only; all data was resident on the GPU before timing.

**F4. What is a CUDA event, and what is `cudaStreamWaitEvent`?**
> [!success]- Answer
> An event is a marker recorded into a stream; the GPU timestamps it and marks it complete when the stream reaches it. `cudaEventElapsedTime` gives the GPU time between two events. `cudaStreamWaitEvent(s, e)` makes stream s wait (on the GPU, not the CPU) until event e completes: a device-side dependency between streams. It's the tool I'd use to replace the host-side `cudaStreamSynchronize` barrier between layers in my column split.

**F5. What is a CUDA graph?**
> [!success]- Answer
> A recorded DAG of kernels, copies and dependencies that is instantiated once and launched as one unit. It removes most per-launch CPU overhead and lets the driver optimise dependencies. For workloads of many tiny kernels (my N = 32 stream variants, or LLM decode) it's the standard fix for launch-bound regimes. I didn't use it here. [FUTURE]

**F6. Roughly how expensive is a kernel launch?**
> [!success]- Answer
> A few microseconds of CPU-side cost per launch, and a few microseconds of GPU-side latency before the kernel starts. That is irrelevant for my 44 ms forward pass at N = 32768 and dominant for the 6.9 µs one at N = 32, which is why every extra launch and sync made the stream variants 4× slower there. PyTorch's attention floor of ~0.11 ms at small N is the same effect over several kernels. [THEORY with MEAS examples]

**F7. What is concurrent kernel execution and what does it need?**
> [!success]- Answer
> Two or more kernels running on the GPU at the same time, on different SMs or sharing SMs. It needs: the kernels in different (non-default, or non-blocking) streams, and enough free resources: if one kernel's blocks occupy every SM, the next kernel's blocks wait until some finish. Hardware supports up to many concurrent kernels, but resources are the practical limit.

**F8. What is L2 locality, and why does it matter for the column split?**
> [!success]- Answer
> Reuse of data while it's still in the ~50 MB L2 cache shared by all SMs. If the blocks that need the same data run close together in time, only the first fetches it from HBM. In the column split, each kernel's 1,024 blocks fit in one wave, so all 32 batch rows of a W chunk are resident together and share its lines through L2. The baseline spreads the same W chunk's users across four waves, by which time it has been evicted. [RECON hypothesis]

**F9. Which saved tensors does backward need for a Linear + ReLU layer?**
> [!success]- Answer
> The layer input A_prev, for the weight gradient dW = A_prevᵀ·dZ, and either Z (or A) for the ReLU derivative, which zeroes dZ where Z ≤ 0. The weights are needed too, for dA_prev = dZ·Wᵀ, but those are always resident. In my code: `mm_tiled_AT(a3, dZ4, dW4)` and `k_relu_deriv(z3, dZ3)`.

**F10. What is gradient checkpointing, in one sentence, and its cost?**
> [!success]- Answer
> Store only some activations in forward and recompute the others from the nearest stored one during backward; the cost is roughly one extra forward pass of compute over the recomputed segments (+35.5 ms, +26% of backward, in my case).

**F11. What is √n checkpointing?**
> [!success]- Answer
> For a chain of n layers, put a checkpoint every √n layers: you keep √n checkpoints and rebuild one segment of √n layers at a time, so peak activation memory is ~2√n instead of n, for about one extra forward (Chen et al., 2016). Recursively applied it reaches O(log n) memory at higher compute. With 4 layers my project is too small for the asymptotics; the assignment fixed the checkpoints.

**F12. What does `cudaMemGetInfo` return, and what's the catch for measuring peak memory?**
> [!success]- Answer
> Free and total device memory in bytes, device-wide. Before/after deltas give what your process allocated, if nothing else on the GPU allocates in between, and if all allocations are made up front (otherwise you'd need to sample at the true peak). My Part 2 preallocates everything, so the delta is the peak; on a shared GPU another process could contaminate it, and the values are MiB despite "MB" labels.

**F13. How does PyTorch avoid `cudaMalloc` costs, and how does that relate to your arena?**
> [!success]- Answer
> Its caching allocator keeps freed blocks and reuses them for later allocations of similar size instead of returning memory to the driver. With `torch.utils.checkpoint`, recomputed activations are allocated when needed and freed after use, and the allocator reuses that memory: the same effect my single aliased arena gets by hand. The difference is that I had to prove the lifetimes don't overlap; PyTorch frees tensors when their last reference dies.

**F14. Define attention and its shapes in your benchmark.**
> [!success]- Answer
> For each batch element and head: O = softmax(Q·Kᵀ/√D)·V with Q, K, V of shape [N, D]. In my benchmark the tensors are [B = 8, NH = 16, N, D = 64], FP32, contiguous. The score matrix Q·Kᵀ is N × N per (batch, head), which is what makes standard attention O(N²) in memory.

**F15. Why subtract the max in softmax?**
> [!success]- Answer
> e^s overflows float32 for s above ~88. Subtracting the row max makes every exponent ≤ 0, so every term is in (0, 1] and at least one is 1; the e^(−m) factor cancels between numerator and denominator, so the result is unchanged. The catch is needing the max before starting, which online softmax removes.

**F16. What is online softmax?**
> [!success]- Answer
> A one-pass softmax: process the row in pieces, keeping a running max m, a running sum l of e^(s − m) and (for attention) a running weighted sum o. When a piece has a larger max, rescale l and o by e^(m_old − m_new) before adding the piece's terms. At the end o / l equals the exact softmax-weighted sum (Milakov & Gimelshein 2018; FlashAttention uses it per key tile).

**F17. What does "IO-aware" mean in FlashAttention?**
> [!success]- Answer
> Designing the algorithm around memory traffic between HBM and on-chip SRAM rather than FLOPs: attention at long sequence lengths is limited by reading and writing the N × N matrices, so FlashAttention tiles the computation to keep scores in SRAM and never writes them to HBM, even at the price of recomputation in backward. The FLOP count barely changes; the paper shows HBM accesses fall from Θ(N·D + N²) for standard attention to Θ(N²·D²/M), where M is the SRAM size, and nothing N × N is ever written.

**F18. What is dynamic shared memory?**
> [!success]- Answer
> Shared memory whose size is given at launch time (the third `<<< >>>` parameter) rather than in the declaration: `extern __shared__ float smem[];`. One buffer per block, which the kernel carves into arrays by pointer offsets: mine is Qi | Kj | Vj | S, 28,800 B at D = 64. Above 48 KB per block it requires `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, ...)`, which my host code does once if needed.

**F19. What does `__shfl_xor_sync(0xffffffff, v, off)` do?**
> [!success]- Answer
> Every participating lane exchanges `v` with lane `lane XOR off` and gets the partner's value, register to register, without shared memory. `0xffffffff` says all 32 lanes participate (they must all execute the call). A butterfly over offsets 16, 8, 4, 2, 1 with `fmaxf` or `+` leaves the warp-wide max or sum in every lane after 5 steps.

**F20. `__syncwarp()` vs `__syncthreads()`?**
> [!success]- Answer
> `__syncwarp()` synchronises the lanes of one warp and orders their memory accesses; `__syncthreads()` does the same for every thread of the block. Since Volta, lanes of a warp can be scheduled independently, so code where lanes exchange data through shared memory needs at least `__syncwarp()`. I use it after each warp writes its own rows of S, and block barriers around the shared K/V tiles.

**F21. What is a bank conflict, and why does padding fix it?**
> [!success]- Answer
> Shared memory has 32 banks; word a is in bank a mod 32; lanes hitting different addresses in the same bank are served in turn. With a row stride that's a multiple of 32 words (my K tile at 64), reading down a column puts every lane in the same bank: 32-way. Padding the stride to 65 shifts each row by one bank, so a column read hits 32 different banks. The padding column is never read.

**F22. What is swizzling?**
> [!success]- Answer
> Storing element (r, c) of a shared tile at column c XOR f(r) (or a similar permutation) so that column reads spread across banks without wasting space. Unlike padding, it keeps rows aligned to 16 bytes, so vector loads still work. CUTLASS and modern attention kernels use it. My kernel uses padding, the simplest fix, at the cost of 128 B per block and no `LDS.128` on Kj rows.

**F23. `__expf` vs `expf`?**
> [!success]- Answer
> `__expf` is a hardware-intrinsic approximation (special-function unit), much faster and slightly less accurate; `expf` is the accurate library version. Softmax tolerates the error well; my kernel's max absolute difference vs PyTorch's FP32 explicit path is 6.7e-7. Compiling with `--use_fast_math` would substitute it everywhere.

**F24. How do `fmaxf` and `exp` treat NaN, and why did it matter in your quiz kernel?**
> [!success]- Answer
> `fmaxf(NaN, x)` returns x (it ignores a NaN operand), but `exp(NaN − x)` is NaN. In the lab quiz (story S5) the running max was initialised with `cudaMemset(0xFF)`: all bits set is a NaN, not −∞. The max update hid it, but the rescale factor e^(m_old − m_new) became NaN and poisoned l. I seeded m = −∞ on the first tile instead. The `d06i-flash-dbg` animation's last step shows it.

**F25. What does `__launch_bounds__(256)` do?**
> [!success]- Answer
> Tells the compiler the kernel will never be launched with more than 256 threads per block (optionally, a minimum number of blocks per SM), so it can budget registers per thread accordingly instead of assuming 1,024 threads. My FlashAttention kernel declares it; it lands at 62 registers.

**F26. What does `__restrict__` promise?**
> [!success]- Answer
> That the pointer is the only way the kernel accesses that memory (no aliasing with the other pointer arguments), so the compiler can keep loaded values in registers, reorder loads and use the read-only data path. `mm_thin` and `flash_forward_kernel` mark all pointer arguments `__restrict__`. Passing aliased buffers anyway is undefined behaviour.

**F27. What is a C ABI and `extern "C"`, and how does Python call your kernel?**
> [!success]- Answer
> `extern "C"` turns off C++ name mangling so the function has a plain symbol, `flash_attention_forward`, in `libflash.so`. Python's ctypes loads the shared library and calls it with raw device pointers taken from PyTorch tensors (`tensor.data_ptr()`) plus integer sizes. There's no type checking at that boundary.

**F28. What are tensor cores, and why would they change your attention kernel?**
> [!success]- Answer
> Matrix units that do small matrix multiply-accumulates per instruction in FP16/BF16/TF32/FP8, many times the FP32 CUDA-core throughput. With them, QKᵀ and PV become warp-level MMAs, the compute ceiling rises several-fold, and attention becomes HBM-bound at long sequences: exactly where FlashAttention's IO savings turn into speed. My FP32 CUDA-core kernel is limited by shared memory instead, which is why it's slower than PyTorch despite being "flash".

**F29. What is occupancy for your FlashAttention kernel, and what limits it?**
> [!success]- Answer
> 50% theoretical: 62 registers (64 allocated) × 256 threads = 16,384 per block → 4 blocks per SM = 32 warps of 64. Shared memory (28,800 B per block) would allow ~7 blocks, so registers bind. From the occupancy API; not measured.

**F30. What is local memory, and where does it appear in your FlashAttention kernel?**
> [!success]- Answer
> Per-thread memory in device DRAM (cached), used for spills and runtime-indexed arrays. `o_acc[4][4]` (64 bytes) is indexed by `jj`, which depends on the runtime D, so it lives in local memory: `STACK:64` and `LDL/STL` in the SASS. The accesses happen once per key tile, not per FMA, so the cost is probably small; making D a template parameter would fix it.

### 7.2 Implementation and optimisation (30)

**I1. Walk me through the three forward-pass variants.**
> [!success]- Answer
> Baseline: two `mm_thin` launches on the default stream over all 32 rows, grid (128, 32) at 32K. Batch split: four streams, each with 8 rows (pointer offsets into X, A1, Y) but the full W1, W2, both layers in the same stream, then host syncs. Column split: four streams, each with all 32 rows and a quarter of the output columns (offset W1 + c0, b1 + c0, A1 + c0), a host barrier, then the same for layer 2. 44.43, 52.28 and 25.65 ms at 32K.

**I2. Explain `W1 + c0` with N still passed as the width.**
> [!success]- Answer
> W1 is row-major with row stride N. `W1 + c0` points at column c0 of row 0; element (k, c0 + j) is at `W1 + c0 + k·N + j`. The kernel computes `B[(k0 + kk)·N + col]` with its `N` argument as the row stride, so passing the full N keeps the stride right while the pointer offset selects the column window. The grid only covers `cols` columns. The comment at `:313–317` says this; the catch is that the kernel's `col < N` guard then doesn't stop threads past the window (the N ≤ 512 bug).

**I3. Grid sizes for each variant at N = 32768?**
> [!success]- Answer
> Baseline: ((32768 + 255)/256, 32) = (128, 32) = 4,096 blocks per layer. Batch split: (128, 8) = 1,024 blocks per stream per layer. Column split: (8192/256, 32) = (32, 32) = 1,024 blocks per stream per layer. Each block is 256 threads.

**I4. Why does the batch split need no barrier between its layers?**
> [!success]- Answer
> Each stream's layer 2 needs only its own 8 rows of A1, which its own layer-1 kernel wrote earlier in the same stream; in-stream order guarantees layer 1 finished. Rows don't mix across a matmul's output rows. Columns do (every output column needs all input columns), which is why the column split needs the barrier.

**I5. How would you add a correctness check to Part 1?**
> [!success]- Answer
> After timing each variant, copy Y back and compare with a reference: the baseline's Y (cheap, same kernel), and better, a cuBLAS or PyTorch result for one width, using max relative error like PA2. Also clear Y between variants so a broken variant can't inherit the previous output. It would have caught the N ≤ 512 column-split bug immediately. [FUTURE]

**I6. Fix the column-split bug.**
> [!success]- Answer
> Give the kernel a separate column bound: add an argument `ncols` (the chunk width) and guard with `col < ncols` while keeping N as the row stride (leading dimension). For the batch split and baseline, pass `ncols = N`. Then a 256-thread block that's wider than its chunk leaves the extra threads idle instead of writing into a neighbour's columns or past the end of A1/Y.

**I7. Why does `mm_thin` stage A in shared memory but read W straight from global?**
> [!success]- Answer
> A's row slice is reused by all 256 threads of the block (each multiplies it with its own column), so staging 32 floats once saves 255 redundant loads per value. W's elements are used by exactly one thread in the block (its column), so there's no intra-block reuse to exploit; their reuse is across the 32 row-blocks, which shared memory can't capture because those are different blocks. That reuse is left to L1/L2, which is the root of the locality story.

**I8. How would you restructure `mm_thin` to reuse W across rows explicitly?**
> [!success]- Answer
> Make one block handle several (or all 32) batch rows for its column chunk: stage a K-slice of the rows' activations (32 × BK) in shared memory and keep 32 accumulators per thread (one per row), so each W element loaded once feeds 32 FMAs from registers. That's register blocking over the batch, the same idea as my matmul's 4 × 4 patches and FlashAttention's 4 rows per warp. It would make W traffic one pass per layer by design. [FUTURE]

**I9. Which tensors are kept, dropped and recomputed in checkpointed mode?**
> [!success]- Answer
> Kept (own buffers): A2 and Z4. Dropped after forward: Z1, A1, Z2 (segment A, overwritten in the arena by segment B). Recomputed in backward: Z3, A3 from A2 (segment B), then Z1, A1, Z2 from X (segment A), both into the same arena.

**I10. Compute the arena size and the checkpointed activation total at N4096.**
> [!success]- Answer
> n1 = n2 = n3 = 4096 × 4096 floats = 64 MiB each. Segment A = Z1 + A1 + Z2 = 3 × 64 = 192 MiB; segment B = Z3 + A3 = 128 MiB; arena = max = 192 MiB. Total = A2 64 + Z4 16 + arena 192 = 272 MiB, vs 400 in the baseline (−32%).

**I11. What enforces the ordering rule at line 513?**
> [!success]- Answer
> Only the source order plus stream semantics. All kernels go to the default stream, so they execute in issue order on the GPU; the code issues every use of z3/a3 (layer-4 and layer-3 gradients) before the segment-A rebuild. Nothing checks it: reordering two lines would silently produce wrong gradients. A safer design would make the aliasing explicit, for example by nulling `z3`/`a3` after their last use or using an allocator that tracks lifetimes.

**I12. Why custom GEMMs instead of cuBLAS in PA3?**
> [!success]- Answer
> cuBLAS keeps its own workspace allocations, which would show up in the `cudaMemGetInfo` deltas and blur the memory comparison (documented at `:17–18`). The cost is speed: the forward pass runs at roughly 8.7 TFLOP/s (the 1-to-1 tiled kernel's shared roof), and backward is 3.4× forward because the transposed-operand kernels are uncoalesced.

**I13. Why scale the weights by 1/√fan_in?**
> [!success]- Answer
> With U(−1, 1) weights on 4096-wide layers, each layer multiplies the activation scale by roughly √fan_in, so the training loop hit inf/NaN within a few steps. NaN ≠ NaN, so the two modes' checksums could never match and the equality check was useless (story S4). Scaling keeps activations O(1) through the layers. Nit: W4 uses `isq(H2)` instead of `isq(H3)`, harmless since H2 = H3 in every config.

**I14. Why is backward 3.4× the forward instead of ~2×?**
> [!success]- Answer
> Backward per layer does two GEMMs (dW = A_prevᵀ·dZ and dA_prev = dZ·Wᵀ), roughly 2× the forward FLOPs. The rest comes from implementation: `mm_tiled_AT` and `mm_tiled_BT` read one operand transposed, so their tile loads are uncoalesced. A cuBLAS backward (with transpose flags) wouldn't pay that. [RECON]

**I15. How does the forward pass differ between the two modes?**
> [!success]- Answer
> Only in where it writes: `z1 = checkpointed ? rZ1 : Z1` and so on for a1, z2, z3, a3 (`:465–469`). Same kernels, same order, same arithmetic. In checkpointed mode z3/a3 overwrite the start of the arena where z1 lived, which is fine because forward never reads z1 again after layer 2.

**I16. Map block (qt = 10, h = 3, b = 0), warp 2, lane 7 to data.**
> [!success]- Answer
> `base = (b·NH + h)·N·D = 3 · 8192 · 64 = 1,572,864` (offset of head 3's Q, K, V); `qbase = 10 · 32 = 320`; `r0 = 2 · 4 = 8`, so the warp owns query rows 328–331. In the score phase lane 7 owns key `j·32 + 7` of tile j; in the PV phase it owns features 7 and 39. At the end it writes O[328…331][7] and [39].

**I17. Derive the 28,800 bytes.**
> [!success]- Answer
> Qi: 32 × 64 = 2,048 floats; Kj: 32 × 65 = 2,080 (padded); Vj: 32 × 64 = 2,048; S: 32 × 32 = 1,024. Total 7,200 floats × 4 B = 28,800 B, under the 48 KB default, so no opt-in.

**I18. Why is x the outer loop in the score phase?**
> [!success]- Answer
> So that one `Kj[c·65 + x]` load (lane c's key, feature x) is reused for all 4 query rows of the warp: 4 FMAs per Kj load, plus 4 broadcast Qi loads. If r were the outer loop, each row would reload the whole key row: 2 loads per FMA. This loop order is the "register blocking over query rows" that took the kernel from 339 to 160 ms.

**I19. Walk through the softmax block for one row.**
> [!success]- Answer
> s = a[r]·scale (or −∞ for lanes past kv_len); m_tile = butterfly max over the warp; p = e^(s − m_tile) (0 past kv_len); l_tile = butterfly sum; S[row][lane] = p; m_new = max(m_run, m_tile); e_run = e^(m_run − m_new) (0 on the first tile since m_run = −∞); e_tile = e^(m_tile − m_new); l_run = e_run·l_run + e_tile·l_tile; m_run = m_new. The PV phase then uses e_run and e_tile to rescale o.

**I20. Why is `e_run` 0 on the first tile, and why is that correct?**
> [!success]- Answer
> m_run starts at −∞ and m_tile is finite (at least lane 0 has a real key), so m_new is finite and e^(−∞ − m_new) = 0. Multiplying the initial l = 0 and o = 0 by 0 changes nothing, so the first tile's terms simply become the state. It's the natural "empty state" value: −∞ is the identity of max, and e^(−∞) = 0 is the identity for the rescale.

**I21. Explain the PV loop and the role switch.**
> [!success]- Answer
> After the softmax, S holds p for the warp's 4 rows × 32 keys. Now lane x owns output features x and x + 32: for each key c it reads `Vj[c·64 + x]` once (adjacent lanes → adjacent banks) and the broadcast `S[row][c]` for 4 rows, accumulating pv[4]. Then o = o·e_run + pv·e_tile. The switch from "lane = key" to "lane = feature" is what makes both GEMMs reduction-free per lane except for the softmax statistics.

**I22. Why divide by l only at the end?**
> [!success]- Answer
> Keeping o unnormalised means each tile costs one multiply-add per output (the rescale), and the division happens once per output element at the end (`inv = 1/l_run`, then a multiply). Normalising every tile would cost a division per element per tile and would have to be undone at the next rescale. Same exact result, less work.

**I23. Why does the host reject D > 128?**
> [!success]- Answer
> Each lane holds `o_acc[RPW][4]`: 4 feature slots (x, x + 32, x + 64, x + 96), which covers D ≤ 128. Larger D would need more slots per lane (and more registers or local memory) and more shared memory per tile. The check returns −1 instead of silently computing garbage.

**I24. Why ctypes instead of a PyTorch C++ extension, and what did it cost?**
> [!success]- Answer
> Not documented; reconstructed: it's the smallest build (a plain `.so`, no PyTorch headers or build system) and enough for a benchmark. Costs: no dtype/contiguity/device checks (wrong inputs mean garbage or a crash), no autograd, and the kernel launches on the default stream rather than PyTorch's current stream, which is correct here only because PyTorch is also using the default stream. A `torch.utils.cpp_extension` op would fix all three. [RECON]

**I25. How is the PyTorch explicit baseline run at N = 8192, and why?**
> [!success]- Answer
> The full score tensor would be 34.4 GB, above the `--budget-gb 16` limit, so `bench_attention.py:140–143` halves the batch chunk until it fits: 8 → 4 (still 17 GB) → 2. The CSV records `standard_batch_chunk=2`. Its output is the correctness reference, and its time (105.08 ms) includes a 4-iteration Python loop over chunks, slightly slower than one unchunked call would be (which would need 34 GB).

**I26. How do you check the FlashAttention output?**
> [!success]- Answer
> Max absolute difference against the PyTorch explicit FP32 path on the same random inputs: 6.7e-7 at N = 8192, at most 7.2e-7 across the sweep. Differences come from summation order (online rescaling, tile order) and `__expf`. The check is the whole output tensor, unlike PA2's prefix check.

**I27. What's the FLOP count of your attention, and the TFLOP/s?**
> [!success]- Answer
> Two GEMMs of 2·N²·D FLOPs each per (batch, head): 4·B·NH·N²·D = 4 · 8 · 16 · 8192² · 64 ≈ 2.2 × 10¹² FLOPs. At 160 ms that's 13.75 TFLOP/s, about 20% of the 66.9 FP32 peak. (Softmax exponentials are ignored in this count, as is conventional.)

**I28. Which accesses had to be checked after padding, and why?**
> [!success]- Answer
> All shared accesses, because a layout change can create a conflict elsewhere: the Kj store (consecutive x within a row → consecutive words, fine), the Kj score read (now conflict-free), Qi and S reads (broadcasts), the S store (consecutive), the Vj read (lane = feature → consecutive, so V needs no padding). And the store and load must use the same stride, which they do (`KS` in both), so correctness is unchanged.

**I29. What did FlashAttention-2 change, and how much of it is in your kernel?**
> [!success]- Answer
> It parallelises over the sequence (query blocks) as well as batch and heads, splits work between warps by query rows so warps don't share partial results through shared memory, and reduces non-matmul FLOPs (e.g. deferring the normalisation). My grid over query tiles (3,358 → 2,377 ms) and warp-per-rows split are the first two ideas; I also normalise once at the end. Missing: tensor cores, `cp.async` prefetch, and FA-2's tuned tile shapes. [RECON]

**I30. How would you add the causal mask?**
> [!success]- Answer
> Skip key tiles entirely above the diagonal (j·32 > last query of the block), process tiles entirely below it unmasked, and in diagonal tiles set s = −∞ for keys after the query. Care is needed so a fully-masked row in a tile doesn't produce e^(−∞ − (−∞)) = NaN: either guarantee each processed tile has at least one unmasked key per row (true when the diagonal tile is included) or guard m_tile = −∞. It also halves the work, so speed per useful FLOP is unchanged. [FUTURE]

### 7.3 Difficult follow-ups (20)

**D1. "Your baseline gets 7.1× slower for 4× the work from 16K to 32K. What could cause superlinear scaling?"**
> [!success]- Answer
> The working set crossing a cache or translation limit. At 16K each W is 1.07 GB; at 32K 4.29 GB. Candidates: L2 reuse across waves breaking down (W chunks evicted before the next wave needs them), TLB reach (more distinct pages touched per wave than the TLB covers, so page walks), and DRAM page/row-buffer locality. All three are about access order, which is why reordering the work (column split) fixes it. I can't rank them without counters. [THEORY]

**D2. "If it's locality, can you get the 1.73× without streams?"**
> [!success]- Answer
> I'd predict yes. Either run the four column chunks one after another on one stream (prep exercise 2), or write one kernel whose grid makes `blockIdx.x` the batch row and `blockIdx.y` the column chunk, so blocks dispatched together share a W chunk. If either matches ~26 ms, the honest resume line is "partitioning by output column", not "stream parallelism". Not run.

**D3. "What if the batch were 1,024 instead of 32?"**
> [!success]- Answer
> Then the problem becomes a normal square-ish GEMM: plenty of reuse of W within a block if you tile over rows, and the grid fills the GPU without streams. `mm_thin` (one row per block) would be the wrong kernel; a tiled or register-blocked GEMM would be right, and the batch split would no longer multiply W traffic in proportion (each stream's 256 rows would reuse W well). The thin kernel and the stream question only make sense because batch 32 is skinny. [THEORY]

**D4. "Why not overlap layer 1 and layer 2 across streams in the column split?"**
> [!success]- Answer
> Layer 2 needs every A1 column. You could start partial layer-2 work as each stream's A1 chunk finishes: stream s's A1 columns contribute a partial sum A1[:, chunk_s]·W2[chunk_s, :] to all of Y (split-K over streams), combined with atomics or a final reduction, so no global barrier. That adds traffic (4 partial Y's or atomic contention) for an overlap that probably isn't worth much when each layer already fills the GPU. With `cudaStreamWaitEvent` I'd at least move the barrier from the host to the device. [FUTURE]

**D5. "Checkpointing saved 32% on 4 layers. What would it do on 100 layers?"**
> [!success]- Answer
> With checkpoints every √100 = 10 layers, keep 10 checkpoints and rebuild one 10-layer segment at a time: ~20 activations' worth instead of 100, about 80% saved, for about one extra forward. The arena trick generalises: one buffer sized to the largest segment, reused for every segment in backward order. My 4-layer case can only save so much because the arena (192 MiB) is close to half the baseline. [THEORY]

**D6. "Can checkpointing change the results?"**
> [!success]- Answer
> In my code, no: the recompute runs identical kernels on identical inputs in the same order, so it's bitwise identical, as the checksums show. In general it can if the forward is non-deterministic (atomics, some cuDNN algorithms) or stochastic (dropout): the recomputed activations must use the same random mask, which is why PyTorch's checkpoint saves and restores the RNG state by default.

**D7. "Is recomputing segment B actually necessary?"**
> [!success]- Answer
> In this schedule, no, and I found that while preparing. In checkpointed mode the forward writes z3, a3 into the arena last, and nothing touches the arena before backward starts, so at line 495 the arena already holds valid z3, a3. Skipping that recompute would save the layer-3 GEMM, about 0.4 of a forward ≈ 16 ms of the 35.5 ms overhead, at the same 272 MiB. It's only valid because forward and backward run back to back; in a framework with other work in between, you'd need it. Not measured. [RECON]

**D8. "In online softmax, what if every score in a tile is −∞?"**
> [!success]- Answer
> Then m_tile = −∞ and p = e^(−∞ − (−∞)) = e^NaN = NaN, which would poison l and o. My kernel can't hit it: each processed tile has kv_len ≥ 1, so at least lane 0 has a finite score, and lanes past kv_len are masked with −∞ only alongside a real one. With causal or padding masks it becomes possible, and kernels guard it (treat m_tile = −∞ as "skip this tile" or use a large negative finite number instead of −∞). [THEORY]

**D9. "Is your FlashAttention output identical to the standard path?"**
> [!success]- Answer
> Not bit-identical: the sums are done in a different order (tile by tile with rescaling) and with `__expf`, so FP32 rounding differs; the max absolute difference is 6.7e-7. Mathematically it's exact (no approximation like sparse or low-rank attention); numerically it's within FP32 noise.

**D10. "FlashAttention is famous for being faster. Why is yours slower?"**
> [!success]- Answer
> The paper's speedup comes from cutting HBM traffic when attention is memory-bound: FP16 tensor-core GEMMs are so fast that reading and writing the N² matrices dominates. My kernel is FP32 on CUDA cores with both GEMMs reading operands from shared memory: it's bound by shared-memory bandwidth at 13.75 TFLOP/s, while PyTorch's explicit path runs its GEMMs in cuBLAS at much higher efficiency. In FP32 the explicit path's extra HBM traffic (~137 GB of score traffic, ~30 ms at 4.8 TB/s by a rough count) costs less than my kernel's inefficiency. [THEORY, back-of-envelope]

**D11. "Why does padding break vector loads?"**
> [!success]- Answer
> `LDS.128` loads 4 consecutive floats from a 16-byte-aligned address. With a row stride of 65 floats (260 bytes), row c starts at byte 260c, which is 16-byte aligned only when c is a multiple of 4. So the compiler can't use 128-bit loads along Kj rows. A swizzle keeps rows at a stride of 64 (aligned) and permutes columns instead.

**D12. "Why not store K transposed instead of padding?"**
> [!success]- Answer
> With Kᵀ in shared memory (feature-major), the score read `KjT[x·32 + c]` is consecutive across lanes: conflict-free. But the conflict moves to the store: loading K from global row by row writes down columns of KjT. The store happens once per tile per element, while the read happens D = 64 times per lane per tile, so transposing would also work and costs less than the original conflict; padding was simply the one-line fix. [RECON]

**D13. "Your kernel is at 50% occupancy. Would more occupancy help?"**
> [!success]- Answer
> Probably not much. The kernel sits on the shared-memory bandwidth roof, and more resident warps would issue more shared loads into the same saturated banks. What would help is fewer shared loads per FMA (more rows per warp, operands in registers) or tensor cores. Occupancy helps latency-bound kernels; this one is throughput-bound on shared memory. [THEORY]

**D14. "What's the HBM traffic of your kernel vs the standard path?"**
> [!success]- Answer
> Mine: each of the 32,768 blocks reads its 8 KB Q tile and its head's full K and V (2 MB each): at most ~137 GB if nothing hits L2 (the 256 query tiles of a head share the same K, V, so L2 should catch a lot), and writes O once (268 MB). The standard path additionally writes the 34.4 GB score matrix, reads and writes it for softmax, and reads it for PV: on the order of 137 GB of score traffic on top. Both are rough counts, not measured. [THEORY]

**D15. "How would you implement the backward pass?"**
> [!success]- Answer
> As in FlashAttention: save only O and the per-row logsumexp (m + log l) from forward. In backward, for each tile, recompute S = QKᵀ and P = exp(S − logsumexp) from Q, K and the saved statistics, then compute dV += Pᵀ·dO, dP = dO·Vᵀ, dS = P ∘ (dP − rowsum(dO ∘ O)), dQ += dS·K, dK += dSᵀ·Q. Nothing N × N is stored: it's checkpointing applied to attention. Not implemented. [FUTURE]

**D16. "What if inputs were FP16?"**
> [!success]- Answer
> Load FP16 Q, K, V (half the bytes), compute QKᵀ and PV on tensor cores with FP32 accumulation, and keep the softmax statistics (m, l) and the rescaling in FP32 for range and accuracy. The shared layout changes (fragments for `mma.sync`, swizzles), and the bottleneck moves toward HBM, where flash's IO savings finally pay off. The correctness check would need a tolerance appropriate for FP16. [FUTURE]

**D17. "What happens at D = 128? D = 256?"**
> [!success]- Answer
> D = 128: shared memory = (32·128 + 32·129 + 32·128 + 1,024) × 4 = 53,376 B > 48 KB, so the host opt-in (`cudaFuncSetAttribute` to 200 KB) fires once; each lane uses all 4 `o_acc` slots; it should work, with fewer blocks per SM by shared memory (~4, the same as the register limit). D = 256: rejected (`return −1`), because each lane can hold only 4 feature slots. Not benchmarked; D = 64 is the only measured size. [RECON]

**D18. "Your first layout used 128 blocks on 132 SMs. What should you have predicted?"**
> [!success]- Answer
> That the GPU would be under-filled: grid (B, NH) = 8 × 16 = 128 blocks of one warp each, so at most 128 warps on the whole chip (under 1 per SM's 64-warp capacity), no latency hiding and 4 SMs idle. Moving to a grid over query tiles gave 32,768 blocks. It only gave 1.4× because the next bottleneck (bank conflicts) was already waiting, a pattern I saw three times in this project.

**D19. "Why did the matmul and attention both end up at ~half of the vendor's speed?"**
> [!success]- Answer
> Same root cause: both kernels feed FMAs from shared memory with modest register reuse (4 × 4 outputs per thread in matmul; 4 rows per warp in attention), while cuBLAS uses larger register tiles, vectorised conflict-free shared loads and double-buffered async copies. Both sit near a shared-memory roof I can compute. The fix is the same family of techniques in both cases. [RECON]

**D20. "You never ran a profiler. How do you know any of this?"** (hostile)
> [!success]- Answer
> Nsight Compute needed root on the shared server (`ERR_NVGPUCTRPERM`), so I used three substitutes and I'm explicit about which claims rest on which: address arithmetic for coalescing and banks (hand-computed for the hot loops, and the predicted fixes worked: 7.0× from the padding); roofline-style models whose predictions matched measurements (8.4 vs 8.6, 13.4 vs 13.75 TFLOP/s; 35.5 vs 35.54 ms for checkpointing); and static analysis of the binaries (`cuobjdump`: registers, local memory). Occupancy figures are theoretical. Where I couldn't verify (the 1.73× mechanism), I say "likely" and name the experiment.

### 7.4 Debugging and code analysis (10)

**B1. Change `KS = D + 1` back to `KS = D`. What happens to correctness and time?**
> [!success]- Answer
> Correctness: unchanged, because the store (`:107`) and the read (`:119`) use the same stride; the host still allocates the D + 1 size, which is harmless. Time: the score loop returns to a 32-way conflict. Per feature step that's one Kj read costing 32 passes plus 4 broadcast Qi reads (~36) instead of ~5; with the PV phase unchanged (~5), expect roughly (36 + 5)/(5 + 5) ≈ 4× slower, ~500–650 ms. Less than the original 7× because 4 rows per warp amortise each Kj read. A prediction, not a measurement (prep exercise 1).

**B2. Seed `m_run` with 0 instead of −∞. Is the result still correct?**
> [!success]- Answer
> Mathematically yes: any reference value cancels in o / l. Numerically no, for rows whose scores are all very negative: with m stuck at 0, every p = e^(s − 0) can underflow to 0 (e.g. s = −200), giving l = 0 and o / l = 0/0 = NaN. Seeding with −∞ makes the first tile set m to the true tile max, so at least one p is 1. (The Day 6 rebuild exercise asks you to break it this way.)

**B3. The quiz harness initialised `m` with `cudaMemset(ptr, 0xFF, …)`. What goes wrong, and how did you find it?**
> [!success]- Answer
> All-ones bits are a NaN, not −∞. `fmaxf(NaN, m_tile)` returns m_tile, so m_new looks right, but e_run = e^(NaN − m_new) = NaN and l = NaN·0 + … = NaN, so every output is NaN. I found it by reading the evaluator rather than the spec (story S5) and fixed it by seeding m_prev = −∞ on the first tile instead of trusting the initial buffer.

**B4. Delete the `__syncwarp()` at line 153. What can go wrong?**
> [!success]- Answer
> Lane c writes `S[row][c]`, and in the PV loop every lane reads all 32 entries of the row, including those written by other lanes. Since Volta, lanes of a warp aren't guaranteed to execute in lockstep, so a lane could read an S entry before its writer stored it: a race that may rarely show up. `__syncwarp()` both reconverges the lanes and orders the shared-memory accesses. `compute-sanitizer --tool racecheck` would flag it.

**B5. Delete the `__syncthreads()` at line 103 (before loading the next K/V tile).**
> [!success]- Answer
> A fast warp could start overwriting Kj and Vj with tile j + 1 while a slow warp is still in tile j's PV loop reading Vj (or its score loop reading Kj): wrong outputs, timing-dependent. It's the "barrier 2" of the matmul tile loop (Day 5 §4.5), placed at the top of the next iteration instead of the bottom of this one.

**B6. Drop the `e_run` rescale of `o_acc` (keep it for `l_run`). What breaks?**
> [!success]- Answer
> Whenever a later tile has a larger max than the earlier ones, the old output stays scaled relative to the old max while l is rescaled, so old tiles get too much weight. If the row max happens to be in the first tile, e_run = 1 for every later tile and the bug is invisible: a nasty data-dependent error. In the toy (s = [2, 1 | 3, 0]) the result becomes (2.104 + 2.0)/1.553 = 2.643 instead of 1.786.

**B7. Remove the `lane < kv_len` guards in the last tile when N isn't a multiple of 32. What happens?**
> [!success]- Answer
> Lanes past the last real key would compute scores against whatever Kj rows hold from the previous tile (the loader only writes kv_len rows), and those ghost keys would get real softmax weight and pull in stale V rows: wrong outputs for every query, with no crash. The −∞ score for those lanes is what gives them zero weight.

**B8. In the column split, someone "fixes" the bug by passing `cols` instead of `N` as the kernel's `N` argument. Does it work?**
> [!success]- Answer
> No. Inside `mm_thin`, `N` is both the column bound and the **row stride** of W and the output: `B[(k0 + kk)·N + col]`, `C[m·N + col]`. Passing `cols` makes the kernel walk W with the wrong stride, reading the wrong elements and writing the wrong output rows. The correct fix separates the two meanings: keep N as the leading dimension and add a separate column bound.

**B9. Someone moves the segment-A rebuild (`:514–523`) above the layer-3 gradient (`:507–511`). What happens, and how would you catch it?**
> [!success]- Answer
> The rebuild writes Z1 over the arena where z3 lives (`rZ1 = rZ3 = arena`), so `k_relu_deriv(z3, dZ3)` would use Z1's values as the ReLU mask (and rA1 = arena + n1 = rA3 overwrites a3 too): wrong dZ3 and therefore wrong dW3, dW2, dW1, with no crash. dW4 was computed before the move, so W4's first update is still right: the W4 checksum would only drift over later steps, through the corrupted W1–W3. Catch it by comparing every gradient tensor with the non-checkpointed mode after a single step.

**B10. Your peak-memory number jumps by 2 GB on one run. What happened?**
> [!success]- Answer
> `cudaMemGetInfo` is device-wide, so another process allocating on the same GPU between the before and after calls would be counted (it's a shared server). Other causes: a first CUDA call lazily creating a context or library workspace inside the measured window. Fixes: measure on a GPU nobody else is using, make the "before" call after context creation and warm-up, or use allocator-level statistics (e.g. a memory pool's used-bytes counter) that see only your allocations. [RECON]

### 7.5 Performance analysis (10)

**P1. How would you prove two kernels actually overlapped?**
> [!success]- Answer
> Look at a timeline: Nsight Systems shows each kernel's start and end per stream; overlap is visible directly. Without it, record events at the start and end of each stream's work and compare intervals, or compare the multi-stream time against the sum of the individual kernel times (if total ≈ sum, there was no overlap). For my column split, the one-stream experiment answers the question more directly.

**P2. Waves: compute them for the baseline at N = 32768 and say what they imply.**
> [!success]- Answer
> 256 threads per block, up to 2,048 threads per SM → up to 8 blocks per SM × 132 SMs = 1,056 blocks per wave. The baseline has 128 × 32 = 4,096 blocks, ~3.9 waves. So the GPU is fully occupied for most of the kernel: a second stream has no idle SMs to use, and each wave sweeps all of W. [THEORY]

**P3. What's the minimum time for the forward pass at N = 32768, and what does the gap say?**
> [!success]- Answer
> Reading W1 and W2 once (2 × 4.29 GB) at 4.8 TB/s takes ≥ 1.8 ms; the FLOPs (1.4 × 10¹¹) at 66.9 TFLOP/s take ≥ 2.1 ms. The baseline takes 44.4 ms and the column split 25.7: far from both bounds, so neither raw bandwidth nor compute is the limit. The kernel's access pattern (W re-reads, one global load per FMA) is. A register-blocked thin GEMM could get much closer. [THEORY]

**P4. Predict the checkpointing overhead before running it.**
> [!success]- Answer
> Count recomputed GEMM work relative to the forward: layers 3, 1, 2 = 1 + 0.25 + 1 = 2.25 of 2.5 units = 0.9 forward. With forward measured at 39.5 ms, the prediction is 35.5 ms. Measured: +35.54 ms. Bias/ReLU kernels are small next to the GEMMs, which is why it matches so closely.

**P5. Chen et al. say checkpointing costs ~1 extra forward, usually quoted as ~30%. Why is yours 26% of backward and ~20% of the step?**
> [!success]- Answer
> The rule of thumb assumes backward ≈ 2× forward, so one extra forward ≈ 1/3 of the step. Here backward is 3.4× forward (the transposed-operand kernels are slow), so the same 0.9 forward is a smaller fraction: 35.5 / 136.15 = 26% of backward, 35.5 / 175.65 ≈ 20% of the step. And I recompute 0.9, not 1.0, of a forward because layer 4 isn't recomputed.

**P6. Why does PyTorch's explicit path, which moves 34 GB of scores, beat your kernel?**
> [!success]- Answer
> Rough budget: its two GEMMs (2.2 × 10¹² FLOPs) at cuBLAS-like ~50 TFLOP/s take ~44 ms; the score traffic (write, softmax read/write, read: ~137 GB) at 4.8 TB/s takes ~29 ms; plus a 4-chunk Python loop. That's in the range of the measured 105 ms. My kernel avoids the score traffic but does the FLOPs at 13.75 TFLOP/s: ~160 ms. In FP32, compute efficiency dominates the saved traffic. [THEORY, back-of-envelope]

**P7. Where is the crossover between your kernel and PyTorch, and what does it mean?**
> [!success]- Answer
> My kernel is faster only at N ≤ 128; from N = 256 PyTorch wins. Below 256 PyTorch has a ~0.11 ms floor from launching several kernels (matmul, softmax, matmul, plus overheads), while mine is one launch. So it's a launch-overhead effect in a regime that doesn't matter, not better attention, and I report it that way.

**P8. Is a single 1.73× measurement enough to claim it?**
> [!success]- Answer
> It's enough to report as a measurement at one width, not to generalise. To make it robust: repeat each variant several times and report spread; add widths between 16K and 32K (where does the gain start?); run the one-stream control. I'd present it as "1.73× at width 32K, 1.02× at 16K" until then.

**P9. Which memory metric should you report for checkpointing, and why?**
> [!success]- Answer
> Both, with their denominators: activation memory (400 → 272 MiB, −32%), which is what checkpointing targets, and the total training footprint (962 → 834 MiB, −13.3%), which is what decides whether a job fits. Reporting only the larger percentage with the wrong label (as my resume did) overstates the effect; reporting only the total hides how well the technique worked on what it can touch.

**P10. Your FlashAttention went 3,358 → 2,377 → 339 → 160 ms. For each step, name the bottleneck removed and the evidence.**
> [!success]- Answer
> (1) Under-filled GPU: 128 single-warp blocks on 132 SMs → 32,768 blocks; evidence: 1.4×, smaller than expected, which pointed to another limit. (2) 32-way bank conflict on the Kj reads → stride 65; evidence: the bank arithmetic and 7.0×. (3) Shared-memory bandwidth at 2 loads per FMA → 4 rows per warp (5 loads per 4 FMAs); evidence: the model roof 8.4 vs 6.5 measured before, 13.4 vs 13.75 after (2.1×). Next bottleneck: the same roof, at 13.75 TFLOP/s.

---

## Sources

The study note ([[Day 06 - CUDA II#A. Theory]]) has the summaries: Mark Harris, *How to Overlap Data Transfers in CUDA C/C++*; CUDA Programming Guide, *Asynchronous Concurrent Execution*; Chen et al., *Training Deep Nets with Sublinear Memory Cost* (2016); the PyTorch `torch.utils.checkpoint` docs; Dao et al., *FlashAttention* (2022) and *FlashAttention-2* (2023); Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018).

## My notes and fumbles

- 
