# Project report: CUDA kernel optimisation and efficient ML inference (CS794, Systems for ML)

> **For the LLM reading this.** This report describes coursework by G Jayavardhan (M.Tech CSE, IIT Bombay) for **CS794:
> Systems for Machine Learning** (instructor Prof. Mythili Vutukuru, Spring 2026). He must defend it in interviews. It was
> written from the repository's CUDA source, results CSVs, RESULTS.md and SETUP.md.
>
> Labels:
> - **[CODE]**: what the code does, with `file:line`.
> - **[MEASURED]**: a number from the results.
> - **[DOC]**: a reason stated in the repo.
> - **[RECON]**: added by the report writer, including findings from reading the code that the repo does not state.
>
> If something is not covered here, say so rather than invent it.

---

## 1. Summary

Four programming assignments (PA1–PA4) and two lab quizzes, all on one NVIDIA **H200** (shared server). The resume claims come from **PA2** and **PA3**.

- **PA2, matrix multiplication (`matmul/matmul_bench.cu`).** Square FP32 SGEMM `C = A·B`, built up through 7 GPU kernels:
  naive → coalesced → shared-memory tiling (three variants) → 2D register blocking, with cuBLAS as the ceiling and a CPU
  triple loop as the floor.
  - Best custom kernel (TILE 64, R 4): **25.71 TFLOP/s at N = 32768**. That is **51.3× the naive GPU kernel**,
    **48.2% of cuBLAS**, and 38.4% of theoretical FP32 peak.
- **PA3, MLP (`mlp/mlp_bench.cu`).**
  - (a) 2-layer MLP forward, B = 32, width N up to 32K. Batch-split across 4 CUDA streams never helps. A
    **column split across 4 streams is 1.73× faster at N = 32768**.
  - (b) A 4-layer MLP training step with **gradient checkpointing**: keep only A2 and Z4, recompute the rest, and
    reuse one scratch arena. This cuts peak memory of the whole step by **13.3%** (128 MB) at a backward-pass cost of +26%.
- **PA4, FlashAttention forward (`flashattention/flash_lib.cu`).** Custom FP32 kernel with Br = Bc = 32 and an online
  softmax. Tuned **21×** (3358 → 160 ms at N = 8192), mainly by fixing a 32-way shared-memory bank conflict. Faster
  than explicit PyTorch attention only for N ≤ 128.
- **PA1, KV cache + radix tree on nanoGPT/GPT-2 (`kvcache/`).** Implemented but **never benchmarked**.
- **Lab quizzes 2 and 3 (`prob2/`, `prob3/`).** Seven CUDA kernels; all pass their evaluators and are clean under `compute-sanitizer memcheck`.

Repo and toolchain:
- **Repo:** `/data/jayavardhan/projects/sysml`, a single commit (`16870fb`, 2026-08-18, "kernel").
- **Toolchain:** CUDA 13.0, gcc 13.3, `nvcc -O3 -arch=sm_90`. PyTorch 2.13.0+cu130 in mamba env `sysml` (Python 3.11).
- **H200 numbers:** 132 SMs, 1980 MHz, 141 GB HBM3e. FP32 non-tensor peak = 132 × 128 × 2 × 1.98 GHz = **66.91 TFLOP/s**.
- **`ncu` was unavailable.** Performance counters need root (`ERR_NVGPUCTRPERM`), so nothing here is hardware-profiled.
  Occupancy figures come from the CUDA occupancy API.

---

## 2. Resume bullets vs evidence

| Bullet | Verdict | Evidence |
|---|---|---|
| "Optimized CUDA matrix multiplication through coalescing, shared-memory tiling, and 2D register blocking, achieving 51.3× speedup over the naive GPU kernel and 48.2% of cuBLAS performance on an H200" | **Supported.** | `results/pa2_matmul.csv`, N = 32768. Naive 140,486 ms (0.50 TF) vs `block2d_t64r4` 2,737 ms (25.71 TF): 140486/2737 = 51.3×. cuBLAS 1,318 ms (53.37 TF): 25.71/53.37 = 48.2%. Kernel time only (CUDA events), no host transfers. |
| "Accelerated MLP inference 1.73× using column-wise CUDA stream parallelism at large dimensions" | **Supported at one size only.** | N = 32768, B = 32: baseline 44.43 ms vs column split 25.65 ms = 1.73×. At N ≤ 16384 it is neutral or slower (6.293 vs 6.186 ms at 16384, slower below). See §6.3 for an unexplained jump in the baseline at 32768. |
| "…while reducing peak activation memory 13.3% through gradient checkpointing" | **Number right, label wrong.** | 13.3% (962 → 834 MB, 128 MB saved) is the peak memory of the **whole training step**, including 320 MB of parameters and gradients. Measured on **activations only**, the reduction is **32%** (400 → 272 MB). Say "13.3% of peak training-step memory (32% of activation memory)". Cost: backward +35.5 ms (+26%). |

---

## 3. GPU background (teach this first)

### 3.1 Execution model
- A **kernel** is launched over a **grid** of **thread blocks**. A block runs on one **SM** (Streaming Multiprocessor;
  the H200 has 132). Its threads are grouped into **warps** of 32 that execute in lockstep (SIMT).
- `threadIdx`, `blockIdx`, `blockDim` and `gridDim` identify each thread's work.
- Each SM has a fixed register file, shared memory, and limits on resident blocks and warps.
  **Occupancy** = active warps ÷ maximum warps per SM. Higher occupancy hides memory latency.

### 3.2 Memory hierarchy (fast → slow)
registers (per thread) → **shared memory** (per block, on-chip, ~100+ KB per SM, banked) → L1 / L2 cache → **global memory** (HBM3e, ~4.8 TB/s on the H200).

### 3.3 Coalescing
When the 32 lanes of a warp access 32 **consecutive** 4-byte words, the hardware serves the warp with a few 128-byte
transactions. If the lanes are strided by N, each lane needs its own transaction, which is up to 32× the traffic.

### 3.4 Shared-memory banks
Shared memory has **32 banks**, each 4 bytes wide; address `i` (in words) lives in bank `i mod 32`. If several lanes of
a warp touch different addresses in the same bank, the accesses are serialised (an *n-way bank conflict*). If all lanes
read the same address, it is a broadcast with no conflict. The standard fix is **padding** a row stride from 32 or 64
to 33 or 65, so column accesses spread across banks.

### 3.5 Arithmetic intensity and the roofline
Arithmetic intensity is FLOPs per byte moved from memory. A kernel is **memory-bound** if its intensity is below
peak FLOP/s ÷ bandwidth (the "ridge point"), and compute-bound otherwise.
- Naive matmul does 2 FLOPs per two 4-byte global loads, an intensity of about 0.25 FLOP/B.
- Tiling with tile size T reuses each loaded element T times, multiplying intensity by about T.
- Register blocking lets one shared-memory load feed R FMAs.

### 3.6 Matmul cost
C = A·B with N×N matrices costs **2N³ FLOPs** (one multiply and one add per inner-product term). TFLOP/s = 2N³ / time.

### 3.7 CUDA streams
A stream is an in-order queue of GPU work. Kernels in **different** streams may run concurrently if resources allow.
The default stream serialises work. Splitting work across streams only helps if each piece leaves resources idle that
another can use.

### 3.8 Gradient checkpointing
Backpropagation needs each layer's forward activations. Checkpointing **keeps only some activations** and **recomputes**
the others during backward. This trades compute (backward gets slower) for memory (a lower peak).

### 3.9 FlashAttention
Standard attention materialises the N×N score matrix S = QKᵀ, so memory is O(N²). FlashAttention tiles Q, K and V into
shared memory and keeps a running max and sum per query row (**online softmax**), so S is never written to global memory.

---

## 4. PA2: matrix multiplication (`matmul/matmul_bench.cu`, 506 lines)

All matrices are N×N, row-major, FP32. Each GPU kernel is checked against cuBLAS on the first min(N², 4M) elements of C
(max relative error reported as 0.0) [CODE `:445-452`, `max_rel_err`].

### 4.1 The kernels, in order

**1. `mm_naive` (`:44-54`).**
- `row = blockIdx.x*32 + threadIdx.x`, `col = blockIdx.y*32 + threadIdx.y`. Consecutive `threadIdx.x` values map to
  consecutive **rows**.
- Inner loop: `acc = fmaf(A[row*N+k], B[k*N+col], acc)`.
- Within a warp, lanes read `A[row*N + k]` at addresses N apart and write `C[row*N+col]` N apart. That is uncoalesced:
  one transaction per lane.
- Launch: 32×32 blocks.

**2. `mm_coalesced` (`:60-70`).**
- Identical arithmetic. Only the mapping is swapped: `col = blockIdx.x*32 + threadIdx.x`.
- Now a warp's lanes read **consecutive columns** of B and write consecutive elements of C (coalesced). They all read
  the same `A[row*N+k]` (a broadcast).
- **10.9× faster than naive** [MEASURED].

**3. `mm_tiled_1to1<32>` (`:77-101`): shared-memory tiling.**
- `__shared__ float sA[32][32], sB[32][32]`.
- For each K-tile t:
  - each thread loads **one** element of A's tile and one of B's tile;
  - `__syncthreads()`;
  - each thread does 32 FMAs out of shared memory;
  - `__syncthreads()`.
- Each global element is loaded once per block and reused 32 times. **+1.6× over coalesced** [MEASURED].

**4/5. `mm_tiled_strided<64,32>` and `mm_tiled_contiguous<64,32>` (`:117-225`).**
- The tile (64×64) is larger than the block (32×32 threads), so `F = TILE/BLK = 2`. Each thread loads F×F = 4 elements
  per tile and owns a 2×2 block of outputs in registers (`acc[F][F]`).
- The two variants differ **only** in which elements a thread fetches:
  - **strided:** element `(ty + s*BLK, tx + u*BLK)`. Lanes stay adjacent, so every load is a coalesced warp transaction.
  - **contiguous:** element `(ty*F + s, tx*F + u)`. Each lane takes an F-wide run, so the warp's addresses are strided
    by F and touch about F× more sectors.
- Measured: strided 19.36 vs contiguous 18.57 TF, **1.04×** in favour of strided, the expected direction [MEASURED].
- Both are **37–39× naive**. That is more than `tiled_1to1`, because 4 outputs per thread raise intensity.

**6. `mm_block2d<TILE,R>` (`:231-287`): 2D register blocking (the Lab Quiz 2 Q3 kernel).**
- Each thread owns an **R×R block** of C: `reg[R][R]`, with `rowBase = blockIdx.y*TILE + ty*R` and `colBase = blockIdx.x*TILE + tx*R`.
- Load phase: each thread loads R×R elements of each 64×64 tile.
- Compute phase, for each k in the tile:
  - load a column of R values `av[i] = sA[ty*R+i][k]` and a row of R values `bv[j] = sB[k][tx*R+j]` into registers;
  - do the **outer product** `reg[i][j] += av[i]*bv[j]` (`:262-273`).
- **2R shared loads feed R² FMAs.** For R = 4 that is 8 loads for 16 FMAs, vs 2 loads per FMA in `tiled_1to1`.
- Launches:
  - `block2d_t64r2`: 32×32 threads, 4 outputs each;
  - **`block2d_t64r4`: 16×16 = 256 threads, 16 outputs each** (`:475-483`).
- R = 4 beats R = 2 by 1.56× [MEASURED]. Total: **51.3× naive, 48.2% cuBLAS**.
- [RECON] Why R = 2 (16.5 TF) is *slower* than the strided tile (19.4 TF): its load pattern `sc = tx*R + j` is the
  "contiguous" pattern (lanes strided by R), and it has fewer outputs per thread than R = 4. So it pays the uncoalesced
  load cost without enough reuse to make up for it.

**7. cuBLAS (`:432-441`).**
- `cublasSgemm` is column-major. Computing **B·A** in column-major layout gives row-major A·B, so the call passes `dB` first.
- It is also the correctness reference.

**CPU baseline (`:292-304`).**
- Single-thread triple loop at `-O3`, measured up to N = 4096 and extrapolated as N³ beyond (about 33.7 h at N = 32768).
- At N = 4096 the best GPU kernel is **42,783×** faster.

### 4.2 Timing harness (`:309-345`)
- `time_kernel`: warm-up launches, then one timed launch with CUDA events. If one launch × iterations fits the 30 s
  budget, it times `iters − 1` more launches and averages.
- `tflops = 2N³ / (ms·1e-3) / 1e12`.
- The naive kernel is skipped if its projected time exceeds a budget (`:457-464`).

### 4.3 Results at N = 32768 [MEASURED, `results/pa2_matmul.csv`]
| Kernel | ms | TFLOP/s | vs naive | % cuBLAS | % peak |
|---|---:|---:|---:|---:|---:|
| GPU naive | 140,486 | 0.50 | 1.0× | 0.9% | 0.7% |
| coalesced | 12,927 | 5.44 | 10.9× | 10.2% | 8.1% |
| tiled 1-to-1 | 8,148 | 8.64 | 17.2× | 16.2% | 12.9% |
| tiled contiguous | 3,790 | 18.57 | 37.1× | 34.8% | 27.7% |
| tiled strided | 3,635 | 19.36 | 38.7× | 36.3% | 28.9% |
| block2d T64 R2 | 4,263 | 16.51 | 33.0× | 30.9% | 24.7% |
| **block2d T64 R4** | **2,737** | **25.71** | **51.3×** | **48.2%** | **38.4%** |
| cuBLAS | 1,318 | 53.37 | 106.6× | 100% | 79.8% |

The ratios are stable across N. At N = 1024 the R4 kernel already reaches 22.1 TF vs cuBLAS 36.3 TF.

### 4.4 Why still only 48% of cuBLAS [RECON]
cuBLAS uses things these kernels don't:
- larger register tiles (e.g. 8×8 per thread);
- vectorised `float4` loads;
- double buffering, which overlaps the next tile's load with compute;
- warp-level tiling;
- tuned launch shapes.

No tensor cores are used on either side; this is FP32 SGEMM. Next steps would be float4 loads, double-buffered shared
memory, bigger R, and a transposed A tile to avoid bank conflicts in `sA[ty*R+i][k]`.

---

## 5. Lab quizzes (correctness gate) [MEASURED]
| Quiz | Kernel | Result |
|---|---|---|
| LQ2 q1 | conv2d + ReLU | 10/10, 10/10 |
| LQ2 q2 | shared-memory transpose | 10/10 (incl. m = 13, 31, 138, 511, 517, 16384) |
| LQ2 q3 | 2D register-blocked matmul | 10/10 |
| LQ3 q1 | sliding-window attention | 4/4 |
| LQ3 q2 | standard attention | 4/4 |
| LQ3 q3 | gradient checkpointing | 4/4 + 4/4 |
| LQ3 q4 | FlashAttention (bonus) | 4/4 |

Harness traps found [DOC]:
- LQ3 q4's launcher passes no dynamic shared-memory size, so `extern __shared__` would be 0 bytes and a static
  allocation is needed. The documented layout needs 96·D + 1024 floats = 52 KB at D = 128, which is over the 48 KB static limit.
- LQ3 q2 launches `<<<grid, 32>>>`, not D threads, so the guards must be `x < N` and `x < D`, and no thread may return
  before a `__syncthreads()`.
- The extractors find a kernel by the first textual occurrence of its name.

---

## 6. PA3: MLP, streams, gradient checkpointing (`mlp/mlp_bench.cu`, 665 lines)

All GEMMs are hand-written. cuBLAS keeps an internal workspace that would pollute the `cudaMemGetInfo` memory readings [DOC `:17-18`].

### 6.1 Part 1 kernel: `mm_thin<BN=256, BK=32>` (`:46-80`)
- Computes `C[M,N] = A[M,K]·B[K,N] (+bias, optional ReLU)` for a small M (batch 32).
- One block per (row m, chunk of 256 columns): `grid = (ceil(N/256), M)`, 256 threads.
- The row's slice of A is staged in shared memory (`sA[BK]`) in chunks of 32. Consecutive threads read consecutive
  columns of B, so B loads are coalesced.
- [DOC] A square tile would waste 24 of every 32 tile rows once the batch is split four ways.

### 6.2 The three forward variants (`:253-299`)
2-layer MLP: `A1 = ReLU(X·W1 + b1)`, `Y = A1·W2 + b2`, X = [32, N], W = [N, N].
- **(a) baseline:** two `mm_thin` launches on the default stream.
- **(b) batch split:** 4 streams, each processing 8 rows (`r0 = s*chunk`) for both layers.
- **(c) column split (not in the brief, added by him):** layer 1 is split by output column (each stream gets
  `W1 + c0`, `b1 + c0`, `A1 + c0` and N/4 columns). Then **synchronise all streams**, because layer 2 needs all of A1.
  Layer 2 is then split by column the same way. Passing `W1 + c0` with leading dimension N selects a column sub-block of a row-major matrix (`:313-317`).
- Measurement: 3 warm-up runs, then CUDA events around `iters` runs (`:301-311`).

### 6.3 Results, B = 32 (ms) [MEASURED]
| N | baseline | batch split | column split |
|---|---:|---:|---:|
| 32 | 0.0069 | 0.0278 | 0.0338 |
| 1024 | 0.0990 | 0.1200 | 0.1253 |
| 8192 | 1.504 | 1.526 | 1.549 |
| 16384 | 6.293 | 6.847 | 6.186 |
| 32768 | 44.43 | 52.28 | **25.65** (1.73×) |

- **Batch split never helps** [DOC]. At B = 32 each stream gets 8 rows, but every stream still streams the **entire
  weight matrix** (the same weight traffic), plus 4× the launches. Below N ≈ 2048 everything is launch-latency bound
  (7 µs at N = 32).
- **Column split** gives each stream its own slice of the output *and of the weights*, so the concurrent kernels have
  independent work and memory. It pays only at the largest size [DOC].
- **[RECON] Something to be ready for.** From 16384 to 32768 the work and weight bytes grow 4×, but the baseline time
  grows **7×** (6.29 → 44.43 ms). The column split grows a normal ~4×. So the 1.73× at 32768 coincides with an anomaly
  in the baseline, plausibly cache locality: each stream touches a 4× smaller weight slice, which changes L2 reuse.
  It may not be pure stream concurrency. Without `ncu` this was not profiled. Say "at N = 32768 the column split was
  1.73× faster; I couldn't profile why, and the baseline's super-linear jump suggests memory locality plays a role".

### 6.4 Part 2: 4-layer MLP training step with checkpointing (`:337-560`)
- Forward: `Z1 = X·W1 + b1, A1 = ReLU(Z1)`, …, `Z4 = A3·W4 + b4`. Loss is MSE, then SGD.
- **Baseline** keeps all of Z1, A1, Z2, A2, Z3, A3, Z4 resident.
- **Checkpointed** keeps only **A2 and Z4** (`:412-420`) and rebuilds:
  - **segment B** (Z3, A3) from A2 at the start of backward (`:493-501`);
  - **segment A** (Z1, A1, Z2) from X after layers 4 and 3 are done (`:513-523`).
- **The key trick, one arena** (`:441-462`). Backward runs layer 4 → 1, so segment B is dead before segment A is rebuilt.
  - One buffer of `max(2·n3, 2·n1 + n2)` floats serves both: `rZ1 = arena`, `rA1 = rZ1+n1`, `rZ2 = rA1+n1`, and `rZ3 = arena`, `rA3 = rZ3+n3` (aliased).
  - **Bug story [DOC]:** the first version allocated one buffer per dropped activation. That holds exactly what the
    baseline holds and saved **0 MB**.
- Backward kernels:
  - `k_mse_grad` → `dZ4 = 2(Z4 − Y)/n`;
  - `dW = Aᵀ·dZ` via `mm_tiled_AT`;
  - `dA_prev = dZ·Wᵀ` via `mm_tiled_BT`;
  - `k_relu_deriv` (zero where Z ≤ 0), `k_bias_grad` (column sums), `k_sgd`.
- Peak memory = `cudaMemGetInfo` free before − free after allocating everything. Every buffer is preallocated, so this
  equals the step's peak (`:429-434`, `:546-550`).
- Weights are initialised with scale 1/√fan_in. Without it, 4096-wide layers blow up to inf/NaN and the checksum cross-check becomes useless [DOC `:396-400`].

### 6.5 Results (100 iterations) [MEASURED]
| batch, dims | mode | fwd ms | bwd ms | peak MB | saved | bwd overhead |
|---|---|---:|---:|---:|---:|---:|
| 1024, 512-1024³-512 | base / ckpt | 0.772 / 0.772 | 2.631 / 3.255 | 70 / 62 | 8 MB (11.4%) | +23.8% |
| 2048, 1024-2048³-1024 | base / ckpt | 5.798 / 5.799 | 19.863 / 24.668 | 274 / 242 | 32 MB (11.7%) | +24.2% |
| **4096, 1024-4096³-1024** | base / ckpt | 39.495 / 39.479 | 136.147 / 171.689 | 962 / 834 | **128 MB (13.3%)** | **+35.54 ms (+26.1%)** |

- **Forward time is unchanged:** checkpointing does not skip forward work, it just doesn't keep the results.
- Both modes give **identical post-update weight checksums** (−84.0866, −259.7291, −379.7093).
- **Accounting for the 4096 config [RECON, checked].** Nb = 4096, H = 4096, so n1 = n2 = n3 = 16.8M floats = 64 MB, and n4 = 4096×1024 = 16 MB.
  - Baseline activations = 6×64 + 16 = **400 MB**.
  - Checkpointed = A2 (64) + Z4 (16) + arena max(3×64, 2×64) = 192 → **272 MB**.
  - Saved 128 MB, which is **32% of activations** and 13.3% of the whole step (the step also holds 320 MB of parameters and gradients that checkpointing cannot touch).

### 6.6 Two code findings the repo doesn't state [RECON, from reading the code; not run on a GPU]
1. **`mm_tiled_AT` does not compute Aᵀ·B** (`mlp_bench.cu:109-126`).
   - It is meant to compute `C[row][col] = Σ_k A[k][row]·B[k][col]`, with A stored [K, M].
   - The load writes `sA[tx][ty] = A[(t*TS+tx)*M + row]`, i.e. `sA[i][j] = A[t*TS+i][rowBase+j]`.
   - The FMA then reads `sA[ty][k]`, which is `A[t*TS+ty][rowBase+k]`. Correct would be `sA[k][ty]` = `A[t*TS+k][row]`.
   - So the transpose is applied twice and every `dW` is wrong.
   - **What it does not affect:** timing, memory, and the equality of checksums between modes, because both modes call the same wrong kernel.
   - **What it does affect:** the training step does not compute the true gradient.
   - Simplest fix: keep the load and change the FMA to `acc = fmaf(sA[k][ty], sB[k][tx], acc)`, since `sA[k][ty]` = `A[t*TS+k][row]`.
   - If asked, say honestly that the checksum check could not catch it, because it compares two modes that share the kernel; a check against a CPU or PyTorch gradient would have.
2. **One redundant recompute.** In checkpointed mode the forward pass writes z3 and a3 into the arena, and nothing overwrites the arena before backward starts. Yet backward recomputes segment B from A2 (`:493-498`). That layer-3 GEMM is unnecessary in this schedule.
   - Estimate by FLOP share for the 4096 config: the recompute GEMMs are A2·W3 (4096³ × 2 = 137 GFLOP), X·W1 (34 GFLOP) and A1·W2 (137 GFLOP).
   - 35.5 ms for 309 GFLOP is about 8.7 TFLOP/s, which matches the tiled kernel's speed.
   - So the redundant layer-3 GEMM is about **16 ms of the 35.5 ms overhead (~45%)**.
   - Removing it would leave the memory saving unchanged and cut the overhead to about +14%. This is an estimate and was not measured.

### 6.7 Choosing what to checkpoint [RECON]
Keeping A2 (the middle) splits the network into two recompute segments of about equal size. Keeping Z4 (the output pre-activation) means the loss gradient needs no recompute. The general rule is checkpoints every √L layers, which gives O(√L) memory for one extra forward.

---

## 7. PA4: FlashAttention forward (`flashattention/flash_lib.cu`, 257 lines)

### 7.1 The kernel (`flash_forward_kernel`, `:46-184`)
- FP32, **no causal mask**, shapes [B, NH, N, D]. Defines: BR = BC = 32, **RPW = 4 query rows per warp**, 8 warps
  (256 threads) per block. `grid = (Tr = ceil(N/32), NH, B)`.
- Shared layout: `Qi[32·D] | Kj[32·(D+1)] | Vj[32·D] | S[32·32]` = **28,800 B** at D = 64.
  **`Kj` has row stride D+1** (`KS = D+1`, `:55`).
- Stage the Q tile once. Then for each K/V tile j:
  1. **Load** Kj and Vj cooperatively; `__syncthreads()`.
  2. **Scores.** Lane c owns key c. The loop over x (feature) is outermost, so **one `Kj[c*KS+x]` load feeds the 4 query
     rows** of this warp (`:118-126`).
  3. **Per-row online softmax via warp shuffles** (`:128-152`):
     - `m_tile` = warp max via `__shfl_xor_sync` butterflies (offsets 16, 8, 4, 2, 1);
     - `p = exp(s − m_tile)`; `l_tile` = warp sum;
     - store p in `S[row][lane]`;
     - `m_new = max(m_run, m_tile)`, `e_run = exp(m_run − m_new)`, `e_tile = exp(m_tile − m_new)`;
     - `l_run = e_run·l_run + e_tile·l_tile`.
  4. **PV.** Each lane owns feature coordinates `lane`, `lane+32`. The loop over c is innermost, so one `Vj` load feeds
     4 rows. Then `o_acc = o_acc·e_run + pv·e_tile` (`:155-172`).
- At the end, `O = o_acc / l_run` (`:175-183`).
- `flash_attention_forward` is exposed as `extern "C"`, called from PyTorch via ctypes. It requires D ≤ 128 (`o_acc` holds ≤ 4 slots per lane).

### 7.2 Optimisation history at N = 8192 [MEASURED]
| Version | Time | Why |
|---|---:|---|
| 1. Quiz layout: grid(B, NH), 32 threads, query loop inside the kernel | 3358 ms | only 128 single-warp blocks on 132 SMs |
| 2. grid(Tr, NH, B): query loop becomes grid parallelism | 2377 ms | 32,768 blocks, but still limited by (3) |
| 3. **Pad Kj stride to D+1** | **339 ms** (7.0×) | Lane c reads `Kj[c*64 + x]`, so every lane hits bank (c·64 mod 32) = 0: a **32-way bank conflict** in the hottest loop. With stride 65, lane c hits bank (c·65 + x) mod 32 = (c + x) mod 32, all distinct. |
| 4. **4 query rows per warp** | **160 ms** | At 1 row per warp there were 2 shared loads per FMA, so measured 6.5 TFLOP/s was exactly the shared-bandwidth roofline (about 8 TF). With 4 rows, one Kj/Vj load feeds 4 FMAs. |

That is 21× overall.

### 7.3 Results (B = 8, NH = 16, D = 64, FP32, 10 iterations) [MEASURED]
| N | PyTorch standard | custom flash | PyTorch fused SDPA | score tensor |
|---|---:|---:|---:|---:|
| 32 | 0.119 | **0.017** | 0.023 | 0.5 MB |
| 128 | 0.109 | **0.051** | 0.025 | 8 MB |
| 256 | **0.111** | 0.183 | 0.082 | 34 MB |
| 1024 | **1.476** | 2.625 | 1.032 | 537 MB |
| 8192 | **105.08** | 160.05 | 62.62 | 34.4 GB |

- Max abs difference ≤ 7.2×10⁻⁷ (the requirement was 10⁻³).
- Crossover at N = 256. The small-N "win" is launch overhead: PyTorch's explicit path pays about a 0.11 ms floor for 3 kernels.
- Kernel stats: 62 registers per thread, 4 blocks per SM, **50% theoretical occupancy**. 13.75 TFLOP/s at N = 8192 (20.5% of peak).
- Remaining gap: PyTorch's path is cuBLAS SGEMM, much more register-blocked. This kernel reads both GEMM operands from
  shared memory and uses no tensor cores [DOC].
- The flash kernel's memory win is real: no N×N tensor. At N = 8192 the standard path needs 34.4 GB for scores.

---

## 8. PA1 (not run)
`kvcache/model.py` (nanoGPT-style GPT-2 with a per-layer KV cache threaded through `Block` and `GPT.forward`, `flash=False`), `kvcache/radix_tree.py` (edge-compressed radix tree over token ids for prefix sharing), and `kvcache/bench_pa1.py`. GPT-2 124M weights load, but **the benchmark sweep was never executed** [DOC]. Don't claim PA1 numbers.

---

## 9. Design decisions

| Decision | Why | Source |
|---|---|---|
| Kernel-only timing with CUDA events | isolates the kernel from PCIe transfers | [DOC] |
| cuBLAS as both reference and ceiling | trusted result and practical upper bound | [DOC] |
| Peak computed from SM count × 128 lanes × 2 × clock | FP32 non-tensor peak for honest % | [DOC] |
| TILE 64, R 4, 16×16 threads | 16 outputs per thread; best measured | [MEASURED] |
| Column split for streams (beyond the brief) | the batch split can't add parallelism at B = 32 | [DOC] |
| Own GEMMs in PA3 | cuBLAS workspace pollutes `cudaMemGetInfo` | [DOC] |
| Checkpoint A2 and Z4 | the brief requires it; splits into two segments | [DOC] |
| One aliased arena | two segments never live at once; per-buffer allocation saved 0 | [DOC] |
| 1/√fan_in init | stable training, so checksums are meaningful | [DOC] |
| Kj padded to D+1 | removes the 32-way bank conflict | [DOC] |
| 4 rows per warp | escape the shared-bandwidth roofline | [DOC] |
| Static shared memory in quiz q4 | the harness passes no dynamic size | [DOC] |

---

## 10. Limitations
- No hardware profiling (`ncu` needs root). Bank-conflict and roofline claims are reasoned from measured TFLOP/s and code, not counters.
- FP32 only; no tensor cores (TF32/BF16 would be far faster).
- The stream speedup appears only at N = 32768, and its cause is not isolated (§6.3).
- The PA3 gradient kernel `mm_tiled_AT` is incorrect (§6.6).
- PA1 never benchmarked. Single GPU, shared machine.

---

## 11. Likely interview questions
1. *Why does coalescing alone give 10.9×?* → §3.3 and §4.1: same FLOPs, a 32× difference in transactions for B and C.
2. *What is the arithmetic intensity of naive vs tiled vs register-blocked?* → about 0.25 FLOP/B naive; tiling multiplies it by the tile size; register blocking cuts shared loads to 2R per R² FMAs.
3. *Draw which thread computes which outputs in `block2d<64,4>`.* → thread (tx, ty) owns rows `by·64 + ty·4 … +3` and cols `bx·64 + tx·4 … +3`; 256 threads × 16 = 64×64.
4. *Why is strided loading better than contiguous?* → lanes stay adjacent, so loads coalesce.
5. *How did you compute 48.2% and 51.3%?* → §2.
6. *What would close the gap to cuBLAS?* → §4.4.
7. *Why did batch-splitting streams not help?* → same weight traffic per stream, 4× launches, little row parallelism.
8. *Explain gradient checkpointing and your memory accounting.* → §6.4–6.5: 400 → 272 MB of activations, 13.3% of the step.
9. *Why one arena?* → segments are never live together; the first version saved 0.
10. *Is your backward correct?* → be honest about `mm_tiled_AT` (§6.6) and how you would validate (compare against PyTorch autograd).
11. *Explain the bank conflict and the D+1 fix.* → §7.2.
12. *Why did your FlashAttention lose to PyTorch at large N?* → shared-bandwidth bound, no tensor cores, cuBLAS register blocking; 50% occupancy.
13. *What is the online softmax and why is it exact?* → rescale by exp(m_old − m_new); the final division by l.
14. *Why can't you measure achieved occupancy?* → `ncu` permission; theoretical occupancy only.
15. *What does `__shfl_xor_sync` do?* → a butterfly exchange of registers within a warp; reductions in 5 steps without shared memory.

---

## 12. Numbers cheat sheet
| Number | Meaning |
|---|---|
| 66.91 TFLOP/s | H200 FP32 peak (132 × 128 × 2 × 1.98 GHz) |
| 0.50 → 25.71 TF | naive → best custom kernel at N = 32768 |
| 51.3× | best custom vs naive GPU |
| 48.2% / 38.4% | of cuBLAS (53.37 TF) / of peak |
| 10.9×, 1.6×, 3.0× | coalescing, tiling, register blocking steps |
| 1.04× | strided vs contiguous tile loads |
| 42,783× | best GPU vs single-thread CPU at N = 4096 |
| 1.73× | column-split streams at N = 32768 (44.43 → 25.65 ms) |
| 13.3% / 32% | memory saved: whole step (962 → 834 MB) / activations (400 → 272 MB) |
| +26.1% | backward overhead of checkpointing (136.1 → 171.7 ms) |
| 21× | FlashAttention tuning (3358 → 160 ms at N = 8192) |
| 7.0× | from the D+1 bank-conflict padding alone |
| N = 256 | crossover where custom flash becomes slower than explicit PyTorch |
| 28,800 B, 50% | flash shared memory per block, theoretical occupancy |

---

## Appendix A: Key code, annotated

Real line numbers from `matmul/matmul_bench.cu`, `mlp/mlp_bench.cu` and `flashattention/flash_lib.cu`. `⋮` = left out; `◀` = plain-English note. `#pragma unroll` lines are left out¹.

### Reading the code: CUDA / C++ constructs
| Construct | Meaning |
|---|---|
| `__global__ void k(...)` | a kernel: runs on the GPU, launched from the CPU |
| `k<<<grid, block>>>(args)` | launch `grid` blocks of `block` threads each. `dim3(32,32)` = 2D, 1,024 threads per block |
| `blockIdx.x/y`, `threadIdx.x/y`, `blockDim` | this block's index in the grid; this thread's index in its block; block size |
| `__shared__ float sA[T][T]` | on-chip memory shared by the threads of one block (fast, about 100 KB per SM) |
| `__syncthreads()` | barrier: every thread of the block waits here (so a tile is fully loaded before use) |
| `fmaf(a, b, c)` | fused multiply-add a·b + c in one instruction, one rounding |
| `template<int TILE, int R>` | compile-time parameters; each instantiation (e.g. `<64,4>`) is a separate kernel with constants folded in |
| `(size_t)row * N` | 64-bit index (N² can exceed 2³¹ at N = 32768) |
| `extern __shared__ float smem[]` | dynamically sized shared memory (size given as the 3rd launch parameter) |
| `__shfl_xor_sync(mask, v, off)` | exchange a register with lane `lane ^ off` in the same warp (no shared memory) |
| `__expf` | fast, lower-precision exp intrinsic |
| `cudaStream_t`, `<<<g, b, 0, st[s]>>>` | a work queue; kernels in different streams may run concurrently |
| `cudaMemGetInfo(&free, &total)` | free and total device memory in bytes |

¹ `#pragma unroll` asks the compiler to copy a loop body N times (no loop counter), which helps when the trip count is a compile-time constant.

### A.1 Naive vs coalesced: only the mapping changes (`matmul_bench.cu`)
```cuda
44  __global__ void mm_naive(const float* A, const float* B, float* C, int N) {
46      int row = blockIdx.x * blockDim.x + threadIdx.x;   // ◀ consecutive threads → consecutive ROWS
47      int col = blockIdx.y * blockDim.y + threadIdx.y;
⋮
51      for (int k = 0; k < N; ++k)
52          acc = fmaf(A[(size_t)row * N + k], B[(size_t)k * N + col], acc);   // ◀ lanes of a warp read A N floats apart: 32 transactions
53      C[(size_t)row * N + col] = acc;                    // ◀ writes also N apart: uncoalesced
⋮
60  __global__ void mm_coalesced(const float* A, const float* B, float* C, int N) {
62      int col = blockIdx.x * blockDim.x + threadIdx.x;   // ◀ SWAPPED: consecutive threads → consecutive COLUMNS
63      int row = blockIdx.y * blockDim.y + threadIdx.y;
    ⋮   // same loop: now a warp reads 32 adjacent B values (1 transaction) and broadcasts one A value → 10.9×
```

### A.2 Shared-memory tiling, 1 element per thread (`mm_tiled_1to1<32>`)
```cuda
77  template<int TILE>
78  __global__ void mm_tiled_1to1(const float* A, const float* B, float* C, int N) {
80      __shared__ float sA[TILE][TILE];                 // ◀ 32×32 tile of A in shared memory
81      __shared__ float sB[TILE][TILE];
83      const int tx = threadIdx.x, ty = threadIdx.y;
84      const int row = blockIdx.y * TILE + ty;          // ◀ the output element this thread owns
85      const int col = blockIdx.x * TILE + tx;
88      for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {  // ◀ walk along K one tile at a time
89          int aCol = t * TILE + tx;
90          int bRow = t * TILE + ty;
91          sA[ty][tx] = (row < N && aCol < N) ? A[(size_t)row * N + aCol] : 0.0f;  // ◀ each thread loads ONE A element
92          sB[ty][tx] = (bRow < N && col < N) ? B[(size_t)bRow * N + col] : 0.0f;  // ◀ and ONE B element
93          __syncthreads();                               // ◀ tile complete before anyone reads it
96          for (int k = 0; k < TILE; ++k)
97              acc = fmaf(sA[ty][k], sB[k][tx], acc);     // ◀ 32 FMAs from shared memory: each loaded value reused 32×
98          __syncthreads();                               // ◀ don't overwrite the tile while others still read it
99      }
100     if (row < N && col < N) C[(size_t)row * N + col] = acc;
```
This is block matrix multiplication: C_tile = Σ_t A_tile(row, t) · B_tile(t, col).

### A.3 Bigger tile than block: strided vs contiguous loads (only the changed lines)
```cuda
117 template<int TILE, int BLK>                        // ◀ launched as <64,32>: 32×32 threads, 64×64 tile
118 __global__ void mm_tiled_strided(...) {
120     constexpr int F = TILE / BLK;                  // ◀ F = 2: each thread loads 2×2 and computes 2×2 outputs
128     float acc[F][F];                               // ◀ 4 accumulators in registers
139             int r = ty + s * BLK;                  // ◀ STRIDED: thread loads rows ty, ty+32 …
140             int c = tx + u * BLK;                  // ◀ … cols tx, tx+32: adjacent lanes stay adjacent → coalesced
⋮
194             int r = ty * F + s;                    // ◀ CONTIGUOUS variant (mm_tiled_contiguous)
195             int c = tx * F + u;                    // ◀ lane tx loads cols 2tx, 2tx+1: lanes strided by 2 → more sectors
```
Measured: strided 19.36 vs contiguous 18.57 TFLOP/s at N = 32768.

### A.4 2D register blocking: the 51.3× kernel (`mm_block2d<64,4>`, launched with 16×16 threads)
```cuda
231 template<int TILE, int R>
232 __global__ void mm_block2d(const float* A, const float* B, float* C, int N) {
234     __shared__ float sA[TILE][TILE];               // ◀ 64×64 = 16 KB each
235     __shared__ float sB[TILE][TILE];
237     const int tx = threadIdx.x, ty = threadIdx.y;  // ◀ 0..15 each
238     const int rowBase = blockIdx.y * TILE + ty * R;   // ◀ this thread owns rows rowBase..rowBase+3
239     const int colBase = blockIdx.x * TILE + tx * R;   // ◀ and cols colBase..colBase+3 → 16 outputs
241     float reg[R][R];                               // ◀ 4×4 accumulators in REGISTERS
247     for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
248         const int kOff = t * TILE;
⋮           // 249-259: each thread loads R×R = 16 elements of the A tile and of the B tile (256 threads × 16 = 64×64)
260         __syncthreads();
262         for (int k = 0; k < TILE; ++k) {
263             float av[R], bv[R];
265             for (int i = 0; i < R; ++i) av[i] = sA[ty * R + i][k];   // ◀ 4 values of A's column k
267             for (int j = 0; j < R; ++j) bv[j] = sB[k][tx * R + j];   // ◀ 4 values of B's row k
269             for (int i = 0; i < R; ++i)
271                 for (int j = 0; j < R; ++j)
272                     reg[i][j] = fmaf(av[i], bv[j], reg[i][j]);         // ◀ OUTER PRODUCT: 8 shared loads → 16 FMAs
273         }
274         __syncthreads();
275     }
⋮   // 277-286: write the 4×4 block of C with bounds checks
480 dim3 gb((N + 63) / 64, (N + 63) / 64);   // ◀ host side: one block per 64×64 output tile
481 dim3 bb(16, 16);
482 record("block2d_t64r4", time_kernel([&]{ mm_block2d<64,4><<<gb,bb>>>(dA,dB,dC,N); }, 1, iters, budget));
```
| Name | Meaning (R = 4, TILE = 64) |
|---|---|
| `R` | outputs per thread per dimension (4 → 16 outputs) |
| `rowBase`, `colBase` | top-left of the thread's 4×4 output patch |
| `av`, `bv` | register copies of a column of A and a row of B |
| `reg` | the 16 running sums |

### A.5 Timing harness and TFLOP/s (`matmul_bench.cu`)
```cuda
322 CUDA_CHECK(cudaEventRecord(beg));     // ◀ GPU timestamps around the launch (kernel time only)
323 launch();
324 CUDA_CHECK(cudaEventRecord(end));
325 CUDA_CHECK(cudaEventSynchronize(end));
327 CUDA_CHECK(cudaEventElapsedTime(&first, beg, end));
⋮
347 static double tflops(int N, double ms) {
349     return (2.0 * (double)N * N * N) / (ms * 1e-3) / 1e12;   // ◀ 2N³ FLOPs
⋮
391 double peak = (double)prop.multiProcessorCount * 128.0 * 2.0 * ((double)clock_khz * 1e3) / 1e12;  // ◀ 132×128×2×1.98 GHz = 66.9
⋮
436 cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dB, N, dA, N, &beta, dC, N);
    // ◀ cuBLAS is column-major: computing B·A in column-major = A·B in row-major
```

### A.6 The MLP forward and the column-split streams (`mlp_bench.cu`)
```cuda
46  template<int BN, int BK>                 // ◀ BN = 256 columns per block, BK = 32 K-chunk
47  __global__ void mm_thin(const float* A, const float* B, const float* bias, float* C, int M, int K, int N, int relu) {
53      __shared__ float sA[BK];             // ◀ one row-slice of A (the batch row), reused by 256 columns
55      const int m   = blockIdx.y;          // ◀ batch row
56      const int col = blockIdx.x * BN + threadIdx.x;   // ◀ output column → B reads are coalesced
60      for (int k0 = 0; k0 < K; k0 += BK) {
62          for (int t = threadIdx.x; t < BK; t += BN) sA[t] = (k0 + t < K) ? A[(size_t)m * K + k0 + t] : 0.0f;
66          __syncthreads();
69              int kmax = (K - k0 < BK) ? (K - k0) : BK;   // ◀ last chunk may be short
70              for (int kk = 0; kk < kmax; ++kk) acc = fmaf(sA[kk], B[(size_t)(k0 + kk) * N + col], acc);
73          __syncthreads();
⋮
77      float v = acc + (bias ? bias[col] : 0.0f);
78      C[(size_t)m * N + col] = relu ? (v > 0.0f ? v : 0.0f) : v;    // ◀ fused bias + ReLU
⋮
280 auto streamed_col = [&] {                // ◀ (c) column split: each stream owns N/4 output columns
281     for (int s = 0; s < NS; ++s) {
282         int c0 = s * cchunk;
285         dim3 g((cols + BN - 1) / BN, B);
286         mm_thin<BN,BK><<<g, blk, 0, st[s]>>>(X, W1 + c0, b1 + c0, A1 + c0, B, N, N, 1);   // ◀ pointer offset = column sub-block
288     }
289     for (int s = 0; s < NS; ++s) cudaStreamSynchronize(st[s]);   // ◀ layer 2 needs ALL of A1
290     ...                                  // ◀ 290-298: same split for layer 2 (A1 · W2)
```

### A.7 Gradient checkpointing with one aliased arena (`mlp_bench.cu`)
```cuda
412 CUDA_CHECK(cudaMalloc(&A2, n2 * sizeof(float)));   // checkpoint   ◀ kept in BOTH modes
413 CUDA_CHECK(cudaMalloc(&Z4, n4 * sizeof(float)));   // checkpoint
414 if (!checkpointed) { /* 415-419: Z1, A1, Z2, Z3, A3 each get their own buffer */ }
⋮
455 if (checkpointed) {
456     size_t segA = 2 * n1 + n2;                     // ◀ Z1, A1, Z2 (layers 1-2)
457     size_t segB = 2 * n3;                          // ◀ Z3, A3 (layer 3)
458     size_t arena_floats = segA > segB ? segA : segB;   // ◀ ONE buffer = the larger segment
459     CUDA_CHECK(cudaMalloc(&arena, arena_floats * sizeof(float)));
460     rZ1 = arena;  rA1 = rZ1 + n1;  rZ2 = rA1 + n1; // ◀ segment A layout
461     rZ3 = arena;  rA3 = rZ3 + n3;                  // ◀ segment B ALIASES the same memory
462 }
⋮
493 if (checkpointed) {                                // ◀ backward, step 1: rebuild segment B from A2
495     mm_tiled<<<g2(Nb,H3), dim3(TS,TS)>>>(A2, W3, rZ3, Nb, H2, H3);
⋮   // 496-497 bias + ReLU into rA3; 503-511: layer 4 and 3 gradients use rZ3/rA3
514 if (checkpointed) {                                // ◀ z3/a3 are dead now → overwrite the arena with segment A
515     mm_tiled<<<g2(Nb,H1), dim3(TS,TS)>>>(X, W1, rZ1, Nb, IN, H1);
518     mm_tiled<<<g2(Nb,H2), dim3(TS,TS)>>>(rA1, W2, rZ2, Nb, H1, H2);
```
Note from §6.6: in this schedule the forward pass already left z3/a3 in the arena, so the recompute at `:495` is redundant.

### A.8 The bug in `mm_tiled_AT` (`mlp_bench.cu:109-126`)
```cuda
109 __global__ void mm_tiled_AT(const float* A, const float* B, float* C, int M, int K, int N) // A is [K, M]; want C = Aᵀ·B
114     int row = blockIdx.y * TS + ty, col = blockIdx.x * TS + tx;
117     int aR = t * TS + tx, bR = t * TS + ty;
118     sA[tx][ty] = (aR < K && row < M) ? A[(size_t)aR * M + row] : 0.f;   // ◀ stores sA[i][j] = A[t·TS+i][rowBase+j]
119     sB[ty][tx] = (bR < K && col < N) ? B[(size_t)bR * N + col] : 0.f;
122     for (int k = 0; k < TS; ++k) acc = fmaf(sA[ty][k], sB[k][tx], acc);   // ◀ reads A[t·TS+ty][rowBase+k]: WRONG
    // ◀ correct operand is A[t·TS+k][row] = sA[k][ty] → fix: fmaf(sA[k][ty], sB[k][tx], acc)
```

### A.9 FlashAttention: the bank-conflict fix and online softmax (`flash_lib.cu`)
```cuda
55  const int KS = D + 1;                 // ◀ PADDED row stride: 65 instead of 64
57  extern __shared__ float smem[];
58  float* Qi = smem;                     // ◀ [32 × D]
59  float* Kj = Qi + BR * D;              // ◀ [32 × (D+1)]
60  float* Vj = Kj + BC * KS;             // ◀ [32 × D]
61  float* S  = Vj + BC * D;              // ◀ [32 × 32] probabilities of this tile
63  const int lane = threadIdx.x;         // ◀ 0..31
64  const int w    = threadIdx.y;         // ◀ warp 0..7; owns query rows r0..r0+3
⋮
118 if (lane < kv_len) {
119     const float* kptr = Kj + lane * KS;   // ◀ lane c reads row c: address c·65 + x → bank (c + x) mod 32, all different
120     for (int x = 0; x < D; ++x) {
121         const float kv = kptr[x];
123         for (int r = 0; r < RPW; ++r)
124             a[r] = fmaf(Qi[(r0 + r) * D + x], kv, a[r]);   // ◀ one K load feeds 4 query rows
⋮
131     float s = (lane < kv_len) ? a[r] * softmax_scale : -INFINITY;
133     float m_tile = s;
135     for (int off = 16; off > 0; off >>= 1)
136         m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, off));   // ◀ warp-wide max in 5 steps
138     float p = (lane < kv_len) ? __expf(s - m_tile) : 0.0f;
⋮   // 140-143: warp-wide sum l_tile the same way; 145: S[row][lane] = p
147     const float m_new = fmaxf(m_run[r], m_tile);
148     e_run[r]  = __expf(m_run[r] - m_new);   // ◀ rescale old accumulators
149     e_tile[r] = __expf(m_tile   - m_new);   // ◀ rescale this tile
150     l_run[r]  = e_run[r] * l_run[r] + e_tile[r] * l_tile;
⋮
171     o_acc[r][jj] = o_acc[r][jj] * e_run[r] + pv[r] * e_tile[r];   // ◀ output accumulator update
⋮
182     O[base + (size_t)qrow * D + x] = o_acc[r][jj] * inv;          // ◀ inv = 1 / l_run: final normalisation
```
With stride 64, lane c reads `Kj[c·64 + x]`, so bank = (c·64 + x) mod 32 = x mod 32 for **every** lane. That is a 32-way conflict, and fixing it gave 7.0× on its own.
