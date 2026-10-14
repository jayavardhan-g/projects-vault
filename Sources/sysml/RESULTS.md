# CS794 — Measured Results

Hardware: **NVIDIA H200 SXM**, 132 SMs, 1980 MHz, 141 GB HBM3e.
Theoretical FP32 (non-tensor) peak: **66.91 TFLOP/s** = 132 SM × 128 FP32 lanes × 2 flop/FMA × 1.98 GHz.
Toolchain: CUDA 13.0, gcc 13.3, PyTorch 2.13.0+cu130, env `sysml` (Python 3.11).
All GPU work pinned to an explicitly idle GPU on this shared machine.

Raw data: `results/*.csv`. Figures: `results/*.png`.

---

## Lab quizzes — correctness gate

All seven `student.cu` files pass their provided evaluators, and every merged
binary is clean under `compute-sanitizer memcheck` (0 errors).

| Quiz | Question | Result |
|---|---|---|
| LQ2 q1 | conv2d + ReLU | conv **10/10**, relu **10/10** |
| LQ2 q2 | shared-memory transpose | **10/10** (incl. m = 13, 31, 138, 511, 517, 16384) |
| LQ2 q3 | 2D register-blocked matmul | **10/10** |
| LQ3 q1 | sliding-window attention | **4/4** |
| LQ3 q2 | standard attention | **4/4** |
| LQ3 q3 | gradient checkpointing | **4/4** + evaluator2 kernel tests **4/4** |
| LQ3 q4 | FlashAttention (bonus) | **4/4** |

Run with `bash tools/run_quiz.sh <2|3> <q> [--sanitize]`.

### Three harness traps worth knowing

1. **LQ3 q4's launcher passes no dynamic shared-memory size** —
   `flash_forward_kernel<<<grid, block>>>` (`evaluator.cu:88`). The `extern
   __shared__` the writeup prescribes would therefore name a **zero-byte**
   buffer. A static allocation is required. The grading script confirms this
   independently: it rewrites `constexpr int D_MAX = 128;` to `80`, because the
   documented `Qi|Kj|Vj|S` layout needs 96·D + 1024 floats, which is 52 KB at
   D = 128 and overflows the 48 KB static ceiling.
2. **LQ3 q2's launcher is `<<<grid, 32>>>`**, not `<<<grid, D>>>` as the block
   dimension `(D,1,1)` in the writeup implies. Guards must be `x < N` for the
   score phase and `x < D` for the output phase, and no thread may return early
   — all 32 must reach both `__syncthreads()`.
3. **LQ3 q3's evaluator defines neither `forward_pass` nor `backward_pass`**;
   the student block has to supply the entire MLP. Separately, `evaluator2.cu`
   extracts *only* `bias_gradient` and `relu_derivative`, so those two must
   compile standalone.

Also note the LQ2 extractors are textual: they locate a kernel by the **first**
occurrence of its name anywhere in the file, and stop the `#define` scan at the
first literal `__global__`. Naming a kernel in a header comment above its
definition breaks extraction.

---

## PA2 — CUDA matrix multiplication

`results/pa2_matmul.csv`, `pa2_time_vs_n.png`, `pa2_tflops_vs_n.png`.
Kernel time only (CUDA events; H2D/D2H excluded). Every GPU kernel matches
cuBLAS bit-exactly up to N = 16384 (max relative error 0.0).

### Throughput at N = 32768

| Implementation | Time (ms) | TFLOP/s | vs naive | % of cuBLAS | % of peak |
|---|---:|---:|---:|---:|---:|
| CPU naive (`-O3`, 1 thread) | 121 468 000 *(extrap.)* | 0.0006 | — | — | — |
| GPU naive (32×32, uncoalesced) | 140 486 | 0.50 | 1.0× | 0.9 % | 0.7 % |
| GPU coalesced | 12 927 | 5.44 | **10.9×** | 10.2 % | 8.1 % |
| Tiled, 1-to-1 | 8 148 | 8.64 | 17.2× | 16.2 % | 12.9 % |
| Tiled, contiguous load | 3 790 | 18.57 | 37.1× | 34.8 % | 27.7 % |
| Tiled, strided load | 3 635 | 19.36 | 38.7× | 36.3 % | 28.9 % |
| 2D register block, TILE 64 R 2 | 4 263 | 16.51 | 33.0× | 30.9 % | 24.7 % |
| **2D register block, TILE 64 R 4** | **2 737** | **25.71** | **51.3×** | **48.2 %** | **38.4 %** |
| cuBLAS SGEMM | 1 318 | 53.37 | 106.6× | 100 % | 79.8 % |

### What each optimisation bought

* **Coalescing alone: 10.9×.** Same arithmetic, same occupancy — only the
  thread→element mapping changes, so consecutive lanes walk consecutive columns
  and a warp's access becomes one transaction instead of 32.
* **Shared-memory tiling: a further 1.6×**, by reusing each loaded element
  TILE times.
* **Register blocking: a further 3.0×** on top of tiling. R = 4 (16 outputs per
  thread) beats R = 2 by 1.56×; the win is arithmetic intensity, one shared read
  feeding R² FMAs.
* **Strided vs contiguous cooperative loads: 1.04×** in favour of strided
  (19.36 vs 18.57 TFLOP/s), the expected direction — strided keeps a warp's
  lanes adjacent so each tile load is one coalesced transaction, while the
  contiguous variant gives each lane an F-wide run and strides the warp by F.

**Best custom kernel reaches 48.2 % of cuBLAS and 38.4 % of theoretical peak.**

### CPU baseline

Measured at N ≤ 4096; larger N extrapolated as N³ (dashed in the figure).

| N | CPU time | Note |
|---|---:|---|
| 1024 | 5.64 s | measured |
| 2048 | 33.57 s | measured |
| 4096 | 237.24 s | measured |
| 8192 | ≈ 32 min | extrapolated |
| 16384 | ≈ 4.2 h | extrapolated |
| 32768 | ≈ 33.7 h | extrapolated |

At N = 4096 the best custom GPU kernel is **42 783×** faster than the
single-threaded CPU triple loop.

---

## PA3 — MLP, CUDA streams, gradient checkpointing

`results/pa3_mlp.csv`, `pa3_forward.png`, `pa3_checkpointing.png`.

### Part 1 — forward pass, B = 32 (ms)

| N | baseline | 4 streams (batch split) | 4 streams (column split) |
|---|---:|---:|---:|
| 32 | 0.0069 | 0.0278 | 0.0338 |
| 1024 | 0.0990 | 0.1200 | 0.1253 |
| 8192 | 1.504 | 1.526 | 1.549 |
| 16384 | 6.293 | 6.847 | **6.186** |
| 32768 | 44.43 | 52.28 | **25.65** |

**Splitting the batch across 4 streams never helps and costs up to 4× at small
N.** That is the honest result and it is not surprising: at B = 32 there is very
little row parallelism to divide, so each stream launches a kernel with a
quarter of the rows and the same weight-matrix traffic, while paying 4× the
launch overhead. Below N ≈ 2048 the whole forward pass is launch-latency bound
(baseline is 7 µs at N = 32), so any extra launches dominate.

**Splitting the output columns across 4 streams is 1.73× faster at N = 32768**
(25.65 ms vs 44.43 ms). This is the decomposition that actually adds
parallelism: each stream owns a distinct slice of the output and of the weight
matrix, so the concurrent kernels have independent work and independent memory
footprints. It only pays off once the problem is large enough for concurrency
to matter — at N ≤ 16384 it is neutral or slightly negative.

### Part 2 — gradient checkpointing, 4-layer MLP, 100 iterations

Checkpoints retained: **A2 and Z4**. Z1, A1, Z2, Z3, A3 are recomputed.

| Config (batch, dims) | Mode | Fwd (ms) | Bwd (ms) | Peak (MB) | Saved | Bwd overhead |
|---|---|---:|---:|---:|---:|---:|
| 1024, 512-1024³-512 | baseline | 0.772 | 2.631 | 70 | — | — |
| | checkpointed | 0.772 | 3.255 | 62 | **8 MB (11.4 %)** | **+0.625 ms (+23.8 %)** |
| 2048, 1024-2048³-1024 | baseline | 5.798 | 19.863 | 274 | — | — |
| | checkpointed | 5.799 | 24.668 | 242 | **32 MB (11.7 %)** | **+4.805 ms (+24.2 %)** |
| 4096, 1024-4096³-1024 | baseline | 39.495 | 136.147 | 962 | — | — |
| | checkpointed | 39.479 | 171.689 | 834 | **128 MB (13.3 %)** | **+35.54 ms (+26.1 %)** |

Both modes produce **identical** post-update weight checksums
(−84.08664117, −259.7290833, −379.7093289), so the recomputation is exact.

The forward pass is unchanged in time (39.495 vs 39.479 ms) — checkpointing does
not skip forward work, it only declines to *keep* the results.

**Where the saving comes from.** The two recomputed segments are never live
simultaneously: backward runs layer 4 → 1, so segment B (Z3, A3) is consumed
before segment A (Z1, A1, Z2) is rebuilt. One scratch arena sized
`max(2·n₃, 2·n₁+n₂)` serves both. Allocating one buffer per dropped activation
instead holds exactly what the baseline holds and saves **nothing** — that was
a real bug in the first version of this benchmark, which reported 0 MB saved.

Peak here counts the whole training step including 320 MB of parameters and
gradients, which checkpointing cannot touch. Measured against the *activation*
footprint alone, the 4096 config drops 400 MB → 272 MB, a **32 % reduction**.

---

## PA4 — FlashAttention

`results/pa4_attention.csv`, `pa4_runtime.png`. B = 8, N_H = 16, D = 64, FP32,
10 iterations each.

### Runtime (ms)

| N | PyTorch standard | Custom flash | PyTorch fused | max abs diff | score tensor |
|---|---:|---:|---:|---:|---:|
| 32 | 0.119 | **0.017** | 0.023 | 7.2e-07 | 0.5 MB |
| 64 | 0.110 | **0.018** | 0.022 | 6.0e-07 | 2 MB |
| 128 | 0.109 | **0.051** | 0.025 | 7.2e-07 | 8 MB |
| 256 | **0.111** | 0.183 | 0.082 | 6.0e-07 | 34 MB |
| 1024 | **1.476** | 2.625 | 1.032 | 6.9e-07 | 537 MB |
| 4096 | **24.17** | 40.30 | 15.50 | 5.5e-07 | 8.6 GB |
| 8192 | **105.08** | 160.05 | 62.62 | 6.7e-07 | 34.4 GB |

Numerical consistency: **max absolute difference ≤ 7.2 × 10⁻⁷** everywhere,
three orders of magnitude inside the 10⁻³ requirement.

### Crossover

The custom kernel is **faster than explicit PyTorch attention for N ≤ 128** and
slower from **N = 256** onward — a single crossover at **N = 256**. The small-N
win is not an attention-efficiency win: PyTorch has a ~0.11 ms floor there from
launching three separate kernels, so the comparison below N = 256 measures
launch overhead, not attention.

### Kernel characteristics

| Metric | Value |
|---|---|
| Shared memory per block | **28 800 B** (Qi 8192 + Kj 8320 + Vj 8192 + S 4096) |
| Registers per thread | 62 |
| Threads per block | 256 (8 warps × 32) |
| Max active blocks / SM | 4 |
| Theoretical occupancy | **50 %** |
| Achieved throughput at N = 8192 | 13.75 TFLOP/s (20.5 % of peak) |

> **`ncu` could not be used.** Profiling on this machine returns
> `ERR_NVGPUCTRPERM` — GPU performance counters require root, which we do not
> have. So achieved occupancy and DRAM throughput are **not** measured here; the
> numbers above come from the CUDA occupancy API (`cudaFuncGetAttributes`,
> `cudaOccupancyMaxActiveBlocksPerMultiprocessor`) and are *theoretical*
> occupancy. Enabling `ncu` needs an admin to set
> `NVreg_RestrictProfilingToAdminUsers=0`.

### Optimisation history — 21× on the custom kernel

Each step, with the measurement that motivated the next (N = 8192):

| Version | Time | Why |
|---|---:|---|
| 1. Quiz layout: `grid(B, NH)`, 32 threads, query loop inside kernel | 3358 ms | 128 blocks of one warp on a 132-SM GPU |
| 2. `grid(Tr, NH, B)` — query loop becomes grid parallelism | 2377 ms | 32768 blocks, but still bottlenecked by (3) |
| 3. Pad `Kj` row stride to D+1 | **339 ms** | stride D = 64 put every lane on bank 0 — a **32-way shared-memory bank conflict** on the hottest loop; 7.0× on its own |
| 4. 4 query rows per warp | **160 ms** | at 1 row/warp the inner loops issued 2 shared loads per FMA and measured 6.5 TFLOP/s — exactly the shared-bandwidth roofline. Register-blocking over rows lets one Kj/Vj load feed 4 FMAs |

The remaining 1.5× gap to explicit PyTorch (and 2.6× to fused) is expected: that
path is cuBLAS SGEMM, far more aggressively register-blocked, whereas this
kernel uses no tensor cores and reads both GEMM operands from shared memory.
The same ratio shows up in PA2 — the best hand-written kernel there also lands
around 48 % of cuBLAS.

---

## PA1 — not run

The implementation is complete and in `pa1_kvcache/` (nanoGPT-style GPT-2 with
`flash=False`, per-layer KV cache threaded through `Block`/`GPT.forward`, and an
edge-compressed radix tree over token ids for prefix sharing), and GPT-2 124M
weights load correctly from HuggingFace. **The benchmark sweep was not executed**
— at the user's direction, to avoid large runs against real pretrained models.

To run it later:

```bash
conda activate sysml
export HF_HOME=/mnt/warm_storage/jayavardhan/harshith/.cache/huggingface
cd pa1_kvcache && CUDA_VISIBLE_DEVICES=<idle gpu> python bench_pa1.py --iters 3
```

It writes `results/pa1_kvcache.csv` (baseline vs KV cache: total time, ms/token,
peak activation memory over prompt × generation-length combinations) and
`results/pa1_radix.csv` (shared-prefix batch vs independent batch).
