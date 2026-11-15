---
type: interview-guide
day: 5
date: 2026-11-15
tags: [interview, cuda, matmul, mock]
---
# Day 5, interview edition: the CUDA matmul project (explain it, defend it, survive the follow-ups)

Part of [[Interview plan (Nov 2026)]]. Built from [[Day 05 - CUDA I]] (the study guide) and [[CUDA kernels - prep]] (decisions, stories, numbers sheet). Repo `sysml`, one commit `16870fb`, file `matmul/matmul_bench.cu`. Line numbers below are checked against that commit. Part 2 of the project (streams, checkpointing, FlashAttention) is in [[Day 06 - CUDA II (interview)]].

> [!important] What this note is for
> In an interview you will be asked to explain **your** matmul kernels from scratch, then pushed: *why this tile size, what if N isn't a multiple of it, can it race, is it always faster, how would you go further.* This note is organised so that every answer comes back to the code and the CSV. Nothing here is generic CUDA trivia unless it explains one of your lines or one of your numbers.

**How to use it (about 5 hours in total):**

| pass | what | time |
|---|---|---|
| 1 | §0 (the one-page project), then §9 (say it out loud) | 40 min |
| 2 | §1–§6, one optimisation at a time: card → animation → code walk → question bank | 2.5 h |
| 3 | §7 harness and §8 performance analysis | 50 min |
| 4 | §10 mock interview: cover the answers, say yours out loud, then open the callout | 1 h+ |

**Every claim carries a label.** Use the same discipline when you speak.

| label | means | example |
|---|---|---|
| **[IMPL]** | it is in the code at `16870fb` | `mm_block2d<64,4>` gives each thread a 4 × 4 patch |
| **[MEAS]** | a number in `results/pa2_matmul.csv` | 2,736.6 ms, 25.71 TFLOP/s |
| **[THEORY]** | what CUDA theory or a hand count predicts; not measured | "33 vs 2 cache lines per step → 16.5×" |
| **[RECON]** | reasoning read off the code or binaries by you, not documented by the project | "the loader is stride-R, which costs sectors" |
| **[FUTURE]** | not done; a possible next step | double buffering with `cp.async` |

**Animations.** Each animated figure has a step-through player: click **▶ Step through it** under it (← / → keys step, space plays). New figures for this note are `d05i-*`, made by `Days/figures/_gen/cudafigs.py`.

---

## 0. The project on one page

### 0.1 The problem, in your words

> [!quote] Say this
> "Matrix multiplication is the core of every ML workload, and on a GPU the arithmetic is cheap: the hard part is **feeding** the arithmetic units. I wrote a chain of FP32 matmul kernels on an H200, each fixing the bottleneck the previous one exposed, measured each one against cuBLAS, and explained every step with a memory-traffic count."

**The ten questions an interviewer can ask about the whole project, answered once:**

| question | answer | label |
|---|---|---|
| What problem? | C = A·B for square FP32 matrices, N = 1024 … 32768, as fast as possible on CUDA cores, and understanding *why* each version is as fast as it is. | [IMPL] |
| Why a GPU? | Each of the N² outputs is independent and needs N multiply-adds: 3.5 × 10¹³ FMAs at N = 32768. The H200 has 132 SMs × 128 FP32 lanes = 16,896 FMA lanes per clock (66.9 TFLOP/s peak). A CPU core has a handful. | [THEORY] |
| Baseline? | `mm_naive` (`:44`): one thread per output, consecutive threads own consecutive **rows**. 140,486 ms, 0.50 TFLOP/s at N = 32768. | [IMPL] [MEAS] |
| Its bottleneck? | Uncoalesced loads of A: a warp's 32 lanes read 32 addresses N floats apart → 32 cache lines per load instead of 1; and zero designed reuse (2N global loads per output). | [THEORY] |
| What did you add? | (1) coalescing, (2) shared-memory tiling, (3) several outputs per thread with two loader patterns, (4) 2D register blocking. | [IMPL] |
| Why should each help? | Each moves the hot data one rung up the memory ladder: HBM lines used fully → reuse from shared memory → reuse from registers. | [THEORY] |
| What happens in hardware? | Fewer distinct 128-byte lines per warp instruction; fewer HBM bytes per FMA; fewer shared-memory reads per FMA. | [THEORY] |
| Trade-offs? | Shared memory and registers per thread are finite, so tiles and register blocks lower occupancy (block2d R = 4 runs at 50%); bigger tiles also waste work at ragged edges. | [IMPL] [RECON] |
| Alternatives? | cuBLAS/CUTLASS; tensor cores (TF32/BF16); warp tiling, vector loads, double buffering, swizzled shared layouts. | [FUTURE] |
| Limitations? | 48.2% of cuBLAS; no tensor cores; no profiler (ncu needed root); single sweep, n = 1 for most kernels at N = 32768; correctness check covers only the first 4M outputs. | [MEAS] [RECON] |

### 0.2 The measured ladder

![[d05i-ladder.svg]] [[d05i-ladder.html|▶ Step through it]]

All at N = 32768, FP32, H200, kernel time by CUDA events **[MEAS]**:

| # | kernel (`matmul_bench.cu`) | time | TFLOP/s | step gain | what changed **[IMPL]** | the bottleneck it removed / exposed **[THEORY]** |
|---|---|---|---|---|---|---|
| 1 | `mm_naive` :44 | 140,486 ms | 0.50 | | lanes walk rows | 32 lines per warp load of A |
| 2 | `mm_coalesced` :60 | 12,927 ms | 5.44 | **10.9×** | lanes walk columns | now 2 lines per step; still 2N loads per output |
| 3 | `mm_tiled_1to1<32>` :78 | 8,148 ms | 8.64 | 1.59× | 32 × 32 shared tiles | HBM traffic ÷ 32; now 2 shared reads per FMA (~8.4 roof) |
| 4 | `mm_tiled_strided<64,32>` :118 | 3,635 ms | 19.36 | 2.24× | 64-tile, 2 × 2 outputs per thread, coalesced loader | fewer shared reads per FMA |
| 5 | `mm_tiled_contiguous<64,32>` :173 | 3,790 ms | 18.57 | (control) | same maths, stride-2 loader | isolates the global access pattern: 4% |
| 6 | `mm_block2d<64,2>` :232 | 4,263 ms | 16.51 | (slower) | 2 × 2 in registers, 1,024 threads | 1 block per SM, contiguous loader |
| 7 | `mm_block2d<64,4>` :232 | **2,736.6 ms** | **25.71** | 1.33× over #4 | 4 × 4 in registers, 256 threads | 0.5 shared reads per FMA; 50% occupancy |
| — | cuBLAS SGEMM :436 | 1,318.5 ms | 53.37 | | reference and ceiling | 80% of peak |

**The headline numbers, derived (say "measured against what" every time):**
- **51.3×** = 140,486 / 2,736.6, against the deliberately uncoalesced naive kernel. Against the coalesced naive kernel it is **4.7×** (12,927 / 2,736.6). The ratio is 45× at N = 1024 and 50–51× from N = 2048 up **[MEAS]**.
- **48.2%** = 25.71 / 53.37 TFLOP/s of cuBLAS SGEMM, both FP32 on CUDA cores, default math mode (no TF32) **[MEAS]**. 48–50% for every N ≥ 2048, 61% at N = 1024.
- **38.4% of peak** = 25.71 / 66.9; cuBLAS is at 80% **[MEAS]** over **[THEORY]** peak.
- "Register blocking 3.0×" (RESULTS.md) is against the 1-to-1 tile. Against the strided 64-tile kernel, which already has 2 × 2 outputs per thread, it is **1.33×**. Quote the chain: 10.9× → 1.59× → 2.24× → 1.33×.

### 0.3 What you implemented, measured, predicted, and did not do

| | |
|---|---|
| **Implemented [IMPL]** | six kernels plus two template configs; a harness that runs cuBLAS first as the reference, times every kernel with CUDA events under a 30 s budget, checks max relative error, computes TFLOP/s and the device's FP32 peak, and writes `results/pa2_matmul.csv`. |
| **Measured [MEAS]** | time, TFLOP/s, n (iterations) and max relative error per kernel per N; register counts from `cuobjdump` of the committed binary (62, 40, 32 registers). |
| **Predicted, not measured [THEORY]/[RECON]** | lines per warp load (33 vs 2); HBM intensity (0.25, 8, 16 FLOP/B); the shared-memory roof (8.4 and 33.5 TFLOP/s); occupancy (50%, 50%, 100%); the 2-way bank conflicts in `mm_block2d`; why you stop at 48%. |
| **Not done [FUTURE]** | tensor cores; vector loads written by hand; double buffering / `cp.async`; warp tiling; swizzled layouts; autotuning; Nsight Compute profiles; repeated runs with error bars. |

> [!warning] Two repo facts to know before anyone asks
> - git has **one commit**: all "history" (what you tried first) comes from `RESULTS.md` and code comments.
> - Folders were renamed after the runs (`pa2_matmul/` → `matmul/`), so `SETUP.md`'s build commands fail as written. The code and CSVs are intact.

---

## 1. The execution model, and where your kernels put each output

### 1.1 The idea

![[d5-6-hierarchy.svg]] [[d5-6-hierarchy.html|▶ Step through it]]

| level | what it is | in your kernels **[IMPL]** |
|---|---|---|
| **grid** | everything one launch runs: a 2-D array of blocks | `mm_naive` at N = 32768: 1024 × 1024 blocks |
| **block** | threads that run on **one** SM, share `__shared__` memory, and can `__syncthreads()` | 32 × 32 = 1,024 threads (naive … strided); 16 × 16 = 256 (block2d R = 4) |
| **warp** | 32 consecutive threads that issue **one instruction together** | in a 32 × 32 block: one `threadIdx.y`, all 32 `threadIdx.x` |
| **thread (lane)** | one copy of the kernel body with its own registers | owns 1, 4 or 16 outputs |

**The one rule that every coalescing and bank argument uses:** threads are numbered with `threadIdx.x` fastest, so **a warp is 32 consecutive `threadIdx.x` values with the same `threadIdx.y`** (in a 16 × 16 block a warp spans two `threadIdx.y` values).

![[d05s-grid.svg]] [[d05s-grid.html|▶ Step through it]]

### 1.2 The launch shapes **[IMPL]** (`matmul_bench.cu:454–483`)

| kernel | grid at N = 32768 | block | outputs / thread | outputs / block | index formula |
|---|---|---|---|---|---|
| `mm_naive` | 1024 × 1024 | 32 × 32 | 1 | 32 × 32 | `row = bx·32 + tx`, `col = by·32 + ty` |
| `mm_coalesced` | 1024 × 1024 | 32 × 32 | 1 | 32 × 32 | `col = bx·32 + tx`, `row = by·32 + ty` |
| `mm_tiled_1to1<32>` | 1024 × 1024 | 32 × 32 | 1 | 32 × 32 | `row = by·TILE + ty`, `col = bx·TILE + tx` |
| strided / contiguous `<64,32>` | 512 × 512 | 32 × 32 | 2 × 2 | 64 × 64 | strided: `ty + i·32`; contiguous: `ty·2 + i` |
| `mm_block2d<64,2>` | 512 × 512 | 32 × 32 | 2 × 2 | 64 × 64 | `rowBase = by·64 + ty·2` |
| `mm_block2d<64,4>` | 512 × 512 | 16 × 16 | 4 × 4 | 64 × 64 | `rowBase = by·64 + ty·4`, `colBase = bx·64 + tx·4` |

Check it on the whiteboard: 1024 × 32 = 32,768 ✓; 512 × 16 × 4 = 32,768 ✓.

#### From the launch line to the maths (how to read every "How it is called" box below)

Every kernel is launched from the per-N loop of the harness, for example (`matmul_bench.cu:454–467`) **[IMPL]**:

```cpp
dim3 b32(32, 32);                                   // :454  block = 32 × 32 threads
dim3 g32((N + 31) / 32, (N + 31) / 32);             // :455  grid  = ceil(N/32) × ceil(N/32) blocks
record("tiled_1to1", time_kernel([&]{ mm_tiled_1to1<32><<<g32,b32>>>(dA,dB,dC,N); }, 1, iters, budget));  // :467
```

- `[&]{ … }` is a lambda. `time_kernel` calls it for the warm-ups and the timed runs (§7), so the code inside the braces **is** the launch.
- `<32>` is a template argument: a compile-time constant baked into the kernel (here TILE).
- `<<<g32, b32>>>` means "run g32.x × g32.y blocks of b32.x × b32.y threads". Every thread runs the same body; only its `(blockIdx, threadIdx)` differ.
- `(dA, dB, dC, N)` are ordinary arguments, the same for every thread: three device pointers to flat row-major N² arrays, and N.

**Read any launch in four steps:**

| step | question | `mm_tiled_1to1<32>` at N = 32768 |
|---|---|---|
| 1 | how many threads, in what shape? | grid 1024 × 1024 blocks, block 32 × 32 threads → 2³⁰ threads |
| 2 | which entries of C does thread (bx, by, tx, ty) own? | the kernel's index lines: `row = by·32 + ty`, `col = bx·32 + tx` |
| 3 | what does it compute for them? | C[row][col] = Σₖ A[row][k] · B[k][col] |
| 4 | do the owned entries cover C exactly once? | 1024 · 32 = 32,768 rows and columns, one thread each ✓ |

**Step 3 never changes.** Every kernel computes C[i][j] = Σₖ A[i][k]·B[k][j]. What changes from kernel to kernel is step 2 (which thread owns which entries) and *where and in what order* the sum is accumulated (registers, shared tiles, outer products). So each kernel section below has a **"How it is called, on an example"** box: the real call from the harness, then a 4 × 4 call small enough to do by hand on the toy matrices of [[Day 05 - CUDA I]]:

$$A = \begin{bmatrix}1&2&0&1\\0&1&2&1\\2&0&1&0\\1&1&0&2\end{bmatrix}\quad B = \begin{bmatrix}1&0&2&1\\2&1&0&0\\0&1&1&2\\1&2&0&1\end{bmatrix}\quad C = A \cdot B = \begin{bmatrix}6&4&2&2\\3&5&2&5\\2&1&5&4\\5&5&2&3\end{bmatrix}$$

> [!note] The hand-sized calls are not in the repo
> They are the same kernel bodies with a smaller block or template argument, so every index line is unchanged; only the numbers shrink. Their values are hand counts, checked with a line-for-line CPU port of each kernel's index arithmetic **[THEORY]**. cuBLAS is called differently (a library call, not a `<<< >>>` launch); its 2 × 2 example is in §7.3.

### 1.3 What happens when you write `kernel<<<grid, block>>>(args)` **[THEORY]**

1. **Host side.** The CPU call packs the arguments and pushes a launch command into a stream's queue (the default stream here). It **returns immediately**: the launch is asynchronous. That is why timing needs events (§7).
2. **Front end.** When the GPU reaches the command, its work distributor creates the grid: `grid.x · grid.y` blocks waiting to run.
3. **Block scheduling.** Blocks are handed to SMs that have room for them: enough free registers (`threads × regs`), shared memory and thread slots. A block, once placed, stays on its SM until it finishes. Blocks may run in **any order**, which is why your kernels never need another block's result.
4. **Warp scheduling.** Inside an SM, each block is split into warps. The SM's warp schedulers pick a ready warp each cycle and issue its next instruction to 32 lanes. A warp that waits for memory is simply not picked; another one runs. This is **latency hiding**, and it is why occupancy matters at all.
5. **Waves.** If the grid has more blocks than fit at once (1,048,576 blocks for `mm_naive` at N = 32768), new blocks start as old ones finish.
6. **Completion.** The kernel ends when its last block ends. The next kernel in the same stream starts after that.

### 1.4 Question bank: execution model

**L1. What is a warp?**
> [!success]- Answer
> 32 consecutive threads of a block that the SM issues as one unit: one instruction, 32 lanes. On NVIDIA GPUs the width is fixed by the hardware. It matters to you for two reasons: (1) memory: the 32 addresses of one warp load are combined into as few cache lines as possible (coalescing, §3); (2) control flow: if lanes take different branches, the warp runs both paths with lanes masked (divergence). In your 32 × 32 blocks a warp is one `threadIdx.y` and all 32 `threadIdx.x`; in `mm_block2d<64,4>`'s 16 × 16 block, a warp is two consecutive `threadIdx.y` rows of 16 lanes each.

**L1. Thread vs block vs grid?**
> [!success]- Answer
> A thread is one instance of the kernel body with private registers. A block is a group of up to 1,024 threads placed on one SM; its threads share `__shared__` memory and can barrier with `__syncthreads()`. A grid is all the blocks of one launch; blocks cannot synchronise with each other inside a kernel and run in any order. In `mm_naive` at N = 32768: 1,048,576 blocks × 1,024 threads, one output each.

**L1. What is an SM, and how many does the H200 have?**
> [!success]- Answer
> A streaming multiprocessor: the unit that runs blocks. It has its own registers (65,536 × 4 B), shared memory / L1 (up to 228 KB), warp schedulers and 128 FP32 lanes. The H200 has 132, so 132 × 128 = 16,896 FMAs per clock, which at 1.98 GHz and 2 FLOPs per FMA is the 66.9 TFLOP/s your harness prints (`:385–392`).

**L2. Why are your block shapes 32 × 32 and 16 × 16, not, say, 24 × 24?**
> [!success]- Answer
> The x-dimension should be a multiple of 32 so every warp is full and lines up with 32 consecutive columns: a 24-wide block would make warps straddle two rows of the block and leave 8-lane gaps in each coalesced load. 32 × 32 = 1,024 is the per-block maximum. 16 × 16 = 256 is what `mm_block2d<64,4>` needs: 64 / 4 = 16 threads per side. Square tiles are the simplest to reason about for square matrices. [RECON]

**L2. Why is `blockIdx.x` mapped to columns in every kernel except `mm_naive`?**
> [!success]- Answer
> Because `threadIdx.x` must vary along contiguous memory, and in row-major storage the contiguous direction is the column index. Mapping `x → col` puts 32 adjacent columns in one warp, so the B loads and C stores coalesce. `mm_naive` maps `x → row` on purpose; it is the assignment's uncoalesced baseline.

**L3. "Your naive kernel launches a million blocks. Isn't that a problem?"**
> [!success]- Answer
> No. Blocks are cheap to schedule and run in waves; a million blocks just means many waves over 132 SMs. What matters is the work per block and whether each block's loads are efficient. The naive kernel is slow because of its memory access, not its block count. The opposite problem, too few blocks, does show up in the project: the first FlashAttention layout had 128 blocks on 132 SMs (Day 6).

**L3. "What if a block needs more registers or shared memory than an SM has?"**
> [!success]- Answer
> The launch fails ("too many resources requested for launch"), or for static shared memory over 48 KB the kernel doesn't even compile. The project hits exactly this limit in prep-pack exercise 5: `mm_block2d<128,8>` would need two static 128 × 128 float tiles = 128 KB > 48 KB, so it fails to compile; you'd need `extern __shared__`, a launch-time size, and `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`. Short of failing, more resources per block just means fewer blocks per SM (lower occupancy, §6). [RECON]

**L3. "Can blocks of your kernel run in a different order? Does it matter?"**
> [!success]- Answer
> Yes, CUDA guarantees no order, and no, it doesn't matter for correctness: each block reads only A and B (never written) and writes a disjoint part of C. No block reads another block's output. It can matter for **performance**, through L2 locality: blocks that run together and share rows of A or columns of B get cache hits. You didn't control block order (no swizzling); cuBLAS does. [THEORY] [FUTURE]

---

## 2. The baseline: `mm_naive`

### 2.1 Component card

| | |
|---|---|
| **Problem** | compute C = A·B, one thread per output, the simplest correct kernel |
| **Bottleneck** | (a) `row` comes from `threadIdx.x`, so a warp's A loads are 32 addresses **N floats apart** → 32 lines per load; (b) no designed reuse: 2N global loads per output |
| **Hardware level** | each warp-step requests 32 lines for A + 1 broadcast line for B = 33 lines for 64 useful floats; the C store is also scattered |
| **Measured** | 140,486 ms, 0.50 TFLOP/s at N = 32768, n = 1 (one 140 s run) |
| **Why keep it** | the assignment defines it as the baseline; it is the "before" of the coalescing story |

![[d05s-naive.svg]] [[d05s-naive.html|▶ Step through it]]

### 2.2 The code, line by line (`matmul_bench.cu:44–54`)

```cpp
44  __global__ void mm_naive(const float* A, const float* B, float* C, int N)
45  {
46      int row = blockIdx.x * blockDim.x + threadIdx.x;
47      int col = blockIdx.y * blockDim.y + threadIdx.y;
48      if (row >= N || col >= N) return;
49
50      float acc = 0.0f;
51      for (int k = 0; k < N; ++k)
52          acc = fmaf(A[(size_t)row * N + k], B[(size_t)k * N + col], acc);
53      C[(size_t)row * N + col] = acc;
54  }
```

| line | what it does | why it is written this way | what breaks if you change it |
|---|---|---|---|
| 44 | `__global__`: a kernel, launched from the host, run by every thread of the grid. `const float*` inputs are read-only. | Pointers are device addresses (`dA`, `dB`, `dC` from `cudaMalloc`). | Passing a host pointer → illegal address at the first load. |
| 46 | global row index = block offset + position in block | `blockIdx.x * blockDim.x` skips the rows owned by earlier blocks | Swap x/y here and on 47 → you get `mm_coalesced` (10.9× faster). |
| 47 | global column index from the y dimension | | |
| 48 | edge guard: threads past the matrix exit | N may not be a multiple of 32 | Remove it with N = 1000 → out-of-bounds reads and writes. An early `return` is safe **only because this kernel has no `__syncthreads()`**. |
| 50 | `acc` lives in a register | the running sum never touches memory until line 53 | |
| 51–52 | dot product of row `row` of A with column `col` of B, one FMA per k | `fmaf` = a·b + c in one instruction, one rounding; counts as 2 FLOPs | Dropping `(size_t)`: `row * N + k` is 32-bit `int`; fine at N = 32768 (max 1.07 × 10⁹) but overflows for N > 46,340. |
| 53 | one store per thread | | |

#### How it is called, on an example

**Real call** (`:460`) **[IMPL]**: `mm_naive<<<g32, b32>>>(dA, dB, dC, N)`. At N = 32768: grid (1024, 1024), block (32, 32), one thread per entry of C. The harness skips it when its projected time passes 400 s (`:457–464`).

**Hand-sized call [THEORY]:** `mm_naive<<<dim3(2, 2), dim3(2, 2)>>>(dA, dB, dC, 4)`. The kernel reads its block size from `blockDim`, so a 2 × 2 block needs no code change: 4 blocks × 4 threads = 16 threads for 16 outputs.

With `blockDim = 2` the index lines become **row = 2·bx + tx, col = 2·by + ty**. Who computes each entry of C, written `(bx,by) / (tx,ty)`:

| | col 0 | col 1 | col 2 | col 3 |
|---|---|---|---|---|
| **row 0** | (0,0) / (0,0) | (0,0) / (0,1) | (0,1) / (0,0) | (0,1) / (0,1) |
| **row 1** | (0,0) / (1,0) | (0,0) / (1,1) | (0,1) / (1,0) | (0,1) / (1,1) |
| **row 2** | (1,0) / (0,0) | (1,0) / (0,1) | (1,1) / (0,0) | (1,1) / (0,1) |
| **row 3** | (1,0) / (1,0) | (1,0) / (1,1) | (1,1) / (1,0) | (1,1) / (1,1) |

**The maths of the launch:** the map (bx, by, tx, ty) ↦ (i, j) = (2bx + tx, 2by + ty) sends the 16 threads onto the 16 entries of C one to one, and each thread evaluates one dot product. Example: block (1, 0), thread (0, 1) → row 2, col 1:

$$C[2][1] = \underbrace{[2, 0, 1, 0]}_{A[2][\,:\,]} \cdot \underbrace{[0, 1, 1, 2]}_{B[\,:\,][1]} = 0 + 0 + 1 + 0 = 1 \;✓$$

The two threads with consecutive `tx` in block (0, 0) (a toy "warp") own C[0][0] and C[1][0]: a **column** of C, i.e. rows of A that are N floats apart in memory. That is the uncoalesced pattern of §3, in miniature.

### 2.3 Traced with real numbers

**Toy, N = 4** (the matrices from the study guide), the thread with `row = 0`, `col = 3`:

| k | A index `row·N + k` | A value | B index `k·N + col` | B value | `acc` |
|---|---|---|---|---|---|
| 0 | 0 | 1 | 3 | 1 | 1 |
| 1 | 1 | 2 | 7 | 0 | 1 |
| 2 | 2 | 0 | 11 | 2 | 1 |
| 3 | 3 | 1 | 15 | 1 | **2** = C[0][3] ✓ |

The A index steps by 1 (along a row); the B index steps by N (down a column).

**Real size, N = 32768**, block (bx = 2, by = 1), thread (tx = 5, ty = 3): `row = 2·32 + 5 = 69`, `col = 1·32 + 3 = 35`, so this thread computes `C[69][35]`.

**Memory addresses.** With `dA` at byte address `a₀`, element `A[69][k]` is at byte `a₀ + 4·(69·32768 + k)`. At k = 0 that is `a₀ + 9,043,968`. Its warp neighbour (tx = 6) owns row 70: byte `a₀ + 9,175,040`, **131,072 bytes away**. Every lane of the warp is 128 KB from the next one, so no two lanes share a 128-byte line.

### 2.4 Question bank: the baseline

**L1. What does `blockIdx.x * blockDim.x + threadIdx.x` compute?**
> [!success]- Answer
> The thread's global index along x: the number of threads in all earlier blocks along x, plus its position in its own block. Block 2, thread 5, blocks of 32: 2·32 + 5 = 69. In `mm_naive` that number is used as the output **row**; in every other kernel x is used for the column.

**L2. Why is an early `return` safe here but not in the tiled kernels?**
> [!success]- Answer
> `mm_naive` never calls `__syncthreads()`, so a thread that leaves early cannot leave anyone waiting. In `mm_tiled_1to1` every thread must reach both barriers in every iteration: a returned thread would leave the block's barrier waiting for it, which is undefined behaviour (a hang or a wrong result). That's why the tiled kernels keep out-of-range threads alive, load `0.0f`, and guard only the final store (`:100`). See the animation in §4.6.

**L2. Why `(size_t)` in the index?**
> [!success]- Answer
> `row * N + k` in `int` arithmetic overflows past 2³¹ − 1 ≈ 2.1 × 10⁹. At N = 32768 the largest index is 32768² − 1 ≈ 1.07 × 10⁹, which fits, but for N > 46,340 it wouldn't. Casting one operand to 64-bit `size_t` before multiplying makes the whole expression 64-bit. It costs a little index arithmetic and removes a silent-corruption bug class.

**L3. "Isn't the B access in the naive kernel uncoalesced too? It walks down a column."**
> [!success]- Answer
> Per **thread**, yes: each thread's B index jumps by N each iteration. But coalescing is about the 32 addresses of one **warp instruction**. In `mm_naive`'s warp all lanes share `col` and `k`, so they read the **same** B address: one line, broadcast. The expensive one is A: lanes differ in `row`, so 32 addresses N apart → 32 lines. Per step: 32 + 1 = 33 lines for 64 useful floats.

**L3. "Why include a kernel you know is bad? Isn't 51.3× inflated?"**
> [!success]- Answer
> The assignment defines that kernel as the baseline, and it is the cleanest way to show what coalescing alone buys (10.9×). But I quote both numbers: 51.3× against the uncoalesced naive kernel and **4.7×** against the coalesced one, and I point to cuBLAS as the real yardstick: 48.2%. A ratio against a strawman says little on its own; the honest headline is "48% of cuBLAS".

---

## 3. Coalescing: `mm_naive` → `mm_coalesced` (10.9×)

### 3.1 Component card

| | |
|---|---|
| **Problem** | naive warp loads touch 32 cache lines for 32 useful floats |
| **Optimisation [IMPL]** | swap which index comes from `threadIdx.x` (`:62–63`): `col = blockIdx.x·blockDim.x + threadIdx.x` |
| **Why it should help [THEORY]** | lanes now read 32 consecutive floats of B (1 line) and one shared address of A (broadcast, 1 line): 2 lines per step instead of 33 → up to 16.5× fewer line requests |
| **Measured [MEAS]** | 140,486 → 12,927 ms = **10.9×** |
| **Why not 16.5× [THEORY]** | the naive warp's 32 lines at step k are the same lines it needs at steps k + 1 … k + 31 (next floats of the same rows): some survive in L1, which softens the naive penalty. Not measured: no profiler. |
| **Trade-off** | none: same instructions, same registers. It's a free fix. |
| **Limitation** | still 2N global loads per output, no reuse by design → only 5.44 TFLOP/s |

### 3.2 How global memory is fetched **[THEORY]**

Global memory is moved in **32-byte sectors**, grouped into **128-byte lines** (32 floats). For one warp load instruction the hardware looks at the 32 lane addresses and fetches every line they touch:

$$\text{lines fetched} = \#\{\lfloor 4 \cdot \text{index}_\ell / 128 \rfloor : \ell = 0 \ldots 31\}$$

| pattern across the 32 lanes | lines | in your code |
|---|---|---|
| 32 consecutive floats | 1 | `B[k*N + col]`, `col` from `threadIdx.x` |
| the same address | 1 (broadcast) | `A[row*N + k]`, `row` equal across the warp |
| N floats apart | 32 | `A[row*N + k]`, `row` from `threadIdx.x` (naive) |

![[d5-7-coalescing.svg]] [[d5-7-coalescing.html|▶ Step through it]]

![[d05s-warp.svg]] [[d05s-warp.html|▶ Step through it]]

### 3.3 The diff and one warp's addresses

```cpp
// mm_naive :46–47                                  // mm_coalesced :62–63
int row = blockIdx.x * blockDim.x + threadIdx.x;    int col = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;    int row = blockIdx.y * blockDim.y + threadIdx.y;
```

Block (2, 1), warp `threadIdx.y = 3`, N = 32768, step k = 0:

| | `mm_naive` | `mm_coalesced` |
|---|---|---|
| `row` across lanes | 64 … 95 | 35 for all |
| `col` across lanes | 35 for all | 64 … 95 |
| A float index | 2,097,152 · 2,129,920 · … (step 32,768) | 1,146,880 for all (broadcast) |
| B float index | 35 for all (broadcast) | 64, 65, …, 95 |
| byte span of the B load | 4 B | 128 B: bytes 256 … 383 = exactly line 2 |
| lines per step | 32 (A) + 1 (B) = **33** | 1 + 1 = **2** |
| C store | 32 lines (scattered) | 1 line |

**The shapes changed, not the work:** each thread still reads a 1 × N row of A and an N × 1 column of B. What changed is the shape **a warp** reads at one step: naive reads a 32 × 1 column piece of A; coalesced reads a 1 × 32 row piece of B.

#### How it is called, on an example

**Real call** (`:466`) **[IMPL]**: `mm_coalesced<<<g32, b32>>>(dA, dB, dC, N)`: the launch is character for character the same as `mm_naive`'s. Only the two index lines inside differ, so the difference is entirely in step 2 (who owns what).

**Hand-sized call [THEORY]:** `mm_coalesced<<<dim3(2, 2), dim3(2, 2)>>>(dA, dB, dC, 4)`. Index lines: **col = 2·bx + tx, row = 2·by + ty**. Owner table, `(bx,by) / (tx,ty)`:

| | col 0 | col 1 | col 2 | col 3 |
|---|---|---|---|---|
| **row 0** | (0,0) / (0,0) | (0,0) / (1,0) | (1,0) / (0,0) | (1,0) / (1,0) |
| **row 1** | (0,0) / (0,1) | (0,0) / (1,1) | (1,0) / (0,1) | (1,0) / (1,1) |
| **row 2** | (0,1) / (0,0) | (0,1) / (1,0) | (1,1) / (0,0) | (1,1) / (1,0) |
| **row 3** | (0,1) / (0,1) | (0,1) / (1,1) | (1,1) / (0,1) | (1,1) / (1,1) |

It is the naive table **transposed**. The same 16 dot products are computed; they are just dealt out to threads differently. The thread that computed C[2][1] in the naive launch (block (1, 0), thread (0, 1)) now computes

$$C[1][2] = \underbrace{[0, 1, 2, 1]}_{A[1][\,:\,]} \cdot \underbrace{[2, 0, 1, 0]}_{B[\,:\,][2]} = 0 + 0 + 2 + 0 = 2 \;✓$$

and the consecutive-`tx` pair in block (0, 0) now owns C[0][0] and C[0][1]: a **row** of C, adjacent columns of B, adjacent addresses in memory. Same maths, coalesced access.

### 3.4 Question bank: coalescing

**L1. What is memory coalescing?**
> [!success]- Answer
> When the 32 lanes of a warp access global memory in one instruction, the hardware combines their addresses into as few 32-byte sectors / 128-byte lines as possible. If lane ℓ reads float `base + ℓ`, the warp needs one 128-byte line: every fetched byte is used. If lanes are N floats apart, it needs 32 lines and uses 4 bytes of each. Your 10.9× from swapping two lines is coalescing.

**L1. Is coalescing about threads or warps?**
> [!success]- Answer
> Warps. The merge happens over the addresses of **one warp instruction**. Two threads in different warps reading neighbouring floats don't coalesce with each other (they may still share a cache line by luck).

**L2. Your model says 16.5×, you measured 10.9×. Which is wrong?**
> [!success]- Answer
> Neither; the model ignores caches. The naive warp fetches 32 lines at step k, and steps k + 1 … k + 31 read the **next floats of the same 32 rows**, which sit in those same lines. Whatever survives in L1 is a hit, so the naive kernel pays less than 32 lines per step on average. The measured 10.9× sits between "no caching" (16.5×) and "perfect caching". Without Nsight Compute I can't give the hit rate, so I present 16.5× as an upper bound. [THEORY] [MEAS]

**L2. How would a profiler show this?**
> [!success]- Answer
> Nsight Compute reports sectors per request for global loads: about 4 sectors (128 B) per request for coalesced float loads, near 32 for the naive A load. You couldn't run it (`ERR_NVGPUCTRPERM`, needs root on the shared server), so the argument is by address arithmetic, checked against the measured times. Say that plainly.

**L3. "Could you have fixed the naive kernel without swapping the mapping, e.g. by transposing A?"**
> [!success]- Answer
> Yes: the rule is "whatever varies with `threadIdx.x` must move along contiguous memory". Storing A transposed (column-major) would make `A[k][row]` contiguous across lanes. But that costs an extra transpose pass and memory, while swapping the mapping is free. The swap also makes the C store coalesced, which a transpose of A alone would not.

**L3. "After coalescing, your kernel is at 5.4 TFLOP/s but the roofline says 0.25 FLOP/B × 4.8 TB/s = 1.2. How is that possible?"**
> [!success]- Answer
> The 0.25 FLOP/B is the textbook count with **no reuse at all**: every operand fetched from HBM for every FMA. In reality, the A value is a broadcast (one fetch for 32 FMAs) and B lines are shared through L1/L2 by many warps and blocks working on neighbouring outputs. Caches create reuse the formula ignores, so the real HBM traffic is far lower. That's why I use the roofline for direction, not exact predictions, and why the next step (tiling) makes that reuse **explicit** instead of leaving it to the cache. [THEORY]

---

## 4. Shared-memory tiling: `mm_tiled_1to1<32>` (1.59×)

### 4.1 Component card

| | |
|---|---|
| **Problem** | every element of A is needed by a whole row of outputs (N uses), but the coalesced kernel fetches it once per use: 2N loads per output |
| **Optimisation [IMPL]** | a block cooperatively copies a TILE × TILE piece of A and of B into `__shared__` memory, syncs, and every thread reads its operands from there (`:77–101`) |
| **Why it should help [THEORY]** | global loads per output fall from 2N to 2N / TILE: 65,536 → 2,048 at TILE = 32; HBM intensity rises from 0.25 to TILE / 4 = 8 FLOP/B |
| **Hardware level** | global loads are still coalesced (lanes `tx` read adjacent columns); shared reads in the inner loop are a broadcast (`sA[ty][k]`) and 32 different banks (`sB[k][tx]`) |
| **Measured [MEAS]** | 12,927 → 8,148 ms = 1.59× |
| **New bottleneck [THEORY]** | 2 shared reads per FMA. An SM serves ~32 words per clock but does 128 FMAs per clock → at most 16 FMAs/clock = 1/8 of peak ≈ 8.4 TFLOP/s. Measured 8.64: right at the roof. |
| **Trade-offs** | two barriers per tile (threads wait for the slowest); 8 KB of shared memory per block; idle threads at ragged edges |
| **Alternatives** | rely on L1/L2 caching (what coalesced did); larger tiles (next part); `cp.async` to overlap loads and compute [FUTURE] |

### 4.2 Where the reuse comes from, and the formula

![[d5-8-tiling.svg]] [[d5-8-tiling.html|▶ Step through it]]

Toy N = 4, TILE = 2: naive loads 16 outputs × 8 = **128** floats; tiled loads 4 blocks × 2 steps × 8 = **64**. Each loaded float served TILE = 2 threads.

$$\text{global loads per output} = \frac{2N}{\text{TILE}}, \qquad \text{HBM intensity} = \frac{2N \text{ FLOPs}}{4 \cdot 2N/\text{TILE} \text{ bytes}} = \frac{\text{TILE}}{4}\ \text{FLOP/B}$$

![[d05s-march.svg]] [[d05s-march.html|▶ Step through it]]

![[d05s-reuse.svg]] [[d05s-reuse.html|▶ Step through it]]

### 4.3 Exactly which thread loads which element

![[d05i-loadmap.svg]] [[d05i-loadmap.html|▶ Step through it]]

Each of the TILE² threads loads **one** element of each tile per step: thread (tx, ty) copies `A[row][t·TILE + tx]` into `sA[ty][tx]` and `B[t·TILE + ty][col]` into `sB[ty][tx]`. At real size, block (bx = 2, by = 1), thread (tx = 3, ty = 5) computes `C[37][67]` and at step t = 0 copies `A[37][3]` → `sA[5][3]` and `B[5][67]` → `sB[5][3]`. Lanes `tx` = 0…31 of a warp read 32 adjacent columns of A's row 37 and of B's row 5: both loads are one line each.

### 4.4 The code, line by line (`matmul_bench.cu:77–101`)

![[d05i-tiled-dbg.svg]] [[d05i-tiled-dbg.html|▶ Step through it]]

```cpp
77  template<int TILE>
78  __global__ void mm_tiled_1to1(const float* A, const float* B, float* C, int N)
79  {
80      __shared__ float sA[TILE][TILE];
81      __shared__ float sB[TILE][TILE];
82
83      const int tx = threadIdx.x, ty = threadIdx.y;
84      const int row = blockIdx.y * TILE + ty;
85      const int col = blockIdx.x * TILE + tx;
86
87      float acc = 0.0f;
88      for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
89          int aCol = t * TILE + tx;
90          int bRow = t * TILE + ty;
91          sA[ty][tx] = (row < N && aCol < N) ? A[(size_t)row * N + aCol] : 0.0f;
92          sB[ty][tx] = (bRow < N && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
93          __syncthreads();
94
95          #pragma unroll
96          for (int k = 0; k < TILE; ++k)
97              acc = fmaf(sA[ty][k], sB[k][tx], acc);
98          __syncthreads();
99      }
100     if (row < N && col < N) C[(size_t)row * N + col] = acc;
101 }
```

| line | what | why | what breaks if changed or removed |
|---|---|---|---|
| 77 | `template<int TILE>` | TILE is a compile-time constant: it can size static arrays and let the compiler fully unroll line 96 | A runtime `tile` argument would need `extern __shared__` and a loop the compiler can't unroll. |
| 80–81 | two static shared arrays, one copy **per block** | on-chip, visible to all threads of the block; 2 × 32 × 32 × 4 B = 8 KB | Static shared memory is capped at 48 KB per block: TILE = 128 (2 × 64 KB) would not compile (§6.6). |
| 83 | thread position in the block | used for both the load slot and the output | |
| 84–85 | output owned: `row` from `ty`, `col` from `tx` | `col` from `tx` keeps B loads and C stores coalesced | Swap them → the A and B tile loads become 32-line loads again. |
| 87 | `acc` in a register | | |
| 88 | loop over tiles along K; `(N + TILE − 1) / TILE` is ceil(N / TILE) | a ragged last tile is still visited | Plain `N / TILE` would silently drop the last N mod TILE terms for N = 1000. |
| 89–90 | which column of A / row of B this thread fetches in step t | the block's TILE² threads cover the TILE² cells exactly once | |
| 91–92 | load one element of each tile, or 0.0f if out of range | the ternary prevents out-of-bounds reads **and** keeps the thread alive for the barriers | Replacing the ternary with an early `return` → threads skip a barrier: undefined (hang or wrong). |
| 93 | **barrier 1**: everyone has *written* the tile | line 97 reads cells written by other threads | Remove it → a thread may read a cell before its owner wrote it (old tile or garbage). |
| 95–97 | the inner product over this tile, all from shared memory | `sA[ty][k]` is one address per warp (broadcast); `sB[k][tx]` is 32 consecutive words = 32 different banks | |
| 98 | **barrier 2**: everyone has *finished reading* the tile | the next iteration overwrites `sA`, `sB` | Remove it → a fast warp overwrites the tile while a slow warp still reads it: a race, wrong only sometimes. |
| 100 | guarded store | out-of-range threads computed a meaningless 0 | |

#### How it is called, on an example

**Real call** (`:467`) **[IMPL]**: `mm_tiled_1to1<32><<<g32, b32>>>(dA, dB, dC, N)`. The template argument and the block must agree: the kernel indexes with `TILE`, not `blockDim`, and each thread loads exactly one cell of each tile, so "block = TILE × TILE" is a contract of the call that nothing in the code checks. At N = 32768: 1024 × 1024 blocks, each owning one 32 × 32 tile of C.

**Hand-sized call [THEORY]:** `mm_tiled_1to1<2><<<dim3(2, 2), dim3(2, 2)>>>(dA, dB, dC, 4)`. Four blocks, each owning a 2 × 2 tile of C; inside a block, thread (tx, ty) owns C[2·by + ty][2·bx + tx] (the coalesced owner table above). The loop at `:88` runs ⌈4 / 2⌉ = 2 times.

**The maths: block matrix multiplication.** Cut A, B and C into TILE × TILE blocks, A_{I,t}, B_{t,J}, C_{I,J}. Then

$$C_{I,J} = \sum_{t=0}^{N/\text{TILE} - 1} A_{I,t}\, B_{t,J}$$

exactly like ordinary matrix multiplication, with numbers replaced by small matrices. **CUDA block (bx, by) computes C_{by, bx}**, and **iteration t of the loop is term t of the sum**: lines 91–92 copy A_{by,t} into `sA` and B_{t,bx} into `sB`, and the inner `k` loop adds each thread's entry of the small product `sA · sB` to its `acc`.

Toy, block (bx = 1, by = 0) → C_{0,1} = rows 0–1, columns 2–3 of C:

| t | `sA` = A_{0,t} | `sB` = B_{t,1} | `sA · sB` | `acc` of the 4 threads after step t |
|---|---|---|---|---|
| 0 | [[1, 2], [0, 1]] | [[2, 1], [0, 0]] | [[2, 1], [0, 0]] | [[2, 1], [0, 0]] |
| 1 | [[0, 1], [2, 1]] | [[1, 2], [0, 1]] | [[0, 1], [2, 5]] | **[[2, 2], [2, 5]]** ✓ = C rows 0–1, cols 2–3 |

One thread of it: (tx = 1, ty = 0) owns entry (0, 1) of the tile = **C[0][3]**:
- t = 0: `sA[0][0]·sB[0][1] + sA[0][1]·sB[1][1]` = 1·1 + 2·0 = **1**
- t = 1: 0·2 + 1·1 = 1 → `acc` = **2** = C[0][3] ✓

Its dot product of length 4 was split into 2 partial dot products of length TILE = 2, one per tile step. At real size, block (2, 1) computes C_{1,2} = Σ A_{1,t} B_{t,2} over t = 0 … 1023: 1,024 products of 32 × 32 matrices, each thread adding 32 terms per step.

### 4.5 Synchronisation and races

![[d05i-race.svg]] [[d05i-race.html|▶ Step through it]]

- `__syncthreads()` is a barrier for **one block**: no thread continues until all threads of the block have arrived. It also makes shared-memory writes before it visible to all threads after it.
- The kernel needs **two** per tile, because there are two hazards: read-before-write (barrier 1) and write-before-everyone-has-read (barrier 2).
- **Can it race?** Not as written: each shared cell is written by exactly one thread per step, every read happens between the two barriers, and global memory is only written once per output by its owner. The races appear only if a barrier is removed, or if a thread skips one.
- **The cost.** The fastest warp waits for the slowest at each barrier, and loads and maths never overlap: load tile → wait → compute → wait. That idle time is what double buffering removes in cuBLAS-class kernels [FUTURE].

### 4.6 Edge cases: N not divisible by TILE

![[d05i-edge.svg]] [[d05i-edge.html|▶ Step through it]]

1. **Grid round-up.** `(N + TILE − 1) / TILE` blocks per side, so every output has a thread. Extra threads own nothing.
2. **K round-up.** The tile loop visits the ragged last tile.
3. **Zero padding.** Out-of-range loads store 0.0f. A zero only ever multiplies into a sum, so it adds nothing.
4. **No early return.** Only the store is guarded (`:100`).
5. **In your runs the guards never fire:** every benchmark N (1024 … 32768) is a multiple of 64. They are correctness insurance, not a performance factor. [RECON]

### 4.7 The new bottleneck: shared-memory bandwidth, and banks

Every FMA in line 97 reads **two** floats from shared memory. **[THEORY]**
- An SM serves about 32 four-byte words per clock (32 banks) but issues 128 FMAs per clock. At 2 reads per FMA it can feed 16 FMAs per clock: **1/8 of peak** = 66.9 / 8 ≈ **8.4 TFLOP/s**.
- Measured 8.64. The kernel sits on this roof. **[MEAS]** The fix has to cut shared reads per FMA (§5, §6).
- (The compiler partly vectorises the `sA` reads; the prep pack's static SASS count is 0.8 FFMA per shared-load instruction. The 8.4 estimate is a source-level model that happens to land within 3% of the measurement.) **[RECON]**

**Banks.** Shared memory has 32 banks; 4-byte word `a` lives in bank `a mod 32`. One warp instruction is served in one pass if its lanes hit 32 different banks, or the same address (broadcast). Two different addresses in one bank are served one after the other (a conflict). In the inner loop: `sA[ty][k]` = one address for the warp (broadcast); `sB[k][tx]` = word `32k + tx` → bank `tx`, 32 different banks. **No conflicts**: the 1-to-1 kernel is slow because of the *number* of shared reads, not conflicts.

### 4.8 Question bank: tiling and synchronisation

**L1. What is shared memory?**
> [!success]- Answer
> Fast on-chip memory inside each SM (up to 228 KB per SM on the H200), allocated per block, visible to all threads of that block, and gone when the block ends. It's ~20–30 cycles away instead of hundreds for HBM. Your tiled kernels use it as a software-managed cache: load a tile once from HBM, then read it TILE times.

**L1. What is the purpose of `__syncthreads()`?**
> [!success]- Answer
> A barrier for all threads of one block, plus a guarantee that shared-memory writes before it are visible after it. In `mm_tiled_1to1` it separates "everyone writes the tile" from "everyone reads the tile" (line 93), and "everyone has read it" from "someone overwrites it" (line 98). It can't synchronise different blocks.

**L2. Why do multiple threads need to cooperate to load a tile?**
> [!success]- Answer
> Because every thread needs a whole row of `sA` and a whole column of `sB` (TILE values each), but if each thread loaded its own operands we'd be back to 2N global loads per output. Splitting the TILE² cells one per thread means the block loads each element **once** and every thread reuses the TILE − 1 elements loaded by others. Cooperation is the reuse. It is also why the barriers are needed.

**L2. Why TILE = 32 for the 1-to-1 kernel?**
> [!success]- Answer
> In the 1-to-1 design the tile equals the block, and a block holds at most 1,024 threads = 32 × 32. So 32 is the largest 1-to-1 tile; it also makes a warp exactly one tile row, so loads are one line and `sB` reads are conflict-free. To go past 32 you must give each thread more than one cell, which is §5's design. [RECON]

**L2. What happens if you increase the tile size?**
> [!success]- Answer
> HBM traffic per output falls as 2N / TILE (intensity TILE / 4), so bigger is better for HBM. But: (1) shared memory per block grows as 2·TILE²·4 B (8 KB at 32, 32 KB at 64, 128 KB at 128, past the 48 KB static limit); (2) fewer blocks fit per SM; (3) with 1 output per thread you can't exceed 32 anyway; (4) once intensity passes the HBM ridge (~14 FLOP/B, reached at TILE = 64) the limit is inside the SM, so a bigger tile buys little. Registers (§6) are the next lever.

**L3. "What happens if N is not a multiple of TILE?"**
> [!success]- Answer
> The kernel stays correct: the grid and the K loop both round up, out-of-range loads become 0.0f (which contributes nothing to any dot product), and only the store is guarded. Threads past the edge stay alive so they reach both barriers. Cost: idle threads in the last block row/column and some wasted FMAs on zeros. None of your benchmark sizes triggers it (all are multiples of 64). The animation in §4.6 walks N = 5, TILE = 2.

**L3. "Can your kernel produce wrong results because of a race?"**
> [!success]- Answer
> Not as written. Every shared cell has exactly one writer per step; all reads of a tile happen between barrier 1 and barrier 2; each output is written once by its owner; A and B are read-only. The two failure modes I'd look for are exactly the ones the barriers prevent: reading before the tile is complete, and overwriting before everyone has read. A third classic one is a barrier inside divergent code; the kernel avoids it by never returning early. I didn't run `compute-sanitizer --tool racecheck` on the PA2 binary; the quiz binaries were clean under `memcheck`. [IMPL] [RECON]

**L3. "Why did tiling give only 1.59× when it cut global loads by 32×?"**
> [!success]- Answer
> Because global loads were no longer the main limit after coalescing: the caches were already giving the coalesced kernel a lot of reuse. Removing the HBM bottleneck exposed the next one, shared-memory bandwidth: 2 shared reads per FMA caps the SM at about 1/8 of peak ≈ 8.4 TFLOP/s, and the kernel measures 8.64. Fewer global accesses don't help if you then saturate the next level. That's the cue for register blocking.

---

## 5. Several outputs per thread: strided vs contiguous loaders (2.24×, and a controlled 4%)

### 5.1 Component card

| | |
|---|---|
| **Problem** | at 1 output per thread the tile is capped at 32 and each FMA needs 2 shared reads |
| **Optimisation [IMPL]** | TILE = 64 with a 32 × 32 block: F = TILE / BLK = 2, so each thread owns F × F = 4 outputs in `acc[2][2]` and loads 4 cells of each tile (`:118`, `:173`) |
| **Why it should help [THEORY]** | HBM intensity 16 FLOP/B (past the ~14 ridge); per k, one `sA` value in a register feeds F FMAs, so shared reads per FMA fall to 1 + 1/F = 1.5 as written |
| **Measured [MEAS]** | strided 3,635 ms (19.36 TFLOP/s), **2.24×** over 1-to-1; contiguous 3,790 ms (18.57), 4% slower |
| **The controlled pair** | identical maths, shared footprint, registers and occupancy; only the two loader lines differ, so the 4% is purely the global access pattern |
| **Trade-off** | 32 KB of shared memory per block; 4 accumulators per thread |

![[d05s-loaders.svg]] [[d05s-loaders.html|▶ Step through it]]

### 5.2 The code, and the two-line difference

```cpp
// mm_tiled_strided :134–158 (inside the t loop)
for (int s = 0; s < F; ++s)
    for (int u = 0; u < F; ++u) {
        int r = ty + s * BLK;           // strided within the tile
        int c = tx + u * BLK;
        int aR = row0 + r,      aC = t * TILE + c;
        int bR = t * TILE + r,  bC = col0 + c;
        sA[r][c] = (aR < N && aC < N) ? A[(size_t)aR * N + aC] : 0.0f;
        sB[r][c] = (bR < N && bC < N) ? B[(size_t)bR * N + bC] : 0.0f;
    }
__syncthreads();
for (int k = 0; k < TILE; ++k)
    for (int i = 0; i < F; ++i) {
        float a = sA[ty + i * BLK][k];                       // one A value in a register…
        for (int j = 0; j < F; ++j)
            acc[i][j] = fmaf(a, sB[k][tx + j * BLK], acc[i][j]);   // …feeds F FMAs
    }

// mm_tiled_contiguous :194–195 changes only the slot mapping:
int r = ty * F + s;             // contiguous run per thread
int c = tx * F + u;
```

| line(s) | what | why | what breaks |
|---|---|---|---|
| `r = ty + s·BLK`, `c = tx + u·BLK` | slot (s, u) of thread (tx, ty) fills tile cell (r, c) | for fixed u, lanes tx = 0…31 hit columns 0…31: **one line** per warp instruction | |
| `c = tx·F + u` (contiguous) | each thread fills an F × F block of adjacent cells | lanes are F = 2 floats apart: columns 0, 2, …, 62 → **2 lines** per instruction, half of each unused | |
| `float a = sA[…][k]` | one A value held in a register, reused for F FMAs | that is the start of register reuse | |
| `sB[k][tx + j·BLK]` | B is re-read per row i in the source | the compiler *may* reuse it (nothing writes shared in between); never inspected | Writing B into registers explicitly is §6. |

**Which outputs one thread owns** (block (2, 1), thread (3, 5), `row0 = 64`, `col0 = 128`): strided owns C[69][131], C[69][163], C[101][131], C[101][163] (scattered, 32 apart); contiguous owns the 2 × 2 patch C[74…75][134…135].

**Why only 4% apart [THEORY]:** the u = 1 instruction of the contiguous loader reads the other half of the same two lines, which is usually still in L1; and loading is a small share of these kernels' time (most time is the inner loop). The pair is valuable as a controlled experiment, not as a big win.

#### How it is called, on an example

**Real calls** (`:469–472`) **[IMPL]**:
```cpp
dim3 g64((N + 63) / 64, (N + 63) / 64);                          // :470  (512, 512) at N = 32768
mm_tiled_strided<64,32><<<g64, b32>>>(dA, dB, dC, N);            // :471
mm_tiled_contiguous<64,32><<<g64, b32>>>(dA, dB, dC, N);         // :472
```
Two template arguments: `TILE = 64` is the side of the C tile a block owns; `BLK = 32` is the block side and must equal `blockDim`. F = TILE / BLK = 2 outputs per thread in each direction. The grid is ⌈N / 64⌉, not ⌈N / 32⌉, because each block now covers 64 × 64 of C with 32 × 32 threads.

**Hand-sized call [THEORY]:** `mm_tiled_strided<4,2><<<dim3(1, 1), dim3(2, 2)>>>(dA, dB, dC, 4)`, and the same with `mm_tiled_contiguous<4,2>`. One block owns all of C (TILE = 4 = N); 4 threads × F² = 4 outputs each; the tile loop runs ⌈4 / 4⌉ = 1 time.

**The maths** is the block sum of §4.4 again; what's new is that one thread holds F × F entries of the block's result. Which ones:
- strided: thread (tx, ty) owns C[row0 + ty + 2i][col0 + tx + 2j], i, j ∈ {0, 1}
- contiguous: thread (tx, ty) owns C[row0 + 2ty + i][col0 + 2tx + j]

Owner of each entry, `(tx,ty)` (the same maps say which thread **loads** each cell of `sA`/`sB`, because each kernel uses one (r, c) formula for loading and computing):

| strided | c0 | c1 | c2 | c3 | | contiguous | c0 | c1 | c2 | c3 |
|---|---|---|---|---|---|---|---|---|---|---|
| **r0** | (0,0) | (1,0) | (0,0) | (1,0) | | **r0** | (0,0) | (0,0) | (1,0) | (1,0) |
| **r1** | (0,1) | (1,1) | (0,1) | (1,1) | | **r1** | (0,0) | (0,0) | (1,0) | (1,0) |
| **r2** | (0,0) | (1,0) | (0,0) | (1,0) | | **r2** | (0,1) | (0,1) | (1,1) | (1,1) |
| **r3** | (0,1) | (1,1) | (0,1) | (1,1) | | **r3** | (0,1) | (0,1) | (1,1) | (1,1) |

Thread (1, 0):
- strided: C[0][1] = 4, C[0][3] = 2, C[2][1] = 1, C[2][3] = 4. For example `acc[1][1]` = C[2][3] = Σₖ `sA[2][k]·sB[k][3]` = 2·1 + 0·0 + 1·2 + 0·1 = **4** ✓.
- contiguous: the 2 × 2 patch C[0…1][2…3] = [[2, 2], [2, 5]] ✓.

**Where the register reuse shows up:** for fixed k and i the code reads `a = sA[ty + 2i][k]` once and multiplies it into `sB[k][tx]` and `sB[k][tx + 2]`: one shared read of A feeds F = 2 FMAs. In the toy: thread (1, 0) at k = 0, i = 0 reads `a` = A[0][0] = 1 once and uses it for both C[0][1] and C[0][3].

### 5.3 Question bank

**L2. Why give each thread several outputs?**
> [!success]- Answer
> Two reasons. (1) It lets the tile be bigger than the 1,024-thread block (64 × 64 with 32 × 32 threads), which doubles HBM intensity to 16 FLOP/B, past the ridge. (2) Within a thread, one loaded operand can feed several FMAs from a register, which cuts shared reads per FMA. Measured: 8,148 → 3,635 ms (2.24×).

**L2. Why is the strided loader coalesced and the contiguous one not?**
> [!success]- Answer
> Coalescing depends on what consecutive lanes touch in **one** instruction. Strided: for a fixed slot u, lane tx reads column `tx + u·32`, so 32 lanes read 32 adjacent floats = 1 line. Contiguous: lane tx reads column `tx·2 + u`, so lanes are 2 floats apart, spanning 64 floats = 2 lines per instruction with half of each wasted. Each thread's own cells are contiguous in the second case, which feels natural but is the wrong axis for a warp.

**L3. "If the contiguous loader requests 2× the lines, why is it only 4% slower? Doesn't that undercut your coalescing story?"**
> [!success]- Answer
> No; it shows where the time goes. The second instruction (u = 1) reads the other half of the same two lines, typically from L1, so the real extra HBM traffic is small. And at TILE = 64 the kernel is no longer HBM-bound: most of the time is the shared-memory inner loop, which is identical in both. Coalescing gave 10.9× when global loads *were* the bottleneck (naive → coalesced); here they are not. It's a single run without error bars, so I'd call 4% "small and probably real", not more.

**L3. "Why keep a slower kernel in the results?"**
> [!success]- Answer
> Because it's a controlled experiment: same arithmetic, same shared-memory footprint, same registers and occupancy; only the global access pattern differs. It isolates one variable. It also matters later: the final `mm_block2d` kernel inherited the contiguous-style mapping (`tx·R + j`) from the lab-quiz kernel it was built from, and this pair tells me how much that costs (a few percent, by this evidence). [IMPL] [RECON]

---

## 6. 2D register blocking: `mm_block2d<64,4>` (the final kernel)

### 6.1 Component card

| | |
|---|---|
| **Problem** | shared-memory reads per FMA cap the SM: 2 per FMA in the 1-to-1 kernel → ~8.4 TFLOP/s roof |
| **Optimisation [IMPL]** | each of 256 threads owns an R × R = 4 × 4 patch of C in registers (`reg[4][4]`); per k it loads a column piece `av[4]` of `sA` and a row piece `bv[4]` of `sB` into registers and does the 16-FMA outer product (`:232–286`) |
| **Why it should help [THEORY]** | shared reads per FMA = 2R / R² = 2 / R = **0.5** at R = 4, 4× fewer than the 1-to-1 kernel → shared roof ≈ 33.5 TFLOP/s; more independent accumulators also give instruction-level parallelism |
| **Hardware level** | 16 accumulators + 8 operands + indices → 62 registers per thread (cuobjdump); the compiler vectorised the shared loads to `LDS.128` (SASS, [RECON]) |
| **Measured [MEAS]** | 2,736.6 ms, 25.71 TFLOP/s: 1.33× over strided, 3.0× over 1-to-1, 51.3× over naive, 48.2% of cuBLAS |
| **Trade-offs** | 62 → 64 registers × 256 threads → 4 blocks per SM → **50% occupancy**; contiguous-style loader (stride-R global loads); 2-way bank conflicts on `av`/`bv` reads by hand count |
| **Alternatives** | R = 8 (more registers, lower occupancy); warp tiling; transposed `sA` for vector reads; double buffering [FUTURE] |
| **Limitations** | 38% of peak; no overlap of loads and maths; scalar strided global loads |

### 6.2 The idea: an outer product per k

![[d5-9-regblock.svg]] [[d5-9-regblock.html|▶ Step through it]]

Per k: `av` is R × 1, `bv` is 1 × R, `reg += av · bv` is R × R. Each loaded value is used R times. Toy: `av = [1, 0]`, `bv = [2, 1]` gives [[2, 1], [0, 0]]: 4 products from 4 loads.

![[d05s-regblock.svg]] [[d05s-regblock.html|▶ Step through it]]

### 6.3 The code, line by line

```cpp
// which outputs (:237–241)
const int tx = threadIdx.x, ty = threadIdx.y;
const int rowBase = blockIdx.y * TILE + ty * R;
const int colBase = blockIdx.x * TILE + tx * R;
float reg[R][R];                        // zeroed by two unrolled loops

for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {          // :247
    const int kOff = t * TILE;
    for (int i = 0; i < R; ++i)                              // the load (:250–258)
        for (int j = 0; j < R; ++j) {
            int sr = ty * R + i, sc = tx * R + j;
            int aR = rowBase + i, aC = kOff + sc;
            int bR = kOff + sr,   bC = colBase + j;
            sA[sr][sc] = (aR < N && aC < N) ? A[(size_t)aR * N + aC] : 0.0f;
            sB[sr][sc] = (bR < N && bC < N) ? B[(size_t)bR * N + bC] : 0.0f;
        }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {                         // the compute (:262–273)
        float av[R], bv[R];
        for (int i = 0; i < R; ++i) av[i] = sA[ty * R + i][k];
        for (int j = 0; j < R; ++j) bv[j] = sB[k][tx * R + j];
        for (int i = 0; i < R; ++i)
            for (int j = 0; j < R; ++j)
                reg[i][j] = fmaf(av[i], bv[j], reg[i][j]);
    }
    __syncthreads();
}
for (int i = 0; i < R; ++i) {                                // the store (:277–286)
    int r = rowBase + i;  if (r >= N) continue;
    for (int j = 0; j < R; ++j) { int c = colBase + j; if (c < N) C[(size_t)r * N + c] = reg[i][j]; }
}
```
(Every inner loop carries `#pragma unroll` in the source.)

| piece | what | why | what breaks if changed |
|---|---|---|---|
| `rowBase`, `colBase` | top-left output of **this thread**: rows `by·64 + ty·4 …+3`, cols `bx·64 + tx·4 …+3` | each thread owns a contiguous 4 × 4 patch | |
| `float reg[R][R]` + `#pragma unroll` | 16 accumulators in **registers** | with R a template constant and every index loop unrolled, every `reg[i][j]` index is known at compile time, so the array maps to registers | A runtime R, or a loop not unrolled, would put `reg` in local memory (off-chip): a large slowdown. (This happened to `o_acc` in FlashAttention, Day 6.) |
| `kOff = t·TILE` | first K index of tile t | | |
| load loop | the thread fills a 4 × 4 block of cells of each tile: rows `ty·4 + i`, cols `tx·4 + j` | the lab-quiz mapping it was built from | Lanes `tx` are 4 floats apart on the global side: **stride-R loads** (half or more of each line unused per instruction). [RECON] |
| `__syncthreads()` × 2 | same two hazards as §4.5 | | |
| `av[i] = sA[ty·4 + i][k]` | 4 values down column k of `sA` | the A side of the outer product | |
| `bv[j] = sB[k][tx·4 + j]` | 4 adjacent values along row k of `sB` | adjacent → the compiler emitted one 128-bit `LDS.128` [RECON] | |
| 16 `fmaf`s | `reg += av · bv` | 8 shared loads for 16 FMAs = **0.5 per FMA** | R = 2: 4 loads for 4 FMAs = 1 per FMA. |
| store | 16 guarded writes | | |

**Traced at real size**, block (bx = 2, by = 1), thread (tx = 3, ty = 5), N = 32768:
1. `rowBase = 1·64 + 5·4 = 84`, `colBase = 2·64 + 3·4 = 140` → C rows **84–87**, cols **140–143**.
2. 32768 / 64 = **512** tile steps.
3. Load: 16 cells of `sA` (rows 20–23, cols 12–15 of the tile, from A[84…87][t·64 + 12…15]) and 16 of `sB` (from B[t·64 + 20…23][140…143]).
4. Compute: per k, `av = sA[20…23][k]`, `bv = sB[k][12…15]`, 16 FMAs.
5. Check: 512 × 64 × 16 = 524,288 FMAs = 16 outputs × 32,768 terms ✓.

The toy trace (N = 4, R = 2, thread (1, 0) → C[0…1][2…3] = [[2, 2], [2, 5]]) is in the study guide, [[Day 05 - CUDA I#6.2 The code]].

#### How it is called, on an example

**Real calls** (`:475–483`) **[IMPL]**:
```cpp
dim3 gb((N + 63) / 64, (N + 63) / 64);          // :476 / :480  (512, 512) at N = 32768
mm_block2d<64,2><<<gb, b32>>>(dA, dB, dC, N);   // :477  block 32 × 32 = 64/2 per side
dim3 bb(16, 16);                                // :481
mm_block2d<64,4><<<gb, bb>>>(dA, dB, dC, N);    // :482  block 16 × 16 = 64/4 per side
```
The contract is **block side = TILE / R**: each thread covers R columns and R rows of the 64 × 64 tile. That's why R = 4 needs its own 16 × 16 block `bb`, and why both configs share the same grid: a block still owns 64 × 64 of C.

**Hand-sized call [THEORY]:** `mm_block2d<4,2><<<dim3(1, 1), dim3(2, 2)>>>(dA, dB, dC, 4)` (the study guide's toy). One block owns all of C; thread (tx, ty) owns the 2 × 2 patch rows `2ty … 2ty+1`, cols `2tx … 2tx+1` (the contiguous owner map in §5). One tile step.

**The maths: a sum of outer products.** For the patch with top-left (r, c):

$$C[r{:}r{+}R,\; c{:}c{+}R] = \sum_{k=0}^{N-1} \underbrace{A[r{:}r{+}R,\; k]}_{\texttt{av},\ R \times 1}\ \underbrace{B[k,\; c{:}c{+}R]}_{\texttt{bv},\ 1 \times R}$$

Thread (tx = 1, ty = 0): r = 0, c = 2:

| k | `av` = A[0…1][k] | `bv` = B[k][2…3] | `av · bv` (outer product) | `reg` after k |
|---|---|---|---|---|
| 0 | [1, 0] | [2, 1] | [[2, 1], [0, 0]] | [[2, 1], [0, 0]] |
| 1 | [2, 1] | [0, 0] | [[0, 0], [0, 0]] | [[2, 1], [0, 0]] |
| 2 | [0, 2] | [1, 2] | [[0, 0], [2, 4]] | [[2, 1], [2, 4]] |
| 3 | [1, 1] | [0, 1] | [[0, 1], [0, 1]] | **[[2, 2], [2, 5]]** ✓ |

**Tiling vs register blocking, as loop orders.** Both compute the same 16 terms per patch entry. Tiling (§4.4) runs `for i, j: for k:` (one dot product at a time); register blocking runs `for k: for i, j:` (the whole patch moves one rank-1 update per k). Swapping the loops is what lets 2R loaded values (`av`, `bv`) feed R² FMAs.

At real size (`<64,4>`), thread (3, 5) of block (2, 1) computes the 4 × 4 patch C[84…87][140…143] as the sum of 32,768 outer products: 64 per tile × 512 tiles.

### 6.4 Registers, occupancy and why 50% still wins

![[d05i-occupancy.svg]] [[d05i-occupancy.html|▶ Step through it]]

**Occupancy** = resident warps per SM ÷ 64. Limited by whichever runs out first: registers, shared memory, thread slots. **[THEORY]**, using register counts from `cuobjdump` **[RECON]**:

| kernel | threads/block | regs/thread | regs/block | blocks by regs | by threads | by shared (32 KB) | resident warps | occupancy | time |
|---|---|---|---|---|---|---|---|---|---|
| strided `<64,32>` | 1,024 | 32 | 32,768 | 2 | 2 | ~7 | 64 | 100% | 3,635 ms |
| block2d `<64,2>` | 1,024 | 40 | 40,960 | **1** | 2 | ~7 | 32 | 50% | 4,263 ms |
| block2d `<64,4>` | 256 | 62 → 64 | 16,384 | **4** | 8 | ~7 | 32 | 50% | **2,737 ms** |

Both block2d kernels are at 50%, yet R = 4 is 1.56× faster than R = 2, and faster than the 100%-occupancy strided kernel. **Occupancy is a means (hiding latency), not the goal.** With 16 independent accumulators per thread, each warp has plenty of independent FMAs in flight, so fewer warps can keep the FMA units busy: instruction-level parallelism replaces thread-level parallelism.

### 6.5 Bank conflicts and coalescing in `mm_block2d` (the hidden costs) **[RECON]**

With 16 × 16 threads, one warp = two `ty` values × 16 `tx` values.
- `av[i] = sA[ty·4 + i][k]`: two addresses per warp (one per `ty`), 4 rows = 256 words apart → **same bank**: 2-way conflict.
- `bv[j] = sB[k][tx·4 + j]`: words `64k + 4tx + j`; lanes tx and tx + 8 land on the same bank → 2-way (scalar view; with `LDS.128` the hardware splits the warp into phases, so the real cost may differ: not profiled).
- Shared **stores** `sA[ty·4 + i][tx·4 + j]`: lanes 4 words apart plus two `ty` rows 256 words apart → about 4-way (prep pack Q7).
- Global loads `A[…][kOff + tx·4 + j]`: lanes 4 floats apart → each instruction spans 4× the lines it needs.

**How register blocking interacts with coalescing:** giving a thread a contiguous R × R patch is natural for the register outer product, but if the *same* mapping drives the global loader, consecutive lanes are R floats apart and the loads stop coalescing. The fix is to decouple the two: load tiles with a strided (lane-adjacent) mapping, as kernel #4 does, and compute with the patch mapping. You didn't do that; it's one of the reasons for the 48% [FUTURE].

### 6.6 Why you stop at 48.2% of cuBLAS (ranked guesses, not profiled) **[RECON]**

1. **No overlap.** Load → sync → compute → sync. Every tile pays the full global-load latency. cuBLAS double-buffers with `cp.async`/TMA.
2. **Scalar, stride-R global loads** (`LDG.E`); cuBLAS uses 128-bit coalesced loads.
3. **Bank conflicts** on shared stores (~4-way) and reads (2-way).
4. **Tile and register block too small**: 64 × 64 tile, 4 × 4 per thread; cuBLAS-class kernels use 128 × 128+ tiles, 8 × 8+ per thread, warp tiles.
5. **No SASS-level tuning, no autotuning.**

Without Nsight Compute you can't rank them. Your first experiment would be vector loads plus double buffering. [FUTURE]

### 6.7 Question bank: register blocking and occupancy

**L1. What is register blocking?**
> [!success]- Answer
> Giving each thread a small block of outputs (R × R) held in registers, and computing them as an outer product: per k, load R values of A and R of B into registers once, then do R² FMAs. Each loaded value is reused R times from the fastest storage there is. In your kernel R = 4: 8 shared loads feed 16 FMAs.

**L1. What is occupancy?**
> [!success]- Answer
> Resident warps on an SM divided by the maximum (64 warps = 2,048 threads on the H200). Higher occupancy gives the scheduler more warps to switch to while others wait on memory. It's limited by registers per thread × threads per block, shared memory per block, and thread slots. Your final kernel's theoretical occupancy is 50% (register-limited, from the occupancy rules; not measured).

**L2. Why can register pressure reduce occupancy?**
> [!success]- Answer
> The SM has a fixed register file (65,536 × 32-bit). Each resident thread holds its registers for its whole life. At 64 registers (62 rounded up to the allocation granularity) × 256 threads, a block needs 16,384 registers, so only 4 blocks = 1,024 threads fit: 50%. Raise R to 8 and the 64 accumulators push registers past 100 per thread, so occupancy drops further, and past 255 registers the compiler spills to local memory. [THEORY]

**L2. Why is R = 2 slower than the strided kernel when both give 4 outputs per thread?**
> [!success]- Answer
> Same work per thread, worse everything else: `block2d<64,2>` uses 1,024 threads × 40 registers = 40,960 registers per block, so only 1 block fits per SM (50%), while strided at 32 registers fits 2 (100%); R = 2's register reuse (1 shared load per FMA) is no better than strided's; and R = 2 uses the contiguous-style, stride-2 loader. Only at R = 4 is the reuse large enough (0.5 loads per FMA) to beat the strided kernel.

**L3. "Why not R = 8? Or why didn't you choose a different tile size?"**
> [!success]- Answer
> Not tried, so I'd reason it out. R = 8 with TILE = 64 means 8 × 8 = 64 threads per block and 64 accumulators per thread (100+ registers): shared reads per FMA halve again (0.25), but occupancy falls to a few warps per SM and each block's load phase has only 64 threads to issue 8 K loads. TILE = 128 with static shared memory doesn't compile (2 × 64 KB > 48 KB). The cuBLAS-style answer is a 128 × 128 block tile with a short k-slice (8–16) in shared memory, 256 threads at 8 × 8 each, and double buffering. 64/4 was the largest configuration that fitted the static template design and the lab-quiz kernel it grew from. [RECON] [FUTURE]

**L3. "Is your kernel compute-bound?"**
> [!success]- Answer
> By the HBM roofline, yes: intensity 16 FLOP/B is past the ridge of ~14, so HBM shouldn't be the limit. But it runs at 38% of peak, so the FMA units aren't saturated either. The binding limits are inside the SM: shared-memory traffic (my second roof estimate is ~33.5 TFLOP/s for 0.5 reads per FMA), bank conflicts, and exposed load latency between barriers. The precise answer is "past the HBM ridge, limited by on-chip effects", and I'd confirm with Nsight Compute's pipe utilisation if I could run it.

**L3. "How would you optimise it further?"**
> [!success]- Answer
> In order of expected payoff [FUTURE]: (1) decouple the loader from the compute mapping so global loads are lane-adjacent and 128-bit (`float4`); (2) double-buffer the shared tiles with `cp.async` so tile t+1 loads while tile t computes; (3) store `sA` transposed (k-major) so both operand reads are conflict-free `LDS.128`; (4) warp tiling: 128 × 128 block tile split into warp tiles, 8 × 8 per thread; (5) swizzled block order for L2 reuse; (6) autotune the shape per N. Beyond FP32 CUDA cores: TF32 or BF16 tensor cores via `mma.sync`/WMMA, which is a different ceiling entirely. Boehm's worklog shows that path going from ~kernel-5 level (where you are) to over 90% of cuBLAS.

---

## 7. The harness: reference, timing, correctness

### 7.1 Component card

| | |
|---|---|
| **Reference [IMPL]** | cuBLAS SGEMM runs **first** at every N (`:432–440`); its output `hRef` is the correctness reference and its time is the speed ceiling. A CPU reference would take ~34 hours at N = 32768. |
| **Timing [IMPL]** | `time_kernel` (`:312–344`): untimed warm-up launches, then one timed run with CUDA events; if 10 runs fit in 30 s, 9 more back to back between one event pair; returns the mean and n. |
| **Check [IMPL]** | `record` (`:445–448`) copies `dC` back and computes `max_rel_err` (`:353–363`) over the **first 4M elements**. |
| **Output [IMPL]** | one CSV row per kernel and N: `impl, N, time_ms, tflops, iters, max_rel_err`. |
| **Known gaps [RECON]** | check covers only the first 4M outputs (whole matrix up to N = 2048, ~122 rows at 32768; the comment at `:352` claims a whole-matrix sample); `dC` is never cleared between kernels and `cudaGetLastError` is not called after launches, so a kernel that failed to launch would inherit the previous kernel's correct output; n = 1 for most kernels at N = 32768; no repeats, no error bars; clocks not locked. |

### 7.2 Timing, line by line (`matmul_bench.cu:321–344`)

![[d05i-timing.svg]] [[d05i-timing.html|▶ Step through it]]

```cpp
321  // One timed iteration first, so a slow kernel doesn't commit to ten of them.
322  CUDA_CHECK(cudaEventRecord(beg));
323  launch();
324  CUDA_CHECK(cudaEventRecord(end));
325  CUDA_CHECK(cudaEventSynchronize(end));
326  float first = 0.f;
327  CUDA_CHECK(cudaEventElapsedTime(&first, beg, end));
328
329  int extra = 0;
330  double total = first;
331  if (first / 1000.0 * iters < budget_s) {
332      extra = iters - 1;
333      CUDA_CHECK(cudaEventRecord(beg));
334      for (int i = 0; i < extra; ++i) launch();
335      CUDA_CHECK(cudaEventRecord(end));
336      CUDA_CHECK(cudaEventSynchronize(end));
337      float rest = 0.f;
338      CUDA_CHECK(cudaEventElapsedTime(&rest, beg, end));
339      total += rest;
340  }
341
342  CUDA_CHECK(cudaEventDestroy(beg));
343  CUDA_CHECK(cudaEventDestroy(end));
344  return { total / (1 + extra), 1 + extra };
```
(Real line numbers. Lines 318–319, just above, run the `warmup` untimed launches and `cudaDeviceSynchronize()`.)

| call | what it does | why it matters |
|---|---|---|
| `cudaEventRecord(beg)` | enqueues a marker in the stream; the GPU timestamps it **when it reaches it** | the time excludes host work before the kernel |
| `launch()` | a lambda that enqueues one kernel; returns in microseconds | asynchronous: a CPU timer here would measure only the enqueue |
| `cudaEventRecord(end)` | marker after the kernel | brackets exactly the GPU work |
| `cudaEventSynchronize(end)` | the CPU waits until the GPU passes `end` | the elapsed time is undefined before this |
| `cudaEventElapsedTime(&ms, beg, end)` | GPU-clock milliseconds between the markers | kernel time only: no copies, no host code |
| `first / 1000 · iters < budget_s` | would 10 runs fit in 30 s? | naive: 140.5 s × 10 ≫ 30 → n = 1; block2d R = 4: 27.4 s → n = 10 |
| 9 launches between one event pair | back-to-back runs | the mean hides per-launch gaps (irrelevant for multi-second kernels, important for tiny ones) |

`CUDA_CHECK(call)` is a macro in the file that prints file, line and the error string and exits if a call fails. Kernel launches themselves return nothing, so launch errors only show up at the next checked call: one reason `cudaGetLastError` after launches is on the fix list.

### 7.3 The cuBLAS column-major trick (`:432–440`)

```cpp
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
            &alpha, dB, N, dA, N, &beta, dC, N);
```
cuBLAS is column-major. A row-major buffer read as column-major is the transpose. Asking for B·A gives Bᵀ·Aᵀ = (A·B)ᵀ in column-major, which, read row-major, is A·B. No copies, no transposes. 2 × 2 check: A = [[1,2],[3,4]], B = [[5,6],[7,8]] → buffer `[19, 22, 43, 50]` = A·B row-major ✓.

### 7.4 The check (`:353–363`)

`e = |ref − got| / max(|ref|, 1)`: relative for big values, absolute for small ones (so near-zero outputs don't blow up the ratio). `worst` = the max over the first `min(N², 4,000,000)` elements. **[MEAS]:** 0.000e+00 for every GPU kernel up to N = 16384; **4.48 × 10⁻⁵** for all of them at N = 32768. The custom kernels all agree with each other, so the difference is on the cuBLAS side, most likely a different summation order (for example split-K) at that size: a hypothesis, not documented. **Don't say "bit-exact" without "up to N = 16384, on the first 4M outputs".**

### 7.5 Question bank: the harness

**L1. Why use CUDA events instead of a CPU timer?**
> [!success]- Answer
> Kernel launches are asynchronous: the CPU call returns long before the GPU finishes. A CPU timer around the launch measures the enqueue; a CPU timer around launch + `cudaDeviceSynchronize` measures the kernel plus sync and driver overhead. Events are timestamped by the GPU itself when the stream reaches them, so `cudaEventElapsedTime(beg, end)` is the GPU time of exactly the work between them.

**L1. Why warm-up iterations?**
> [!success]- Answer
> The first launch can include one-off costs that aren't the kernel: loading the module onto the device, clocks ramping up from idle, cold L2 and TLBs, and for cuBLAS the algorithm/workspace setup. `time_kernel` runs untimed warm-ups first (1 for your kernels, 2 for cuBLAS) and synchronises before the timed region. [IMPL] [THEORY]

**L2. Your naive kernel was timed once. Is that acceptable?**
> [!success]- Answer
> For that kernel, yes, with the caveat stated. It's a 140-second run, so launch overhead and timer resolution are negligible, and naive sits at 0.50 TFLOP/s at every N from 1024 up, so it's stable. The 51.3× ratio holds at 50–51× from N = 2048 up, which is a better robustness argument than repeats of one size. But I say "single run, no error bars", and I don't lean on small differences such as the 4% loader gap. [MEAS]

**L2. Why time only the kernel, not the copies?**
> [!success]- Answer
> The question is kernel efficiency; copies are identical for every kernel (three 4.29 GB matrices at N = 32768, ~13 GB over PCIe) and would swamp the differences. In a real application you'd keep data resident on the GPU across many matmuls, so kernel time is the relevant number. [IMPL]

**L3. "How do you know every kernel actually ran? Could a failed launch have reported cuBLAS's answer?"**
> [!success]- Answer
> Good catch: in principle yes. `dC` isn't cleared between kernels and there's no `cudaGetLastError` after launches, so a kernel that failed to launch would leave the previous correct output in `dC` and pass the check. In practice every kernel has a distinct time (from 1.3 s to 140 s), which a non-launch wouldn't produce. The fix is two lines: `cudaMemset(dC, 0, bytes)` before each kernel and `CUDA_CHECK(cudaGetLastError())` after each launch. [RECON]

**L3. "Your error is exactly zero up to 16384. Isn't that suspicious?"**
> [!success]- Answer
> It's plausible: every kernel, and cuBLAS at those sizes, accumulates each output in k order with FMA, so the same rounding sequence gives the same bits. At 32768 all custom kernels show the same 4.48e-5, i.e. they still agree with each other and cuBLAS changed, most likely its kernel choice and summation order. It's a hypothesis; the repo doesn't document it. And the check is a prefix of 4M outputs, so "zero error" means "zero on the first 4M outputs". [MEAS] [RECON]

---

## 8. Performance analysis

### 8.1 How to measure a kernel correctly (and what the project did)

| practice | why | in the project |
|---|---|---|
| time on the GPU clock (events or a profiler) | launches are asynchronous | ✓ CUDA events |
| warm up first | exclude one-off costs | ✓ 1–2 untimed launches |
| keep host syncs and copies out of the timed region unless they are part of the workload | measure the thing you mean | ✓ for PA2 (Day 6 has a case where this matters) |
| repeat and report n, mean, spread | noise, clock variation | partly: 10 runs when they fit in 30 s, else n = 1; no spread |
| fix the clocks or report them | boost clocks vary with temperature and power | ✗ no root; peak uses the 1.98 GHz boost clock |
| check correctness of every timed kernel | a fast wrong kernel is worthless | ✓ against cuBLAS, prefix only |
| compare against a strong baseline | ratios against strawmen mislead | ✓ cuBLAS; naive also reported |
| state the configuration | FP32 vs TF32, sizes, GPU | ✓ FP32 CUDA cores, default math mode |

### 8.2 Latency vs throughput

- **Latency** = time for one operation to complete: one HBM load takes hundreds of cycles; one shared-memory load ~20–30. **[THEORY]**
- **Throughput** = operations completed per unit time: TFLOP/s, TB/s.
- GPUs trade latency for throughput: they don't make one load faster, they keep many in flight (many warps, many independent instructions per warp) so the units stay busy. That is what occupancy and ILP buy.
- In your numbers: every kernel's "time" is a throughput measure (TFLOP/s = 2N³ / time). The 140 s naive run is long because of low throughput, not because any single operation is slow.

### 8.3 Arithmetic intensity and the roofline

![[d05s-intensity.svg]] [[d05s-intensity.html|▶ Step through it]]

**Arithmetic intensity** = FLOPs per byte moved from a given memory level. **Roofline**: attainable FLOP/s = min(peak, intensity × bandwidth). **Ridge** = peak / bandwidth.

H200 FP32: peak 66.9 TFLOP/s (132 × 128 × 2 × 1.98 GHz, computed by the harness at `:385–392`), HBM 4.8 TB/s → ridge ≈ **13.9 FLOP/B**. **[THEORY]**

![[d5-10-roofline.svg]]

| kernel | HBM intensity [THEORY] | HBM roof | shared-memory roof [THEORY] | measured [MEAS] | reading |
|---|---|---|---|---|---|
| naive / coalesced | 0.25 (no reuse) | 1.2 | | 0.50 / 5.44 | caches give reuse the count ignores |
| tiled 1-to-1, T = 32 | 8 | 38.4 | 2 reads/FMA → 8.4 | 8.64 | on the shared roof |
| strided, T = 64 | 16 | 66.9 (past ridge) | 1.5 reads/FMA as written → ~11 | 19.36 | above the as-written model: the compiler reuses B values / vectorises (SASS: 1.6 FFMA per LDS) |
| block2d R = 4 | 16 | 66.9 | 0.5 reads/FMA → 33.5 | 25.71 | under its roof: conflicts, latency, overheads |
| cuBLAS | ≥ 32 | 66.9 | | 53.37 | 80% of peak |

(The strided row is the one place the source-level count clearly under-predicts: say so if asked. The "~11" is 66.9 × 1/(8 · 0.75); treat it as a model, and quote the SASS count from the prep pack instead.)

**The second roof**, derived on a whiteboard: shared memory serves 32 banks × 4 B = 128 B per clock per SM → 132 × 128 B × 1.98 GHz ≈ **33.5 TB/s**. An FMA is 2 FLOPs. At 2 shared reads (8 B) per FMA: 2 / 8 = 0.25 FLOP/B × 33.5 = **8.4 TFLOP/s**. At 0.5 reads (2 B) per FMA: 1 FLOP/B → **33.5 TFLOP/s**.

### 8.4 Compute-bound or memory-bound? How to tell

1. **Count intensity** at each level (HBM, shared). Compare with each ridge.
2. **Compare measured with each roof.** If you sit on a roof, that level is the limit (1-to-1 kernel: 8.64 vs 8.4 shared roof).
3. **Perturb.** Change only memory traffic (the strided/contiguous pair) or only arithmetic, and see which moves the time. Your controlled pair says global access patterns barely matter at T = 64.
4. **Profile** (what you'd do with root): Nsight Compute's "speed of light" shows % of peak for memory and compute pipes; a kernel at ~80% memory throughput and ~20% FMA pipe is memory-bound.

### 8.5 How tile size, occupancy, registers and shared memory interact

| knob ↑ | helps | hurts | in your data |
|---|---|---|---|
| TILE | HBM traffic ÷ TILE | shared memory per block ↑ (2·TILE²·4 B), fewer blocks per SM, more ragged-edge waste, 48 KB static cap | 32 → 64 tile (with F = 2): 2.24× |
| outputs per thread (F or R) | shared reads per FMA ↓ (2/R), more ILP | registers per thread ↑ → occupancy ↓; risk of spills | R = 4 at 50% beats strided at 100% |
| threads per block | more warps to hide latency | registers per block ↑: a big block can fit only once per SM | `block2d<64,2>`: 1,024 × 40 regs → 1 block per SM |
| occupancy | latency hiding | (it's an output of the above, not a knob) | not the goal: R = 4 wins at 50% |

**The general shape:** every reuse trick spends on-chip storage (shared memory or registers) to save off-chip traffic, and on-chip storage is what decides occupancy. The art is spending it where it removes the binding bottleneck.

### 8.6 Measured speedup vs theoretical expectation

| step | theory predicted | measured | why they differ |
|---|---|---|---|
| coalescing | ≤ 16.5× (33 → 2 lines) | 10.9× | L1 hits softened the naive kernel |
| 1-to-1 tiling | HBM traffic ÷ 32 | 1.59× | the limit moved to shared memory (8.4 roof) |
| contiguous vs strided loader | 2× the lines per instruction | 4% | second half from L1; loads are a small share of time |
| R = 4 vs 1-to-1 | shared roof 8.4 → 33.5 (4×) | 3.0× | conflicts, exposed latency, overheads |
| final vs cuBLAS | | 48.2% | §6.6 list |

**The rule:** quote the measured number as the result and the theory as the explanation. Never quote a theoretical number as if measured.

### 8.7 Question bank: performance analysis

**L1. What is arithmetic intensity?**
> [!success]- Answer
> FLOPs performed per byte moved from a memory level. For a matmul with no reuse: one FMA (2 FLOPs) needs two 4-byte floats → 0.25 FLOP/B. With TILE × TILE shared tiles each loaded float is used TILE times → TILE / 4 FLOP/B: 8 at TILE = 32, 16 at TILE = 64. You compare it with the machine's ridge (peak ÷ bandwidth ≈ 13.9 FLOP/B for the H200's HBM) to see which side of the roofline you're on.

**L2. How does the roofline help, concretely, in your project?**
> [!success]- Answer
> It told me where to look next. The 1-to-1 kernel's HBM roof is 38 TFLOP/s, yet it ran at 8.6: so HBM wasn't the limit. Applying the same model to shared memory gave an 8.4 TFLOP/s roof, right at the measurement, which pointed at shared reads per FMA as the bottleneck and at register blocking as the fix (0.5 reads per FMA → 33.5 roof, measured 25.7). It's a direction-finder; caches make the exact numbers off (coalesced kernel at 5.4 above its "1.2" roof).

**L3. "Why might a kernel with fewer global-memory accesses still be slower?"**
> [!success]- Answer
> Because global memory may not be the binding limit, and the change that cut global accesses may have cost something else. Three examples from the project: (1) `block2d<64,2>` has the same global traffic per output as the strided kernel but is slower (4,263 vs 3,635 ms) because its register use leaves 1 block per SM and its loader is stride-2; (2) the 1-to-1 tile cut HBM loads by 32× but gained only 1.59× because it then saturated shared memory; (3) more generally, extra barriers, bank conflicts, lower occupancy or register spills can all outweigh saved HBM bytes. Find the binding roof first.

**L3. "Why can shared-memory tiling improve performance but also reduce occupancy?"**
> [!success]- Answer
> Tiling trades on-chip storage for off-chip traffic. Each resident block holds its tiles in the SM's shared memory for its whole life, so bigger tiles mean fewer blocks fit. At TILE = 64 your kernels use 32 KB per block: ~7 blocks per SM by shared memory alone, so in your kernels registers bind first. But TILE = 128 (128 KB of double tiles) would allow one block per SM, a hard occupancy cut. The kernel is faster as long as the saved HBM traffic matters more than the lost latency hiding; past the HBM ridge, it usually doesn't.

**L3. "Is your optimisation always faster? What about small N?"**
> [!success]- Answer
> No. At small N the grid is too small to fill 132 SMs: at N = 1024, `mm_block2d<64,4>` launches only 16 × 16 = 256 blocks, under 2 per SM, so many SMs idle and per-block overheads (barriers, the load phase) weigh more. The data shows the shape: the speedup over naive is 45× at N = 1024 vs 50–51× from 2048 up, and the share of cuBLAS is 61% at N = 1024 vs 48–50% above (cuBLAS picks shapes for small problems that you don't). At tiny N a launch-bound regime takes over entirely: any kernel costs a few microseconds of launch. [MEAS] [THEORY]

---

## 9. Say it out loud

### 9.1 The 30-second introduction

> "On an H200 I wrote FP32 matrix-multiply kernels in CUDA, starting from a naive one-thread-per-output kernel and fixing one bottleneck at a time: coalescing the global loads, staging tiles in shared memory, and finally 2D register blocking, where each thread computes a 4 × 4 patch from registers. The final kernel runs at 25.7 TFLOP/s: 51× the naive kernel and 48% of cuBLAS SGEMM, and I can show with a memory-traffic count why each step helped and why I stop at 48%."

### 9.2 The 2-minute technical explanation

> "The setup is one H200, FP32 on CUDA cores, no tensor cores, so cuBLAS SGEMM is the fair comparison. I time kernels with CUDA events, check every kernel against cuBLAS, and run N up to 32768.
>
> The naive kernel maps neighbouring threads to neighbouring **rows**, so a warp's 32 loads of A are 128 KB apart: 32 cache lines per load. Swapping the thread mapping so lanes walk columns makes each load one line. That two-line change gave 10.9×.
>
> But each output still read 2N floats from memory. With 32 × 32 shared-memory tiles, a block loads each element once and reuses it 32 times. That gave 1.6×, less than you'd expect, because the bottleneck moved: every FMA now needed two shared-memory reads, and that caps an SM at about an eighth of peak, ~8.4 TFLOP/s. I measured 8.6.
>
> So the last step cuts shared reads per FMA. Each thread keeps a 4 × 4 block of outputs in registers; per k it loads 4 values of A and 4 of B and does 16 FMAs: 0.5 reads per FMA. That got to 25.7 TFLOP/s at 50% occupancy, which is faster than my 100%-occupancy kernel, because more work per thread beats more threads.
>
> I stop at 48% of cuBLAS because my kernel doesn't overlap loads with compute, its global loads are scalar and partly strided, and it has some bank conflicts. Those are the next steps."

### 9.3 The 5-minute walkthrough (structure; expand each with §2–§8)

1. **Frame (30 s).** H200, FP32, cuBLAS as reference and ceiling, events, N up to 32768, CSV per run.
2. **Execution model (45 s).** Grid of blocks, block on one SM, warp of 32 lanes; "a warp is 32 consecutive `threadIdx.x`" is the rule everything else uses.
3. **Naive → coalesced (60 s).** Draw one warp's addresses: 32 rows (33 lines per step) vs 32 columns (2 lines). 16.5× predicted, 10.9× measured, L1 explains the gap.
4. **Tiling (60 s).** Draw the tile march; loads per output 2N → 2N/TILE; the two barriers and what each prevents; then the surprise: only 1.59×, and the shared-memory roof (8.4 vs 8.64 measured).
5. **Register blocking (60 s).** The outer product per k; 2/R reads per FMA; occupancy 50% and why it still wins; the strided/contiguous controlled pair as evidence that loads aren't the limit anymore.
6. **Numbers and honesty (45 s).** 51.3× vs naive but 4.7× vs coalesced; 48.2% of cuBLAS; n = 1 for most kernels at 32K; correctness checked on the first 4M outputs; no profiler, so bottlenecks are by count.
7. **Next steps (20 s).** Vector loads, double buffering, warp tiling, tensor cores.

### 9.4 The whiteboard version of the kernel architecture

Draw these four pictures, left to right, and talk over them:

```
 (1) who owns what                     (2) one tile step                     (3) one thread, one k
 C = 32768²                            HBM ──coalesced──► sA 64×64           av[4] = sA[ty·4+0..3][k]
 ┌──────┬──────┐                             └─────────► sB 64×64           bv[4] = sB[k][tx·4+0..3]
 │blk   │blk   │  block = 64×64 of C      __syncthreads()   (all written)     reg[4][4] += av ⊗ bv
 │(0,0) │(1,0) │  256 threads 16×16       for k in 0..63:  outer products     8 shared loads → 16 FMAs
 ├──────┼──────┤  thread = 4×4 of C       __syncthreads()   (all read)
 │      │      │  512×512 blocks          next tile (512 of them)           (4) the ladder
 └──────┴──────┘                                                              0.50 → 5.44 → 8.64 → 19.4 → 25.7 | 53.4
```

Then write the three numbers that explain the ladder: **33 → 2 lines** (coalescing), **2N → 2N/T loads** (tiling), **2 → 2/R shared reads per FMA** (register blocking).

### 9.5 Likely follow-ups (with the section that answers each)

| follow-up | where |
|---|---|
| "Walk me through which thread computes which output." | §1.2, §6.3 trace |
| "Why is the B load in naive fine?" | §2.4 |
| "Why 10.9× and not 32×?" | §3.4 |
| "Why two barriers? Remove one, what happens?" | §4.5 |
| "N = 1000?" | §4.6 |
| "Why did tiling only give 1.6×?" | §4.8 |
| "Why is lower occupancy faster?" | §6.4 |
| "Why 48% and not 90%?" | §6.6 |
| "Is 51× fair?" | §2.4, §0.2 |
| "How did you time / check it?" | §7 |
| "Where are you on the roofline?" | §8.3 |
| "Why not tensor cores?" | §10.3 |

### 9.6 Claims to avoid unless your data supports them

| don't say | why | say instead |
|---|---|---|
| "51× faster" (alone) | it's against a deliberately uncoalesced kernel | "51× over the naive kernel, 4.7× over a coalesced naive kernel, 48% of cuBLAS" |
| "48% of cuBLAS" without "SGEMM / FP32" | cuBLAS with TF32 tensor cores is far faster | "48% of cuBLAS SGEMM, FP32 on CUDA cores" |
| "bit-exact with cuBLAS" | only up to N = 16384, on the first 4M outputs; 4.48e-5 at 32768 | "max relative error 0 up to 16K on the checked prefix, 4.5e-5 at 32K" |
| "register blocking gave 3×" | that's against the 1-to-1 tile | "1.33× over my best plain tiled kernel; 3× over the 1-to-1 tile" |
| "my kernel is compute-bound" | it's at 38% of peak | "past the HBM ridge; limited on-chip, by shared traffic and latency" |
| "I profiled it" / "Nsight showed…" | ncu was never run (needed root) | "by address arithmetic, checked against measured times" |
| "50% achieved occupancy" | it's theoretical, from register counts | "50% theoretical occupancy" |
| "averaged over 10 runs" (for all kernels) | n = 1 for most kernels at 32K | "10 runs where they fit in 30 s; otherwise a single multi-second run" |
| the CPU speedup (42,783×) | single-threaded ijk loop, extrapolated past 4096 | don't quote it |
| "my kernel is optimal / near cuBLAS" | it's at kernel-5 level of Boehm's worklog | "the standard first half of the optimisation path" |

---

## 10. Mock interview: 100 questions with model answers

**How to drill:** read the question, answer out loud in under a minute, *then* open the answer. Mark each strong / OK / weak in [[#My notes and fumbles]]. The topic banks in §1–§8 are the first layer; these go wider and combine concepts. Answers are written the way you would say them, in the first person.

### 10.1 Fundamentals (30)

**F1. What does `__global__` mean, and what happens when the host calls such a function?**
> [!success]- Answer
> It marks a kernel: device code that the host launches with `<<<grid, block>>>`. Every thread of the grid runs the same body with different `blockIdx`/`threadIdx`. The host call only enqueues the launch in a stream and returns immediately; the GPU runs it later. That asynchrony is why my harness times with CUDA events and synchronises before reading results.

**F2. Host memory vs device memory: how does your harness move data?**
> [!success]- Answer
> `malloc` gives host buffers `hA, hB, hC, hRef`; `cudaMalloc` gives device buffers `dA, dB, dC` in HBM; `cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice)` uploads once per N (`:429–430`), and `record` downloads `dC` into `hC` after each kernel to check it. Kernels only ever see device pointers. At N = 32768 each matrix is 4.29 GB.

**F3. What is SIMT?**
> [!success]- Answer
> Single Instruction, Multiple Threads: you write scalar code for one thread; the hardware groups 32 threads into a warp and issues each instruction for all 32 lanes at once. Unlike SIMD you don't write vectors explicitly, and lanes can diverge (at a cost). Coalescing is the memory side of SIMT: one load instruction, 32 addresses.

**F4. What is warp divergence, and where could it happen in your kernels?**
> [!success]- Answer
> When lanes of one warp take different branches, the warp executes both paths in turn with inactive lanes masked, so throughput drops. In my kernels the only branches are the edge guards (`row < N`, the load ternaries, the store `if`), and they diverge only in the last block row/column when N isn't a multiple of the tile. At my benchmark sizes they never diverge.

**F5. Why is a block limited to 1,024 threads?**
> [!success]- Answer
> It's a hardware limit on threads per block on current NVIDIA GPUs: a block must fit on one SM with all its registers and shared memory, and `__syncthreads()` must be able to track every thread. That limit is why `mm_tiled_1to1` stops at a 32 × 32 tile: with one output per thread, the tile can't be bigger than the block.

**F6. Describe the memory hierarchy, and which levels each of your kernels uses deliberately.**
> [!success]- Answer
> Registers (per thread, ~1 cycle) → shared memory / L1 (per SM, up to 228 KB, ~20–30 cycles) → L2 (~50 MB, whole chip) → HBM (141 GB, 4.8 TB/s, hundreds of cycles). Naive and coalesced use only registers for `acc` and rely on caches for reuse. The tiled kernels add shared memory for operand reuse. `mm_block2d` adds registers for operand reuse (`av`, `bv`) and for 16 accumulators. Each step moves reuse one rung up.

**F7. What lives in registers in `mm_block2d<64,4>`?**
> [!success]- Answer
> The 16 accumulators `reg[4][4]`, the 8 operands `av[4]` and `bv[4]` of the current k, and indices (`rowBase`, `colBase`, loop counters, addresses). That's 62 registers per thread by `cuobjdump`. It stays in registers only because R is a template constant and every loop over `i`, `j` is unrolled, so every array index is a compile-time constant.

**F8. What is local memory?**
> [!success]- Answer
> Per-thread memory that physically lives in device memory (cached in L1/L2). The compiler uses it for register spills and for arrays indexed with runtime values. It's much slower than registers. My matmul kernels avoid it; my FlashAttention kernel doesn't quite: `o_acc[4][4]` ends up in local memory (`STACK:64`, `LDL/STL` in the SASS) because D is a runtime argument (Day 6).

**F9. Shared memory vs L1 cache?**
> [!success]- Answer
> On recent NVIDIA GPUs they share the same on-chip SRAM, split by configuration. L1 is managed by hardware (it caches global loads automatically, and you can't control what stays). Shared memory is managed by the program: you decide what goes in, when, and who reads it, and you synchronise explicitly. Tiling uses shared memory because the reuse pattern is known exactly; the coalesced kernel's accidental reuse came from L1/L2.

**F10. What is the unit of a global memory transaction?**
> [!success]- Answer
> 32-byte sectors, grouped into 128-byte cache lines (32 floats). A warp load costs the number of distinct sectors/lines its 32 addresses touch. 32 consecutive floats = 4 sectors = 1 line; 32 floats N apart = 32 lines with 4 useful bytes each.

**F11. What is a broadcast?**
> [!success]- Answer
> When several lanes of a warp read the same address, the hardware fetches it once and delivers it to all of them. In global memory that's one line; in shared memory one bank access with no conflict. `mm_coalesced`'s A load (`row` identical across the warp) and `mm_tiled_1to1`'s `sA[ty][k]` read are broadcasts.

**F12. What is a shared-memory bank conflict?**
> [!success]- Answer
> Shared memory is split into 32 banks; 4-byte word `a` is in bank `a mod 32`. If two lanes of one warp instruction access *different* addresses in the *same* bank, the accesses are served one after another; a k-way conflict costs k passes. Same address = broadcast, no conflict. My 1-to-1 inner loop is conflict-free (`sB[k][tx]` → bank `tx`); `mm_block2d` has 2-way conflicts by hand count; FlashAttention had a 32-way conflict that a one-line padding fixed for 7×.

**F13. What does `fmaf` do and why is it 2 FLOPs?**
> [!success]- Answer
> Fused multiply-add: `a·b + c` in one instruction with a single rounding at the end. It performs a multiply and an add, so the convention (cuBLAS, papers, my `tflops()`) counts it as 2 FLOPs. It's also slightly more accurate than separate multiply and add.

**F14. How many FLOPs is an N × N matmul? Compute it for your largest N.**
> [!success]- Answer
> N² outputs × N terms × 2 FLOPs = 2N³. At N = 32768: 2 × 32768³ = 7.04 × 10¹³ FLOPs. My final kernel's 2.7366 s gives 7.04e13 / 2.7366 = 25.7 TFLOP/s; cuBLAS's 1.3185 s gives 53.4 (`tflops()` at `:347–350`).

**F15. Row-major vs column-major, and the index formula?**
> [!success]- Answer
> Row-major stores a row contiguously: element (i, j) at `i·N + j`; moving along a row is +1, down a column is +N. Column-major (Fortran, cuBLAS) stores columns contiguously: (i, j) at `i + j·N`. My kernels use row-major; I pass operands to cuBLAS swapped (B, A) so its column-major product comes out as my row-major A·B.

**F16. What is `dim3`?**
> [!success]- Answer
> A 3-component size type for grids and blocks; unspecified components default to 1. `dim3 b32(32, 32)` is a 32 × 32 block; `dim3 g32((N+31)/32, (N+31)/32)` a 2-D grid. The kernel reads them back as `blockDim` and `gridDim`, and positions as `blockIdx`, `threadIdx`.

**F17. What does `#pragma unroll` do in your kernels?**
> [!success]- Answer
> It asks the compiler to replicate a loop body for each iteration, removing the counter and branch. With compile-time trip counts (TILE, R, F) it fully unrolls, which removes loop overhead, exposes independent FMAs for scheduling, and is what allows `reg[R][R]`, `av[R]`, `bv[R]` to live in registers (every index becomes a constant).

**F18. Why are TILE and R template parameters?**
> [!success]- Answer
> So they're compile-time constants: they size the static `__shared__` arrays, let loops unroll fully, and keep small arrays in registers. Each instantiation (`<32>`, `<64,2>`, `<64,4>`) is a separate compiled kernel. Cost: the static shared-memory cap of 48 KB per block (so TILE = 128 with two tiles doesn't compile), and no runtime choice of tile.

**F19. What limits static shared memory, and how do you get more?**
> [!success]- Answer
> Statically declared `__shared__` arrays are capped at 48 KB per block. To use more (the H200's SM has up to 228 KB), you declare `extern __shared__`, pass the size as the third launch parameter, and opt in with `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`. My FlashAttention host code has exactly that opt-in (it never fires at D = 64), and a lab-quiz harness trap was that the grader passed no dynamic size, making `extern __shared__` zero bytes.

**F20. Compute the H200's FP32 peak.**
> [!success]- Answer
> 132 SMs × 128 FP32 lanes × 2 FLOPs per FMA × 1.98 GHz = 66.9 TFLOP/s. My harness computes the same thing from `cudaGetDeviceProperties` (SM count) and `cudaDeviceGetAttribute(cudaDevAttrClockRate)` (`:385–392`); the 128 lanes per SM is a constant in the code. It uses the boost clock, so the real sustained peak may be a bit lower.

**F21. What is the ridge point of the H200 for FP32 from HBM?**
> [!success]- Answer
> Peak ÷ bandwidth = 66.9 TFLOP/s ÷ 4.8 TB/s ≈ 13.9 FLOP/B. A kernel needs more than ~14 FLOPs per HBM byte to have any chance of being compute-bound. Tiled matmul with TILE = 64 has 16 FLOP/B, just past it; TILE = 32 has 8, below it.

**F22. What is latency hiding?**
> [!success]- Answer
> Keeping the execution units busy while some operations wait. A global load takes hundreds of cycles; instead of stalling, the SM switches to another ready warp (thread-level parallelism) or issues independent instructions from the same warp (instruction-level parallelism). Enough of either hides the latency; that's what occupancy and register blocking each provide.

**F23. What is instruction-level parallelism, and where does your final kernel get it?**
> [!success]- Answer
> Independent instructions within one thread that can be in flight at once. `mm_block2d<64,4>`'s 16 FMAs per k update 16 different accumulators, so none waits for another; the 8 operand loads are independent too. That's why it runs well at 50% occupancy: each warp brings 16 independent FMA chains.

**F24. What is a wave of blocks?**
> [!success]- Answer
> The set of blocks resident on the GPU at the same time: blocks per SM (from occupancy) × 132 SMs. A grid larger than one wave runs in several. The last, partial wave can leave SMs idle (the tail effect), which matters for small grids. For `mm_block2d<64,4>`: 4 blocks per SM × 132 = 528 blocks per wave; at N = 32768 the grid has 262,144 blocks, ~500 waves, so the tail is negligible. At N = 1024 it has 256 blocks: less than one wave.

**F25. What does `cudaDeviceSynchronize()` do, and when do you need it?**
> [!success]- Answer
> The host blocks until all previously issued GPU work in all streams has finished. You need it (or an event/stream sync) before using results on the host, before timing with host clocks, and after warm-up in my `time_kernel` (`:319`) so warm-up work doesn't leak into the timed region. `cudaMemcpy` (non-async) also synchronises implicitly.

**F26. How are CUDA errors reported? What does `CUDA_CHECK` do?**
> [!success]- Answer
> API calls return a `cudaError_t`; kernel launches return nothing, so launch errors are retrieved with `cudaGetLastError()` and execution errors surface at the next synchronising call. `CUDA_CHECK(call)` is a macro in my file that prints file, line and the error string and exits on failure. I wrap every API call but don't call `cudaGetLastError()` after launches: a known gap.

**F27. What is cuBLAS SGEMM, and what are "leading dimensions"?**
> [!success]- Answer
> NVIDIA's single-precision general matrix multiply: C = α·op(A)·op(B) + β·C in column-major storage. The leading dimension (`lda`, `ldb`, `ldc`) is the stride between consecutive columns in memory, which lets you pass sub-matrices. For my square contiguous matrices all are N. I call it with α = 1, β = 0 and operands swapped for the row-major trick.

**F28. Tensor cores vs CUDA cores?**
> [!success]- Answer
> CUDA cores do scalar FP32 FMAs (128 per SM per clock here). Tensor cores do small matrix multiply-accumulates per instruction (for example 16 × 8 × 16 tiles) in lower precisions (FP16, BF16, TF32, FP8), at many times the throughput: the H200's dense TF32 tensor peak is about 495 TFLOP/s vs 67 FP32. My whole project is FP32 on CUDA cores, and cuBLAS ran in default math mode (no TF32), so 48% is SGEMM vs SGEMM.

**F29. Compute-bound vs memory-bound?**
> [!success]- Answer
> Compute-bound: the arithmetic units are the limit; faster memory wouldn't help. Memory-bound: data movement from some memory level is the limit; faster ALUs wouldn't help. You decide by comparing intensity with the ridge at each level and measured speed with each roof. My naive kernel is memory-bound (HBM); my 1-to-1 tiled kernel is bound by shared-memory bandwidth; my final kernel is past the HBM ridge but limited on-chip.

**F30. Theoretical vs achieved occupancy?**
> [!success]- Answer
> Theoretical occupancy is computed from resource use (registers, shared memory, threads per block) and the SM's limits: the maximum warps that *could* be resident. Achieved occupancy is what the profiler measures over the run, lower if blocks finish unevenly or the grid is small. My 50% / 100% figures are theoretical, from `cuobjdump` register counts and the occupancy rules; I never measured achieved occupancy (no ncu).

### 10.2 Implementation and optimisation (30)

**I1. Walk me through your kernel chain in one minute.**
> [!success]- Answer
> Naive, lanes on rows, 0.50 TFLOP/s. Coalesced, lanes on columns, 5.44 (10.9×). 32 × 32 shared tiles, 8.64 (1.59×), stuck on shared bandwidth. 64-tile with 2 × 2 outputs per thread and a coalesced loader, 19.36 (2.24×), with a contiguous-loader twin at 18.57 as a controlled experiment. Register-blocked 4 × 4 per thread, 25.71 (1.33×). cuBLAS 53.37. So 51.3× over naive, 48.2% of cuBLAS.

**I2. Which output does thread (tx = 3, ty = 5) of block (bx = 2, by = 1) compute in each kernel?**
> [!success]- Answer
> - **Naive** (row from x): row = bx·32 + tx = 67, col = by·32 + ty = 37 → C[67][37].
> - **Coalesced** and **1-to-1 `<32>`** (col from x): col = 67, row = 37 → C[37][67].
> - **Strided `<64,32>`**: row0 = by·64 = 64, col0 = bx·64 = 128; cells (ty + 32s, tx + 32u) = (5, 3), (5, 35), (37, 3), (37, 35) → C[69][131], C[69][163], C[101][131], C[101][163].
> - **Contiguous `<64,32>`**: cells (ty·2 + s, tx·2 + u) = rows 10–11, cols 6–7 → C[74…75][134…135].
> - **`block2d<64,4>`**: rowBase = 1·64 + 5·4 = 84, colBase = 2·64 + 3·4 = 140 → C[84…87][140…143].
>
> (The study guide's naive example uses thread (5, 3) of the same block: C[69][35].)

**I3. Global loads per output for each kernel at N = 32768?**
> [!success]- Answer
> Naive and coalesced: 2N = 65,536 floats (as issued; caches cut real HBM traffic). 1-to-1, TILE = 32: 2N/32 = 2,048. TILE = 64 kernels (strided, contiguous, block2d): 2N/64 = 1,024. The 64-tile kernels all have the same global traffic per output; they differ in how it's issued and in shared/register reuse.

**I4. Shared reads per FMA for each tiled kernel?**
> [!success]- Answer
> As written in the source: 1-to-1: 2 (one `sA`, one `sB` per FMA). Strided/contiguous F = 2: (F + F²)/F² = 1.5, since A is held in a register and B named again per row. block2d: 2R/R² = 2/R: 1 at R = 2, 0.5 at R = 4. The compiler may do better than the source (vectorised `LDS.128`, reused B values); the static SASS counts in the prep pack are 0.8, 1.6, 2.7, 2.7 and 8 FFMAs per shared-load instruction.

**I5. Why exactly does the 1-to-1 kernel need two barriers per tile? Could one suffice?**
> [!success]- Answer
> Two hazards: reading a cell before its owner has written it (barrier 1, `:93`) and overwriting the tile before everyone has finished reading (barrier 2, `:98`). With a single buffer you need both. With **two** buffers (double buffering) you can drop to one barrier per tile: load tile t+1 into buffer B while computing on buffer A, then one barrier swaps roles. That's a standard optimisation I didn't implement. [FUTURE]

**I6. How do you make sure `reg[4][4]` stays in registers?**
> [!success]- Answer
> R is a template constant and every loop indexing `reg`, `av`, `bv` has `#pragma unroll` with a compile-time trip count, so every array index is a constant and the compiler can assign each element its own register. I checked the result with `cuobjdump -res-usage`: 62 registers and no local-memory stack for `block2d<64,4>`. If I indexed with a runtime variable, the array would go to local memory, as `o_acc` did in my FlashAttention kernel.

**I7. What did you learn from the SASS / cuobjdump?**
> [!success]- Answer
> Register counts (62 for `block2d<64,4>`, 40 for `<64,2>`, 32 for strided), which give the occupancy numbers; that the compiler vectorised shared loads to `LDS.128` in the register-blocked kernel (the 4 adjacent `bv` values become one 128-bit load); and, for FlashAttention, that `o_acc` lives in local memory. Those are static observations of the committed binary, not runtime measurements. [RECON]

**I8. Explain the strided loader with numbers.**
> [!success]- Answer
> TILE = 64, BLK = 32, F = 2. Thread (tx, ty) fills cells r = ty + 32s, c = tx + 32u for s, u ∈ {0, 1}. For a fixed slot (s, u), the 32 lanes of a warp (tx = 0…31, one ty) fill columns u·32 + 0…31 of one tile row: 32 adjacent floats in global memory, one 128-byte line. Thread (3, 5) fills cells (5, 3), (5, 35), (37, 3), (37, 35).

**I9. Why does `block2d` use the contiguous-style mapping if strided loads better?**
> [!success]- Answer
> It was the Lab Quiz 2 Q3 kernel, re-templated (prep pack decision 7), and its compute mapping (thread owns rows `ty·R + i`, cols `tx·R + j`) was also used for loading. I kept it rather than re-deriving a separate loader. The cost is stride-R global loads and ~4-way bank-conflicted shared stores. The strided/contiguous pair suggests it's a few percent, not the main gap to cuBLAS. [RECON]

**I10. How would you decouple the load mapping from the compute mapping?**
> [!success]- Answer
> Treat the 256 threads as a flat array for loading: thread `tid = ty·16 + tx` loads cells `tid + 256·m` of the 64 × 64 tile (m = 0…15), so consecutive threads load consecutive columns: fully coalesced, and with `float4` four columns per instruction. After the barrier, compute with the patch mapping as now. The tile layout in shared memory is the same, so the compute code doesn't change. [FUTURE]

**I11. How does the ternary guard interact with `__syncthreads()`?**
> [!success]- Answer
> The ternary lets out-of-range threads write 0.0f instead of reading out of bounds, so they stay in the loop and reach both barriers. All threads take the same control path through the barriers; only the data differs. If I'd written `if (aCol >= N) return;` instead, those threads would skip the barrier and the block would be in undefined territory.

**I12. Explain the cuBLAS row-major trick in one sentence, and why you need it.**
> [!success]- Answer
> A row-major buffer read as column-major is the transpose, so asking cuBLAS for B·A computes Bᵀ·Aᵀ = (A·B)ᵀ in column-major, which is exactly A·B when read back row-major. It avoids transposing 4 GB matrices or changing my kernels' layout.

**I13. Why does cuBLAS run first?**
> [!success]- Answer
> It is both the correctness reference and the speed ceiling: its output `hRef` is what every later kernel is compared with, and its time gives the "% of cuBLAS". A CPU reference would take about 34 hours at N = 32768, and comparing against a strawman wouldn't tell me how good the kernel is.

**I14. What does `max(|ref|, 1)` do in the error formula?**
> [!success]- Answer
> It makes the error relative for large values and absolute for small ones. Outputs near zero would otherwise make a tiny absolute difference look like a huge relative error. With inputs in [−0.5, 0.5] and K = 32768 the outputs are sums of many products, so both regimes occur.

**I15. How did you derive 51.3× and 48.2%?**
> [!success]- Answer
> From the CSV at N = 32768: naive 140,486 ms ÷ block2d R = 4 2,736.6 ms = 51.3×; block2d 25.71 TFLOP/s ÷ cuBLAS 53.37 = 48.2% (equivalently 1,318.5 / 2,736.6 ms). TFLOP/s = 2N³ / time. Both kernels in the 48.2% have n = 10; naive has n = 1.

**I16. What is the 30-second budget for, and what did it cost?**
> [!success]- Answer
> It stops slow kernels from running ten times: one timed run first, and nine more only if ten fit in 30 s. It saved hours (naive would have taken 23 minutes per size). Cost: at N = 32768 every kernel except cuBLAS and block2d R = 4 has n = 1, and so does naive at 16384. I state that whenever I quote those numbers.

**I17. Why is the 128 in the peak formula a constant and not queried?**
> [!success]- Answer
> CUDA doesn't expose "FP32 lanes per SM" as a device attribute; it's an architecture fact (128 on sm_80/86/90 according to the comment at `:390`). The SM count and clock are queried. So the peak is correct for this GPU but would need the constant changed on an architecture with a different count.

**I18. Why is a 64-wide tile the sweet spot in your design?**
> [!success]- Answer
> It puts HBM intensity at 16 FLOP/B, just past the ridge, so HBM stops being the limit; two 64 × 64 float tiles are 32 KB, under the 48 KB static cap; and 64 = 32 × 2 or 16 × 4 maps cleanly onto 32 × 32 or 16 × 16 blocks. 128 would exceed the static limit with my template design. [RECON]

**I19. How many blocks per SM and what occupancy for each tiled kernel?**
> [!success]- Answer
> From registers per thread × threads per block against 65,536 registers and 2,048 threads: strided (32 regs × 1,024) → 2 blocks, 100%; `block2d<64,2>` (40 × 1,024 = 40,960) → 1 block, 50%; `block2d<64,4>` (62 → 64 × 256 = 16,384) → 4 blocks, 50%. Shared memory (32 KB per block) would allow ~7 in each case, so registers bind. All theoretical.

**I20. Why is the 1-to-1 inner loop conflict-free?**
> [!success]- Answer
> In one warp (one `ty`, tx = 0…31): `sA[ty][k]` is the same address for all lanes, a broadcast; `sB[k][tx]` is word `32k + tx`, bank `tx`: 32 lanes, 32 banks. One pass each. The kernel is slow because of the number of shared reads (2 per FMA), not conflicts.

**I21. Show the 2-way conflict in `mm_block2d<64,4>`'s `av` reads.**
> [!success]- Answer
> A 16 × 16 block puts two `ty` values in one warp. `av[i] = sA[ty·4 + i][k]` is word `(ty·4 + i)·64 + k`; for ty and ty + 1 the rows differ by 4, so the words differ by 256, which is 0 mod 32: same bank, different addresses → 2-way. Lanes with the same ty read the same address (broadcast). [RECON, hand count]

**I22. How would you remove that conflict?**
> [!success]- Answer
> Pad the row stride of `sA` from 64 to 65 floats (the same trick as my FlashAttention fix): rows 4 apart are then 260 words apart, 260 mod 32 = 4, different banks. Or store `sA` transposed (k-major), so `av` becomes 4 adjacent words read with one `LDS.128`, which is what cuBLAS-style kernels do. Padding breaks 16-byte alignment of rows, so it conflicts with vector loads; the transpose doesn't. [FUTURE]

**I23. How would you vectorise the global loads, and what does that require?**
> [!success]- Answer
> Load `float4` (16 bytes) per thread per instruction instead of `float`: 4× fewer load instructions and fully used 128-byte lines when consecutive threads load consecutive `float4`s. It requires 16-byte-aligned addresses (row starts aligned, which holds when N is a multiple of 4 and the base from `cudaMalloc` is aligned), and a loader mapping where each thread's 4 elements are contiguous along the row while lanes are adjacent `float4`s. [FUTURE]

**I24. Sketch double buffering for your kernel.**
> [!success]- Answer
> Allocate two copies of `sA`/`sB`. Before the loop, load tile 0 into buffer 0. In iteration t: issue the loads of tile t+1 into buffer (t+1) mod 2 (ideally with `cp.async`, which copies global → shared without going through registers), compute on buffer t mod 2, then wait for the async copies and one `__syncthreads()`. Loads and FMAs now overlap, and the per-tile exposed latency disappears. Cost: 2× shared memory (64 KB at TILE = 64, which needs dynamic shared memory). [FUTURE]

**I25. Why does `(N + TILE − 1) / TILE` appear in both the launch and the kernel?**
> [!success]- Answer
> It's integer ceil(N / TILE). In the launch it makes enough blocks to cover a ragged last row/column of C; in the kernel's t loop it visits the ragged last tile along K. Without it, N = 1000 with TILE = 32 would compute only 992 of the 1,000 terms and leave a strip of C unwritten.

**I26. How would your kernel change for non-square matrices, M × K times K × N?**
> [!success]- Answer
> Three sizes instead of one: grid = (ceil(N/TILE), ceil(M/TILE)), the t loop runs ceil(K/TILE) times, and the guards compare rows with M, A-columns/B-rows with K, and columns with N; the index arithmetic uses the right leading dimension (K for A, N for B and C). The tiled structure is identical. Very skinny shapes (M = 32, as in my Day 6 MLP) call for a different decomposition, which is why `mm_thin` exists.

**I27. What would TF32 tensor cores change?**
> [!success]- Answer
> The inner loop would use warp-level matrix instructions (`wmma` or `mma.sync`) on fragments loaded from shared memory, each computing a small tile product in one instruction. Throughput per SM is several times higher, so the bottleneck shifts back to feeding data: shared-memory layout, async copies and swizzling become critical. Accuracy also changes: TF32 has a 10-bit mantissa, so I'd need a tolerance-based check instead of my near-zero error. Not done. [FUTURE]

**I28. What is the purpose of the toy N = 4 example in your notes?**
> [!success]- Answer
> Every kernel computes C[0][3] = 2 from the same four products (1·1, 2·0, 0·2, 1·1) in the same order; only where the operands are read from changes (registers, global, shared). Tracing it by hand for each kernel proves I understand the index arithmetic, and it's what I'd draw if asked to trace a thread.

**I29. Your contiguous loader reads columns 0, 2, 4…: how many lines per instruction exactly?**
> [!success]- Answer
> Lanes 0…31 read columns 0, 2, …, 62 of one tile row (slot u = 0): a span of 64 floats = 256 bytes = 2 lines, each half used. The u = 1 instruction reads columns 1, 3, …, 63: the same two lines, usually hitting L1. So per pair of instructions it's 4 line requests vs the strided loader's 2.

**I30. If you had one more day on this kernel, what would you do?**
> [!success]- Answer
> Get a profiler working first (or on a machine where I have counters), because ranking the remaining bottlenecks by hand is guesswork. Then: decouple the loader and vectorise it (`float4`, coalesced), and double-buffer with `cp.async`. Those two attack the two things I'm most sure about: exposed load latency and wasteful global loads. Then transpose `sA` for conflict-free vector reads.

### 10.3 Difficult follow-ups (20)

**D1. "51× over a kernel you made slow on purpose isn't an achievement. What's your real result?"**
> [!success]- Answer
> Agreed that the ratio alone isn't the point. The real result is 25.7 TFLOP/s FP32, 48% of cuBLAS SGEMM and 38% of peak, and 4.7× over a properly coalesced naive kernel. The 51× is the assignment's metric; I report it with what it's measured against. What I'd defend is the explanation: each step's gain matches a traffic count, and I can say why I stop at 48%.

**D2. "cuBLAS is at 80% of peak. Why not 100%, and does that make your peak number wrong?"**
> [!success]- Answer
> No real kernel issues an FMA every cycle on every lane: there are loads, address arithmetic and synchronisation instructions competing for issue slots, and the boost clock (1.98 GHz) in my peak formula may not be sustained under full load (I couldn't lock clocks). So 66.9 is an upper bound and both percentages are slightly understated. cuBLAS at 80% is typical for well-tuned SGEMM. [THEORY]

**D3. "Your shared-memory roof says 8.4 and you measured 8.64. Isn't that a convenient coincidence?"**
> [!success]- Answer
> Partly, and I'd say so. The model (32 words per clock per SM, 2 reads per FMA) is a source-level count; the compiler vectorises some of those reads, so the instruction-level picture is different. The point it supports is qualitative: after tiling, the kernel is limited by shared-memory traffic, not HBM (its HBM roof is 38 TFLOP/s). And the prediction from the same model, that cutting reads per FMA by 4× should help a lot, was confirmed: 3.0×.

**D4. "Single runs, no error bars. Why should I believe any of these differences?"**
> [!success]- Answer
> For the large ratios, because they're far beyond plausible run-to-run noise (a few percent) and they're consistent across N: 51.3× is 50–51× from N = 2048 up, 48% of cuBLAS holds from 2048 to 32768. For small differences, you shouldn't: the 4% between the strided and contiguous loaders is a single run each, so I call it "small, probably real" and never build an argument on it alone.

**D5. "Maybe the compiler already register-blocked your strided kernel. Then what did block2d add?"**
> [!success]- Answer
> It may have done part of it: in the strided kernel the source re-reads `sB[k][…]` per row, and the compiler is free to reuse those values since nothing writes shared memory in between; the static SASS count (1.6 FFMA per shared load vs 1.33 as written) suggests some reuse happened. block2d makes the reuse explicit for both operands and raises the block from 2 × 2 to 4 × 4, reaching 8 FFMAs per shared-load instruction in the SASS. The measured gain over strided is 1.33×. I'd quote that, not the 3.0× over 1-to-1. [RECON]

**D6. "If occupancy isn't the goal, why not run one warp per SM?"**
> [!success]- Answer
> Because latency hiding still has to come from somewhere. One warp per SM can't hide hundreds of cycles of global-load latency even with lots of ILP, and the barrier phases would leave the SM idle. The goal is enough parallelism (warps × independent instructions per warp) to cover latency. My kernel gets it at 50% occupancy with 16 independent FMA chains per thread; at much lower occupancy it would start losing.

**D7. "What happens with a 33 × 33 matrix? Or 32768 × 32?"**
> [!success]- Answer
> 33 × 33 with `block2d<64,4>`: one block, 256 threads, most of the tile is zero padding; correct, but under 1% of the GPU is busy and launch latency dominates; cuBLAS would pick a small-matrix kernel. 32768 × 32 (tall-skinny) isn't supported by my square-only kernels; generalising to M, N, K is mechanical (§10.2 I26), but square tiles waste most of their rows when one dimension is 32, which is exactly why my Day 6 MLP uses a different "thin" kernel.

**D8. "What happens if the number of threads exceeds the available resources?"**
> [!success]- Answer
> Two different cases. Per block: if a block needs more threads (> 1,024), registers or shared memory than an SM can hold, the launch fails (or doesn't compile, for static shared memory over 48 KB). Per grid: more threads than the GPU can hold at once is normal; blocks wait and run in waves as earlier blocks finish. The software can't oversubscribe an SM; the block scheduler only places a block where it fits.

**D9. "How does register blocking interact with memory coalescing?"**
> [!success]- Answer
> They pull the thread mapping in opposite directions. Register blocking wants each thread to own a contiguous R × R patch, so `tx·R + j` indexes its columns; coalescing wants consecutive lanes on consecutive addresses, i.e. stride 1 across `tx`. If one mapping drives both, the loads become stride-R, which is what my `mm_block2d` does (inherited from the quiz kernel). The resolution is to use different mappings for the load phase (flat, lane-adjacent, vectorised) and the compute phase (patches), since shared memory decouples them.

**D10. "Your final kernel uses 32 KB of shared memory and 64 registers per thread. Which one limits occupancy, and what if you doubled the tile?"**
> [!success]- Answer
> Registers: 64 × 256 = 16,384 per block → 4 blocks, while shared memory would allow ~7. Doubling to TILE = 128 with the same R = 4 needs 32 × 32 = 1,024 threads (one block per SM by registers: 64 × 1,024 = 65,536) and 128 KB of double tiles, which doesn't compile as static shared memory and would allow only one block anyway. So it's both a compile-time and an occupancy problem; the standard fix is a 128 × 128 output tile but a short k-slice in shared memory. [THEORY]

**D11. "Prove your 1-to-1 kernel isn't HBM-bound."**
> [!success]- Answer
> Count its HBM traffic upper bound: N² outputs × 2N/32 floats × 4 B = 32768² × 2,048 × 4 B ≈ 8.8 TB per run, over 8.148 s ≈ 1.08 TB/s, under a quarter of the 4.8 TB/s HBM bandwidth even with zero L2 reuse. Its HBM roof (8 FLOP/B × 4.8 TB/s) is 38 TFLOP/s against 8.64 measured. Whatever limits it, it isn't HBM; the shared-memory count says what it is. [THEORY]

**D12. "Your error at 32768 is 4.5e-5. Is your kernel wrong?"**
> [!success]- Answer
> All six custom kernels show exactly the same 4.48e-5 against cuBLAS, so they agree with each other; with 32,768-term FP32 sums, a different summation order on cuBLAS's side (it likely picks a different kernel, possibly split-K, at that size) easily produces relative differences of that order. That's a hypothesis, not documented. To settle it I'd compare against an FP64 reference on a sample of outputs; I didn't.

**D13. "What if two blocks had to add into the same output?"**
> [!success]- Answer
> That's split-K: divide K among blocks to get more parallelism for small M × N, then combine partial sums with `atomicAdd` or a second reduction kernel. Atomics on floats make results order-dependent (non-deterministic last bits); a separate reduction keeps them deterministic. My kernels never do this: each output has exactly one owner, so no atomics and deterministic results. [THEORY]

**D14. "Would your kernel be 48% of cuBLAS on an A100 or an RTX card?"**
> [!success]- Answer
> I can't claim that; I only measured an H200. The mechanisms carry over (same warp size, banks, coalescing rules), but the ratios depend on the balance of SM count, shared-memory bandwidth, cache sizes and clocks, and cuBLAS picks per-GPU kernels. Boehm's similar kernel on an RTX A6000 sat around two-thirds of cuBLAS at 4092², so the share can move by tens of percent between GPUs.

**D15. "Why write matmul at all when cuBLAS exists?" (hostile)**
> [!success]- Answer
> You don't in production: you call cuBLAS or CUTLASS. You write it to understand what limits GPU kernels, because the same three ideas (coalescing, on-chip reuse, register blocking) decide the speed of every custom kernel you can't get from a library: fused attention, custom activations, quantised GEMMs. My FlashAttention kernel used exactly these lessons (a bank conflict worth 7×, register blocking over query rows worth 2.1×).

**D16. "Is your kernel deterministic?"**
> [!success]- Answer
> Yes. Each output is accumulated by one thread in a fixed k order with FMAs, no atomics, no inter-block reductions, so the same inputs give the same bits every run. cuBLAS may not be bit-identical across versions or sizes (different algorithms), which is consistent with the 4.48e-5 at 32K.

**D17. "Shared memory is 228 KB per SM, but you say the limit is 48 KB. Which is it?"**
> [!success]- Answer
> Both, at different levels: 228 KB is the SM's capacity (shared by all resident blocks, configurable against L1); 48 KB is the per-block limit for **statically declared** `__shared__` arrays. A single block can use more (up to ~227 KB on Hopper) only through dynamic shared memory and an explicit opt-in attribute. My matmul tiles are static, so 48 KB is the binding limit for my template design.

**D18. "If you started over, what would you change?"**
> [!success]- Answer
> Three things. Make the measurement stronger from day one: clear `dC` and check `cudaGetLastError` after every launch, check the whole matrix (sampled) or against FP64, and repeat runs for error bars. Write the final kernel's loader myself instead of inheriting the quiz kernel's stride-R mapping. And plan for dynamic shared memory so the tile size isn't capped by the 48 KB static limit.

**D19. "You say L1 softened the naive penalty. Prove it."**
> [!success]- Answer
> I can't from my data; it's the most plausible explanation of 10.9× vs 16.5×. The experiment: compile the naive kernel with global loads bypassing L1 (`-Xptxas -dlcm=cg`, cache only in L2) and re-time it. If the naive time rises toward 16.5× of the coalesced one, the L1 explanation holds. With profiler access, the L1 hit rate of the A load answers it directly. [FUTURE]

**D20. "Why is your speedup over naive smaller at N = 1024 (45×) than at 32768 (51×)?"**
> [!success]- Answer
> At N = 1024 the grids are small: `block2d<64,4>` launches 256 blocks, under one wave on 132 SMs at 4 blocks per SM, so part of the GPU idles and fixed per-block costs weigh more; the naive kernel's caches also do relatively better on a 4 MB matrix that fits in L2. From N = 2048 up both kernels fill the GPU and the ratio stabilises at 50–51×. [MEAS] [THEORY]

### 10.4 Debugging and code analysis (10)

**B1. In `mm_tiled_1to1`, someone swaps line 84 and 85 to `row = blockIdx.y*TILE + tx; col = blockIdx.x*TILE + ty;`. What happens?**
> [!success]- Answer
> It becomes **wrong**. Lines 91–92 use `row` and `col` to decide what goes into `sA[ty][tx]`. After the swap, `sA[ty][tx] = A[by·T + tx][t·T + tx]`: the row index now depends on `tx`, not `ty`. In the inner loop `sA[ty][k]` is then `A[by·T + k][t·T + k]`, a diagonal of A, not row `row`. No crash, no warning, just wrong numbers. The lesson: the load mapping and the compute mapping must agree on what `sA[r][c]` means. A swap made consistently everywhere would be correct but would put lanes on rows of C and lose coalescing on B and C, back to the naive problem.

**B2. Remove the second `__syncthreads()`. What are the symptoms, and how would you catch it?**
> [!success]- Answer
> Usually small, intermittent errors: some outputs off, different from run to run or size to size, because a fast warp occasionally overwrites `sA`/`sB` with tile t+1 while a slow warp still reads tile t. It might even pass a test by luck. `compute-sanitizer --tool racecheck` reports shared-memory read-after-write / write-after-read hazards directly; comparing against cuBLAS over the whole matrix several times would also expose it.

**B3. Replace the load ternaries with `if (row >= N || aCol >= N) return;`. What happens?**
> [!success]- Answer
> For N a multiple of TILE, nothing changes (no thread returns). For ragged N, some threads leave before `__syncthreads()`: undefined behaviour, in practice a hang or garbage, and the threads that did stay read stale cells that the returned threads never wrote. The fix is the original design: keep everyone in the loop, load zeros, guard only the store.

**B4. Change the tile loop bound to `t < N / TILE`. Which inputs break?**
> [!success]- Answer
> Any N not divisible by TILE: the last N mod TILE terms of every dot product are dropped, silently. N = 1000, TILE = 32: 31 full tiles cover k < 992, so 8 terms per output are lost. All benchmark sizes are multiples of 64, so the CSV would look perfect: exactly why edge cases need their own test.

**B5. Remove `(size_t)` from the naive kernel and run N = 50,000. What happens?**
> [!success]- Answer
> `row * N + k` is computed in 32-bit `int`; at row ≥ 42,950 (42,950 × 50,000 > 2³¹) it overflows to a negative or wrapped index, so the kernel reads wrong addresses or faults with an illegal address. At N ≤ 46,340 every index fits (46,340² < 2³¹), which is why the bug wouldn't show in my sweep.

**B6. How could a kernel that never ran still pass your correctness check?**
> [!success]- Answer
> If its launch failed (bad configuration, too many resources), nothing is written to `dC`, which still holds the previous kernel's correct output; `record` copies it back and `max_rel_err` reports a tiny error. My harness neither clears `dC` between kernels nor calls `cudaGetLastError()` after launches. The distinct timings show every kernel ran, but the fix is two lines.

**B7. How would you detect that `reg[R][R]` spilled to local memory?**
> [!success]- Answer
> Compile with `-Xptxas -v` (prints registers, spill stores/loads and stack frame per kernel), or inspect the binary with `cuobjdump -res-usage` (shows `REG`, `STACK`) and the SASS for `LDL`/`STL` instructions. That's how I know `block2d<64,4>` uses 62 registers with no stack, and that my FlashAttention `o_acc` has a 64-byte stack and `LDL/STL`.

**B8. In the 1-to-1 inner loop, a typo makes it read `sB[tx][k]` instead of `sB[k][tx]`. Besides the wrong answer, what happens to performance?**
> [!success]- Answer
> Lane tx reads word `tx·32 + k`: bank `(32·tx + k) mod 32 = k` for every lane: 32 different addresses in one bank, a 32-way conflict on every FMA. It's the same pattern as my FlashAttention K-tile bug (stride 64), and padding the row to 33 would fix the bank pattern (the math would still be wrong).

**B9. What does `compute-sanitizer` check, and which tool would you use for which bug?**
> [!success]- Answer
> `memcheck` (default): out-of-bounds and misaligned global/shared accesses, invalid frees. `racecheck`: shared-memory data races (missing barriers). `synccheck`: illegal barrier use, e.g. `__syncthreads` in divergent code. `initcheck`: reads of uninitialised global memory. For B2 I'd use racecheck, B3 synccheck, B5 memcheck. My lab-quiz binaries were all clean under memcheck.

**B10. Someone instantiates `mm_tiled_strided<48,32>`. What happens?**
> [!success]- Answer
> `F = TILE / BLK = 48 / 32 = 1` by integer division, so each thread loads one cell per tile and owns one output, but the tile is 48 × 48: only the 32 × 32 corner of each shared tile is loaded and computed, and the shared arrays' other cells hold garbage that the inner loop (k < 48) reads. Wrong results, no error. There's no `static_assert(TILE % BLK == 0)` in the code; I'd add one. [RECON]

### 10.5 Performance analysis (10)

**P1. How do you measure kernel performance correctly? Answer with your harness as the example.**
> [!success]- Answer
> Time on the GPU clock with CUDA events around only the kernel; warm up first and synchronise before the timed region; repeat and report n (mine: 10 when ten fit in 30 s, else 1); check each timed kernel's output; convert to a throughput metric (TFLOP/s = 2N³/time) so sizes compare; compare with a strong baseline (cuBLAS); state precision and configuration. What I'd add: locked clocks, repeated runs with spread, and profiler counters.

**P2. Latency vs throughput, in your kernels?**
> [!success]- Answer
> Latency is how long one access takes (HBM hundreds of cycles, shared ~20–30, register ~1); throughput is how many FLOPs or bytes per second the whole GPU completes. My optimisations don't make any single load faster; they cut how many slow accesses each FMA needs and keep enough independent work in flight (warps, 16 accumulators per thread) that latency is hidden. The metric I optimise is throughput: TFLOP/s.

**P3. Compute the naive kernel's global-load efficiency per warp step.**
> [!success]- Answer
> Per step k, a warp does 32 FMAs (one per lane). Naive: it needs 32 different A floats plus 1 B float (broadcast) = 33 useful floats = 132 bytes, but requests 32 lines for A + 1 for B = 33 × 128 B = 4,224 bytes: about **3%** of the fetched bytes are used, before caching. Coalesced: it needs 1 A float (broadcast) + 32 B floats = the same 132 useful bytes, and requests 2 lines = 256 bytes: about **52%**, with the B line fully used.

**P4. How much data per second does the coalesced kernel *request*, and what does that tell you?**
> [!success]- Answer
> It issues 2N loads per output: 32768² × 65,536 floats × 4 B ≈ 2.8 × 10¹⁴ bytes in 12.9 s ≈ 22 TB/s, about 4.5× the HBM bandwidth. So most of those requests must be served by L1/L2, not HBM. That's the quantitative version of "caches give reuse the roofline count ignores", and why it beats its naive 1.2 TFLOP/s roof. [THEORY, from MEAS time]

**P5. How do you identify whether a kernel is compute-bound or memory-bound without a profiler?**
> [!success]- Answer
> Compute intensity at each memory level and compare with the ridge; compute the bandwidth the kernel *must* be using (bytes ÷ time) and compare with the hardware's; then perturb: change only memory behaviour (my strided vs contiguous pair) or only arithmetic and see what moves. My 1-to-1 kernel: HBM use ≤ 1.1 TB/s (not HBM-bound), shared-memory model says 8.4 roof vs 8.64 measured (shared-bound).

**P6. Why do you report TFLOP/s rather than milliseconds?**
> [!success]- Answer
> TFLOP/s normalises for problem size (2N³ grows 8× per doubling of N), so kernels and sizes compare directly, and it can be put against the hardware peak (66.9) and the roofline. Milliseconds are what I measured; TFLOP/s is derived, `2N³ / (ms·10⁻³) / 10¹²`.

**P7. Explain how tile size, occupancy, register usage and shared memory interact, with your kernels.**
> [!success]- Answer
> Bigger tiles cut HBM traffic (2N/TILE) but cost shared memory (2·TILE²·4 B per block); more outputs per thread cut shared traffic (2/R reads per FMA) but cost registers; both reduce how many blocks fit per SM, i.e. occupancy, which reduces latency hiding. My final kernel spends 32 KB and 64 registers per thread, lands at 50% occupancy, and is still the fastest, because the reuse removed the binding bottleneck (shared bandwidth) while ILP covered the lost warps.

**P8. Is 66.9 TFLOP/s achievable?**
> [!success]- Answer
> Not in practice. It assumes every lane issues an FMA every cycle at the boost clock, with no loads, address arithmetic or barriers. cuBLAS reaches 80% (53.4). And the clock under sustained load may be below 1.98 GHz (not locked; no root), so the true peak may be lower than my number, which would make my percentages slightly conservative.

**P9. What do your results across N tell you that the single headline doesn't?**
> [!success]- Answer
> Stability and regime. The 51× ratio is 45× at N = 1024 and 50–51× from 2048 up; the cuBLAS share is 61% at 1024 and 48–50% from 2048 up. So the headline isn't a one-size artefact, and small sizes are a different regime (under one wave of blocks, cache-resident matrices, cuBLAS small-matrix kernels). Reporting only N = 32768 would hide both facts.

**P10. Without a profiler, design an experiment to rank the bottlenecks of `mm_block2d<64,4>`.**
> [!success]- Answer
> Perturbation, one factor at a time, each a small code change: (1) loads only: skip the FMA loop (keep a dummy dependence) → the load/barrier floor; (2) compute only: run the FMA loop on registers without reloading tiles → the arithmetic floor; (3) swap in a strided, coalesced loader → the global-pattern cost; (4) pad `sA`'s row to 65 → the bank-conflict cost; (5) add double buffering → the exposed-latency cost. The gaps between these times rank the causes. Each is a measurement I didn't do; it's how I'd answer "where does the other 52% go?" [FUTURE]

---

## Sources

The study guide ([[Day 05 - CUDA I#Sources, and what they say]]) has the summaries. Main one: Simon Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance* (2022): your kernels follow his kernels 1–5; his later steps (vectorised loads, autotuning, warptiling) are your [FUTURE] list. NVIDIA *CUDA C++ Programming Guide* (programming model, memory hierarchy, shared-memory banks) and *Best Practices Guide* (coalescing). Williams, Waterman and Patterson, "Roofline" (CACM 2009).

## My notes and fumbles

- 
