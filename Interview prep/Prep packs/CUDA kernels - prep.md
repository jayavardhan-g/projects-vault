> Prep pack for **CUDA kernels**, written 2026-11-05 from `sysml/INTERVIEW_PREP.md`. The vault copy is the one to study and edit.

# Interview Prep — CS794: CUDA Kernels and Efficient ML Inference

Written 2026-11-05 from a full read of `CLAUDE.md`, `RESULTS.md`, `SETUP.md`, every
source file, every `results/*.csv`, the git log, and a static look at the committed
binaries (`cuobjdump`). Nothing was re-run. No GPU job was launched.

**Read this first.**

* Git has exactly **one commit** (`16870fb "kernel"`, 2026-08-18). It adds all 57 files
  at once, binaries included. So git holds **no development history**. All "history"
  below comes from `RESULTS.md` and code comments. When I say "documented", I mean one
  of those. When I say "reconstructed", I read it off the code myself.
* The folders were renamed after the runs. Docs say `pa2_matmul/`, `pa3_mlp/`,
  `pa4_flash/`, `pa1_kvcache/`, `cs794_labquiz{2,3}_files/`. The repo has `matmul/`,
  `mlp/`, `flashattention/`, `kvcache/`, `prob2/`, `prob3/`. **The build commands in
  `SETUP.md` and `tools/run_quiz.sh` fail as written.** The code is there; the paths are stale.
* Resume problems, in short: the "13.3% peak activation memory" label is wrong (13.3% is
  total training memory; activation memory fell 32%). The 1.73× is real but measured at
  one width only, and its stated mechanism ("stream parallelism") is not proven. Details
  are in section 5.

---

## 1. The pitch

### 60-second version

> This was a GPU systems course project on an NVIDIA H200. I wrote CUDA kernels for
> the core pieces of ML workloads and measured each one against vendor code.
>
> The main piece is matrix multiplication. I started from a naive kernel. Then I fixed
> memory coalescing, added shared-memory tiling, and added 2D register blocking. The
> best kernel runs 51× faster than the naive one. It reaches 48% of cuBLAS, at 25.7
> TFLOP/s in FP32.
>
> I also did three smaller pieces. First, CUDA streams for an MLP forward pass.
> Splitting the batch never helped. Splitting the output columns was 1.73× faster at
> the largest width. Second, gradient checkpointing. It cut activation memory by
> about a third and made the backward pass 26% slower. Third, a FlashAttention
> kernel. Fixing one shared-memory bank conflict there gave 7× on its own.
>
> What I'm proudest of is the measurement. I report the results that didn't work and
> explain why.

### 3-minute version

> **Setup.** It's one H200: 132 SMs, an FP32 peak of about 67 TFLOP/s on the CUDA
> cores, and 4.8 TB/s of HBM. Everything is FP32, with no tensor cores, so the
> comparison with cuBLAS SGEMM is fair. I timed kernels with CUDA events and left
> host-device copies out. Every kernel was checked against a reference.
>
> **Matmul.** The naive kernel maps neighbouring threads to neighbouring *rows*. So
> every warp load of A is 32 separate memory transactions. Swapping the thread
> mapping made the loads coalesced. That alone gave 10.9×. Next, I staged 32×32 tiles
> in shared memory. Each element loaded from DRAM then gets reused 32 times. Then I
> went to 64×64 tiles where each thread computes several outputs. The final kernel
> has each of 256 threads hold a 4×4 block of C in registers. Each shared-memory read
> now feeds several FMAs instead of one, so the kernel stops starving on shared-memory
> bandwidth. That got me to 25.7 TFLOP/s, against 53.4 for cuBLAS. The ratio holds at
> about 48% from N = 2048 to 32768.
>
> I know why I stop there. My kernel loads a tile, syncs, computes, and syncs again,
> so loads and math never overlap. My global loads are scalar and partly strided. And
> I only get 50% occupancy. cuBLAS-class kernels use bigger tiles, vector loads,
> double-buffered async copies, and warp-level tiling.
>
> **Streams.** The MLP has batch 32. Splitting 32 rows across 4 streams makes every
> stream read the whole weight matrix again. It also adds launches. So it never
> helped, and at small widths it was 4× slower. Splitting the output *columns* gives
> each stream its own quarter of the weights. At width 32768 that took 44 ms down to
> 25.7 ms. One caveat: the single-stream kernel already fills the GPU at that size.
> So I believe the win comes from memory locality, not from extra parallelism. I
> couldn't prove it, because the profiler needs root on that machine.
>
> **Checkpointing.** It's a 4-layer MLP. I kept two activations and recomputed the
> rest during the backward pass. The first version saved zero bytes, because I gave
> every recomputed tensor its own buffer. The fix was to see that the two recomputed
> segments are never alive at the same time, so one shared scratch buffer is enough.
> Activation memory fell from 400 to 272 MB. The backward pass got 26% slower. That
> matches the extra forward GEMMs almost exactly.
>
> **FlashAttention.** It uses 32×32 tiles and an online softmax, so the N×N score
> matrix is never stored. The big lesson came from a bank conflict. The key tile had a
> row stride of 64 floats, so all 32 lanes hit bank 0. Padding the stride to 65 gave
> 7×. Moving to four query rows per warp gave another 2×. That's 21× over the first
> version. It still runs 1.5× slower than PyTorch's cuBLAS-based attention at long
> sequences, and I can explain the gap.

---

## 2. Architecture

There is no single program. There are four standalone benchmarks, plus seven
lab-quiz kernels with grader harnesses. Each benchmark writes a CSV.
`tools/plot_all.py` turns the CSVs into PNGs, and `RESULTS.md` interprets them.

```
                          ┌──────────────── H200 (one GPU, pinned via CUDA_VISIBLE_DEVICES) ─────────────┐
                          │                                                                              │
 PA2  matmul/matmul_bench.cu ──► main :365 ──► per N: init A,B :420 ──► cuBLAS ref :435 ─► hRef          │
                          │                     │                                                        │
                          │                     ├─► time_kernel :312 (CUDA events, budget :331)          │
                          │                     │     mm_naive :44 → mm_coalesced :60 → mm_tiled_1to1 :78│
                          │                     │     → mm_tiled_strided :118 / _contiguous :173         │
                          │                     │     → mm_block2d<64,2|4> :232                          │
                          │                     ├─► record :445 → max_rel_err :353 → CSV row             │
                          │                     └─► cpu_naive_matmul :292 (N ≤ --cpu-max)                │
                          │                                                         → results/pa2_matmul.csv
                          │                                                                              │
 PA3  mlp/mlp_bench.cu ───► main :635                                                                    │
                          │   ├─ part1 :198  (2-layer fwd, B=32, N=32..32768)                            │
                          │   │     mm_thin<256,32> :47   one block = (one row, 256 cols)                │
                          │   │     baseline :254 | batch split :262 | column split :280                 │
                          │   │     measure :301 (3 warmup, 20 iters)                                    │
                          │   └─ part2 :595 → part2_run :352  (4-layer train step, ×100)                 │
                          │         activations :411-420 | arena :453-462 | peak via cudaMemGetInfo :548 │
                          │         forward :464 | backward :487 (mm_tiled :90, _AT :109, _BT :128)      │
                          │         checksum(W4) :572                                → results/pa3_mlp.csv
                          │                                                                              │
 PA4  flashattention/bench_attention.py ─► main :99                                                      │
                          │   q,k,v = randn[B=8,NH=16,N,D=64]  (fp32)                                     │
                          │   pytorch_standard :58  (matmul→softmax→matmul, batch-chunked :140)          │
                          │   custom_flash :46 ──ctypes──► build/libflash.so                             │
                          │        flash_lib.cu: flash_attention_forward :189 → kernel :47               │
                          │   pytorch_fused :77 (F.scaled_dot_product_attention)                         │
                          │   timeit :81 (3 warmup, 10 iters)                       → results/pa4_attention.csv
                          │                                                                              │
 PA1  kvcache/bench_pa1.py (NEVER RUN)                                                                   │
                          │   model.py: GPT :142, CausalSelfAttention.forward :60 (cache concat :69-72)  │
                          │   radix_tree.py: RadixTree.match_prefix :55, insert :87                      │
                          └──────────────────────────────────────────────────────────────────────────────┘

 tools/plot_all.py  ── reads results/*.csv ──► results/*.png
 tools/run_quiz.sh  ── extract student.cu → compile evaluator → run (optionally compute-sanitizer)
 prob2/q{1,2,3}, prob3/q{1..4}  ── lab quizzes (conv2d, transpose, 2D-blocked matmul, attention×3, checkpointing)
```

### End-to-end trace: one FlashAttention call at N = 8192

1. `bench_attention.py:135-137` makes `q, k, v` as `[8, 16, 8192, 64]` FP32 tensors,
   contiguous, on the GPU.
2. `:140-143` picks a batch chunk for the reference path. The score tensor is
   8·16·8192²·4 B = 34.4 GB. That exceeds `--budget-gb 16`, so the chunk halves to 4
   (still 17 GB, too big), then to 2. The CSV records `standard_batch_chunk=2`.
3. `:146` runs `pytorch_standard`, the explicit N×N path, and keeps its output as the
   reference.
4. `:147` calls `custom_flash` (`:46`). It allocates `o` and passes four raw device
   pointers through ctypes into `flash_attention_forward` (`flash_lib.cu:189`).
5. On the host side (`flash_lib.cu:192-212`):
   * It rejects D > 128 (`:192`), because each lane holds at most 4 output slots.
   * Tr = Tc = 8192/32 = 256.
   * Shared memory is 32·64 + 32·65 + 32·64 + 32·32 floats = 28,800 B (`:198`).
     That is under 48 KB, so no opt-in is needed (`:202`).
   * It launches `grid(Tr=256, NH=16, B=8)` = 32,768 blocks of `(32, 8)` = 256 threads,
     with dynamic shared memory (`:210-212`).
6. Inside one block (`:47`):
   * The block owns 32 query rows of one (batch, head). Warp `w` owns rows `4w..4w+3`
     (`:72`).
   * All 256 threads stage the 32×64 Q tile into shared memory (`:81-85`).
   * Loop over 256 KV tiles (`:98`):
     - `__syncthreads`. Then all threads load K into `Kj`, using row stride **65**, and
       load V into `Vj` (`:103-110`).
     - **Scores** (`:118-126`): lane `c` owns key `c`. For each feature `x`, the lane
       reads `Kj[c*65 + x]` once and does 4 FMAs, one per query row. The `Qi` reads are
       warp-wide broadcasts.
     - **Online softmax** (`:130-152`), per row:
       - the tile max via 5 `__shfl_xor_sync` steps;
       - `p = exp(s - m_tile)`;
       - the tile sum via 5 more shuffles;
       - write `p` into the warp's own rows of `S`;
       - update the running max and sum, and compute the rescale factors
         `e_run`, `e_tile`.
     - `__syncwarp` (`:153`). Only this warp reads its own rows of `S`.
     - **PV** (`:157-172`): lane owns feature `x` and `x+32`. For each key `c`, it reads
       `Vj[c*64 + x]` once and does 4 FMAs. Then
       `o_acc = o_acc*e_run + pv*e_tile`.
   * At the end, each row is divided by `l_run` and written to global O (`:175-183`).
7. Back in Python, `:149` takes the max abs difference against the reference. The
   result is 6.7e-7.
8. `:154-156` times all three paths over 10 iterations each, using CUDA events. The
   custom kernel takes 160.05 ms.
9. The CSV row is written (`:175-178`). The crossover summary prints (`:184-200`).
   `tools/plot_all.py` draws `pa4_runtime.png`.

---

## 3. Design decisions

The format for each: **chosen** / alternatives / why / cost / source.

### PA2 — matmul

1. **cuBLAS runs first and is both the speed ceiling and the correctness reference.**
   * Alternative: a CPU reference.
   * Why: a CPU reference at N = 32768 would take about 34 hours.
   * Cost: the check is relative to cuBLAS, not to exact math.
   * Row-major layout uses the trick of computing B·A in column-major order
     (`matmul_bench.cu:433-437`).
   * *Documented* (code comment `:432-433`).
2. **The correctness check only looks at the first 4M elements** (`:447`,
   `min(elems, 4000000)`).
   * That covers the whole matrix only up to N = 2048. At N = 32768 it covers the
     first 122 rows.
   * The comment at `:352` says "sampled over the whole matrix". That is not what the
     code does.
   * *Reconstructed.* Know this before you say "bit-exact".
3. **Time budget: one timed iteration first, ten only if they fit in 30 s**
   (`:321-340`).
   * Cost: at N = 32768 every kernel except `block2d_t64r4` and cuBLAS has **n = 1**,
     and so does naive at 16384 (CSV `iters` column).
   * *Documented* (code comment).
4. **The CPU baseline is measured up to N = 4096 and extrapolated as N³ beyond that.**
   * *Documented* (`SETUP.md`, `RESULTS.md`).
   * Cost: the CPU numbers past 4096 are not measurements. It is also a single-threaded
     ijk loop, which is a strawman baseline.
5. **TILE is a template parameter, so the shared tiles are static** (`:228-229`).
   * Alternative: dynamic `extern __shared__`, which the quiz version uses.
   * Why: the compiler can fully unroll, and it auto-vectorized the shared loads to
     `LDS.128`, which I confirmed in the SASS.
   * Cost: the static limit is 48 KB, so TILE = 128 with two tiles cannot compile.
   * *Documented* (comment). The SASS observation is *reconstructed*.
6. **The strided and contiguous loaders are kept as a controlled pair.** They do the
   same arithmetic with the same shared footprint. Only the global access pattern
   differs (`:103-116`).
   * *Documented.*
7. **The final kernel is the Lab Quiz 2 Q3 kernel, re-templated** (`:227-232`). Its
   thread mapping is the "contiguous" one: thread → `ty*R+i`, `tx*R+j`.
   * Cost: it inherits that loader's uncoalesced, stride-R global loads and
     bank-conflicted stores. *Reconstructed.*
8. **Cost of the kernel shapes** (*reconstructed* from `cuobjdump -res-usage` and the
   occupancy rules):
   * `block2d<64,2>` uses 1024 threads × 40 registers. Only 1 block fits per SM, which
     is 50% occupancy.
   * `tiled_strided` uses 1024 × 32 registers. That fits 2 blocks, which is 100%.
   * `block2d<64,4>` uses 256 × 62 registers, rounded to 64. That fits 4 blocks, which
     is 50%.

### PA3 — streams and checkpointing

9. **Custom GEMMs everywhere, no cuBLAS** (`mlp_bench.cu:17-18`).
   * Why: cuBLAS keeps an internal workspace that pollutes `cudaMemGetInfo`.
   * Cost: absolute times are slow. The backward pass is 3.4× the forward instead of
     about 2×, because the transposed-operand kernels load uncoalesced (`_AT :118`,
     `_BT :138`).
   * *Documented* (the reason). The cost is *reconstructed*.
10. **`mm_thin`: one block per (row, 256-column chunk) for the B = 32 forward pass**
    (`:37-45`).
    * Alternative: square tiles.
    * Why: square tiles would waste 24 of every 32 tile rows once the batch is split
      4 ways.
    * Cost: each row-block streams its own copy of the W column slice. Reuse of W
      across the 32 rows depends entirely on L1/L2 cache hits.
    * *Documented* (the reason). The cost is *reconstructed*.
11. **The column-split variant was added beyond the brief** (`:8-10`). The two layers
    need a host barrier between them, because layer 2 needs all of A1 (`:276-278`,
    `:289`).
    * *Documented.*
12. **The stream variants call `cudaStreamSynchronize` inside the timed loop**
    (`:273`, `:289`, `:298`). The baseline does not. It queues 2 launches per
    iteration with no host sync.
    * Cost: at small N the stream numbers include host round-trips. That is part of the
      "4× slower".
    * *Reconstructed.*
13. **The checkpoints are A2 and Z4**, as the assignment prescribes (`CLAUDE.md` PA3).
    Z4 is the output, which MSE needs. A2 splits the network into two 2-layer segments.
    * *Documented.*
14. **A single recompute arena sized `max(2·n3, 2·n1+n2)`, aliased by both segments**
    (`:441-462`).
    * Alternative: one buffer per dropped activation. That saves nothing.
    * Cost: a strict ordering constraint in backward (`:487-491`, `:513`). Every use of
      z3/a3 must come before the segment-A recompute.
    * *Documented* (RESULTS + comment).
15. **Peak memory = the `cudaMemGetInfo` delta after every buffer is allocated**
    (`:358-360`, `:548-550`).
    * Since everything is a `cudaMalloc` made in advance, the footprint equals the
      peak. It is deterministic, so there is no variance.
    * Cost: it is device-wide, so another process on the same GPU would contaminate it.
      Values are MiB (2²⁰), though they are labelled "MB".
    * *Reconstructed.*
16. **Weights are scaled by 1/sqrt(fan_in)** (`:396-404`).
    * Why: without it the training loop hit inf/NaN within a few steps, and the
      equality checksum became useless.
    * *Documented.* Nit: W4 uses `isq(H2)` instead of `isq(H3)`. That is harmless,
      because H2 = H3 in every config.
17. **Equality check = the sum of W4 after 103 SGD steps** (`:570-579`).
    * Cost: it checks one tensor only. Both modes use identical kernels in identical
      order, so bitwise equality is expected rather than a strong test.
    * *Documented* (the reason). The cost is *reconstructed*.

### PA4 — FlashAttention

18. **Work mapping: `grid(Tr, NH, B)`, 8 warps per block, 4 query rows per warp.**
    In the score phase lanes own keys. In the PV phase lanes own features.
    * Alternative: the quiz layout, `grid(B, NH)` with one thread per query row.
    * *Documented* (`flash_lib.cu:6-26`, `RESULTS.md`).
19. **Pad the Kj row stride to D+1** (`:55`).
    * Alternatives: an XOR swizzle, or storing K transposed. Transposed storage would
      move the conflict to the store, which happens once per tile, not D times.
    * Cost: 128 extra bytes per block. It also breaks 16-byte row alignment, so Kj rows
      can't use `LDS.128`.
    * *Documented* (the fix). The alternatives and cost are *reconstructed*.
20. **Dynamic shared memory with `cudaFuncSetAttribute` opt-in above 48 KB**
    (`:198-208`).
    * The opt-in never fires at D = 64.
    * *Reconstructed.*
21. **A C ABI loaded by ctypes**, not a PyTorch C++ extension (`bench_attention.py:34-55`).
    * Cost: no dtype, contiguity or device checks. No autograd. It launches on the
      default stream, which is only right because PyTorch's current stream is also the
      default stream here.
    * *Reconstructed* (no doc gives a reason).
22. **`__expf` fast intrinsic and warp-shuffle reductions** (`:136-143`).
    * Cost: a slightly less accurate exp. The measured error is 7e-7.
    * *Reconstructed.*
23. **Occupancy comes from the CUDA occupancy API, not ncu** (`:230-256`).
    * Forced: ncu gives `ERR_NVGPUCTRPERM` without root.
    * *Documented* (`SETUP.md`, `RESULTS.md`).
24. **The explicit-PyTorch baseline is batch-chunked under a memory budget**
    (`bench_attention.py:140-143`).
    * Cost: at N = 8192 the baseline loops 4 times in Python. That makes it slightly
      slower than one unchunked call, which would need 34 GB.
    * *Documented.*
25. **Crossover reported honestly as a launch-overhead artifact** (`:181-200`).
    * *Documented.*

### PA1 — KV cache and radix tree

26. **KV cache grows by `torch.cat` each step** (`model.py:69-72`).
    * Alternatives: a preallocated `[B, H, T_max, d]` buffer with a write index, or
      paged blocks (vLLM).
    * Cost: every step copies the whole cache, so O(T²) bytes moved overall and a new
      allocation each step.
    * *Reconstructed.*
27. **Radix tree with compressed edges. A node stores the KV for its full path, and any
    prefix is a slice** (`radix_tree.py:1-11`).
    * *Documented.*
28. **`expand_kv` makes the broadcast contiguous** (`radix_tree.py:19-27`).
    * Why: `torch.cat` would materialize it anyway.
    * Cost: each batch row holds its own copy of the prefix KV. So prefix sharing saves
      prefill compute, not decode memory.
    * *Documented* (the reason). The cost is *reconstructed*.
29. **The benchmark was not executed** "at the user's direction, to avoid large runs
    against real pretrained models" (`RESULTS.md:235-241`).
    * *Documented.*

---

## 4. Hardest problems, as STAR stories

### S1. The 32-way bank conflict in FlashAttention (documented: `RESULTS.md:216-226`, `flash_lib.cu:11-26`)

* **Situation.** I moved the kernel from the quiz layout, with 128 blocks of one warp,
  to a grid over query tiles, with 32,768 blocks. I expected a big speedup.
  N = 8192 went only from 3358 ms to 2377 ms.
* **Task.** Find the real bottleneck without a profiler. ncu needs root.
* **Action.** I worked out the shared-memory addresses by hand. In the score loop,
  lane `c` reads `Kj[c*64 + x]` for a fixed `x`. The bank is `(c*64 + x) mod 32`, which
  equals `x mod 32` for every lane. So all 32 lanes hit the same bank at different
  addresses, which is a 32-way conflict on the hottest loop. I padded the row stride to
  65. The bank becomes `(c + x) mod 32`, so all lanes are distinct.
* **Result.** 2377 → 339 ms, which is **7.0×** from a one-line change. Correctness was
  unchanged, because the same stride is used for the store and the load.

### S2. Hitting the shared-memory roofline, then register blocking over rows (documented)

* **Situation.** After the padding fix, the kernel measured 6.5 TFLOP/s.
* **Task.** Decide whether more tuning could help, or whether something fundamental
  capped it.
* **Action.** I counted instructions. At one query row per warp, each FMA needs two
  shared loads: one Kj and one Qi broadcast. Shared memory serves one 128-byte
  wavefront per clock per SM. That caps throughput near 8.4 TFLOP/s:
  132 × 1.98 GHz × 32 flops/clk/SM. So I was already near the ceiling. I gave each
  warp 4 query rows. Now one Kj or Vj load feeds 4 FMAs, which is 5 loads per 4 FMAs
  instead of 2 per 1.
* **Result.** 339 → 160 ms (2.1×), or 13.75 TFLOP/s. My own back-of-envelope check: the
  new ceiling under the same model is about 13.4 TFLOP/s. So the kernel sits on the
  shared-memory roof again, and the next step needs fewer shared loads per FMA.
  The total from the first version is **21×**.

### S3. Gradient checkpointing that saved 0 MB (documented: `RESULTS.md:156-161`, `mlp_bench.cu:441-452`)

* **Situation.** The first checkpointed version reported **0 MB saved**.
* **Task.** Make checkpointing actually reduce memory, and prove it is still correct.
* **Action.** I had given each recomputed activation its own permanent buffer. That
  holds exactly what the baseline holds. The fix rests on one observation. Backward
  runs from layer 4 down to layer 1. So segment B (Z3, A3) is fully consumed before
  segment A (Z1, A1, Z2) is rebuilt. One arena of `max(2·n3, 2·n1+n2)` floats serves
  both, and the code orders every use of z3/a3 before the rebuild. To check
  correctness, I compared the post-update weight checksums of both modes.
* **Result.** At the 4096 config, the training footprint fell 962 → 834 MiB (−13.3%).
  Activations fell 400 → 272 MiB (−32%). Checksums match to 10 digits. The backward
  pass got 35.5 ms slower (+26.1%).

### S4. Training went to NaN, which killed the cross-check (documented: `mlp_bench.cu:396-400`)

* **Situation.** The two-mode equality check relies on final weight checksums. With
  U(−1,1) weights on 4096-wide layers, the loop reached inf/NaN within a few steps.
  NaN ≠ NaN, so the check was meaningless.
* **Action.** I scaled each weight matrix by 1/sqrt(fan_in). That keeps the activation
  scale roughly constant through the layers.
* **Result.** Finite, matching checksums in all three configs (−84.09, −259.73, −379.71).

### S5. Grader harness traps (documented: `RESULTS.md:29-50`, `prob3/q4/student.cu:36-53`)

* **Situation.** The lab quiz harnesses didn't match their writeups. All seven had to
  pass.
* **Action.** I read the evaluators, not just the spec:
  * **LQ3 q4:** the launcher passes no dynamic shared-memory size. `extern __shared__`
    would therefore be a **zero-byte** buffer. I used a static 48 KB buffer, partitioned
    at runtime.
  * **LQ3 q4:** the grader rewrites `D_MAX = 128` to `80`. At D = 128 the specified
    layout needs 52 KB, which is over the static ceiling. So I added a fallback path
    that skips staging Q.
  * **LQ3 q4:** `m` is initialized with `cudaMemset(0xFF)`. That bit pattern is
    **NaN**, not −inf. `fmaxf` hides it, but `expf(NaN − m_new)` = NaN poisons `l`.
    So I seed `m_prev = −inf` on the first tile.
  * **LQ3 q2:** the launcher uses 32 threads, not D. So the guards differ by phase, and
    no thread may return early before a `__syncthreads`.
  * **LQ3 q3:** the evaluator defines neither the forward nor the backward pass, and
    `evaluator2` extracts only two kernels. So those two must compile standalone.
  * **LQ2 extractors** find a kernel by the first textual occurrence of its name. A
    comment naming the kernel above its definition breaks extraction.
* **Result.** 7/7 quizzes pass, and every merged binary is clean under
  `compute-sanitizer memcheck`.

### S6. Batch-split streams never helped (documented: `RESULTS.md:123-135`)

* **Situation.** The brief asked for the forward pass split over 4 streams. At every
  width it was slower: up to 4× at small N, and 18% slower at N = 32768.
* **Task.** Explain it, and find a decomposition that actually helps.
* **Action.** With B = 32, a batch split gives each stream 8 rows. Each stream still
  reads the **entire** W1 and W2. That is the dominant traffic: N² floats against 32·N
  for X. It also launches 4× as many kernels. I added a column split, where each stream
  owns a quarter of the output columns and a quarter of W.
* **Result.** 1.73× at N = 32768 (44.43 → 25.65 ms). It is neutral at 16384 and slower
  below. I reported the negative result as the headline. *Reconstructed caveat to add
  in interviews:* the baseline already launches 4096 blocks, about 4 waves, at that
  size. So the gain is most likely memory locality, not concurrency (see Q22).

### S7. Toolchain on a shared box without root (documented: `SETUP.md:31-42`)

* The quiz scripts hardcode `-ccbin=/usr/bin/gcc-11`, which doesn't exist. I wrote
  `tools/run_quiz.sh` to replicate their steps and pin the GPU.
* `cudaDeviceProp::clockRate` is gone in CUDA 13. I switched to
  `cudaDeviceGetAttribute(cudaDevAttrClockRate)`.
* ncu is blocked, so I used the occupancy API and stated plainly that the occupancy
  figures are theoretical.

---

## 5. Numbers sheet

Common setup: H200 SXM, 132 SMs, 1980 MHz boost clock (clocks not locked; no root),
CUDA 13.0, FP32, kernel time by CUDA events, single sweep (no repeats, no error bars).
The current repo contains the CSVs that support every number below. **I did not re-run
anything.** To re-run you must first fix the stale paths in `SETUP.md`.

| Claim | Where it comes from | How measured | Caveats | Repo supports it? |
|---|---|---|---|---|
| **51.3× over naive GPU** (resume) | `pa2_matmul.csv` N=32768: 140,485.9 / 2,736.6 ms | Kernel time; naive **n = 1** (140 s run), block2d_r4 n = 10 | Naive is deliberately uncoalesced (assignment-defined). Ratio is 45× at N=1024 and 50–51× from 2048 up, so it is robust. Against the coalesced kernel it's 4.7×; against the best plain tiled kernel, 1.33× | **Yes** |
| **48.2% of cuBLAS** (resume) | 25.71 / 53.37 TFLOP/s, N=32768 | cuBLAS n=10, block2d n=10 | Both FP32 CUDA-core paths. cuBLAS default math mode, so no TF32; bit-exact agreement up to N=16384 (on the first 4M elements) supports this. 48–50% for all N ≥ 2048 (61% at 1024). With TF32 tensor cores allowed, cuBLAS would be far faster | **Yes** |
| 38.4% of peak | 25.71 / 66.91 | Peak = 132·128·2·1.98 GHz | Uses boost clock; the real sustained clock is unknown | Yes |
| 10.9× from coalescing | 140,486 / 12,927 at N=32768 | Both n=1 | | Yes |
| "Tiling 1.6×, register blocking 3.0×" (RESULTS) | 8.64/5.44; 25.71/8.64 | | The 3.0× is measured against the 1-to-1 tile. The strided 64-tile kernel already has 2×2 outputs per thread. Against it, R=4 is 1.33× | Yes, but quote the chain (Q5) |
| 42,783× vs CPU (RESULTS only) | 237,241 / 5.545 ms at N=4096 | CPU measured once | Single-thread naive ijk loop, a strawman. Don't put this on a resume | Yes |
| **1.73× MLP "inference"** (resume) | `pa3_mlp.csv` N=32768: 44.43 / 25.65 ms | B=32, 2-layer forward, custom `mm_thin`, 3 warmup + 20 iters, one run | **Only at N=32768.** It is 1.02× at 16384 and slower below. The baseline is my own kernel, not PyTorch; the brief's PyTorch baseline was never implemented. Synthetic weights. Mechanism unproven: the baseline already fills the GPU, and the baseline scales 7.1× from 16K→32K for 4× work, while the column split scales 4.1×. Part 1 has **no correctness check**. The column split is buggy for N ≤ 512, though not at 32768 | **Number yes; mechanism and wording need care** |
| **13.3% "peak activation memory"** (resume) | 1,008,730,112 → 874,512,384 B (962 → 834 MiB) | `cudaMemGetInfo` delta, 4096 config | **Mislabelled.** 13.3% is the whole training footprint (params + grads + dZ buffers + I/O). **Activation memory fell 32%** (400 → 272 MiB, computed from buffer sizes). It's a different workload (4-layer training) from the 1.73× (2-layer forward) | **Number yes; label wrong** |
| +26% backward time | 136.15 → 171.69 ms (+35.54 ms, +26.1%), 100 iters | CUDA events per iteration | Overhead equals the recomputed forward FLOPs: 2.25 of 2.5 GEMM-units × 39.5 ms ≈ 35.5 ms. Configs 1 and 2: +23.8%, +24.2% | Yes |
| FlashAttention 21× | 3358 → 160 ms at N=8192 | Final 160.05 ms in CSV (10 iters) | **The first three versions are not in the repo** (single commit). Only the final number is reproducible. The 21× is against my own first version, not against PyTorch. Padding alone = 7.0× (2377 → 339) | Final yes; history documented only |
| Flash vs PyTorch | 160.05 vs 105.08 (standard) vs 62.62 ms (fused) | 10 iters | The custom kernel is 1.5× slower than standard and 2.6× slower than fused. It wins only at N ≤ 128 (launch-bound regime) | Yes |
| Flash max abs diff ≤ 7.2e-7 | CSV | Against the explicit FP32 path | | Yes |
| 50% theoretical occupancy, 62 regs, 28,800 B smem | Occupancy API; cuobjdump agrees (REG:62) | Not measured | **Theoretical**, not achieved | Yes |
| PA1 speedups | — | — | **Never run. No numbers exist.** | **No** |

### Honest resume wording

* **Bullet 1 (fine as is; tighten):**
  "Optimized FP32 CUDA matrix multiplication on an H200 with coalescing,
  shared-memory tiling and 2D register blocking: 51.3× faster than a naive GPU kernel,
  reaching 48.2% of cuBLAS SGEMM (25.7 vs 53.4 TFLOP/s at N = 32K)."
* **Bullet 2 (rewrite):**
  "Sped up a batch-32 MLP forward pass 1.73× at width 32K by partitioning output
  columns across 4 CUDA streams (batch splitting never helped). Implemented gradient
  checkpointing for a 4-layer MLP, cutting activation memory 32% (13.3% of total
  training memory) for a 26% slower backward pass."

Key fixes in the rewrite:

* "Activation memory 13.3%" becomes 32% activation / 13.3% total.
* "at large dimensions" becomes "at width 32K".
* The bullet no longer implies the two numbers come from one workload.

---

## 6. Prerequisite concepts

**GPU execution model.**
An SM runs blocks; a block is split into warps of 32 threads that issue together. The
H200 has 132 SMs. Each SM holds at most 2048 threads, 64K registers, and up to 228 KB
of shared memory. Occupancy is resident warps divided by the maximum. It is limited by
registers, shared memory, or threads per block. High occupancy hides latency, but it is
not a goal in itself.
*Source:* CUDA C++ Programming Guide (hardware implementation, occupancy).

**Memory hierarchy.**
From fastest to slowest:
* registers (per thread);
* shared memory / L1 (per SM, on-chip);
* L2 (50 MB, shared by all SMs);
* HBM3e (141 GB, 4.8 TB/s).

Each level down is slower and bigger. Fast kernels reuse data at the highest level they can.
*Source:* CUDA C++ Best Practices Guide; NVIDIA H100 architecture whitepaper.

**Coalescing.**
A warp's loads are served in 32-byte sectors. If 32 lanes read 32 consecutive floats,
that's 4 sectors (128 B). If they read 32 floats N apart, that's 32 sectors. Mapping
`threadIdx.x` to the fastest-moving memory index is the fix.
*Source:* Best Practices Guide, "Coalesced Access to Global Memory".

**Shared-memory banks.**
Shared memory has 32 banks, each 4 bytes wide. Bank = (word address) mod 32. Lanes
that hit different addresses in the same bank are serialized. Lanes that hit the same
address get a broadcast. A stride that is a multiple of 32 words causes a 32-way
conflict; padding the row by one word fixes it.
*Source:* CUDA Programming Guide, shared memory section. Mark Harris, "An Efficient
Matrix Transpose in CUDA C/C++" (NVIDIA blog).

**Arithmetic intensity and roofline.**
Intensity = FLOPs per byte moved from a memory level. Attainable throughput is the
minimum of peak compute and intensity × bandwidth. H200's DRAM ridge point is about
67 TFLOP/s ÷ 4.8 TB/s ≈ 14 flop/B. Below that, the kernel is memory-bound.
*Source:* Williams, Waterman, Patterson, "Roofline" (CACM 2009).

**Tiling and register blocking.**
* A T×T shared tile cuts global traffic by T. Global intensity is T/4 flop/B for FP32.
* An R×R register block per thread cuts shared traffic per FMA by R. It also adds
  instruction-level parallelism.

This is the hierarchical GEMM: block tile → warp tile → thread tile.
*Source:* Volkov & Demmel, "Benchmarking GPUs to tune dense linear algebra" (SC'08).
CUTLASS docs, "Efficient GEMM in CUDA". Simon Boehm, "How to Optimize a CUDA Matmul
Kernel for cuBLAS-like Performance" (2022 blog).

**Latency hiding with async copies.**
cuBLAS-class kernels double-buffer shared memory. They load tile k+1 with `cp.async`
(Ampere) or TMA (Hopper) while computing tile k. Without this, every tile pays the full
global-load latency between two `__syncthreads`.
*Source:* CUTLASS docs. NVIDIA H100 whitepaper (TMA).

**CUDA streams.**
A stream is an in-order queue. Kernels in different streams *may* run at the same time
if resources allow. Streams help only when one kernel leaves the GPU underused, or to
overlap copies with compute. They add launch overhead and need syncs.
*Source:* CUDA Programming Guide, "Asynchronous Concurrent Execution". NVIDIA blog,
"GPU Pro Tip: CUDA 7 Streams Simplify Concurrency".

**CUDA event timing.**
`cudaEventRecord` before and after on the same stream measures GPU time between them.
Warm up first, average over iterations, and keep host syncs out of the timed region
unless they are part of the workload.
*Source:* NVIDIA blog, "How to Implement Performance Metrics in CUDA C/C++".

**Gradient checkpointing.**
Store only some activations in the forward pass. Recompute the rest from the nearest
checkpoint during backward. With √n checkpoints for n layers, memory is O(√n) for
about one extra forward pass.
*Source:* Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016).
Griewank & Walther, "Revolve" (2000).

**Online softmax.**
Softmax over a row can be computed in one streaming pass. You keep a running max `m`
and a running sum `l`. When a new block has a larger max, you rescale the old sum (and
the old output) by exp(m_old − m_new). This is what lets attention avoid storing the
N×N matrix.
*Source:* Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018).

**FlashAttention.**
Tile Q, K and V into on-chip memory. Compute scores, the online softmax and P·V per
tile. Never write the N×N matrix to HBM. Memory goes from O(N²) to O(N). It's faster
because attention is HBM-bandwidth bound, not FLOP bound.
*Source:* Dao et al., "FlashAttention" (NeurIPS 2022). Dao, "FlashAttention-2" (2023).

**KV cache.**
In autoregressive decoding, the keys and values of past tokens don't change. Caching
them turns each step from re-encoding T tokens into encoding 1 token against the cache.
Total compute per sequence drops from O(T²) to O(T). The cost is memory that grows
with T.
*Source:* Pope et al., "Efficiently Scaling Transformer Inference" (2022). Kwon et al.,
"PagedAttention / vLLM" (SOSP 2023).

**Radix-tree prefix caching.**
Index cached KV by token prefix in a compressed trie. A new request reuses the KV of its
longest cached prefix and prefills only the rest.
*Source:* Zheng et al., "SGLang: Efficient Execution of Structured Language Model
Programs" (RadixAttention), 2024.

**Tensor cores / TF32.**
Hopper's FP32 CUDA-core peak is about 67 TFLOP/s. Its TF32 tensor-core peak is about
495 TFLOP/s dense. Any "% of cuBLAS" claim must say which path it compares against.
*Source:* NVIDIA H100/H200 datasheets.

---

## 7. Fifty likely questions, with crisp answers

### Matmul (PA2)

**Q1. Walk me through the kernel progression and the arithmetic intensity at each step.**

Global intensity is ideal FLOPs per byte loaded from global by a block. The FFMA-per-LDS
column is a static count from the committed `build/pa2_bench` SASS, not a runtime
measurement.

| Kernel | Global intensity | Per-thread work | FFMA per shared load | Occ. (theor.) | TFLOP/s @32K |
|---|---|---|---|---|---|
| naive | 0.25 flop/B nominal. Per warp per k: 32 sectors for A, 1 for B, so ~0.06 flop/B in sector terms | 1 output | — | 100% | 0.50 |
| coalesced | Per warp per k: A broadcast (1 sector) + B 128 B, so ~0.4 flop/B | 1 | — | 100% | 5.44 |
| tiled 1-to-1, T=32 | T/4 = 8 | 1 | 0.8 | 100% | 8.64 |
| tiled strided, T=64 | 16 | 2×2 | 1.6 | 100% | 19.36 |
| tiled contiguous, T=64 | 16 | 2×2 | 2.7 | 100% | 18.57 |
| block2d T=64 R=2 | 16 | 2×2 | 2.7 | 50% | 16.51 |
| block2d T=64 R=4 | 16 | 4×4 | 8 (LDS.128) | 50% | 25.71 |
| cuBLAS (typical 128×128+) | ≥ 32 | 8×8 or more | high | — | 53.37 |

Why each step helps:

* **Coalescing** cuts transactions per warp from 32 to 1 for A.
* **Tiling** cuts DRAM traffic by T.
* **T=64** puts global intensity above the ~14 flop/B DRAM ridge. From there on, the
  limit is inside the SM.
* **Register blocking** raises FMAs per shared load, so shared bandwidth stops being
  the cap.

**Q2. Why is naive so bad? Isn't the B access fine?**

B is fine: all lanes in a warp share one column, so B is a broadcast. A is the problem.
Lanes differ in `row`, so each lane reads a different row of A, N floats apart. That's
32 sectors per warp load. The C store is also strided.

**Q3. Coalescing gave 10.9×, but the sector math says about 6–16×. Why not exactly 32×?**

L1 caches sectors. The naive kernel's lane reads A[row][k] and then A[row][k+1], which
sits in the same sector. So some of those 32 sectors are L1 hits on the next iterations.
The real ratio depends on cache hit rates I couldn't measure without ncu.

**Q4. Why is block2d R=2 slower than tiled_strided? Both do 4 outputs per thread on a 64-tile.** *(reconstructed)*

Two reasons.

* **Occupancy.** R=2 uses 40 registers × 1024 threads, so only 1 block fits per SM (50%
  occupancy). Strided uses 32 registers, so 2 blocks fit (100%).
* **Mapping.** block2d uses the "contiguous" mapping (`tx*R + j`). Its global loads are
  stride-R across lanes, and its shared stores have bank conflicts.

**Q5. Your RESULTS says "register blocking 3.0×". Is that fair?** *(hostile)*

It's measured against the 1-to-1 tiled kernel. The strided 64-tile kernel already does
2×2 per thread. The honest chain is:

* 10.9× from coalescing,
* 1.59× from the 32-tile,
* 2.24× from the 64-tile with 2×2 per thread,
* 1.33× from going to 4×4 per thread.

**Q6. Why does R=4 beat R=2 by 1.56×?**

Each k step loads R values of A and R of B to do R² FMAs. So FMAs per load = R/2: 1 for
R=2, 2 for R=4. The compiler also vectorized the loads to `LDS.128`, which gives 8
FFMAs per shared load instruction in the SASS. More independent accumulators also means
more ILP.

**Q7. Why do you stop at 48% of cuBLAS?** *(reconstructed — I could not profile)*

1. **No overlap.** It loads, syncs, computes, syncs. Global latency is exposed on every
   tile. There is no double buffering and no `cp.async`/TMA.
2. **Global loads are scalar `LDG.E` with a stride-4 lane pattern** (`tx*R + j`). That
   wastes sectors. cuBLAS uses 128-bit coalesced loads.
3. **Shared stores have about 4-way bank conflicts.** Lanes write words 4 apart.
4. **50% occupancy and a 64×64 block tile.** That's 16 flop/B against cuBLAS's 32+.
5. **No warp-level tiling or register double-buffering, and no SASS-level tuning.**

Without ncu I can't rank these. My first experiment would be vector loads plus double
buffering.

**Q8. What does cuBLAS do differently?**

Typically:

* 128×128 or 128×256 block tiles, split into warp tiles.
* 8×8 or more outputs per thread.
* A stored k-major in shared memory, so both operands are read with conflict-free
  `LDS.128`.
* Multi-stage pipelines with async copies.
* Swizzled block order for L2 reuse.
* Per-shape kernel selection by heuristics.

I didn't check which kernel it picked.

**Q9. Is 48% of cuBLAS good?**

For a course kernel, yes. It's in the range people reach before adding async pipelines
and warp tiling. Boehm's blog walks the same path to about 90%, so I know what the
remaining steps are.

**Q10. How did you verify correctness? You claim bit-exact.**

Each kernel is compared to cuBLAS as max relative error. The value is 0.0 up to
N = 16384, and 4.5e-5 at 32768 for all custom kernels, which agree with each other. Bit
equality is plausible because both accumulate each output in k order with FMA. **Two
caveats:**

* The check covers only the first 4M elements (`:447`). That's the full matrix only up
  to N = 2048.
* I don't clear `dC` between kernels or call `cudaGetLastError` after launches. A kernel
  that failed to launch would inherit the previous result.

The distinct timings show every kernel ran. I'd still fix both.

**Q11. Why does the error become 4.5e-5 at N = 32768 for every custom kernel?**

The custom kernels all match each other, so the difference is on the cuBLAS side. Most
likely cuBLAS picked a different kernel at that size, with a different summation order,
such as split-K. That's a hypothesis.

**Q12. How did you compute the 66.9 TFLOP/s peak?**

132 SMs × 128 FP32 lanes × 2 flops per FMA × 1.98 GHz. That uses the boost clock. The
sustained clock under load may be lower, so % of peak is slightly understated.

**Q13. Why time only the kernel and not the copies?**

The brief asks for kernel time, and copies are the same for every kernel. At N = 32768
the copies are 13 GB. They'd swamp the comparison and say nothing about the kernels.

**Q14. Naive at N = 32768 was measured once. Is that OK?** *(hostile)*

It took 140 s, and naive is at 0.50 TFLOP/s at every N from 1024 up. It's a very stable
kernel. For the 51.3× I'd still say "single 140-second run". Note that the ratio is
45–51× at every size.

**Q15. Why FP32 on CUDA cores and not tensor cores?**

The assignment is about the memory hierarchy. FP32 against SGEMM is the fair matchup.
With TF32 tensor cores, cuBLAS would run several times faster again. The next step is
WMMA/`mma.sync` with TF32 or BF16.

### Streams (PA3 part 1)

**Q16. Why did splitting the batch across streams never help?**

There are three reasons.

1. **The baseline already fills the GPU at large N.** It's 4096 blocks at N = 32768, so
   streams have no idle hardware to use.
2. **Each batch-split stream reads all of W1 and W2.** W is N² floats, against 32·N for
   X. So weight traffic grows, and each kernel sees only 8 rows of reuse per W line.
3. **At small N it's 8 launches plus 4 host syncs** per iteration, against 2 launches.
   The baseline is 7 µs at N = 32, so the overhead dominates.

**Q17. So why did splitting columns help?**

Each stream owns a disjoint quarter of W and of the output. There's no duplicated
weight traffic. At N = 32768 that gave 1.73×.

**Q18. But you just said the GPU was already full. Where does the 1.73× come from?** *(hostile — the key question)*

Honestly: not from added parallelism. Here's the evidence.

* The baseline scales 7.1× in time from N = 16K to 32K for 4× the work.
* The column split scales 4.1×, which is linear.
* So the baseline hits a large-N pathology, and the column split avoids it.

My leading hypothesis is cache locality. In a column-split kernel, `grid.x` is 32
instead of 128. So all 32 row-blocks of a column slice are resident at about the same
time, and they share each W line through L1/L2. In the baseline, only about 8 rows'
worth of blocks are resident together. Each W line must survive across waves in a 50 MB
L2, and the matrix is 4 GB.

TLB reach or DRAM page locality are also candidates. I couldn't separate them without
ncu. The test is to run the four column chunks on **one** stream (exercise 2).

**Q19. Why is the baseline 44 ms? What bounds `mm_thin`?**

Every FMA does one global load of W, so it's about 0.5 flop/B at the L1/L2 level. The
kernel moves 275 GB of W loads through the cache hierarchy. That's about 10.7 TB/s for
the column split, and 6.2 TB/s for the baseline at 32K. Either way it's
memory-hierarchy bound, not FLOP bound: about 3–5 TFLOP/s.

**Q20. Why the barrier between layers in the column split?**

Layer 2's output column j needs all of A1's columns. Layer 1 is split by column, so
every stream must finish before any layer-2 kernel starts.

**Q21. Are the small-N stream numbers a fair comparison?** *(hostile)*

Not fully. The stream variants call `cudaStreamSynchronize` every iteration inside the
timed loop. The baseline just queues launches. So the 4× at small N includes host
round-trips, not only GPU launch overhead.

**Q22. Did you check the outputs of the stream variants?**

No. Part 1 has no correctness check. While preparing, I also found that the column split
is wrong for N ≤ 512. The kernel guard is `col < N` with the full width. So with
`cchunk < 256`, threads write past their chunk, and the last stream writes past the end
of `A1`/`Y`. From N = 1024 up, each chunk is a multiple of 256, so the 32K result is
computed correctly. But the small-N column-split timings come from buggy code. The fix
is to pass `cols` as a separate bound.

**Q23. Where's the PyTorch baseline the brief asked for?**

Not implemented. I'd add `nn.Sequential(Linear, ReLU, Linear)` timed with CUDA events.
PyTorch would use cuBLAS and would almost certainly beat `mm_thin` at large N.

### Checkpointing (PA3 part 2)

**Q24. What exactly did you checkpoint, and why A2 and Z4?**

The assignment prescribed them. Z4 is the network output, which MSE needs. A2 is the
midpoint, so the network splits into two 2-layer segments. Z1, A1, Z2, Z3 and A3 are
recomputed.

**Q25. Explain the +26% backward cost.**

The recompute is layer 3 from A2, then layers 1–2 from X. In units of 4096³ GEMMs, the
layers are 0.25 + 1 + 1 + 0.25 = 2.5 for the forward pass. The recompute is
1 + 0.25 + 1 = 2.25. So 2.25/2.5 × 39.5 ms = 35.5 ms predicted. The measured overhead is
35.54 ms. It's exactly the recomputed GEMMs.

**Q26. Why only 13.3% saved?**

Peak includes:
* 320 MiB of weights and gradients,
* 208 MiB of dZ buffers,
* 32 MiB of X/Y.

Checkpointing can't touch any of that. On activations alone it's 400 → 272 MiB, or
32%. The floor is the arena, which must hold segment A: Z1, A1, Z2 = 192 MiB.

**Q27. Could you have saved more memory without recomputation?** *(hostile, good one)*

Yes. ReLU's derivative only needs the sign, and A > 0 exactly when Z > 0. So you can
drop Z1, Z2 and Z3 entirely and use A. That saves 192 MiB at zero time cost, which is
more than checkpointing saved. Two more ideas:

* Ping-ponging the dZ buffers would save about 80+ MiB.
* Storing a 1-bit mask would be smaller still.

I followed the assignment's design, but I'd point this out.

**Q28. How did you measure peak memory?**

`cudaMemGetInfo` before allocating, and again after every buffer including the arena.
All memory is preallocated with `cudaMalloc`, so footprint equals peak. It's
deterministic. It's device-wide, so I used an idle GPU.

**Q29. How do you know the recompute is correct?**

After 103 SGD steps, both modes give identical W4 checksums to 10 digits in all three
configs. W4's update depends on recomputed A3. Over many steps, it depends on the whole
chain. It's a modest check. A per-tensor gradient comparison would be stronger.

**Q30. The first version saved 0 MB. Why?**

It gave each dropped activation its own buffer, which is the same memory as the
baseline. The saving only exists if the two recomputed segments share storage. That
works because they're never alive at the same time.

### FlashAttention (PA4)

**Q31. Explain the bank conflict and the fix.**

In the score loop, lane c reads `Kj[c*D + x]`. With D = 64, the bank is
`(64c + x) mod 32 = x mod 32` for every lane. That's 32 lanes, 1 bank, 32 different
addresses, so the access is serialized 32 ways. With a stride of 65, the bank is
`(c + x) mod 32`, which is distinct for every lane. That was 2377 → 339 ms, or 7.0×.

**Q32. Why does padding work, and what does it cost?**

Changing the stride from 64 to 65 rotates each row by one bank. It costs 32 floats per
tile. It also breaks 16-byte row alignment, so you can't use `LDS.128` on Kj rows. An
XOR swizzle keeps alignment and also removes conflicts.

**Q33. Is the 21× the bank-conflict fix?** *(hostile)*

No. The padding alone is 7×. The 21× is cumulative over three changes from the quiz
layout:
* 1.4× from grid parallelism,
* 7.0× from padding,
* 2.1× from 4 rows per warp.

The earlier versions aren't in the repo. Only the final 160 ms is reproducible.

**Q34. Why did the grid change only give 1.4×, when it took 128 blocks to 32,768?**

The bank conflict was the real limit. More blocks just meant more warps waiting on the
same serialized shared reads.

**Q35. Why 4 rows per warp?**

At 1 row per warp, each FMA needed 2 shared loads (Kj + Qi). That capped the kernel at
about 8.4 TFLOP/s, and I measured 6.5. With 4 rows, one Kj load feeds 4 FMAs. That's 5
loads per 4 FMAs, and the ceiling rises to about 13.4. I got 13.75 TFLOP/s, so I'm at
the roof again.

**Q36. Why is it still 1.5× slower than explicit PyTorch at N = 8192?**

The explicit path's two big matmuls run on cuBLAS SGEMM. That's heavily register-blocked
and not shared-bandwidth bound. Mine reads both operands from shared memory at about 1.25
loads per FMA. The same ~2× ratio shows up in PA2. I'd also say:

* The win of flash is **memory**. Explicit attention needed a 34 GB score tensor and
  had to be chunked.
* Flash attention wins on speed when attention is HBM-bound, as with FP16 tensor cores
  and long sequences. A CUDA-core FP32 kernel without register tiling doesn't reach
  that regime.

**Q37. Where's the crossover?**

The custom kernel is faster only at N ≤ 128 and slower from 256 on. Below 256, PyTorch
has a floor of about 0.11 ms from launching several kernels. So that win is launch
overhead, not attention efficiency. It never overtakes at real sizes.

**Q38. What's in registers and what's in shared memory?**

Shared memory holds Qi, Kj, Vj and S, 28.8 KB in total. Registers hold the running
max/sum per row and the partial dot products. **Correction found during prep:**
`cuobjdump` shows `STACK:64` and `LDL`/`STL` instructions. `o_acc[4][4]` is exactly 64
bytes, and it lives in local memory, because `jj` is a runtime index (D is a runtime
argument). So "on-register" is only true for the softmax statistics. The traffic is
once per KV tile, not in the inner loop, so the cost is probably small. Fix: make D a
template parameter.

**Q39. Why the shuffle reductions and the `__syncwarp`?**

The 32 lanes hold one tile row's 32 scores. Max and sum take 5 butterfly shuffles each.
Each warp writes and reads only its own rows of S. So `__syncwarp` is enough, and no
block barrier is needed.

**Q40. What's the occupancy, and what limits it?**

It's 50% theoretical. 62 registers × 256 threads, rounded to 64 per thread, gives
16,384 per block, so 4 blocks per SM. Shared memory would allow 7. Registers are the
limit. This is from the occupancy API; I couldn't measure achieved occupancy.

**Q41. What did you skip versus FlashAttention-2?**

* No causal mask, no backward pass, no FP16/BF16 or tensor cores.
* No splitting the KV loop across warps.
* No `cp.async` prefetch.
* No storing of logsumexp for backward.

### KV cache (PA1)

**Q42. Explain your KV cache implementation.**

`CausalSelfAttention.forward` concatenates the new k, v onto the cached ones
(`model.py:69-72`) and returns the extended pair. The mask is sliced to
`[Tk−Tq : Tk, :Tk]`, so new queries sit at their absolute positions (`:92`). Decode
feeds one token per step (`bench_pa1.py:51-54`). `last_only` projects only the last
position through the LM head.

**Q43. What results did you get?**

None. The implementation is complete and the weights load. **The benchmark was never
run.** I can tell you what I expect: per-token cost flat in T with the cache, versus
growing with T without it, and more memory from the cache. But I have no measured
numbers.

**Q44. Weaknesses of your KV cache?**

* `torch.cat` each step copies the whole cache, so O(T²) bytes overall and fragmentation.
  Preallocating, or vLLM-style paging, fixes it.
* No batching of different lengths.
* No eviction.

**Q45. How does the radix tree work, and does it save memory?**

Edges are compressed token runs. A node stores the KV for its full path, and any prefix
is a slice. In my benchmark, the shared prefix is encoded once and broadcast to the
batch. But `expand_kv` makes the broadcast contiguous. So it saves **prefill compute**,
not decode memory. True sharing needs paged KV with reference counts, as in SGLang.

**Q46. Any bugs in the radix tree?** *(found during prep)*

Yes. Say you insert "ABCD" and then "ABXY". The split creates a middle node "AB" with
`kv=None`. A later query "ABZ" misses, even though AB's KV is the first two positions of
ABCD's cache. Fix: on split, set `mid.kv = child.kv` with `mid.length = child.length`.
`match_prefix` already clamps `usable = min(length, matched)`. Separately, the radix
timing in `bench_pa1.py` runs after a warmup that already inserted the prefix. So it
would measure warm-cache hits only.

### General and process

**Q47. How did you work without a profiler?**

* I used first-principles counts: bank math, loads per FMA, FLOPs and bytes.
* I compared them against measured TFLOP/s.
* I used the occupancy API.
* While preparing, I also used `cuobjdump -res-usage` and SASS instruction counts,
  which need no permissions.

I label every unmeasured figure as theoretical.

**Q48. How reliable are your timings?**

* CUDA events, with warmup, 10–100 iterations depending on the benchmark.
* One sweep, no repeats, no error bars, clocks not locked.
* It's a shared machine, but on an idle, pinned GPU.

Big effects (7×, 10.9×, 51×) are safe. Effects of a few percent, like strided versus
contiguous at 1.04×, are within plausible noise.

**Q49. What would you do differently?**

* Add correctness checks to every benchmark, including streams.
* Clear outputs and check launch errors.
* Repeat runs and report variance.
* Keep each optimization step as a commit, so the flash history is reproducible.
* Not rename folders after writing the docs.
* Use ReLU's sign instead of storing Z.
* Get ncu access early from an admin.

**Q50. What's next?**

* **Matmul:** vector loads, double buffering with `cp.async`, 128×128 tiles with warp
  tiling, then TF32/BF16 tensor cores via `mma.sync`/WGMMA.
* **Flash:** a templated D, KV split across warps, prefetch, FP16 tensor cores.
* **Streams:** the one-stream column-split experiment to settle the mechanism.
* **PA1:** run the benchmark, with a preallocated cache.

---

## 8. "It didn't work" / "I didn't do that" — and how to say it

Say it first, say it plainly, then say what you learned or would do. Don't hedge.

| Topic | Honest answer | How to say it well |
|---|---|---|
| PA1 benchmark | Never run. No numbers. | "I built the KV cache and the radix prefix cache and loaded GPT-2 weights, but I never ran the benchmark sweep. So I won't quote a speedup. I can walk you through the code and what I'd expect." |
| Batch-split streams | Never helped; up to 4× slower. | "The batch split never helped. With batch 32, every stream still reads the whole weight matrix. That result taught me to ask whether the GPU is already full before reaching for streams." |
| Stream mechanism | Unproven. | "My 1.73× is real, but I can't prove it's concurrency. The evidence points to cache locality. Here's the one-line experiment that would settle it." |
| FlashAttention vs PyTorch | Never beats it at real sizes. | "My kernel is 1.5× slower than PyTorch's explicit attention at 8K. Its value is memory, not speed. The gap is the same register-blocking gap as in my matmul." |
| The "crossover" | Only in the launch-bound regime. | "It's faster below 256 tokens, but that's PyTorch's launch overhead, not my kernel being better." |
| ncu | Never used. | "Profiling counters needed root. Every occupancy figure I give is theoretical, from the occupancy API." |
| PyTorch MLP baseline | Not implemented. | "The brief asked for one and I didn't add it. cuBLAS would likely beat my thin GEMM." |
| Tensor cores | Not used anywhere. | "Everything is FP32 on CUDA cores. That's a deliberate scope choice. 48% of cuBLAS is against SGEMM, not against tensor-core GEMM." |
| Error bars | None. | "Single sweep, no repeats. I trust the big ratios, not the 4% one." |
| Flash history | Not in git. | "Only the final version is in the repo. The 3358 / 2377 / 339 ms steps are from my notes." |
| Correctness gaps | Part 1 unchecked; column split wrong at N ≤ 512; PA2 check covers a prefix. | "When I re-read my code before this interview I found…" — owning it reads as rigor. |
| Backward pass of attention | Not done. | "Forward only." |

---

## 9. Five self-test exercises

Predict first. Then read the answer.

### Exercise 1 — un-pad the key tile

In `flashattention/flash_lib.cu:55`, change `const int KS = D + 1;` to
`const int KS = D;`. Leave the host shared-memory size alone. What happens to
correctness and to the time at N = 8192?

<details><summary>Answer</summary>

**Correctness:** unchanged. The same stride is used for the store (`:107`) and the read
(`:119`). The host still allocates the D+1 size, which is harmless.

**Speed:** the score phase goes back to a 32-way conflict. Per feature, that's one Kj
read costing 32 wavefronts plus 4 broadcast Qi reads, so about 36 instead of 5. The PV
phase is unaffected (about 5). Overall, expect roughly (36+5)/(5+5) ≈ 4× slower, so
about 500–650 ms instead of 160. It's less than the original 7×, because 4 rows per warp
now amortize each Kj read. This is a model prediction, not a measurement.
</details>

### Exercise 2 — is the 1.73× concurrency or locality?

In `mlp/mlp_bench.cu:286` and `:295`, replace `st[s]` with `st[0]`. Now all four column
chunks run serially on one stream. What do you predict at N = 32768, and what does each
outcome mean?

<details><summary>Answer</summary>

If the time stays near 25–27 ms, the win is **locality** from the column decomposition
and the block order. Concurrency is irrelevant, and the resume should say "partitioning
by output column", not "stream parallelism". If it goes back to about 44 ms, concurrent
execution matters.

My prediction is the former. The baseline already has about 4 waves of blocks, and the
column-split time matches linear scaling from 16K. An even cleaner test: in a single
kernel, make `blockIdx.x` the row and `blockIdx.y` the column chunk.
</details>

### Exercise 3 — give each recomputed tensor its own buffer

In `mlp_bench.cu:456-461`, allocate `segA + segB` floats and point `rZ3`/`rA3` past
segment A instead of aliasing them. Predict the 4096-config saving and the backward time.

<details><summary>Answer</summary>

segA = 192 MiB and segB = 128 MiB, so 320 MiB. That's exactly the five dropped buffers
(Z1, A1, Z2, Z3, A3 at 64 MiB each). Peak equals the baseline, so the **saving is 0 MB**.
Backward time is unchanged, at about +26%, because the recompute still happens. You pay
the time and get nothing. This is the documented first-version bug.
</details>

### Exercise 4 — radix tree miss

With no GPU at all: create a `RadixTree`. Then `insert((1,2,3,4), kv1)`,
`insert((1,2,7,8), kv2)`, and call `match_prefix((1,2,9))`. What does it return? What
one-line change fixes it?

<details><summary>Answer</summary>

It returns `(0, None)`, a miss. The second insert splits the edge at `(1,2)` and creates
`mid` with `kv=None` (`radix_tree.py:108`). The query walks into `mid`, matches 2
tokens, and finds `mid.kv is None`. Then there's no child for token 9, so it breaks.

Fix: at the split, set `mid.kv = child.kv; mid.length = child.length`. Then
`match_prefix` computes `usable = min(child.length, matched) = 2` and returns kv1 sliced
to 2 positions. That's valid, because the first 2 positions of `(1,2,3,4)`'s cache are
exactly the cache of `(1,2)`.
</details>

### Exercise 5 — bigger matmul tile

In `matmul/matmul_bench.cu:479-483`, change `mm_block2d<64,4>` to `mm_block2d<128,8>`,
with grid `(N+127)/128` and block `(16,16)`. What happens when you build and run it?

<details><summary>Answer</summary>

It **fails to compile**. `sA` and `sB` are static `__shared__ float[128][128]`, so
2 × 64 KB = 128 KB. The static shared-memory limit is 48 KB per block ("uses too much
shared data"). To get there, you must switch to `extern __shared__`, launch with the size,
and call `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`. That's the same 48 KB
ceiling as LQ3 q4.

Even then, a 128-tile with 8×8 per thread needs 64 accumulators plus 16 operands.
That's about 100+ registers, so occupancy falls. The cuBLAS-style fix is to keep the 128
block tile with 256 threads, use a k-slice of 8–16 rather than the full 128 in shared
memory, and double-buffer.
</details>

---

## Appendix — issues found while preparing (not in any doc)

1. `SETUP.md` build commands, `tools/run_quiz.sh:38,58`, `RESULTS.md:237,248` and the
   `profile_flash.py` docstring all use pre-rename folder names. They fail as written.
2. `mlp_bench.cu:286-296`, the column split, is incorrect for N ≤ 512. The guard is
   `col < N` (full width) while the chunk is smaller than one 256-wide block. Threads
   write outside their chunk, and the last stream writes past `A1`/`Y`. compute-sanitizer
   would flag it. The N ≥ 1024 results, including the 1.73×, are unaffected.
3. Part 1 of `mlp_bench.cu` has no output check at all.
4. `matmul_bench.cu:447` checks only the first 4M elements, contradicting the comment at
   `:352`. There's no `cudaMemset(dC)` and no `cudaGetLastError` between kernels.
5. `flash_lib.cu`: `o_acc` lives in local memory (`STACK:64`, `LDL`/`STL` in the SASS).
6. `radix_tree.py:108`: a split node gets `kv=None`, so shared-prefix queries that end at
   or diverge after a split miss.
7. `bench_pa1.py:190-192`: the radix timing includes the warmup insert, so timed
   iterations are all warm hits.
8. `mlp_bench.cu:404`: W4 is scaled by `isq(H2)` instead of `isq(H3)`. This is harmless
   in all configs.
9. `RESULTS.md:33` cites `evaluator.cu:88` for the LQ3 q4 launch. In
   `prob3/q4/evaluator.cu` the launch is at line 75; 88 is probably the merged-file line.
