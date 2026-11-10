---
type: study-day
day: 6
date: 2026-11-13
tags: [interview, day, cuda]
---
# Day 6 (Fri Nov 13): CUDA II: streams, checkpointing (PA3) and FlashAttention (PA4)

Part of [[Interview plan (Nov 2026)]]. Repo `sysml` (branch `main`). About 7 h. Builds on [[Day 05 - CUDA I]]. Prep pack:
[[CUDA kernels - prep]] (stories S1–S6 are today's).

**Code walkthrough:** [[Day 06 - code walkthrough]]: one small figure per pipeline, the exact code to read in order, and what each piece does.

**Resume bullet (part 2 of 2):** "Accelerated MLP inference **1.73×** using column-wise CUDA stream parallelism at large dimensions, while
reducing peak activation memory **13.3%** through gradient checkpointing."

**Fix the wording before you defend it** (prep pack §5, "Honest resume wording"):
- **13.3% is the whole training footprint** (962 → 834 MiB). **Activation memory fell 32%** (400 → 272 MiB).
- **1.73× is at width 32K only** (1.02× at 16K, slower below), against your own kernel, and its mechanism is likely cache locality, not concurrency.
- The two numbers come from **different workloads**: a 2-layer forward pass (1.73×) and a 4-layer training step (13.3%).
- Suggested wording: "Sped up a batch-32 MLP forward pass 1.73× at width 32K by partitioning output columns across 4 CUDA streams (batch splitting never helped). Implemented gradient checkpointing for a 4-layer MLP, cutting activation memory 32% (13.3% of total training memory) for a 26% slower backward pass."

**By tonight you can:**
- explain streams and when they overlap, and why the batch split never helped;
- explain checkpointing, the shared recompute arena and the 0 MB bug;
- derive online softmax and explain FlashAttention's tiling;
- tell the bank-conflict story (7.0×) and the shared-memory-roofline story.

## Schedule

| # | Time | Kind | What |
|---|---|---|---|
| 0 | 15 min | recall | Day 5: coalescing, banks, the 51.3× and 48.2% maths |
| 1 | 30 min | theory | CUDA streams and concurrency (A1) |
| 2 | 60 min | code | Part 1: `mm_thin`, baseline, batch split, column split (B1) |
| 3 | 25 min | theory | Gradient checkpointing (A2) |
| 4 | 50 min | code | Part 2: the training step, the arena, peak memory (B2) |
| — | lunch | | |
| 5 | 60 min | theory + maths | FlashAttention and online softmax (A3) |
| 6 | 60 min | code | `flash_lib.cu`: the kernel, phase by phase (B3) |
| 7 | 15 min | skim | `kvcache/` (PA1, never run) and the lab quizzes (`prob2/`, `prob3/`) |
| 8 | 45 min | rebuild | Online-softmax attention in NumPy (C) |
| 9 | 15 min | trace | The bank-conflict arithmetic (D) |
| 10 | 45 min | practice | The questions (E), plus prep pack §8 "It didn't work" |

## A. Theory

### A1. Streams (block 1, 30 min)

**Read:** NVIDIA blog *"How to Overlap Data Transfers in CUDA C/C++"* (Mark Harris), and the "Streams" part of the
CUDA Programming Guide (§6.2.8 "Asynchronous Concurrent Execution").

- A **stream** is an ordered queue of work. Kernels in different streams *may* run concurrently, but only if the GPU has spare SMs.
- **When streams help:** small kernels that leave SMs idle, or overlapping copies with compute.
- **When they don't:** a single kernel already fills all 132 SMs. Then extra streams add launches and host syncs.
- **Your case** (story S6): with batch 32, a **batch split** gives each stream 8 rows, but each stream still reads all of W1 and W2 (N² floats vs 32·N for X). So it multiplies weight traffic and launches. A **column split** gives each stream a quarter of W. At N = 32768 the baseline already launches about 4 waves of blocks, so the 1.73× is most likely **locality** (each stream's W slice stays in L2), not concurrency. Prep pack exercise 2 has the one experiment that would settle it.

### A2. Gradient checkpointing (block 3, 25 min)

**Read:** Chen et al., *"Training Deep Nets with Sublinear Memory Cost"* (2016), §1–3. Or the PyTorch docs for `torch.utils.checkpoint` (10 min).

- Backward needs every layer's activations. **Checkpointing** keeps only a few (the checkpoints), and recomputes the rest from the nearest checkpoint during backward.
- Cost: one extra forward pass over each segment. Your measurement: **+35.5 ms on a 136 ms backward (+26.1%)**. That equals the recomputed forward FLOPs (prep pack §5).
- **Your checkpoints are A2 and Z4** (assignment-prescribed). Z4 is the output MSE needs, and A2 splits the network into two 2-layer segments.
- **The key trick** (story S3): backward runs layer 4 → 1. So segment B (Z3, A3) is fully consumed before segment A (Z1, A1, Z2) is rebuilt. One arena of `max(2·n3, 2·n1+n2)` floats serves both. Separate buffers per recomputed tensor save **nothing**, which was the first version's 0 MB result.

### A3. FlashAttention (block 5, 60 min)

**Read:** Dao et al., *FlashAttention* (2022), §1–3.1 (Algorithm 1). Optional: the FlashAttention-2 paper's §3.1 (the work-partitioning change).
**Video:** Umar Jamil, *"Flash Attention derived and coded from first principles"* (YouTube). Watch the online-softmax part.

**Online softmax, derive on paper.** Keep for each query row a running max `m`, a running sum `l`, and an output accumulator `o`.
For a new tile of scores `s`:
- `m_new = max(m, max(s))`;
- `l_new = l·e^{m − m_new} + Σ e^{s − m_new}`;
- `o_new = o·e^{m − m_new} + Σ e^{s − m_new} · v`;
- at the end, output = `o / l`.

This equals the exact softmax, because every term was rescaled to the same final max.

**Why it matters:** standard attention writes the N×N score matrix to HBM. At N = 8192, B = 8, 16 heads, that is
8·16·8192²·4 B = **34.4 GB**. That's why your PyTorch baseline is batch-chunked. FlashAttention keeps scores in shared memory and never writes them out.
Its value is **memory, not raw speed**: your kernel is 1.5× slower than PyTorch's explicit path at 8K (prep pack §8).

## B. Code

### B1. Part 1: streams (block 2, 60 min, deep)

`mlp/mlp_bench.cu`:
- **47 `mm_thin<BN,BK>`**: one block = one batch row × 256 output columns (comment at 37–45). Why not square tiles at B = 32? (*Square tiles waste 24 of every 32 rows once the batch is split 4 ways.*)
- **198 `part1`**: the 2-layer forward pass, B = 32, N = 32 … 32768.
  - **254** baseline: 2 launches per iteration, no host sync;
  - **262** batch split over 4 streams;
  - **280** column split: each stream owns a quarter of the output columns. There's a host barrier between the layers, because layer 2 needs all of A1 (276–278, 289);
  - **301** `measure`: 3 warm-up, 20 timed iterations.
- Note: the stream variants call `cudaStreamSynchronize` inside the timed loop (273, 289, 298); the baseline doesn't. **Part 1 has no correctness check**, and the column split is wrong for N ≤ 512 (not at 32768).

### B2. Part 2: checkpointing (block 4, 50 min, deep)

- **352 `part2_run`**: a 4-layer MLP training step, run 100 times.
- 396–404: weights scaled by `1/sqrt(fan_in)` (story S4: otherwise NaN, and the checksum check is useless).
- 411–420: the activation buffers. **441–462: the recompute arena.**
- **464** forward; **487** backward, with `mm_tiled` (90), `mm_tiled_AT` (109) and `mm_tiled_BT` (128). Why is backward 3.4× the forward instead of about 2×? (*The transposed-operand kernels load uncoalesced.*)
- 513: the ordering constraint (every use of z3/a3 before the segment-A rebuild).
- **548**: peak memory = the `cudaMemGetInfo` delta after all buffers are allocated. Why is that a fair "peak" here? (*Everything is a `cudaMalloc` made in advance, so footprint = peak.*)
- **572**: equality check = the sum of W4 after the SGD steps.

### B3. FlashAttention (block 6, 60 min, deep)

Follow prep pack §2 "End-to-end trace at N = 8192" side by side with `flashattention/flash_lib.cu`:
- **189 `flash_attention_forward`** (host): rejects D > 128; Tr = Tc = 256; shared memory 28,800 B; grid `(Tr=256, NH=16, B=8)` = 32,768 blocks of 256 threads.
- **47 `flash_forward_kernel`**: the block owns 32 query rows; warp `w` owns rows `4w … 4w+3`.
  - **55**: the K tile's row stride is **D+1 = 65**. Today's key line.
  - 81–85: stage Q. **98**: loop over KV tiles; 103–110: load K (stride 65) and V.
  - **118–126 scores**: lane `c` owns key `c`; one `Kj` read feeds 4 FMAs (4 query rows).
  - **130–152 online softmax**: tile max and sum via 5 `__shfl_xor_sync` steps each; rescale factors.
  - **157–172 PV**: lane owns features `x` and `x+32`. 175–183: divide by `l` and write O.
- `bench_attention.py`: 46 `custom_flash` (ctypes into the `.so`), 58 `pytorch_standard`, 77 `pytorch_fused`, 140–143 the batch chunking.

## C. Rebuild it yourself (block 8, 45 min)

In NumPy, about 50 lines:
1. `attention(Q, K, V)` the standard way (full score matrix, softmax, multiply).
2. `flash_attention(Q, K, V, Bc=32)`: loop over key tiles, keeping `m`, `l`, `o` per query row exactly as in A3. Never build the N×N matrix.
3. Check `np.allclose` for N = 1000 (not a multiple of 32, so you need the tail tile).
4. Break it on purpose: drop the `e^{m − m_new}` rescale of `o` and watch the error. Then seed `m` with 0 instead of −∞ on a row whose scores are all very negative. (Story S5 has the real NaN bug: `cudaMemset(0xFF)` is NaN, not −∞.)

## D. Trace (block 9, 15 min)

32 banks, 4-byte words, bank = `address mod 32`. In the score loop, lane `c` reads `Kj[c*stride + x]` for a fixed `x`.
- With stride 64: list the banks for lanes 0, 1, 2, 31. What is the conflict degree?
- With stride 65: the same. Why does padding by one fix it?
- Why doesn't the same stride break correctness? (*Store and load use the same stride.*)
- Result to quote: 2377 → 339 ms, **7.0×** from one line.

## E. Questions

1. What is a CUDA stream? When do two streams actually run at the same time?
2. Why did the batch split never help? Why did the column split help only at 32K?
3. Can you prove the 1.73× comes from concurrency? (*No. The evidence points to locality; here's the experiment.*)
4. Explain gradient checkpointing. What did you checkpoint, and why those tensors?
5. Your first checkpointing version saved 0 MB. Why? (*Story S3.*)
6. Is 13.3% "peak activation memory"? (*No: 13.3% total, 32% activations. Say it first.*)
7. How did you show checkpointing didn't change training? (*Weight checksums match to 10 digits.*)
8. Derive online softmax. Why is it exact?
9. What does FlashAttention save: time or memory? Is your kernel faster than PyTorch? (*No: 160 vs 105 ms explicit, 63 ms fused.*)
10. Tell me about the bank conflict. (*Story S1: 7.0×.*)
11. After the padding fix, what capped the kernel? (*Story S2: the shared-memory roofline; register blocking over 4 query rows, 339 → 160 ms.*)
12. Why ctypes and not a PyTorch extension? What did that cost you?
13. Hostile: "You never ran the profiler. How do you know any of this?" (*ncu needed root. Occupancy is theoretical, from the occupancy API. Bank and roofline reasoning are by hand, and the measured times match them.*)

## My notes and fumbles

- 
