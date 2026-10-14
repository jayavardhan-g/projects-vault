---
type: project
folder: projects/sysml
repo: github.com/jayavardhan-g/CUDA-Kernel-Optimization-Efficient-ML-Inference
dates: 2026-08-15 (committed 2026-08-18)
status: done — PA1 benchmark not run
kind: coursework — CUDA / GPU
stack: [CUDA 13.0, PyTorch 2.13, H200]
tags: [project, gpu, cuda, coursework]
---
# CUDA kernels — CS794 programming assignments and lab quizzes

**What.** Course assignments for deep-learning systems (CS794), run and measured on an H200 (FP32 peak 66.91 TFLOP/s).

**Key results.**
- **Lab quizzes:** all 7 `student.cu` files pass their evaluators and are clean under `compute-sanitizer`.
- **PA2 matmul** (N=32768): naive 0.50 TFLOP/s, then coalesced **10.9×**, then shared-memory tiling, then 2D register blocking (TILE 64, R 4)
  at **25.7 TFLOP/s, 48% of cuBLAS, 38% of peak**. Every kernel is bit-exact with cuBLAS.
- **PA3 MLP and streams:** splitting the *batch* across 4 streams never helps. Splitting *output columns* is 1.73× faster at N=32768.
  Gradient checkpointing saves 11–13% of peak (32% of activations) for +24–26% backward time, with identical weights.
  The first version saved 0 MB (one buffer per dropped activation).
- **PA4 FlashAttention:** max error 7.2e-7. The custom kernel went from 3358 to 160 ms (**21×**). The big step was padding
  the K-tile stride to D+1, which removed a 32-way bank conflict (7× alone). It is faster than PyTorch only for N ≤ 128.

**Not done.** PA1 (nanoGPT KV cache and radix prefix sharing) is implemented, but its benchmark was never run, at the user's request.

**Lessons** → [[Infrastructure traps]] (`ncu` needs root here), the quiz-harness traps in `RESULTS.md`.

**State.** Done. The env `sysml` was deleted and has no registry record.

**Source.** `RESULTS.md`, `CLAUDE.md` (the assignment brief), `results/*.csv|png`.
