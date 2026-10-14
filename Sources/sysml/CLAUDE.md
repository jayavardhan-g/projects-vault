# Deep Learning Systems Optimization: Execution & Benchmarking Guide

This document outlines the required execution steps to generate the performance metrics, baselines, and improvements for the Deep Learning Systems Optimization project. You must run the associated Python and CUDA files to gather the required data.

---

## Phase 1: Transformer Inference & KV Caching (PA1)

**Goal:** Measure the impact of KV caching and shared-prompt optimization on auto-regressive text generation.

### Execution Steps
1.  **Baseline (No KV Cache):**
    *   Run inference using the standard, unmodified `sample.py` (ensure attention flash flag is `False`).
    *   **Record:** Total inference time, per-token inference time, and peak memory usage for varying input prompt sizes and output generation lengths.
2.  **Implementation 1 (Standard KV Caching):**
    *   Run your updated code where the generation loop and transformer blocks return and pass the accumulated KV cache.
    *   **Record:** The same time and memory metrics as the baseline.
3.  **Implementation 2 (Radix Tree Shared Prompts):**
    *   Run your optimized code passing a batch of prompts with a shared prefix.
    *   **Record:** Total batch inference time and memory usage.

### Required Analysis
*   Compare the execution times and memory consumption between the baseline and KV Cache implementations. Calculate the percentage improvement in speed and reduction in memory for large prompts.
*   Calculate the time saved by reusing the shared prefix KV cache via the Radix Tree compared to processing the batch independently.

---

## Phase 2: CUDA Matrix Multiplication Optimization (PA2)

**Goal:** Profile low-level memory and compute optimizations for $N 	imes N$ matrix multiplication ($C = A 	imes B$).

### Execution Steps
For all steps, vary matrix size $N$ from $1024$ to $32K$ (multiplying by 2).

1.  **CPU Baseline:** Run the naive triple-loop CPU implementation (with `-O3`). Record total execution time.
2.  **Naive GPU Kernel:** Run the unoptimized $32 	imes 32$ thread-block CUDA kernel. Record kernel execution time (`nvprof`/`ncu`) and compute Achieved TFLOP/s.
3.  **Coalesced GPU Kernel:** Run the kernel optimized for contiguous warp access. Record time and TFLOP/s.
4.  **Tiled GPU Kernels (Shared Memory):** Run the implementations utilizing shared memory tiles (1-to-1, strided, and contiguous). Record time and TFLOP/s.
5.  **cuBLAS Upper Bound:** Run the `cublasSgemm` implementation. Record time and TFLOP/s.

### Required Analysis
*   Plot Execution Time vs. $N$ and Achieved TFLOP/s vs. $N$ for all implementations on the same graphs.
*   Calculate the theoretical Peak TFLOP/s of your specific GPU.
*   Record how close (in percentage) your best custom tiled kernel gets to the `cuBLAS` baseline and the theoretical hardware peak.

---

## Phase 3: MLP, CUDA Streams & Gradient Checkpointing (PA3)

**Goal:** Optimize forward pass multiprocessor utilization and manage the compute vs. memory trade-off in the backward pass.

### Execution Steps
1.  **Forward Pass Optimization:**
    *   **Baseline:** Run the standard batched forward pass for the 2-layer MLP ($B=32$, varying $N$ from 32 to 32K). Record GPU kernel execution time using CUDA events.
    *   **Streams:** Run the implementation split into 4 concurrent CUDA streams. Record execution time.
    *   **PyTorch Baseline:** Run the standard PyTorch `nn.Module` forward pass. Record execution time.
2.  **Gradient Checkpointing (Backward Pass):**
    *   **Baseline:** Run the standard 4-layer MLP training loop. Use `cudaMemGetInfo()` to record peak GPU memory consumption and CUDA events to record average forward and backward pass times over 100 iterations.
    *   **Checkpointed:** Run the memory-optimized version (retaining only $A_2$ and $Z_4$). Record memory consumption and execution times.

### Required Analysis
*   Compare the forward pass execution times between the baseline, streamed, and PyTorch implementations.
*   Calculate the exact GPU memory saved (in MB/GB) by utilizing Gradient Checkpointing.
*   Calculate the exact runtime overhead (in ms) added to the backward pass due to activation recomputation.

---

## Phase 4: FlashAttention Custom Kernel (PA4)

**Goal:** Implement hardware-aware scaled dot-product attention and compare it against standard materialization.

### Execution Steps
For all steps, fix $B=8, N_H=16, D=64$ and vary sequence length $N \in \{32, 64, 128 \dots 8192\}$. Run each measurement for at least 10 iterations.

1.  **PyTorch Standard Attention (Correctness Baseline):**
    *   Run the explicit $N 	imes N$ matrix materialization using `torch.matmul` and `torch.softmax`.
    *   **Record:** Runtime (ms) vs. $N$.
2.  **Custom CUDA FlashAttention:**
    *   Run your custom kernel utilizing shared memory tiling ($B_r=32, B_c=32$) and on-register online softmax.
    *   Verify numerical consistency against the baseline (max absolute difference $< 10^{-3}$).
    *   **Record:** Runtime (ms) vs. $N$, achieved occupancy, shared memory usage per block, and global memory throughput (using `ncu`/`nvprof`).
3.  **PyTorch Fused Attention (Upper Bound):**
    *   Run PyTorch's built-in fused attention.
    *   **Record:** Runtime (ms) vs. $N$.

### Required Analysis
*   Plot Runtime (ms) vs. Sequence Length ($N$) for all three implementations.
*   Identify and record the exact sequence length ($N$) at which your custom FlashAttention kernel overtakes the standard PyTorch implementation.
