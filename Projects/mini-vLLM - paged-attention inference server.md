---
type: project
folder: projects/vLLMmini
repo: github.com/jayavardhan-g/mini-vLLM
dates: 2026-08-12 → 2026-08-15
status: done
kind: ML systems — inference serving
stack: [Llama-3.2-1B-Instruct, PyTorch 2.11, Triton, FastAPI, H200]
tags: [project, ml-systems, inference]
---
# mini-vLLM — paged-attention inference server from scratch

**What.** A small vLLM built from scratch: a paged KV-cache block pool, continuous batching with recompute preemption,
a radix-tree prefix cache with refcounted eviction, a **fused Triton decode kernel** reading non-contiguous blocks,
and a FastAPI front end with SSE streaming. It closes the "nanoGPT KV cache" coursework gap.

**Phases.** 0 HF baseline · 1 block manager · 2 naive paged attention (token-for-token equal to HF) · 3 scheduler ·
4 prefix cache · 5 fused decode kernel · 6 benchmarks and real vLLM · 6.5 HTTP server. **421 tests**: 349 CPU and 72 GPU, all green.

**Key results.**
- **3.82×** output throughput over the HF `generate()` baseline (2415 vs 631 tok/s), with TTFT p50 2448 → 179 ms.
  This replaces an earlier 4.4× that came from a confounded workload.
- The fused decode kernel alone gives **7.49×** over the naive gather (paired median). This is the cleanest isolated result.
- **Real vLLM 0.26.0 is 6.2× faster** than this server (paired ratio 0.16×). The gap is the unfused prefill, no CUDA graphs and the Python scheduler loop.
- The prefix cache hits **91% by turn 4** on multi-turn chats, but saves **0% of the time** below about 2k tokens, because prefill there is overhead-bound.
  At 8k context a 90% hit gives a 7.41× faster prefill.
- Paging cuts fragmentation from 16.8% to 3.9% on mixed-length batches. It is slightly worse on uniform prompts.

**Known limits.** Prefill is not chunked or fused, and a prompt above ~16k tokens OOMs because the score matrix is materialised.
A preempted request can outgrow the token budget. The server runs as one process.

**Lessons** → [[Measurement traps]] (arrival-time bug that flattered the baseline, cross-conversation prefix leak, paired ratios on a shared GPU).

**State.** Done. The envs `vllmmini` and `vllm-ref` were path-based on warm storage and are gone. Warm storage also holds
`/mnt/warm_storage/jayavardhan/mini-vllm-artifacts/` (build scratch). Natural next step: chunked and fused prefill.

**Source.** `README.md` (55 KB, per-phase write-up), `PROGRESS.md`, `SETUP.md` (two ways vLLM refuses to start on this box), `results/*.json`.
