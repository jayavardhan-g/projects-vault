---
type: study-day
day: 3
date: 2026-11-10
tags: [interview, day, mini-vllm]
---
# Day 3 (Tue Nov 10): mini-vLLM III: the Triton kernel, benchmarks, server, and a mock

Part of [[Interview plan (Nov 2026)]]. Repo `vLLMmini` (branch `main`). About 6.5 h. Builds on [[Day 02 - mini-vLLM II]].

**Code walkthrough:** [[Day 03 - code walkthrough]]: one small figure per pipeline, the exact code to read in order, and what each piece does.

**By tonight you can:**
- derive online softmax and explain why the fused kernel is 7.49× faster;
- defend every benchmark number: how it was measured, what it hides, and why real vLLM is 6.2× faster;
- answer every mini-vLLM question from Days 1–3 out loud.

## Schedule

| # | Time | Kind | What |
|---|---|---|---|
| 0 | 15 min | recall | Days 1–2: the slot formula, the scheduler loop, the radix cache, the 7.41× table |
| 1 | 50 min | theory | GPU basics for Triton, online softmax, FlashDecoding (A1) |
| 2 | 60 min | code | `paged_attention_triton.py` (B1) |
| 3 | 20 min | rebuild | Online softmax in NumPy (C) |
| 4 | 25 min | skim | `paged_attention.py`, `engine.py`, `model/runner.py` (B2) |
| — | lunch | | |
| 5 | 25 min | theory | Benchmarking a server honestly (A2) |
| 6 | 50 min | code | `benchmarks/serving.py` (B3) |
| 7 | 30 min | skim | `server.py`, `metrics.py`, `workloads.py`, plus README *Phase 6.5* (B4) |
| 8 | 20 min | read | README *Real vLLM: it is 6.2× faster than this* (338) and *Known limitations* (985) |
| 9 | 50 min | practice | **Self-quiz:** answer every question in sections E of Days 1–3 out loud, without notes. Mark each strong, OK or weak |
| 10 | 30 min | fix | Re-read the code and theory behind your weak answers; add them to *Weak topics* in the plan |

## A. Theory

### A1. GPU basics for Triton, and online softmax (50 min)

**Read or watch:**
- the Triton docs tutorials *"Vector Addition"* and *"Fused Softmax"* (triton-lang.org), about 25 min. Learn `program_id`, `tl.arange`, masked `tl.load`/`tl.store`, and that one program handles one tile;
- the *Flash-Decoding for long-context inference* blog post (Tri Dao et al., Oct 2023), about 10 min;
- optional video: Umar Jamil, *"Flash Attention derived and coded from first principles with Triton"*. Watch only the online-softmax derivation (about 30 min of a long video).

**Online softmax.** Process the scores in chunks, keeping a running max `m`, denominator `ℓ` and output `acc`:

```
for each chunk of scores s_j (with values v_j):
    m_new = max(m, max_j s_j)
    ℓ     = ℓ · e^(m − m_new) + Σ_j e^(s_j − m_new)
    acc   = acc · e^(m − m_new) + Σ_j e^(s_j − m_new) · v_j
    m     = m_new
output = acc / ℓ
```

This equals softmax(s)·V exactly, without storing all the scores; subtracting the max prevents overflow. You will see it
again on the CUDA day, in FlashAttention.

**Why the fused kernel is 7.49× faster:**
- **Naive:** `gather_kv` copies the scattered blocks into a new contiguous tensor (read + write all of KV), then runs attention (read KV again).
- **Fused:** each program walks the block table and reads K and V in place, with online softmax. Decode is memory-bound, so fewer bytes moved means less time.
- **Measurement:** the 7.49× is a *paired median* on attention alone (README *Phase 5*, line 385; *Why the reported speedup is a paired median*, 408).
- **Decode only, on purpose** (421): prefill stays naive. That is one reason real vLLM is faster.

### A2. Benchmarking a server (25 min)

- **TTFT** = queueing + prefill. **TPOT** = time between output tokens. **Throughput** = output tokens per second over the whole run.
- **Closed loop:** N clients, each waits for its reply before sending again. **Open loop:** requests arrive on a schedule (for example Poisson).
- **Paired rounds.** The GPU is shared, so its speed drifts. Alternate baseline and yours in pairs, take the ratio inside each pair, and report the median. Drift then cancels.
- **Your confounds** (README *Phase 6*, line 224):
  - an arrival-time bug once flattered the baseline, so the 4.4× was corrected to **3.82×**;
  - varied lengths score lower than uniform ones (249).

## B. Code

### B1. The fused kernel (60 min, deep)

`paged_attention_triton.py`:
- **64 `_paged_attention_decode_kernel`.** Find: the grid (which program handles which sequence and head), the loop over the block table, the masked loads, the online-softmax update, and how GQA maps 32 query heads onto 8 KV heads.
- 150 `DecodeMetadata`, 164 `build_decode_metadata`.
- 196 `can_use_decode_kernel`, 207 `paged_attention_decode`: when it falls back to the naive path.

README:
- *The tests were checked by breaking the kernel* (437);
- *Divergence from the naive path is 12 requests in 32, and it is fine* (452). bf16 rounding differs, so greedy tokens can flip on near-ties.

### B2. Skim (25 min)

- `paged_attention.py`: 36 `write_kv_to_cache`, 68 `gather_kv`, 172 `paged_attention` (the naive reference).
- `engine.py` 111 `step`; `model/runner.py` 81 `_run_step`.
- README *Phase 2* (642) and *The bug this phase actually found* (685).

### B3. Benchmarks (50 min, deep)

`benchmarks/serving.py`: 86 `run_hf`, 168 `run_ours`, **229 `paired_rounds`**, 263 `run_mode`. How is each arrival
generated? Where is the ratio computed?

### B4. Server (30 min, skim)

- `server.py` (FastAPI, SSE streaming), `metrics.py`, `workloads.py`.
- README *Phase 6.5* (104), especially *The first request paid 17 seconds for everyone else* (137). This is a good story.

## C. Rebuild it yourself (20 min)

Make random `s` (length 1000) and `V` (1000×64). Compute in chunks of 16 with the update rule, and assert it matches
`softmax(s) @ V` within 1e-6. Then remove the max subtraction and watch it overflow with scores near 1000.

## E. Questions (for today's self-quiz)

1. Explain online softmax. Why is it exact?
2. Why is the fused kernel 7.49× faster? Is decode compute- or memory-bound?
3. Why Triton and not CUDA C++? (*Faster to iterate, automatic tiling within a program, good enough for a decode kernel. CUDA gives control over shared memory and warps.*)
4. Why does the fused path disagree with the naive one on 12 of 32 requests, and why is that OK?
5. How did you make benchmark numbers trustworthy on a shared GPU?
6. Real vLLM is 6.2× faster. What would you build next, in order? (*Chunked and fused prefill, CUDA graphs, then a C++ or async scheduler.*)
7. What breaks above 16k tokens, and why? (*Prefill materialises the full score matrix.*)
8. Hostile: "A 90% hit rate and 0% time saved. Isn't the cache useless?"

## My notes and fumbles

- 
