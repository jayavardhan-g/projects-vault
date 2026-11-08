---
type: study-day
day: 3
date: 2026-11-09
tags: [interview, day, mini-vllm]
---
# Day 3 (Mon Nov 9): mini-vLLM II: radix prefix cache, Triton kernel, benchmarks

Part of [[Interview plan (Nov 2026)]]. Repo `vLLMmini` (branch `main`). About 7 h. Builds on [[Day 02 - mini-vLLM I]].

**Resume bullet it defends:** "Implemented a radix-tree prefix cache to reuse shared KV states across requests,
accelerating repeated prompt inference by up to **7.4×** through efficient prefix matching and cache reuse."

**By tonight you can:**
- explain the radix cache with refcounted LRU eviction, with the code;
- derive online softmax and say why the fused kernel is 7.49× faster;
- defend every benchmark number: how it was measured, what it hides, and why real vLLM is 6.2× faster.

## Schedule

| # | Time | Kind | What |
|---|---|---|---|
| 0 | 15 min | recall | Yesterday: the block-table formula, the scheduler loop, recompute preemption |
| 1 | 45 min | theory | Radix trees and RadixAttention (A1) |
| 2 | 75 min | code | `prefix_cache.py` plus the scheduler's prefix helpers (B1) |
| 3 | 60 min | rebuild | A toy radix prefix cache (C1) |
| — | lunch | | |
| 4 | 50 min | theory | GPU basics for Triton, online softmax, FlashDecoding (A2) |
| 5 | 60 min | code | `paged_attention_triton.py` (B2) |
| 6 | 20 min | rebuild | Online softmax in NumPy (C2) |
| 7 | 25 min | theory | Benchmarking a server honestly (A3) |
| 8 | 50 min | code | `benchmarks/serving.py`, and skim `server.py`, `metrics.py`, `workloads.py` (B3) |
| 9 | 20 min | trace | The shared-prompt trace (D) |
| 10 | 40 min | practice | The questions (E) |

## A. Theory

### A1. Prefix caching (block 1, 45 min)

**Read:**
- the LMSYS blog *"Fast and Expressive LLM Inference with RadixAttention and SGLang"* (Jan 2024), about 20 min;
- the SGLang paper (Zheng et al., 2024) §3: RadixAttention, the eviction policy and its figure, about 25 min.

**Concepts:**
- **Trie / radix tree.** Each path from the root spells a token prefix. Here a node holds **one full block** of `B`
  token ids plus its physical block id, so matching proceeds block by block.
- **Only full blocks are shared.** A half-filled last block is still being written, and sharing it would need copy-on-write.
  So a request reuses `floor(matched_tokens / B)` blocks and recomputes the rest.
- **Refcounted LRU.** A cached block can be evicted only if no running request uses it (refcount 0) and it is a
  **leaf**; evicting an inner node would orphan its children. Among those, evict the least recently used.
- **Hit rate is not time saved.** Your README headline: *"The cache saves 91% of the tokens and 0% of the time"* (line 286).

**Maths: why 0% below 2k and 7.41× at 8k.** Prefill time is roughly `T(n) ≈ a + b·n + c·n²`:
- `a` is fixed overhead: launches and Python;
- `b·n` comes from the MLP and projections;
- `c·n²` comes from attention.

When n is small, `a` dominates, so skipping 90% of the tokens saves almost nothing. When n is large, the n² term
dominates, so it saves a lot. Your measurements (README around line 305):

| context | no cache | 90% cached | speedup |
|---|---|---|---|
| 512 | 15.5 ms | 15.5 ms | 1.00× |
| 2048 | 49.8 ms | 17.0 ms | 2.92× |
| 8192 | 665.5 ms | 89.8 ms | **7.41×** |

The "up to 7.4×" on your resume is the 8k row. Always say the context length when you quote it.

### A2. GPU basics for Triton, and online softmax (block 4, 50 min)

**Read or watch:**
- the Triton docs tutorials *"Vector Addition"* and *"Fused Softmax"* (triton-lang.org), about 25 min. Learn `program_id`, `tl.arange`, masked `tl.load`/`tl.store`, and that one program handles one tile;
- the *Flash-Decoding for long-context inference* blog post (Tri Dao et al., Oct 2023), about 10 min;
- optional video: Umar Jamil, *"Flash Attention derived and coded from first principles with Triton"*. Watch only the online-softmax derivation (about 30 min of a long video).

**Online softmax maths** (the heart of your kernel, and of FlashAttention on the CUDA day). Process the scores in
chunks, keeping a running max `m`, a running denominator `ℓ` and a running output `acc`:

```
for each chunk of scores s_j (with values v_j):
    m_new = max(m, max_j s_j)
    ℓ     = ℓ · e^(m − m_new) + Σ_j e^(s_j − m_new)
    acc   = acc · e^(m − m_new) + Σ_j e^(s_j − m_new) · v_j
    m     = m_new
output = acc / ℓ
```

This equals softmax(s)·V exactly, without ever storing all the scores. Subtracting the max keeps `e^x` from overflowing.

**Why the fused kernel is 7.49× faster than the naive path:**
- **Naive:** `gather_kv` copies every request's scattered blocks into a new contiguous tensor (read + write all of KV), then runs attention (read KV again).
- **Fused:** each program walks the block table and reads K and V **in place** from the paged cache, with online softmax. Decode is memory-bound, so cutting bytes moved cuts time.
- The 7.49× is a *paired median* on attention alone (README *Phase 5*, line 385, and *Why the reported speedup is a paired median*, 408).
- **Decode only, on purpose** (421): prefill stays on the naive path. That is one reason real vLLM is faster.

### A3. Benchmarking a server (block 7, 25 min)

- **TTFT** = queueing + prefill. **TPOT** = time between output tokens. **Throughput** = output tokens per second over the whole run.
- **Closed-loop vs open-loop load.** Closed: N clients, each waits for its reply before sending again. Open: requests arrive on a schedule (for example Poisson), whether or not the server keeps up.
- **Paired rounds.** The GPU is shared with other users, so its speed drifts. Alternate baseline and yours in pairs, take the ratio inside each pair, and report the median ratio. Drift then cancels out.
- **Your confounds** (README *Phase 6*, line 224, onward):
  - an arrival-time bug once flattered the baseline;
  - the 4.4× → 3.82× correction;
  - why varied lengths score lower than uniform ones (249).

## B. Code

### B1. The prefix cache (block 2, 75 min, deep)

| File, line | Read for |
|---|---|
| `prefix_cache.py` 53 `RadixNode` (63 `is_root`, 67 `is_leaf`, 70 `touch`) | what a node stores; LRU timestamps |
| — 75 `PrefixMatch`, 136 `_blocks_of` | splitting tokens into full blocks |
| — **144 `match`** | walking the tree block by block |
| — 191 `acquire` | increasing refcounts on the matched path |
| — **210 `insert`** | publishing a finished request's blocks |
| — 247 `_evictable_leaves`, **265 `evict`**, 290 `_remove` | refcount-0 leaves, LRU order |
| — 312 `check_invariants` | the rules the cache must always satisfy |
| `scheduler.py` 443 `_match_cached_prefix`, 466 `_first_uncached_chunk`, 523 `_adopt_cached_prefix`, 544 `_publish_blocks` | how admission uses the cache |

README: *Phase 4: radix prefix cache* (763) through 984. The key sections:
- *Blocks are published while the request is still running* (775);
- *Two requests cannot share a prefix neither of them has computed* (809);
- *The correctness bug: admission mutated requests it then refused* (879). This is a great "hardest bug" story.

### B2. The fused kernel (block 5, 60 min, deep)

`paged_attention_triton.py`:
- 64 `_paged_attention_decode_kernel`. Find: the grid (which program handles which sequence and head); the loop over the block table; the masked loads; the online-softmax update from A2; the GQA head mapping.
- 150 `DecodeMetadata`, 164 `build_decode_metadata`: what is packed for the kernel.
- 196 `can_use_decode_kernel`, 207 `paged_attention_decode`: when it falls back.

README *The tests were checked by breaking the kernel* (437), and *Divergence from the naive path is 12 requests in 32, and it is fine* (452). Know why: bf16 rounding differs, so greedy tokens can flip on near-ties.

### B3. Benchmarks and server (block 8, 50 min)

- Deep: `benchmarks/serving.py`: 86 `run_hf`, 168 `run_ours`, **229 `paired_rounds`**, 263 `run_mode`.
- Skim: `server.py` (FastAPI, SSE streaming), `metrics.py`, `workloads.py`.
- README *Real vLLM: it is 6.2× faster than this* (338). Three reasons: no fused or chunked prefill, no CUDA graphs, a Python scheduler loop.
- README *The first request paid 17 seconds for everyone else* (137). This is a good story too.

## C. Rebuild it yourself

**C1. Toy radix cache (60 min).** In about 80 lines, build `match(tokens) -> blocks`, `insert(tokens, blocks)`, `release(blocks)`
and `evict(n)`, with refcounts and LRU over leaves only. Test it:
- two requests that share 2 blocks of system prompt;
- evicting while one of them still runs must not touch the shared blocks.

**C2. Online softmax in NumPy (20 min).** Random `s` (length 1000) and `V` (1000×64). Compute in chunks of 16 with the update
rule, and assert it matches `softmax(s) @ V` within 1e-6. Then remove the max subtraction and watch it overflow with scores near 1000.

## D. Trace (block 9, 20 min)

Block size 4. The system prompt is 10 tokens (2 full blocks + 2 tokens).
1. Turn 1 = system + 6 user tokens: what does `insert` publish?
2. Turn 2 = turn 1 + reply + 5 new tokens: what does `match` return, how many tokens are recomputed, and which refcounts change?
3. Now evict 3 blocks while turn 2 is running: which blocks can go?

## E. Questions

1. How does the radix cache find a shared prefix? Why per block, not per token?
2. Your hit rate is 91% by turn 4. Why doesn't that make things faster? When does it?
3. What exactly is "up to 7.4×"? (*The prefill of an 8k context with 90% cached, vs uncached.*)
4. How is eviction safe while requests are running? Why leaves only?
5. Tell me about the hardest bug in the prefix cache. (*Admission mutated requests it then refused, README 879.*)
6. Explain online softmax. Why is it exact?
7. Why is your fused kernel 7.49× faster? Is decode compute- or memory-bound?
8. Why Triton and not CUDA C++? (*Faster to iterate, automatic tiling within a program, good enough for a decode kernel. CUDA gives control over shared memory and warps.*)
9. Why does the fused path disagree with the naive path on 12 of 32 requests, and why is that OK?
10. How did you make benchmark numbers trustworthy on a shared GPU?
11. Real vLLM is 6.2× faster. What would you build next, in order? (*Chunked and fused prefill, CUDA graphs, then a C++ or async scheduler.*)
12. Hostile: "A 90% hit rate and 0% time saved. Isn't the cache useless?"

## My notes and fumbles

- 
