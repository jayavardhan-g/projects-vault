---
type: study-day
day: 2
date: 2026-11-09
tags: [interview, day, mini-vllm]
---
# Day 2 (Mon Nov 9): mini-vLLM II: continuous batching and the radix prefix cache

Part of [[Interview plan (Nov 2026)]]. Repo `vLLMmini` (branch `main`). About 7 h. Builds on [[Day 01 - mini-vLLM I]].

**Code walkthrough:** [[Day 02 - code walkthrough]]: one small figure per pipeline, the exact code to read in order, and what each piece does.

**Resume bullets:**
- "…and continuous-batching scheduler…achieving **3.82×** throughput over the HuggingFace baseline." (part 2)
- "Implemented a radix-tree prefix cache to reuse shared KV states across requests, accelerating repeated prompt inference by up to **7.4×**…"

**By tonight you can:**
- explain the scheduler loop, admission, the token budget and recompute preemption, with the code;
- explain the radix cache and its refcounted LRU eviction, with the code;
- say exactly what "up to 7.4×" means;
- rebuild a toy radix cache.

## Schedule

| # | Time | Kind | What |
|---|---|---|---|
| 0 | 15 min | recall | Yesterday: KV formula, the slot formula, why paging |
| 1 | 40 min | theory | Continuous batching and preemption (A1) |
| 2 | 90 min | code | `scheduler.py`, minus its prefix-cache helpers (B1) |
| 3 | 30 min | trace | The scheduler trace (D1) |
| — | lunch | | |
| 4 | 45 min | theory | Radix trees and RadixAttention (A2) |
| 5 | 75 min | code | `prefix_cache.py` plus the scheduler's prefix helpers (B2) |
| 6 | 60 min | rebuild | A toy radix prefix cache (C) |
| 7 | 20 min | trace | The shared-prompt trace (D2) |
| 8 | 35 min | practice | The questions (E) |

## A. Theory

### A1. Continuous batching and preemption (40 min)

**Read:**
- the Anyscale blog *"How continuous batching enables 23x throughput in LLM inference while reducing p50 latency"* (2023), about 20 min;
- Orca (Yu et al., OSDI 2022) §3, *iteration-level scheduling*. Only the idea and its main figure, about 15 min.

**Concepts:**
- **Static batching.** A batch runs until its longest request ends, so new requests queue. That is your HF baseline's TTFT p50 of **2448 ms**.
- **Continuous batching.** The scheduler runs **every step**: finished requests leave, waiting ones join. TTFT p50 fell to **179 ms**, and throughput rose **3.82×** (2415 vs 631 tok/s), mostly from more requests in flight per step.
- **Token budget.** Each step's tokens (prefill + one per decode) must stay ≤ `max_num_batched_tokens`, and the blocks must fit.
- **Admission.** Admit a request only if `ceil((prompt_len + 1)/B)` blocks are free above a **watermark**, a reserve kept so running requests can still grow. Without it you get thrashing: admit, then immediately preempt.
- **Preemption.** When a running request needs a block and none is free, evict a victim (the latest admitted).
  - **Swap** copies its KV to the CPU; the cost is bytes ÷ PCIe bandwidth.
  - **Recompute** drops its blocks and re-prefills later.

  You chose **recompute** (README line 742). Prefill is cheap on a 1B model, there's no CPU buffer to manage, and vLLM defaults to it too.

### A2. Prefix caching (45 min)

**Read:**
- the LMSYS blog *"Fast and Expressive LLM Inference with RadixAttention and SGLang"* (Jan 2024), about 20 min;
- the SGLang paper (Zheng et al., 2024) §3: RadixAttention, eviction and its figure, about 25 min.

**Concepts:**
- **Radix tree.** Each root-to-node path is a token prefix. Here a node holds **one full block** of `B` token ids plus its physical block id, so matching goes block by block.
- **Only full blocks are shared.** A half-filled last block is still being written, and sharing it would need copy-on-write. A request reuses `floor(matched / B)` blocks and recomputes the rest.
- **Refcounted LRU.** Evict only **leaves** with refcount 0, least recently used first. Evicting an inner node would orphan its children.
- **Hit rate is not time saved.** README *"The cache saves 91% of the tokens and 0% of the time"* (line 286).

**Maths: why 0% below 2k and 7.41× at 8k.** Prefill time is roughly `T(n) ≈ a + b·n + c·n²`: fixed overhead `a`,
linear MLP and projection cost `b·n`, and quadratic attention cost `c·n²`. For small n, `a` dominates, so skipping
tokens saves almost nothing. For large n, `n²` dominates. Your measurements (README around line 305):

| context | no cache | 90% cached | speedup |
|---|---|---|---|
| 512 | 15.5 ms | 15.5 ms | 1.00× |
| 2048 | 49.8 ms | 17.0 ms | 2.92× |
| 8192 | 665.5 ms | 89.8 ms | **7.41×** |

"Up to 7.4×" is the 8k row. Always say the context length when you quote it.

## B. Code

### B1. The scheduler (90 min, deep)

`scheduler.py`, in this order:
1. 87 `Request`, 80 `RequestStatus`, 190 `SchedulerConfig`: the state and the knobs.
2. 262 `add_request`, 257 `max_blocks_needed`, 55 `RequestTooLargeError`.
3. **296 `schedule`**: the main loop. Running requests first, then waiting ones.
4. 322 `_schedule_running`: decode steps, growing block tables, triggering preemption.
5. 365 `_schedule_waiting`: admission under the token budget and the watermark. 509 `_blocks_to_admit`.
6. 588 `_evict_for`, 616 `_pick_victim`, 626 `_preempt`: recompute preemption.
7. 649 `update`, 688 `_finish`, 704 `abort`.

Then README *Phase 3: continuous batching* (697) and *Preemption is by recompute* (742).

### B2. The prefix cache (75 min, deep)

| File, line | Read for |
|---|---|
| `prefix_cache.py` 53 `RadixNode` (63 `is_root`, 67 `is_leaf`, 70 `touch`) | what a node stores; LRU timestamps |
| — 75 `PrefixMatch`, 136 `_blocks_of` | splitting tokens into full blocks |
| — **144 `match`** | walking the tree block by block |
| — 191 `acquire` | refcounts on the matched path |
| — **210 `insert`** | publishing blocks |
| — 247 `_evictable_leaves`, **265 `evict`**, 290 `_remove` | refcount-0 leaves in LRU order |
| — 312 `check_invariants` | the rules that must always hold |
| `scheduler.py` 443 `_match_cached_prefix`, 466 `_first_uncached_chunk`, 523 `_adopt_cached_prefix`, 544 `_publish_blocks` | how admission uses the cache |
| `block_table.py` 86 `adopt_blocks` | taking over cached blocks |

README *Phase 4: radix prefix cache* (763–984). The key sections:
- *Blocks are published while the request is still running* (775);
- *Two requests cannot share a prefix neither of them has computed* (809);
- **The correctness bug: admission mutated requests it then refused** (879). This is your best "hardest bug" story; know it cold.

## C. Rebuild it yourself (60 min)

In about 80 lines, build a toy radix cache with `match(tokens) -> blocks`, `insert(tokens, blocks)`, `release(blocks)` and
`evict(n)`, using refcounts and LRU over leaves only. Test it:
- two requests that share 2 blocks of system prompt;
- evicting while one of them still runs must not touch the shared blocks.

Then compare your version with `prefix_cache.py`.

## D. Traces

**D1. Scheduler (30 min).** Block size 4, a pool of 8 blocks, a watermark of 1 block. Requests A (prompt 5), B (prompt 9),
C (prompt 3), each generating 4 tokens. For 5 steps, write down:
- which requests are prefill and which are decode;
- every block table;
- the free list.

When does preemption happen, and who is the victim? Check against `_schedule_waiting` and `_pick_victim`.

**D2. Shared prompt (20 min).** Block size 4, and a 10-token system prompt (2 full blocks + 2 tokens).
1. Turn 1 = system + 6 user tokens: what does `insert` publish?
2. Turn 2 = turn 1 + reply + 5 new tokens: what does `match` return, how many tokens are recomputed, and which refcounts change?
3. You evict 3 blocks while turn 2 runs: which ones can go?

## E. Questions

1. Walk me through a request from arrival to its last token.
2. Static vs continuous batching: why does TTFT fall 13.7×? Where does the 3.82× come from?
3. What is the token budget, and why is it needed as well as a block limit?
4. Recompute vs swap: which did you choose, and when would you switch?
5. What is the watermark for?
6. How does the radix cache find a shared prefix? Why per block, not per token?
7. A 91% hit rate by turn 4. Why isn't it faster, and when is it?
8. What exactly is "up to 7.4×"?
9. How is eviction safe while requests run? Why leaves only?
10. Tell me about the hardest bug in this project. (*The admission mutation, README 879.*)
11. Hostile: "vLLM already exists. Why build this?"

## My notes and fumbles

- 
