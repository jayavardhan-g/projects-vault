---
type: study-day
day: 2
date: 2026-11-08
tags: [interview, day, mini-vllm]
---
# Day 2 (Sun Nov 8): mini-vLLM I: paged KV cache and continuous batching

Part of [[Interview plan (Nov 2026)]]. Repo `vLLMmini` (branch `main`). About 7 h.

**Resume bullet it defends:** "Built a paged KV-cache memory manager and continuous-batching scheduler for LLM inference,
reducing GPU memory fragmentation by **4.3×** and achieving **3.82×** throughput over the HuggingFace baseline."

**By tonight you can:**
- draw the architecture: request → scheduler → block tables → paged KV cache → runner;
- compute KV memory and the number of blocks from first principles;
- explain fragmentation, continuous batching and recompute preemption, with the code lines that implement them;
- rebuild the block manager yourself.

## Schedule

| # | Time | Kind | What |
|---|---|---|---|
| 0 | 15 min | recall | Day 1 recap ([[Day 01 - LLM inference basics]]): KV formula, prefill vs decode, the roofline |
| 1 | 30 min | read | The README for the big picture: *Architecture* (line 14), *Model* (466), *The Phase 0 baseline* (554) |
| 2 | 60 min | theory | PagedAttention (section A below) |
| 3 | 75 min | code | `block_allocator.py`, `block_table.py`, `kv_cache.py` (section B1) |
| 4 | 75 min | rebuild | Write your own block manager (section C) |
| — | lunch | | |
| 5 | 40 min | theory | Continuous batching and preemption (section A2) |
| 6 | 90 min | code | `scheduler.py` (section B2) |
| 7 | 25 min | skim | `paged_attention.py`, `engine.py`, `model/runner.py` (section B3) |
| 8 | 30 min | trace | The scheduler trace on paper (section D) |
| 9 | 40 min | practice | The questions out loud (section E); write your fumbles at the bottom |

## A. Theory

### A1. PagedAttention (block 2, 60 min)

**Read:** Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP 2023): §1, §3 (the memory
problem), §4.1–4.3 (PagedAttention, the KV manager, decoding with it). Skip §4.4 onward for now.

**Watch (pick one, 30 min):**
- YouTube: *"Fast LLM Serving with vLLM and PagedAttention"*, the Anyscale talk by the vLLM authors;
- or read the vLLM launch blog post, *"vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention"* (blog.vllm.ai, June 2023).

**Concepts to own:**
- **Why contiguous KV wastes memory.** Without paging, each request reserves `max_len` slots up front. Waste = reserved − used: **internal fragmentation** (inside a reservation) plus **external fragmentation** (free gaps too small to use).
- **Paging.** Split the cache into fixed blocks of `B` tokens. A per-request **block table** maps logical block *i* to a physical block id, like an OS page table. Memory grows one block at a time.
- **Bounded waste.** At most `B − 1` empty slots per sequence, in the last block only. On average about `(B − 1)/2`.
- **Refcounts.** Several requests can point at the same physical block (shared prefixes, tomorrow). A block is freed only when its refcount reaches 0.

**Maths, worked for your model (Llama-3.2-1B, bf16, B = 16):**
- KV bytes per token = 2 (K,V) × 16 layers × 8 KV heads × 64 head_dim × 2 bytes = **32,768 B = 32 KiB**.
- Bytes per block = 16 × 32 KiB = **512 KiB** (all layers together).
- Blocks from a budget: with 40 GB free, `floor(40·10⁹ / 524,288)` ≈ **76,000 blocks** ≈ 1.2M tokens.
- Slot of token position `t` in a request: `block_id = table[t // B]`, `offset = t % B`, `slot = block_id·B + offset`.
  This is `BlockTable.slot_mapping`.
- **Your 4.3×:** padding waste on the `independent` workload went from 16.8% of allocated KV slots to 3.9% (README
  *Fragmentation*, line 321). 16.8 / 3.9 ≈ 4.3. Know that it is *slightly worse* on uniform prompts, and why: with
  equal lengths, contiguous allocation wastes nothing, while paging still wastes part of the last block.

### A2. Continuous batching and preemption (block 5, 40 min)

**Read:**
- the Anyscale blog *"How continuous batching enables 23x throughput in LLM inference while reducing p50 latency"* (2023), about 20 min;
- Orca (Yu et al., OSDI 2022) §3: *iteration-level scheduling*. Read only the idea and Figure 3 or 4, about 15 min.

**Concepts:**
- **Static batching.** A batch runs until its longest request finishes. Short requests wait, and new ones queue.
  That is why your HF baseline's TTFT p50 was **2448 ms**.
- **Continuous (iteration-level) batching.** The scheduler runs **every step**. Finished requests leave and waiting
  ones join at the next step. Your TTFT p50 fell to **179 ms**.
- **Token budget.** Each step's tokens (prefill chunks + one per decode) must stay ≤ `max_num_batched_tokens`, and
  the KV blocks needed must fit.
- **Admission.** Admit a waiting request only if `ceil((prompt_len + 1)/B)` blocks are free, above a **watermark**
  kept in reserve so running requests can still grow.
- **Preemption.** When a running request needs a new block and none is free, evict a victim (the latest admitted). Two options:
  - **swap**: copy its KV to CPU and back later. The cost is bytes ÷ PCIe bandwidth.
  - **recompute**: drop its blocks and re-prefill later. The cost is one prefill of prompt + generated tokens.

  You chose **recompute** (README *Preemption is by recompute*, line 742). On a 1B model, prefill is cheap, it needs no CPU
  buffer management, and vLLM also defaults to recompute.

## B. Code

### B1. The memory manager (block 3, 75 min, deep)

| File, line | Read for |
|---|---|
| `block_allocator.py` 37 `BlockAllocator` | the free list and refcount array |
| — 70 `allocate`, 80 `allocate_many` | why `allocate_many` is all-or-nothing (no partial allocation leaks) |
| — 96 `incref`, 111 `free`, 121 `free_many` | a block returns to the free list only at refcount 0 |
| — 144 `check_invariants` | what "consistent" means: every block is free XOR has refcount > 0 |
| `block_table.py` 75 `blocks_needed_for` | ceiling division |
| — 106 `append_tokens` | when a new block is taken |
| — 128 `locate`, 137 `slot_index`, 147 `slot_mapping` | the formula from A1 |
| — 66 `num_wasted_slots` | the fragmentation metric behind 16.8% → 3.9% |
| — 86 `adopt_blocks` | (it matters tomorrow for the prefix cache) |
| `kv_cache.py` 34 `KVCacheConfig`: 80 `bytes_per_token`, 85 `bytes_per_block`, 88 `num_blocks_for_budget` | match them against your maths in A1 |
| — 98 `blocks_for_free_memory`, 116 `PagedKVCache.__init__`, 166 `k_flat` | the tensor shape of the cache, and why "flat" |

Then read the README's *Phase 1: paged memory manager* (line 602).

### B2. The scheduler (block 6, 90 min, deep)

`scheduler.py`: read in this order.
1. 87 `Request`, 80 `RequestStatus`, 190 `SchedulerConfig`: the state and the knobs.
2. 262 `add_request`, 257 `max_blocks_needed`: when is a request *too large* (55 `RequestTooLargeError`)?
3. **296 `schedule`**: the main loop. Running requests first, then waiting ones.
4. 322 `_schedule_running`: decode steps, growing block tables, triggering preemption.
5. 365 `_schedule_waiting`: admission under the token budget and the watermark. 509 `_blocks_to_admit`.
6. 588 `_evict_for`, 616 `_pick_victim`, 626 `_preempt`: recompute preemption.
7. 649 `update`, 688 `_finish`, 704 `abort`: after the model step.

**Skip for today:** 443 `_match_cached_prefix`, 466 `_first_uncached_chunk`, 523 `_adopt_cached_prefix`,
544 `_publish_blocks` (the prefix cache, tomorrow).

Then read the README's *Phase 3: continuous batching* (697) and *Preemption is by recompute* (742).

### B3. Skim (block 7, 25 min)

- `paged_attention.py`: 36 `write_kv_to_cache`, 68 `gather_kv`, 172 `paged_attention` (the naive reference: gather the blocks into contiguous tensors, then do normal attention).
- `engine.py`: 111 `step` (one scheduler + model iteration).
- `model/runner.py`: 81 `_run_step`.

README *Phase 2* (642), and *The bug this phase actually found* (685).

## C. Rebuild it yourself (block 4, 75 min)

Without looking at the code, write `toy_paging.py` (about 100 lines of pure Python):
1. `BlockAllocator(num_blocks)` with `allocate()`, `allocate_many(n)` (all or nothing), `incref(b)`, `free(b)`; raise when out of blocks.
2. `BlockTable(block_size, allocator)` with `append_tokens(n)`, `slot_mapping()` and `num_wasted_slots()`.
3. A loop: 3 requests of different lengths, each adding 1 token per step. Print the wasted slots each step,
   and compare against contiguous allocation of `max_len`.

**Check:** point `tests/test_block_allocator.py` and `tests/test_block_table.py` at your module (change the import)
and run `pytest -q` on your laptop (CPU only). Each failing test is a concept you missed.

## D. Trace on paper (block 8, 30 min)

Use small numbers so you can do it by hand: block size **4**, a pool of **8 blocks**, watermark 1 block, and requests
A (prompt 5), B (prompt 9), C (prompt 3), each generating 4 tokens. For 5 steps, write:
- which requests are prefill or decode;
- every block table;
- the free list.

At which step does someone get preempted, and who? Check your answers against `_schedule_waiting` and `_pick_victim`.

## E. Questions (block 9, out loud)

1. Walk me through what happens to a request from arrival to its last token.
2. Why is paging needed at all? Why not allocate KV for `max_len` up front?
3. How did you measure fragmentation, and why is paging slightly *worse* on uniform prompts?
4. Block size 16: what changes at 1 and at 256? (*Small: tiny waste, but a bigger block table and less efficient kernels. Large: more waste per sequence.*)
5. Static vs continuous batching: why does TTFT fall 13.7×?
6. What is the token budget, and why do you need it as well as a block limit?
7. Recompute vs swap preemption: which did you choose, and when would you switch?
8. What is the watermark for? What happens without it? (*Thrashing: admit, then immediately preempt.*)
9. How does a block become free when two requests share it?
10. Where does the 3.82× come from: memory, batching, or kernels? (*Mostly batching: more concurrent requests per step. The fused kernel is a separate 7.49× on attention alone.*)
11. What breaks above about 16k tokens, and why? (*Prefill materialises the full score matrix.*)
12. Hostile: "vLLM already exists. Why build this?"

## My notes and fumbles

- 
