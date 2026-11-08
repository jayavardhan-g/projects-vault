---
type: study-day
day: 1
date: 2026-11-08
tags: [interview, day, mini-vllm]
---
# Day 1 (Sun Nov 8): foundations, then mini-vLLM paging

Part of [[Interview plan (Nov 2026)]]. Repo `vLLMmini` (branch `main`). About 6.5 h. **This is your first day of
reading**, so it starts from the foundations that everything later builds on.

**Resume bullet (part 1 of 2):** "Built a paged KV-cache memory manager and continuous-batching scheduler for LLM inference,
reducing GPU memory fragmentation by **4.3×** and achieving **3.82×** throughput over the HuggingFace baseline."
Today covers the *paged memory manager* and the **4.3×**. The scheduler and the 3.82× are tomorrow ([[Day 02 - mini-vLLM II]]).

**By tonight you can:**
- explain how one token flows through a transformer, and why decode is memory-bound;
- compute KV memory, block counts and slot indices by hand;
- explain fragmentation and paging, with the code lines that implement them;
- rebuild the block manager yourself.

## Schedule

| # | Time | Kind | What |
|---|---|---|---|
| 1 | 90 min | theory | Foundations: [[Prereq - LLM inference basics]] sections 1–6, plus the video below |
| 2 | 45 min | code | `model/llama.py`: one token through one layer (B0) |
| 3 | 15 min | practice | The prerequisite's self-check, out loud |
| — | break | | |
| 4 | 20 min | read | The README for the big picture: *Architecture* (line 14), *Model* (466), *The Phase 0 baseline* (554) |
| 5 | 60 min | theory | PagedAttention (A) |
| 6 | 75 min | code | `block_allocator.py`, `block_table.py`, `kv_cache.py` (B1) |
| 7 | 75 min | rebuild | Your own block manager (C) |
| 8 | 20 min | trace | Block tables by hand (D) |
| 9 | 25 min | practice | The questions (E); write your fumbles at the bottom |

**Foundations, block 1 (90 min).** Read [[Prereq - LLM inference basics]] sections 1–6, then:
- **Video, about 30 min:** Umar Jamil, *"LLaMA explained: KV-Cache, Rotary Positional Embedding, RMS Norm, Grouped Query Attention, SwiGLU"*. Watch the KV-cache and grouped-query-attention parts. Llama-3.2-1B, your model, uses all of these.
- **If attention itself is rusty**, read Jay Alammar's *"The Illustrated Transformer"* (blog) first, about 30 min.
- **Optional, any evening:** Horace He, *"Making Deep Learning Go Brrrr From First Principles"*: compute- vs memory- vs overhead-bound work.

## A. Theory: PagedAttention (block 5, 60 min)

**Read:** Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP 2023): §1, §3 (the memory
problem), §4.1–4.3 (PagedAttention, the KV manager, decoding with it).

**Watch or read (30 min, pick one):**
- YouTube: *"Fast LLM Serving with vLLM and PagedAttention"*, the Anyscale talk by the vLLM authors;
- the vLLM launch blog post, *"vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention"* (blog.vllm.ai, June 2023).

**Concepts:**
- **Contiguous KV wastes memory.** Without paging, each request reserves `max_len` slots. Waste = reserved − used: **internal** fragmentation (inside a reservation) plus **external** fragmentation (free gaps too small to use).
- **Paging.** Fixed blocks of `B` tokens. A per-request **block table** maps logical block *i* to a physical block id, like an OS page table. Memory grows one block at a time.
- **Bounded waste.** At most `B − 1` empty slots per sequence, in the last block only; about `(B − 1)/2` on average.
- **Refcounts.** Several requests may point at one physical block (shared prefixes, on Day 2). It is freed only at refcount 0.

**Maths for your model (Llama-3.2-1B, bf16, B = 16):**
- KV bytes per token = 2 × 16 layers × 8 KV heads × 64 head_dim × 2 B = **32 KiB**.
- Bytes per block = 16 × 32 KiB = **512 KiB** (all layers).
- Blocks from a budget: 40 GB free → `floor(40·10⁹ / 524,288)` ≈ **76,000 blocks** ≈ 1.2M tokens.
- Slot of position `t`: `block_id = table[t // B]`, `offset = t % B`, `slot = block_id·B + offset` (`BlockTable.slot_mapping`).
- **The 4.3×:** padding waste on the `independent` workload fell from **16.8%** of allocated KV slots to **3.9%** (README *Fragmentation*, line 321). It is slightly *worse* on uniform prompts: with equal lengths, contiguous allocation wastes nothing, while paging still wastes part of the last block.

## B. Code

### B0. The model (block 2, 45 min)

`model/llama.py`: follow one token through embedding → one decoder layer (RMSNorm → Q/K/V projections → RoPE →
attention with 32 query and 8 KV heads → output projection → SwiGLU MLP) → logits. Write down every tensor shape (hidden
2048, head_dim 64). Where does the code write K and V into the cache?

### B1. The memory manager (block 6, 75 min, deep)

| File, line | Read for |
|---|---|
| `block_allocator.py` 37 `BlockAllocator` | the free list and refcount array |
| — 70 `allocate`, 80 `allocate_many` | why `allocate_many` is all-or-nothing (no partial allocation leaks) |
| — 96 `incref`, 111 `free`, 121 `free_many` | a block returns to the free list only at refcount 0 |
| — 144 `check_invariants` | every block is free XOR has refcount > 0 |
| `block_table.py` 75 `blocks_needed_for` | ceiling division |
| — 106 `append_tokens` | when a new block is taken |
| — 128 `locate`, 137 `slot_index`, 147 `slot_mapping` | the formula from A |
| — 66 `num_wasted_slots` | the fragmentation metric behind 16.8% → 3.9% |
| `kv_cache.py` 34 `KVCacheConfig`: 80 `bytes_per_token`, 85 `bytes_per_block`, 88 `num_blocks_for_budget` | check them against your maths |
| — 98 `blocks_for_free_memory`, 116 `PagedKVCache.__init__`, 166 `k_flat` | the cache tensor's shape, and why "flat" |

Then read README *Phase 1: paged memory manager* (line 602) and *Fragmentation* (321).

## C. Rebuild it yourself (block 7, 75 min)

Without looking at the code, write `toy_paging.py` (about 100 lines of pure Python):
1. `BlockAllocator(num_blocks)` with `allocate()`, `allocate_many(n)` (all or nothing), `incref(b)`, `free(b)`; raise when out of blocks.
2. `BlockTable(block_size, allocator)` with `append_tokens(n)`, `slot_mapping()` and `num_wasted_slots()`.
3. 3 requests of different lengths, each adding 1 token per step. Print the wasted slots each step, and compare against contiguous allocation of `max_len`.

**Check:** point `tests/test_block_allocator.py` and `tests/test_block_table.py` at your module (change the import) and
run `pytest -q` on your laptop (CPU only). Each failing test is a concept you missed.

## D. Trace by hand (block 8, 20 min)

Block size 4, pool of 8 blocks. Request A has 5 tokens, B has 9, C has 3. Each then appends 4 tokens, one at a time. Write down:
- every block table after each append;
- the free list;
- the slot of token 7 of request B;
- the total wasted slots at the end, compared with contiguous allocation at `max_len = 16`.

## E. Questions (block 9, out loud)

1. Why is decode memory-bound and prefill compute-bound? What is the roofline ridge point on an H200?
2. Walk me through the KV-cache size of your model.
3. Why is paging needed? Why not allocate KV for `max_len` up front?
4. How did you measure fragmentation, and why is paging slightly worse on uniform prompts?
5. Block size 16: what changes at 1 and at 256? (*Small: tiny waste, but a bigger table and less efficient kernels. Large: more waste per sequence.*)
6. How does a block become free when two requests share it?
7. What does `allocate_many` being all-or-nothing protect against?

## My notes and fumbles

- 
