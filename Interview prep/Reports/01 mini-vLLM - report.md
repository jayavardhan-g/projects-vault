# Project report: mini-vLLM, a from-scratch LLM inference server

> **For the LLM reading this.** This report describes a project built by G Jayavardhan, an M.Tech CSE student at IIT Bombay.
> He has to defend it in technical interviews as its sole author. Use the report to teach him the project, quiz him,
> or answer questions about it. Everything here comes from the repository's code, README, git history and result
> files. Labels used below:
> - **[CODE]**: what the code does, with `file:line`.
> - **[MEASURED]**: a number from a results file.
> - **[DOC]**: a reason written in the repo's docs or comments.
> - **[RECON]**: reasoning reconstructed by the report writer and not written in the repo.
>
> If a question goes beyond what is written here, say so rather than inventing details.

---

## 1. One-paragraph summary

mini-vLLM is a small LLM inference server written in Python, PyTorch and Triton. It rebuilds the four ideas that make
vLLM/SGLang fast:

1. a **paged KV-cache memory manager**: fixed-size blocks, a per-sequence block table, reference counts;
2. a **continuous-batching scheduler**: iteration-level scheduling, admission tied to free memory, preemption by recompute;
3. a **radix-tree prefix cache**: KV blocks shared between requests with a common prefix, with refcounted LRU eviction;
4. a **fused Triton paged-attention kernel** for the decode step: it reads K/V straight from scattered blocks and uses an online softmax.

It also has a FastAPI HTTP front end with SSE streaming. Model: `meta-llama/Llama-3.2-1B-Instruct` in bfloat16 on one
NVIDIA H200, on a shared server. The model's forward pass is re-implemented over the paged cache, reusing the
HuggingFace weight tensors. The headline is **3.82× output throughput over a naive HuggingFace `generate()`
baseline**. The other key results are the kernel's 7.49× over the naive gather, a 4.3× cut in KV fragmentation, and
the prefix cache's 7.41× at 8k context. It was also measured honestly against real vLLM 0.26.0, which is **6.2× faster** than mini-vLLM.

- **Type:** self project, Summer 2026. All 34 commits by Jayavardhan, 2026-08-12 → 2026-08-15.
- **Repo:** `/data/jayavardhan/projects/vLLMmini`, branch `main` (latest commit `3679394`). Package `src/mini_vllm/`.
- **Size:** about 3,600 lines of source, about 2,500 lines of benchmarks, 421 tests (349 CPU in about 6.5 min, 72 GPU in about 14 min; no skips or xfails).
- **Environment:** Python 3.11, torch 2.11.0+cu128, triton 3.6.0 (bundled with torch), transformers 5.x, FastAPI/uvicorn. The env `vllmmini` was created by path on warm storage. A separate env `vllm-ref` holds vLLM 0.26.0.

---

## 2. Resume bullets and whether the repo supports them

Resume 4 and resume 5 have slightly different wording:

| Resume bullet | Verdict | Where it comes from |
|---|---|---|
| "Built a paged KV-cache memory manager and continuous-batching scheduler for LLM inference, reducing GPU memory fragmentation by 4.3×" | **Supported, with a caveat.** | `results/fragmentation.json`, `benchmarks/fragmentation.py`. On the `independent` workload, KV slots that hold no token drop from 16.8% under pad-to-longest static batching to 3.9% under paging with block size 16 (16.8/3.9 = 4.3×, 904 slots saved). The calculation is deterministic *accounting* (tokenizer only, no GPU), not a measurement of GPU memory. On the `shared_prefix` workload paging is *worse* (0.8% → 2.8%). |
| "…and achieving 3.82× throughput over the HuggingFace baseline" (resume 5) / "contributing to 3.82× overall throughput" (resume 4) | **Supported.** "Contributing to" (resume 4) is the more accurate wording. | `results/serving.json`, `uniform` mode: median of 5 paired per-round ratios = 3.824. Output throughput 631.5 → 2415.0 tok/s. This is the *whole system* (paging + continuous batching + fused decode kernel + prefix cache), so the memory manager and scheduler only *contribute* to it. |
| "Implemented a radix-tree prefix cache to reuse shared KV states across requests, accelerating repeated prompt inference by up to 7.4×" | **Supported as "up to", but know the conditions.** | `results/prefix_cache_value.json`: one request with 90% of an 8,192-token context cached vs the same request with an empty cache, prefill 665.5 ms → 89.8 ms = 7.41×. At 512 tokens the speedup is 1.00×. On the realistic multi-turn workload (prompts ≤ about 2.4k tokens) the cache reused 91% of tokens by turn 4 but saved **0% of wall time (1.00×)**, because prefill below about 2k tokens is overhead-bound. Be ready to explain that crossover (§8.5). |

---

## 3. Background you need first (teach these before the code)

### 3.1 What happens when an LLM generates text
- A decoder-only transformer turns token ids into a hidden vector per token (embedding). The vectors go through N
  layers (16 for Llama-3.2-1B). Each layer is *attention* followed by an *MLP*, with residual connections and RMSNorm.
  The last hidden vector goes through `lm_head` to give logits over the vocabulary (128,256 for Llama 3).
- **Attention:** each token makes a query q, a key k and a value v. Token t's output is softmax(q_t·K^T/√d)·V over
  all earlier tokens (causal).
- **KV cache:** the k and v of past tokens never change, so they are stored and not recomputed. Without a cache,
  generating token t would redo work for all t−1 earlier tokens.
- **Prefill vs decode:** *prefill* processes the whole prompt in one forward pass (many tokens, large matrix
  multiplies, compute-bound). *Decode* produces one token per step per sequence (tiny matmuls, memory-bandwidth or
  launch-overhead bound). TTFT (time to first token) is dominated by prefill. TPOT (time per output token) is dominated by decode.

### 3.2 KV-cache size
bytes per token = 2 (K and V) × layers × KV heads × head_dim × bytes per element.
For Llama-3.2-1B in bf16: 2 × 16 × 8 × 64 × 2 = **32 KiB per token** (`constants.py:8-13`).
- **GQA (grouped-query attention):** Llama-3.2-1B has 32 query heads but only 8 KV heads. Each KV head serves 4 query
  heads, so the cache is 4× smaller than with full multi-head attention. Query head `h` reads KV head `h // 4`.

### 3.3 Why naive serving wastes memory
HF `generate()` with static batching:
- pads every prompt in a batch to the longest one and allocates KV for that padded length (**internal fragmentation**);
- keeps the whole batch until its slowest member finishes (idle slots);
- recomputes a shared system prompt for every request (no reuse).

### 3.4 PagedAttention (Kwon et al., SOSP 2023, the vLLM paper)
This is OS paging applied to KV. The cache is split into fixed-size **blocks** (16 tokens here). Each sequence has a
**block table** mapping logical block i to a physical block id anywhere in the pool. Only the last block of a
sequence can be partly empty, so waste is under `block_size` tokens per sequence. Blocks can be **shared** between
sequences, which needs reference counts. The attention kernel has to *gather* K/V through the block table instead of
walking contiguous memory.

### 3.5 Continuous (iteration-level) batching (Orca, OSDI 2022)
Scheduling is decided again before *every forward pass*. Finished requests leave at once and new requests join as soon
as memory allows. A prefill of one request can share a forward pass with the decodes of others: tokens are flattened
into one ragged batch. When memory runs out, a running request is **preempted**. vLLM offers *recompute* (drop its KV
and redo it later) or *swap* (copy its KV to CPU memory). mini-vLLM uses recompute.

### 3.6 Radix-tree prefix caching (SGLang's RadixAttention, Zheng et al. 2024)
Store computed KV blocks in a tree keyed by token contents. A new request walks the tree from the root along its own
tokens. The deepest match gives blocks it can reuse instead of recomputing. LRU eviction removes cold leaves.
Reference counts stop eviction from freeing a block that a live request is using.

### 3.7 Online softmax / flash-style decoding
To compute softmax(s)·V over a long context without storing all scores, keep a running max `m`, a running denominator
`l` and an accumulator `acc`. For each new chunk:
`m_new = max(m, max(s_chunk))`, `alpha = exp(m − m_new)`, `p = exp(s_chunk − m_new)`,
`l = l·alpha + Σp`, `acc = acc·alpha + p·V_chunk`. At the end, output = `acc / l`.
This gives exactly the softmax result in one pass, with O(head_dim) memory.

### 3.8 Triton
Triton is a Python DSL for GPU kernels. You write one **program**, and the grid launches many copies of it.
`tl.program_id(axis)` tells each copy who it is. `tl.load`/`tl.store` with a `mask` handle out-of-bounds lanes, and
`tl.constexpr` parameters are compile-time constants. It is easier than CUDA C++ because Triton handles the
thread-level details (shared memory, warps) inside a block.

---

## 4. Problem statement and goals [DOC]

The README frames the problem as the one vLLM and SGLang exist to solve. Naive serving allocates KV as one dense,
padded tensor per batch. This wastes memory to fragmentation, caps concurrency below what the GPU could hold, and
recomputes shared prefixes for every request. The project was built in phases, each isolating one variable:

| Phase | What | State |
|---|---|---|
| 0 | Scaffold + naive HF `generate()` baseline | done |
| 1 | Paged KV memory manager (allocator + block table) | done |
| 2 | Naive paged attention + Llama forward over the paged cache, verified against HF | done |
| 3 | Continuous-batching scheduler with recompute preemption | done |
| 4 | Radix prefix cache with refcounted LRU eviction | done |
| 5 | Fused Triton decode kernel | done |
| 6 | Benchmarks vs HF baseline, and vs real vLLM | done |
| 6.5 | FastAPI front end (buffered + SSE streaming) | done |

**Why Llama-3.2-1B and not a Qwen small model [DOC, `constants.py:7-14`, README "Model"].** The choice is about KV
footprint, not answer quality. At 32 KB/token, Llama-3.2-1B puts real pressure on the block pool, so preemption and
eviction actually fire. Qwen2.5-0.5B is 12 KB/token, and an H200 could hold millions of its tokens, so those code
paths would never run. Its 4:1 GQA and head_dim 64 also match what reference paged-attention kernels are written for.

---

## 5. Architecture

```
Client ──HTTP──► server.py (FastAPI, one asyncio loop)
                    │  EngineLoop: many handlers await futures; ONE task calls engine.step()
                    ▼
               engine.py  ContinuousBatchingEngine.step()
                    │ 1. scheduler.schedule()  → which requests, how many tokens each
                    │ 2. flatten into a ragged batch: token_ids, positions, slot_mapping, query_lens
                    │ 3. model forward (PagedLlama) → logits only at each request's last token
                    │ 4. argmax (greedy) → scheduler.update()  (publish blocks, finish, free)
                    ▼
   scheduler.py ──uses──► block_table.py ──uses──► block_allocator.py (refcounted pool of block ids)
        │                                               ▲
        └──uses──► prefix_cache.py (radix tree; holds 1 ref per cached block)
                    
   model/llama.py  PagedLlama.forward: per layer
        RMSNorm → q,k,v proj → RoPE (llama3 scaling) → write_kv_to_cache(slot_mapping)
        → attention: fused Triton decode kernel if every seq has query_len==1, else naive gather
        → o_proj + residual → RMSNorm → SwiGLU MLP + residual
   kv_cache.py  per layer K,V tensors [num_blocks, block_size=16, kv_heads=8, head_dim=64]
```

**Design rule [DOC].** The allocator, block table, prefix cache and scheduler are **pure Python with no torch**. Policy
bugs can then be tested exhaustively on CPU (38 scheduler tests in 1.2 s, a 20k-seed randomised oracle). The engine
only moves tensors between the scheduler and the model.

**File map (source lines):** `constants.py` (41), `runtime.py` (116), `metrics.py`, `workloads.py` (231),
`block_allocator.py` (174), `block_table.py` (178), `kv_cache.py` (197), `paged_attention.py` (250),
`paged_attention_triton.py` (263), `prefix_cache.py` (347), `scheduler.py` (714), `engine.py` (191),
`server.py` (534), `model/llama.py` (311), `model/runner.py` (185, static-batch comparison runner).

---

## 6. Components in detail

### 6.1 `BlockAllocator`: the physical block pool (`block_allocator.py`)
- **State:** `_ref_counts = [0]*num_blocks` and a free list `_free_blocks`. The free list is LIFO, so the most
  recently freed block is reused first and the working set stays hot (`:44-47`).
- `allocate()` pops a block and sets its refcount to 1. It raises `OutOfBlocksError` if the pool is empty (`:70-78`).
- `allocate_many(n)` is **all-or-nothing**. It checks `n <= free` before taking anything, so a failure leaks nothing (`:80-94`).
- `incref(id)` refuses a free block, because that would "resurrect" a block the free list still owns (`:96-109`).
- `free(id)` decrements the count and returns the block to the free list at 0. Double free raises (`:111-119`).
- `free_many` checks every id before changing anything (`:121-132`).
- **Invariant**, checked by `check_invariants()` (`:144-168`) and used in randomised stress tests: *a block is in the free list ⇔ its refcount is 0*.
- **Why refcount from Phase 1, when nothing is shared yet [DOC]:** Phase 4 shares blocks. Adding refcounts later would mean auditing every call site for use-after-free.
- **Why raise instead of deciding [DOC]:** the allocator only reports exhaustion. The scheduler decides whether to defer, evict or preempt.

### 6.2 `BlockTable`: a sequence's "page table" (`block_table.py`)
- Holds `_block_ids` (physical ids in logical order) and `_num_tokens`.
- `blocks_needed_for(n)` = ceil((num_tokens+n)/block_size) − current blocks (`:75-84`). It changes nothing, and the scheduler uses it to price admissions.
- `append_tokens(n)` allocates only the new blocks needed, all-or-nothing (`:106-124`).
- `locate(p)` = `divmod(p, block_size)` → `(block_ids[p//bs], p%bs)`. `slot_index(p) = block_id*block_size + offset` (`:128-145`): a **flat slot index** into the cache viewed as `[num_blocks*block_size, heads, dim]`.
- `adopt_blocks(ids, n)` seeds an *empty* table with cached blocks. It requires whole full blocks (`:86-104`).
- `free()` is idempotent (`:157-169`).
- `num_wasted_slots = capacity − num_tokens` is always < block_size. This is the fragmentation metric.

### 6.3 `PagedKVCache` (`kv_cache.py`)
- One K tensor and one V tensor **per layer**, shape `[num_blocks, block_size, num_kv_heads, head_dim]` (`:131-139`).
  Blocks come first, so a block id indexes the tensor directly.
- `k_flat(layer)` is a *view* `[num_blocks*block_size, heads, dim]` that a flat slot index addresses (`:166-176`).
- `KVCacheConfig.from_model_config` reads `num_key_value_heads`, not `num_attention_heads`. Under GQA this is a 4× difference (`:48-73`).
- `blocks_for_free_memory(fraction=0.9)` sizes the pool from **free** memory (`torch.cuda.mem_get_info`), because the GPU is shared (`:98-113`).
- Why per-layer tensors [DOC]: the runner touches one layer at a time, and separate tensors keep each layer's K and V contiguous for flat slot indexing.

### 6.4 Naive paged attention, the reference (`paged_attention.py`)
- `write_kv_to_cache`: `k_flat[slots] = key` scatters this step's K/V into their physical slots (`:36-65`).
- `gather_kv`: `k_cache[blocks].reshape(-1, heads, dim)[:context_len]` collects one sequence's K/V into a contiguous tensor (`:68-101`). Phase 5 removes exactly this copy.
- `_repeat_kv_heads`: `repeat_interleave(group, dim=1)` expands 8 KV heads to 32. `repeat` would tile instead and pair heads wrongly (`:104-119`).
- `causal_mask(query_len, context_len)`: query i sits at absolute position `context_len − query_len + i`, and `mask = k_pos <= q_pos` (`:122-137`). In decode (query_len = 1) everything is allowed, *because absolute positions are used*.
- `attention_scores`: softmax in float32 whatever the storage dtype (`:140-169`).
- `paged_attention`: a Python loop over sequences in a **flattened ragged batch** (`query` is `[total_tokens, heads, dim]`) (`:172-226`).
- It is deliberately unoptimised [DOC], so it is "obviously correct" and gives the Triton kernel something independent to be checked against.

### 6.5 `PagedLlama`: the forward pass (`model/llama.py`)
- It reuses the HF model's weight tensors (`embed_tokens`, `layers[i].self_attn.q_proj` …) but never calls HF's `forward`. This gives full control of the decode loop without converting checkpoints (`:170-211`).
- **Inputs are flattened and ragged:** `input_ids [total_tokens]`, `positions` (absolute position *within each sequence*), `slot_mapping`, `block_tables`, `query_lens`, `context_lens`, `logits_indices` (`:232-257`).
- Per layer (`:275-306`): RMSNorm (in fp32, matching HF) → q, k, v projections reshaped to heads → `apply_rope` → **write K/V to the cache before attending**, so a sequence sees its own new tokens → attention (fused or naive) → `o_proj` + residual → RMSNorm → SwiGLU MLP `down(silu(gate(x)) * up(x))` + residual.
- `logits_indices` takes the hidden state only at each request's last token before `lm_head` (128,256 wide), so a long prefill does not project every token (`engine.py:129-135`, `llama.py:309-311`).
- **RoPE with llama3 scaling** (`:97-148`): `inv_freq = 1/θ^(2i/d)`, then rescaled by wavelength. Long wavelengths are divided by `factor`, short ones are unchanged, and the middle band is interpolated.
- **Bug found [DOC]:** in transformers 5.x, `config.rope_theta` moved into `config.rope_parameters`. The old `getattr(config, "rope_theta", 10000.0)` silently gave **10000 instead of 500000**, a 50× error. Short prompts still looked fluent. Only a direct comparison with HF's rotary module caught it (74.8% of cos/sin values wrong). `resolve_rope_config` now **raises instead of defaulting** (`:68-94`).
- `decode_meta` (block tables padded into an int32 tensor) is built **once per forward pass, not per layer** (`:271-273`). Doing it 16 times would add back the Python + PCIe overhead the kernel removes.

### 6.6 The fused Triton decode kernel (`paged_attention_triton.py`)
- **Grid:** `(num_seqs, num_q_heads)`. One program computes one (sequence, query head) output vector (`:83-85`, `:242`). At 32 seqs × 32 heads that is 1,024 programs, which fills the H200 at this batch size [DOC].
- Each program: load `q` (head_dim 64) in fp32. Then loop over the context in chunks of `BLOCK_N = 128` token positions (`:101`):
  - `logical_block = offs_n // block_size`, `offset_in_block = offs_n % block_size`;
  - `phys_block = tl.load(block_table[seq, logical_block], mask=valid)`, the paging indirection *per token*, so it does not assume block_size is a power of two (`:108-114`);
  - load K at `phys_block*stride_kb + offset*stride_kp + kv_head*stride_kh + d`, where `kv_head = q_head // num_kv_groups` (GQA) (`:85`, `:116-121`);
  - `scores = sum(q*k)*scale`; slots past `ctx_len` are set to **−inf, not 0**, because the stale K/V of a previous tenant could otherwise dominate the max (`:123-125`);
  - online-softmax update of `m_i`, `l_i`, `acc` (`:127-140`);
  - finally `out = acc / l_i`, stored in the cache dtype (`:142-146`).
- **Decode only [DOC]:** with query_len = 1 the single query is at position `ctx_len−1`, so causal masking reduces to the bounds check. `can_use_decode_kernel` requires *every* sequence to have query_len == 1 and a CUDA device (`:196-204`). Mixed prefill+decode steps fall back to the naive path for the whole step.
- **No split-K / flash-decoding [DOC]:** 1,024 programs already fill the card at these batch sizes. Split-K would need a two-pass log-sum-exp combine. The docstring records it as the next change if profiling shows the GPU idle on long contexts at small batch.
- **Why Triton and not CUDA [RECON]:** the bottleneck was Python-level gather loops and thousands of launches, not a missing low-level trick. Triton gives a fused kernel in about 80 lines, tested against the reference, with no build step (it ships with the torch wheel).

### 6.7 The scheduler (`scheduler.py`)
**Request state** (`:86-140`): `prompt_token_ids`, `output_token_ids`, `table`, `num_computed_tokens` (leading tokens whose KV is in the cache), `num_cached_tokens`, `num_published_blocks`, `num_preemptions`, timing fields. Derived values:
`num_uncomputed_tokens = num_tokens − num_computed_tokens`; `is_prefill = num_uncomputed_tokens > 1`.

**Config** (`:189-211`): `max_num_seqs=256`, `max_num_batched_tokens=8192` (per-iteration token budget), `watermark=0.01` (1% of blocks kept free so a new admission does not force an immediate preemption, which causes thrash).

**`add_request`: reject at the door** (`:262-289`). A request is rejected with `RequestTooLargeError` if (a) prompt + max_new_tokens needs more blocks than the pool minus the watermark, or (b) the prompt alone is over the token budget (prefills are not chunked). Without these checks such a request would wait forever, which looks like a hang.

**`schedule()`** (`:296-320`) serves the **running requests first, then admits waiting ones**. A decode needs at most one new block, while an admission can need hundreds, so letting arrivals win would starve nearly-finished sequences [DOC].

**`_schedule_running`** (`:322-363`). For each running request, compute the blocks needed for its next token(s). While the pool lacks them:
1. **evict cached blocks first** (`_evict_for`). This is cheap because cached blocks are unused;
2. else **preempt a victim**, the most recently admitted request (`self.running[-1]`), so FCFS order survives;
3. if the only candidate is the current request, preempt *it*.

It retries eviction **after every preemption**. With mid-flight publishing, a victim's blocks often drop only to the tree's reference, which makes them *evictable* rather than free. A single eviction pass up front would preempt the whole running set while reclaimable blocks sat in the cache [DOC].

**`_preempt`** (`:626-645`) is preemption by recompute: `table.free()`, `num_computed_tokens = 0`, `num_published_blocks = 0`, and the request goes to the **front** of the waiting queue. It comes back as a prefill of prompt + everything already generated, which gives the same next token (up to bf16 near-ties).

**`_schedule_waiting`** (`:365-441`) uses **decide-then-commit** admission:
1. Stop at `max_num_seqs`. Do not re-admit a request preempted in this same iteration (anti-thrash). Stop if its tokens exceed the remaining budget (FCFS, no skipping ahead).
2. `_match_cached_prefix`: probe the radix tree with `record_stats=False`, **excluding the last token**, so at least one token is left to run and produce logits (`:443-464`).
3. `_blocks_to_admit`: with a hit, only the blocks past the cached prefix are needed (`:509-521`).
4. If the pool minus `needed` would fall under the watermark, evict and **re-match in a loop**. Eviction can remove the very prefix the request matched and *raise* what it needs (`:414-420`).
5. `_first_uncached_chunk`: if another request admitted *this iteration* will compute the same first uncached 16-token block, **defer this one by one iteration** (`:466-507`). Two requests in the same forward pass cannot read each other's KV: that would be a read/write race within one kernel launch. Next iteration the block is published and reusable. `pending` lasts one iteration, so there is no starvation. Deferral is disabled when the cache is off.
6. Only now **commit**: `_adopt_cached_prefix` (acquire refs, `adopt_blocks`, set `num_computed_tokens`) (`:523-542`). Then pop from waiting, mark RUNNING, `_prepare`.

**`_prepare`** (`:599-612`) grows the block table and builds `token_ids`, `positions`, `slot_mapping` for the uncomputed tokens.

**`update`** (`:649-686`), after the forward pass: set `num_computed_tokens = num_tokens`, **publish newly full blocks** to the cache, append the sampled token, and finish on max_new_tokens or EOS. `_finish` publishes and then frees the table. Blocks go back immediately, not at the end of a batch.

**`_publish_blocks`** (`:544-586`) has two safety rules [DOC]:
- **only full blocks**: a full block is never written again, so it is immutable;
- **only computed tokens**: `update` sets `num_computed_tokens` *before* appending the sampled token, whose KV doesn't exist yet. Keying a block on it would let a later request read uninitialised memory as valid KV.

### 6.8 The radix prefix cache (`prefix_cache.py`)
- **Granularity: blocks, not tokens.** Each `RadixNode` owns one physical block. Its key is the block's 16-token tuple, stored in `parent.children[tuple]`. A root-to-node path spells out a prefix (`:52-71`).
- `match(tokens)` walks children by successive 16-token chunks. It **does not take references**, so the scheduler can ask "how much would this save?" without committing (`:144-174`).
- `acquire(match)` increfs each block and touches it and its ancestors for LRU (`:191-206`).
- `insert(tokens, block_ids)` adds new nodes for complete blocks only, and **takes the tree's own reference** (`incref`). Existing nodes keep their identity (`:210-243`).
- **Refcount protocol [DOC]:** the tree holds 1 ref, and each sequence using the block holds another. Eviction only considers blocks with **refcount == 1** (only the tree holds it) **and that are leaves**. Evicting an interior node would orphan its children, which is a leak (`:247-259`). So "never evict a block in use" is *structural*, not a check someone must remember.
- `evict(n)`: sort candidates by `last_accessed` and remove LRU leaves until n are freed. Re-scan, because removing a leaf can expose its parent as a new leaf (`:265-288`).
- `hit_rate` is **token-weighted** (cached tokens reused / queried tokens), not per-request (`:103-113`).
- `check_invariants`: no block cached twice, every cached block has refcount ≥ 1, every node reachable from the root, no cycles (`:312-340`).

### 6.9 The engine (`engine.py`)
`step()` (`:111-150`): `schedule()` → flatten all scheduled requests into one ragged batch → `logits_indices` = last token of each request → `model(...)` under `torch.inference_mode()` → `argmax` (greedy) → `scheduler.update`. If nothing could be scheduled while requests wait, it raises instead of spinning. Decoding is **greedy** so phases can be compared token for token.

### 6.10 Runtime settings that changed results by 24–33× (`runtime.py`) [MEASURED, DOC]
- **`torch.set_num_threads(8)`.** The machine has 224 cores, so torch defaults to 112 threads. Each tiny CPU-side op in decode opens an OpenMP region, and barrier cost across 112 contended threads made decode **315 ms/step vs 9.6 ms/step**, a 33× slowdown. Prefill was unaffected.
- **Disable the cuDNN SDPA backend.** It builds an execution plan on the CPU the first time it sees each shape (4.18 s CPU vs 462 µs GPU). A continuous-batching server changes shape nearly every iteration. First call per shape: 11.6 s with it on, 0.35 s with it off.
- Every entry point calls `configure_torch_runtime()`, so the baseline and the system are measured under identical settings. Otherwise speedups would measure the misconfiguration.

### 6.11 The HTTP server (`server.py`)
- `POST /generate` (JSON or `stream: true` SSE), `GET /health`, `GET /stats`.
- **The engine is not made concurrent; that is the design [DOC].** The block pool and radix tree assume a single caller of `step()`. One background `EngineLoop.run()` task calls `step()`. Many HTTP handlers `await` their own `asyncio.Future`. Handlers and the loop are coroutines on **one event loop**, and the loop never awaits inside a step, so `add_request` can only interleave *between* steps. No locks, no threads. `await asyncio.sleep(0)` once per iteration lets new requests join (`:211-250`).
- Edge cases handled, each with a test:
  - over-budget prompt → **413** with the scheduler's message;
  - a retry with the same id after a rejection works, because nothing is registered until admission accepts (`submit` calls `engine.add_request` first) (`:156-182`);
  - client disconnect → `cancel()` → `scheduler.abort()`, which frees blocks;
  - a dead engine loop fails **every** pending future and unblocks streaming queues (`:238-250`); `/health` reports the stored failure.
- **Streaming without changing the engine contract:** after each step, `_drain_streams` pushes new tokens from each request's `output_token_ids` onto its queue (`:194-209`). Deltas come from decoding the whole output so far and emitting the suffix, **holding back while the decode ends in U+FFFD**. One BPE token can be half a UTF-8 code point.
- **Disconnect must be polled:** `_sse` checks `await http_request.is_disconnected()` after every frame. The first version relied on generator closing; its unit test passed, but against a live server a killed `curl` kept decoding for 12 s.
- **Warmup:** without it the first request took **17.0 s** vs 0.36 s warm (CUDA context, cuBLAS handles, Triton compile). `warmup()` runs one throwaway request at startup, then drops its cache entry and resets the stats.
- **Mutation testing of the server:** 8 deliberate bugs were injected. 5 were missed by the first suite or only caught by hanging. Tests were added and the hangs were rewritten to fail fast.

---

## 7. Worked example (how one step flows)

Block size 16. Request A (prompt 20 tokens) runs alone, then request B arrives with 20 tokens whose first 16 equal A's.
1. A is admitted. `_match_cached_prefix` finds nothing. `_blocks_to_admit` = ceil(20/16) = 2 blocks, say physical #5, #2. `_prepare` gives `positions 0..19`, `slot_mapping = [5*16+0 … 5*16+15, 2*16+0 … 2*16+3]`. Forward pass writes K/V to those slots. Logits are taken at position 19. Greedy token → `update`: `num_computed_tokens=20`. Full blocks = 20//16 = 1, so block #5 is **published** under key `tuple(tokens[0:16])` and refcount(#5) = 2 (sequence + tree).
2. B arrives. Its probe uses `tokens[:-1]` (19 tokens): one full chunk matches node #5. `needed` = ceil(20/16) − 1 = 1 new block. Commit: `acquire` → refcount(#5) = 3; `adopt_blocks([5], 16)`; `num_computed_tokens = 16`. `_prepare` schedules only tokens 16..19 (query_len 4), in a new block #9. Its attention reads positions 0–15 from block #5 and 16–19 from #9.
3. While both A and B are decoding with query_len = 1, every step uses the **fused kernel**. A step where a new arrival is prefilling uses the naive path for everyone.
4. A finishes: `_finish` → publish (no-op) → `table.free()` → refcount(#5) = 2, and #2 returns to the pool if it was never published. When B finishes, #5 drops to 1 (tree only), which makes it an evictable leaf.

---

## 8. Experiments and results

All on 1× H200 (141 GB), shared with other users, Llama-3.2-1B-Instruct bf16, greedy.

### 8.1 Phase 0 baseline (HF `generate()`, static batch 8, 32 requests × 128 tokens) [MEASURED]
| | independent | shared_prefix |
|---|---|---|
| output throughput | 647 tok/s | 656 tok/s |
| TTFT p50 | 30.1 ms | 18.6 ms |
| TPOT p50 | 11.9 ms | 11.6 ms |
| KV padding waste | **16.8%** | 0.8% |
Repeat runs gave 647 vs 1001 tok/s at different host loads. Decode at this size is host/launch-bound, so other users' CPU load moves it about 1.5×. That is why every later headline is a **paired ratio**.

### 8.2 Fragmentation (Phase 1; deterministic accounting, tokenizer only) [MEASURED]
| workload | naive pad-to-longest | paged (block 16) | |
|---|---|---|---|
| independent | 16.8% | **3.9%** | 4.3× less, 904 slots saved |
| shared_prefix | 0.8% | 2.8% | worse by 176 slots |
Paging recovers *padding* waste, which only exists when prompt lengths vary. shared_prefix prompts are 132–139 tokens (near-uniform), so block rounding (≤ 15 slots per sequence) costs more than it saves. The naive 16.8% matches what the Phase 0 baseline measured in real generation (a cross-check).

**Block-size choice** (`constants.py:18-41`):
| block | waste indep/shared | prefix captured |
|---|---|---|
| 8 | 1.9% / 1.4% | 97.3% |
| **16** | 3.9% / 2.8% | 94.1% |
| 32 | 7.0% / 8.2% | 87.8% |
| 64 | 15.9% / 17.4% | 75.3% |
16 cuts waste about 4× while keeping half the block-table bookkeeping of block 8. It is also vLLM's default, which makes numbers comparable.

### 8.3 Continuous vs static batching (Phase 3, naive attention in both arms; 32 requests, output lengths 19–96) [MEASURED]
| | static | continuous |
|---|---|---|
| wall time | 17.18 s | 9.39 s |
| request throughput | 1.86 req/s | 3.41 req/s (**1.83×**) |
| peak blocks | 77 | 175 |
| max concurrent | 8 | 32 |
Peak memory goes **up**: 4× the concurrency for about 2× the blocks. That is the intended trade, and `max_num_seqs` sets it. The gain comes from output-length *variance*.

### 8.4 Fused decode kernel (Phase 5; 32 requests, 5 alternating rounds) [MEASURED]
| | naive gather | fused |
|---|---|---|
| output throughput | 379.1 tok/s | 2839.9 tok/s (**7.49×**) |
| p50 latency | 4.752 s | 0.564 s |
| mean TTFT | 0.0997 s | 0.0999 s (unchanged, as expected, since prefill was not touched) |
- **Why it wins [DOC + RECON]:** the naive path loops over sequences in Python for each of 16 layers. That is about 500 iterations per decode step and thousands of tiny kernel launches. The fused kernel is one launch per layer, so the gain is mostly removing launch and Python overhead, not saving bandwidth.
- **Paired median:** the unpaired medians gave 4.77× on one run whose paired ratios were 7.61, 4.76 and 7.51, because the two arms ran under different machine loads. Arms now alternate within each round, and the headline is the median of per-round ratios.
- **Divergence:** 12 of 32 requests produced different greedy tokens from the naive path. All were near-ties: the largest top-2 logit gap was 0.125 against a 0.5 threshold, and 6 were exactly 0.0. There were no confident disagreements.
- **Mutation testing of the kernel:** three plausible bugs were injected (wrong KV head under GQA, attending to the stale tail of the last block, ignoring the block table). Each failed 19 of 28 tests. At first "ignoring the block table" failed only 2. A lone sequence got a contiguous run starting at block 0, where logical = physical. The tests now fragment the pool first and assert that no sequence starts at block 0.

### 8.5 Headline: the whole system vs the HF baseline (Phase 6; `benchmarks/serving.py`) [MEASURED]
Setup: 32 requests × 128 tokens, `independent` workload, 4,096 blocks, HF static batch 8, 5 rounds with the arms alternating. All requests arrive at t = 0 in both arms, and both arms get identical prompts and identical per-request output lengths.
| | HF | mini-vLLM | |
|---|---|---|---|
| output tok/s | 631.5 | **2415.0** | **3.82×** |
| request/s | 4.93 | 18.87 | 3.83× |
| TTFT p50 | 2448 ms | 179 ms | 13.7× |
| TTFT p99 | 4879 ms | 252 ms | 19.4× |
| latency p50 | 4049 ms | 1690 ms | 2.4× |
- `uniform` lengths are the conservative floor: static batching wastes nothing there, so the result isolates the memory/attention path. With `varied` lengths (0.2–1.0×) the ratio is 3.68×, *lower*, because wall time is set by the longest request and fewer total tokens are delivered. Request throughput shows the difference (19.35 vs 5.26 req/s), and the baseline computes 3,840 tokens to deliver 2,824 (`static_batch_waste_frac = 0.265`).
- **Confounds removed:**
  1. An earlier 4.4× compared a baseline forced to exactly max-new-tokens with a server running varied lengths. It was corrected to identical lengths, which gives 3.82×.
  2. **Arrival bug:** the HF arm timed each request from when *its own batch* started, deleting its queueing time. Fixed: baseline TTFT p50 went from 24.6 ms to 2448 ms. The bug flattered the baseline.
- Before the Phase 5 kernel, the system was **2.9× slower** than the baseline.

### 8.6 Prefix cache [MEASURED]
- **Multi-turn hit rate** (`multi_turn.py`, 8 conversations × 4 turns, each turn = previous prompt + reply + new question):
  | turn | prompt tokens | reused | hit rate |
  |---|---|---|---|
  | 1 | 439 | 0 | 0.0% |
  | 2 | 1080 | 896 | 83.0% |
  | 3 | 1739 | 1536 | 88.3% |
  | 4 | 2389 | 2176 | **91.1%** |
  Overall 81.6% (token-weighted), zero evictions. Turn 1 first read 40.1%. The per-conversation marker had been *appended* to the persona text, so conversations shared a ~40-token head, and those cross-conversation hits were counted as self-reuse. Fixed by prepending `Session {c}.`.
- **But wall-clock speedup on that workload was 1.00×.** `prefix_cache_value.py` explains it. Prefill has a fixed cost (dispatch, launches, 16 layers of small ops):
  | prompt tokens | prefill time |
  |---|---|
  | 128 / 512 / 1024 | 15.6 / 15.5 / 19.1 ms (overhead-bound) |
  | 2048 / 8192 | 49.8 / 665.6 ms (compute-bound) |
  One request with 90% of its context cached vs none:
  | context | no cache | 90% cached | speedup |
  |---|---|---|---|
  | 512 | 15.5 | 15.5 | 1.00× |
  | 1024 | 19.1 | 16.0 | 1.19× |
  | 2048 | 49.8 | 17.0 | 2.92× |
  | 4096 | 173.9 | 28.2 | 6.16× |
  | 8192 | 665.5 | 89.8 | **7.41×** (the resume's "up to 7.4×") |
  **Lesson:** quote a hit rate together with the context length. On this model and card the cache pays off above about 2k tokens of shared context.
- **CPU-oracle hit rates** (8 requests, 135-token prompts sharing a 126-token head): with all requests admitted at once the hit rate went 0.0% → 62.7% after the one-iteration deferral (8 → 9 iterations, 672 prefill tokens saved). Against a warm cache it went 47.8% → 79.1%.

### 8.7 Against real vLLM 0.26.0 (`vllm_reference.py`; 32 × 128 tokens, 5 paired rounds, one asyncio client streaming both servers) [MEASURED]
| | vLLM | mini-vLLM |
|---|---|---|
| output tok/s | 6117.9 | 1007.2 |
| TTFT p50 | 164.9 ms | 1088.4 ms |
| TPOT p50 | 3.73 ms | 23.1 ms |
Median paired ratio **0.16×, so vLLM is 6.2× faster** (rounds 0.15–0.19). Both produced exactly 4,096 tokens per round. The absolute numbers are lower than the 2,415 tok/s headline because a neighbour's job had the card at 100%. The in-process engine ran at 866 tok/s, so the HTTP layer costs nothing.
Where the gap lives:
- TTFT is 6.6× worse: prefill is not fused, materialises `[heads, q, ctx]`, and is not chunked.
- TPOT is 6.2× worse: no CUDA graphs, no split-K, and Python re-entered every iteration.

---

## 9. Hardest problems / real bug stories (all documented in the README and commits)

1. **The prefix cache hit 0% on the workload it was built for.** Blocks were published only when a request *finished*. Continuous batching puts every queued request in flight before the first finishes. Fix 1: publish full, computed blocks after every forward pass. Still 0%: when all requests are admitted in one iteration, none of them has computed the prefix yet. Fix 2: defer a request whose first uncached block is being computed by a request admitted this iteration (`_first_uncached_chunk`). Result: 0% → 62.7% on the oracle.
2. **Admission mutated requests it then refused** (a correctness bug: the cache could serve KV that didn't match its key). `_reuse_cached_prefix` took refs, seeded the table and set `num_computed_tokens` *before* the watermark check that could still refuse. A refused request leaked refs (unevictable blocks), and its stale `num_computed_tokens` made `_prepare` address the wrong blocks later. **Fix:** decide-then-commit (`_match_cached_prefix` probe → `_blocks_to_admit` → `_adopt_cached_prefix`), plus an evict-and-re-match loop. **Found by a CPU oracle** (`tests/test_prefix_cache_oracle.py`): the model is replaced by a hash chain, so each token's "KV" is a hash over its whole prefix, read back through the real block table. Over 20k random seeds there were 38 crashes before and 0 after (6.05M reused tokens, 79.8k evictions exercised). The GPU test could never reach it: it ran 2 requests on 2,048 blocks, so admission was never refused.
3. **RoPE theta silently 10000 instead of 500000** (transformers 5.x config change). Fixed by raising rather than defaulting.
4. **Thread count made decode 33× slower; cuDNN SDPA plan building made new shapes cost 11 s.** Both fixed in `runtime.py` and applied to every arm.
5. **A "diverging" repeated run looked like a cache bug but was a bf16 near-tie.** The top-2 logit gap was 0.125, equal to the max deviation between runs (one bf16 ulp at magnitude ~20). A cache hit computes the suffix in a 6-token pass over a 128-token cached context instead of one 134-token prefill, so the accumulation order differs. Tests now fail only on *confident* disagreement (`tests/greedy_compare.py`, 0.5 logit threshold).
6. **Benchmark confounds:** the baseline arrival-time bug, the max-new-tokens mismatch (4.4× → 3.82×), and unpaired medians (4.77× vs 7.49×).
7. **Turning the cache on "broke" leak tests.** Instead of weakening them, they now assert that every block is *free or cached* and that clearing the cache returns all of them. The weakening was first validated on 20k oracle seeds.

---

## 10. Design decisions (choice → alternatives → why)

| Decision | Alternatives | Why | Source |
|---|---|---|---|
| Block size 16 | 8, 32, 64 | about 4× less waste than baseline, half the bookkeeping of 8, vLLM default | [DOC] `constants.py` |
| Refcounts from day one | add later | sharing in Phase 4; avoids use-after-free audits | [DOC] |
| LIFO free list | FIFO | keeps recently used blocks hot | [DOC] |
| Preempt by recompute | swap to CPU | simpler to get right; composes with the prefix cache (a recomputed sequence often finds its prefix cached) | [DOC] |
| Victim = most recently admitted | longest, shortest, random | preserves FCFS; oldest requests keep progressing | [DOC] |
| Running before waiting | arrivals first | decode needs ≤ 1 block; avoid starving nearly-done requests | [DOC] |
| Watermark 1% | none | avoids admit → immediate preempt thrash | [DOC] |
| Unchunked prefill + reject prompts over budget | chunked prefill | simpler; chunked prefill recorded as the right fix and now the top candidate (TTFT 6.6× worse than vLLM) | [DOC] |
| Block-granular radix tree | token-granular (SGLang) | KV is allocated in blocks; only full blocks are immutable and shareable | [DOC] + [RECON] |
| Evict LRU leaves with refcount == 1 | any LRU block | evicting an interior node orphans children; refcount 1 = only the tree uses it | [DOC] |
| Publish blocks mid-flight | publish on finish | otherwise 0% hits under continuous batching | [DOC] |
| Defer one iteration on a shared uncached chunk | let both point at the same blocks | same-pass read/write race | [DOC] |
| Fused kernel for decode only | fuse prefill too | decode was where the time went; decode needs no causal mask | [DOC] |
| One program per (seq, head), no split-K | flash-decoding split-K | 1,024 programs fill the GPU; avoids a 2-pass LSE combine | [DOC] |
| Triton | CUDA C++ | short, testable, no build step; overhead was the bottleneck, not micro-tuning | [RECON] |
| Torch-free scheduler, allocator and cache | integrated with torch | exhaustive CPU tests, randomised oracle | [DOC] |
| Single engine task on one asyncio loop | thread pool, locks | invariants assume one `step()` caller; no locks needed | [DOC] |
| Greedy decoding | sampling | token-for-token comparison between phases | [DOC] |
| Paired, alternating rounds; median of ratios | absolute medians | shared box drifts ~1.5× | [DOC] |

---

## 11. Known limitations (from the README)

- Prefill still uses the naive Python gather. Any step that mixes in a prefill falls back entirely.
- Prompts above about 16k tokens cannot be prefilled at all: the naive path builds `[heads, q, ctx]` scores, and 32k tokens asks for 128 GiB. Fusing prefill would remove this.
- No chunked prefill. Prompts over `max_num_batched_tokens` (8,192) are rejected.
- A preempted request can grow past the token budget (prompt + generated) and then never be admitted. This is known and deliberately not patched; chunked prefill is the real fix.
- No CUDA graphs, no split-K, Python is re-entered every step. This is why vLLM is 6.2× faster.
- Single process, single GPU, greedy only. No tensor parallelism and no sampling parameters.
- No swap-based preemption.

---

## 12. Likely interview questions (with short answer pointers)

1. *Walk me through what happens to a request.* → §5 and §7: HTTP → `submit` → scheduler waiting queue → admission (prefix match, block pricing, watermark) → `_prepare` → ragged forward → greedy sample → `update`/publish → finish → future resolved.
2. *How is the KV cache laid out, and why blocks-first?* → §6.3.
3. *What is the fragmentation number, and how was it measured?* → §8.2. It is accounting, not GPU memory. Say why shared_prefix got worse.
4. *Why does peak memory go up with continuous batching?* → §8.3.
5. *How do you avoid evicting a block someone is using?* → refcount == 1 and leaf-only, §6.8.
6. *Why not let two same-iteration requests share the prefix immediately?* → same-pass race, §6.7 step 5.
7. *Explain the online softmax in your kernel.* → §3.7 and §6.6. Why −inf and not 0 for stale slots.
8. *Why only decode in the kernel?* → no causal mask needed with q_len = 1.
9. *Why is the kernel 7.49× faster?* → launch and Python overhead removed, not bandwidth.
10. *Your cache gives 91% hits but 1.00× speed. Is it broken?* → overhead-bound prefill below about 2k tokens, §8.6.
11. *Why is vLLM 6.2× faster?* → §8.7.
12. *Recompute vs swap preemption?* Swap costs PCIe bandwidth and CPU memory but avoids recompute FLOPs. Recompute is simpler and the prefix cache softens its cost. [RECON]
13. *How did you make the benchmark fair?* → paired alternating rounds, identical lengths and arrivals, same runtime settings, warmup round.
14. *What bug are you proudest of finding?* → the admission mutation bug and the CPU hash-chain oracle (§9.2).
15. *How would you scale to 10× traffic?* → chunked prefill, fused prefill (FlashAttention-style), CUDA graphs for decode, split-K for long contexts, multiple GPUs (tensor parallelism), sampling support. [RECON]
16. *Why does preemption not change outputs?* → re-prefill of prompt + generated tokens reproduces the same context. Only bf16 accumulation order changes, so only near-ties can flip.
17. *What does `logits_indices` buy?* → the 128,256-wide `lm_head` runs only on the last token of each request.
18. *What is GQA, and where does it show up in your code?* → `num_key_value_heads`, `repeat_interleave`, `kv_head = q_head // groups`.

---

## 13. Numbers cheat sheet

| Number | Meaning |
|---|---|
| 32 KiB/token | Llama-3.2-1B KV size, bf16 (2×16×8×64×2) |
| 16 | tokens per block |
| 16.8% → 3.9% (4.3×) | KV slots wasted, naive vs paged, `independent` workload |
| 1.83× | continuous vs static batching, request throughput (naive attention) |
| 7.49× | fused Triton decode kernel vs naive gather (379 → 2840 tok/s) |
| 3.82× | whole system vs HF baseline (631.5 → 2415 tok/s); 3.68× with varied lengths |
| 13.7× | TTFT p50 improvement (2448 → 179 ms) |
| 91.1% | prefix hit rate at turn 4 of multi-turn chats; 81.6% overall |
| 1.00× / 7.41× | cache speedup at ≤ 1k tokens / at 8k tokens with 90% cached |
| 0.16× (6.2×) | mini-vLLM vs real vLLM 0.26.0 |
| 33× | decode slowdown from torch's default 112 threads |
| 421 | tests (349 CPU + 72 GPU) |
| 20k seeds, 38 → 0 crashes | CPU oracle before and after the admission fix |

---

## Appendix A: Key code, annotated

How to read the excerpts:
- Real line numbers from `src/mini_vllm/` (branch `main`).
- `⋮` marks lines left out, with what they do.
- `◀` notes explain a line in plain English.
- Docstrings and comments are trimmed.
- A names table follows each block.

### Reading the code: Python, PyTorch and Triton constructs used below
| Construct | Meaning |
|---|---|
| `@dataclass` | auto-generates `__init__` etc. from the annotated fields |
| `list.pop()` / `list.append()` | take from / add to the end of a list; used as a LIFO stack |
| `divmod(a, b)` | returns `(a // b, a % b)` |
| `t[idx_tensor] = x` | PyTorch *scatter*: write rows of `x` at the positions in `idx_tensor` |
| `t[idx_tensor]` | PyTorch *gather*: read those rows (a copy) |
| `.view(...)` | reinterpret a tensor's shape without copying |
| `torch.inference_mode()` | no autograd bookkeeping (faster inference) |
| `@triton.jit` | compiles the Python function into a GPU kernel |
| `kernel[grid](args...)` | launches the kernel with `grid` = number of programs per axis |
| `tl.program_id(axis)` | which program (instance) this is, like CUDA's `blockIdx` |
| `tl.arange(0, N)` | a vector `[0, 1, …, N−1]`; Triton works on whole vectors per program |
| `tl.load(ptr, mask=m, other=v)` | vector load; lanes where `m` is False don't touch memory and get `v` |
| `tl.constexpr` | compile-time constant; a new kernel is compiled per value |
| `stride_*` | how many elements to jump to move one step along a tensor dimension (address = base + Σ index·stride) |

### A.1 The block pool: `block_allocator.py`
```python
40  def __init__(self, num_blocks: int) -> None:
⋮
44      self._ref_counts = [0] * num_blocks                    # ◀ one counter per physical block
47      self._free_blocks = list(reversed(range(num_blocks)))  # ◀ free stack; pop() yields 0,1,2… first; LIFO reuse
⋮
70  def allocate(self) -> int:
71      if not self._free_blocks:
72          raise OutOfBlocksError(...)                        # ◀ the scheduler catches this and defers/preempts
76      block_id = self._free_blocks.pop()                     # ◀ O(1): take the most recently freed block
77      self._ref_counts[block_id] = 1                         # ◀ invariant: in free list ⇔ refcount 0
78      return block_id
⋮
96  def incref(self, block_id: int) -> int:                    # ◀ a 2nd user (prefix cache / sharing sequence)
104     if self._ref_counts[block_id] == 0: raise ValueError(...)  # ◀ can't share a free block
108     self._ref_counts[block_id] += 1
⋮
111 def free(self, block_id: int) -> int:                      # ◀ drop ONE reference
114     if self._ref_counts[block_id] == 0: raise ValueError("double free")
116     self._ref_counts[block_id] -= 1
117     if self._ref_counts[block_id] == 0:
118         self._free_blocks.append(block_id)                 # ◀ last user gone → back to the pool
```
| Name | What it is |
|---|---|
| `num_blocks` | pool size; set by `blocks_for_free_memory` (90% of free GPU memory ÷ 512 KiB per block for Llama-1B) |
| `_ref_counts[b]` | how many owners block `b` has (sequences + the prefix tree) |
| `_free_blocks` | stack of block ids with refcount 0 |
| `OutOfBlocksError` | exception defined at the top of the file (`:28`) |

### A.2 The page table: `block_table.py`
```python
75  def blocks_needed_for(self, num_new_tokens: int) -> int:
82      total = self._num_tokens + num_new_tokens
83      blocks_after = (total + self._block_size - 1) // self._block_size   # ◀ ceil(total / 16)
84      return max(0, blocks_after - len(self._block_ids))                  # ◀ extra blocks only; no mutation
⋮
106 def append_tokens(self, num_new_tokens: int) -> list[int]:
119     needed = self.blocks_needed_for(num_new_tokens)
121     new_blocks = self._allocator.allocate_many(needed)    # ◀ all-or-nothing: raises BEFORE changing state
122     self._block_ids.extend(new_blocks)
123     self._num_tokens += num_new_tokens
⋮
128 def locate(self, position: int) -> tuple[int, int]:
134     logical_block, offset = divmod(position, self._block_size)   # ◀ token 37 → (block 2, offset 5)
135     return self._block_ids[logical_block], offset                # ◀ logical block → physical id
⋮
137 def slot_index(self, position: int) -> int:
144     block_id, offset = self.locate(position)
145     return block_id * self._block_size + offset                  # ◀ flat slot in a [num_blocks*16, heads, dim] view
```
**Example:** `_block_ids = [17, 4, 23]`, block size 16. Token 37 → logical block 2, offset 5 → physical block 23 → slot 23·16 + 5 = **373**.

| Name | What it is |
|---|---|
| `_block_ids` | physical block ids in logical order (the "page table") |
| `_num_tokens` | tokens stored so far; only the last block can be partly empty |
| `_allocator` | the `BlockAllocator` from A.1 |

### A.3 Naive paged attention (the reference): `paged_attention.py`
```python
36  def write_kv_to_cache(cache, layer, slot_mapping, key, value):
61      k_flat = cache.k_flat(layer)          # ◀ [num_blocks*16, kv_heads, head_dim] view of the layer's K cache
62      v_flat = cache.v_flat(layer)
63      slots = slot_mapping.to(device=k_flat.device, dtype=torch.long)
64      k_flat[slots] = key.to(k_flat.dtype)  # ◀ scatter: this step's token i goes to slot slots[i]
65      v_flat[slots] = value.to(v_flat.dtype)
⋮
68  def gather_kv(cache, layer, block_ids, context_len):
85      k_cache = cache.k(layer)              # ◀ [num_blocks, 16, kv_heads, head_dim]
87      blocks = torch.as_tensor(block_ids, dtype=torch.long, device=k_cache.device)
89      needed = (context_len + cache.block_size - 1) // cache.block_size
95      blocks = blocks[:needed]
99      k = k_cache[blocks].reshape(-1, k_cache.shape[2], k_cache.shape[3])[:context_len]  # ◀ COPY into contiguous order
100     v = v_cache[blocks].reshape(-1, v_cache.shape[2], v_cache.shape[3])[:context_len]
⋮
122 def causal_mask(query_len, context_len, device):
135     q_pos = torch.arange(context_len - query_len, context_len, device=device).unsqueeze(1)  # ◀ ABSOLUTE positions of the queries
136     k_pos = torch.arange(context_len, device=device).unsqueeze(0)
137     return k_pos <= q_pos                 # ◀ decode (q_len 1): last query sees every key
⋮
214     for blocks, q_len, ctx_len in zip(block_tables, query_lens, context_lens):   # ◀ Python loop per sequence, per layer
219         k, v = gather_kv(cache, layer, blocks, ctx_len)
220         k = _repeat_kv_heads(k, num_q_heads)   # ◀ repeat_interleave 8 KV heads → 32 (GQA)
222         outputs.append(attention_scores(q, k, v, scale, compute_dtype))
```
That Python loop over sequences × 16 layers, with a gather copy each time, is the thousands of small launches per decode step that A.4 removes.

### A.4 The fused decode kernel: `paged_attention_triton.py`
```python
63  @triton.jit
64  def _paged_attention_decode_kernel(q_ptr, k_ptr, v_ptr, out_ptr, block_table_ptr, context_lens_ptr,
    ⋮   scale, stride_* ..., block_size, num_kv_groups, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr):
83      seq = tl.program_id(0)                  # ◀ which sequence this program serves
84      q_head = tl.program_id(1)               # ◀ which query head
85      kv_head = q_head // num_kv_groups       # ◀ GQA: 4 query heads share one KV head
87      ctx_len = tl.load(context_lens_ptr + seq)
89      offs_d = tl.arange(0, HEAD_DIM)         # ◀ [0..63]
90      q = tl.load(q_ptr + seq * stride_qt + q_head * stride_qh + offs_d * stride_qd)
91      q = q.to(tl.float32)
97      m_i = float("-inf")                     # ◀ running max of scores
98      l_i = 0.0                               # ◀ running softmax denominator
99      acc = tl.zeros([HEAD_DIM], dtype=tl.float32)   # ◀ running weighted sum of V
101     for start_n in range(0, ctx_len, BLOCK_N):     # ◀ walk the context 128 tokens at a time
102         offs_n = start_n + tl.arange(0, BLOCK_N)   # ◀ logical token positions of this chunk
103         valid = offs_n < ctx_len
108         logical_block = offs_n // block_size
109         offset_in_block = offs_n % block_size
110         phys_block = tl.load(block_table_ptr + seq * stride_bts + logical_block * stride_btb,
111                              mask=valid, other=0)  # ◀ THE PAGING INDIRECTION, per token
116         kv_base = phys_block[:, None] * stride_kb + offset_in_block[:, None] * stride_kp
117         k = tl.load(k_ptr + kv_base + kv_head * stride_kh + offs_d[None, :] * stride_kd,
118                     mask=valid[:, None], other=0.0).to(tl.float32)   # ◀ [BLOCK_N, HEAD_DIM] read straight from scattered blocks
123         scores = tl.sum(q[None, :] * k, axis=1) * scale             # ◀ q·k for 128 tokens at once
125         scores = tl.where(valid, scores, float("-inf"))             # ◀ stale slots past ctx_len → −inf (not 0!)
127         m_new = tl.maximum(m_i, tl.max(scores, axis=0))
128         alpha = tl.exp(m_i - m_new)                                 # ◀ rescale factor for everything summed so far
129         p = tl.exp(scores - m_new)
131         v_base = ...                                                # ◀ same addressing for V
132         v = tl.load(..., mask=valid[:, None], other=0.0).to(tl.float32)
138         l_i = l_i * alpha + tl.sum(p, axis=0)
139         acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)          # ◀ online softmax update
140         m_i = m_new
142     out = acc / l_i                                                 # ◀ exact softmax(qKᵀ)·V, never materialised
143     tl.store(out_ptr + seq * stride_ot + q_head * stride_oh + offs_d * stride_od,
145              out.to(out_ptr.dtype.element_ty))
⋮
242     grid = (num_seqs, num_q_heads)          # ◀ 32 seqs × 32 heads = 1,024 programs
244     _paged_attention_decode_kernel[grid](query, k_cache, v_cache, out, meta.block_tables, meta.context_lens, scale,
    ⋮       ..., HEAD_DIM=head_dim, BLOCK_N=128, num_warps=4)
```
| Name | What it is (Llama-3.2-1B values) |
|---|---|
| `q_ptr` | query tensor `[num_seqs, 32, 64]`, one token per sequence |
| `k_ptr`, `v_ptr` | one layer's cache `[num_blocks, 16, 8, 64]` |
| `block_table_ptr` | int32 `[num_seqs, max_blocks]`, zero-padded (`build_decode_metadata`, `:164-193`) |
| `context_lens_ptr` | int32 `[num_seqs]`, including this step's token |
| `num_kv_groups` | 32 / 8 = 4 |
| `HEAD_DIM`, `BLOCK_N` | 64 and 128 (compile-time) |

### A.5 Radix prefix cache: `prefix_cache.py`
```python
136 def _blocks_of(self, token_ids):
138     full = len(token_ids) // self.block_size
139     return [tuple(token_ids[i*16:(i+1)*16]) for i in range(full)]   # ◀ only FULL blocks; tuple = hashable key
⋮
144 def match(self, token_ids, record_stats=True) -> PrefixMatch:
155     node = self.root
158     for chunk in self._blocks_of(token_ids):
159         child = node.children.get(chunk)     # ◀ dict lookup by the block's 16 token ids
160         if child is None or child.block_id is None:
161             break                            # ◀ longest cached prefix ends here
162         node = child
163         block_ids.append(child.block_id)
⋮                                             # ◀ note: NO refcount taken here (a probe)
210 def insert(self, token_ids, block_ids, now=None) -> int:
226     for index, chunk in enumerate(chunks):
229         child = node.children.get(chunk)
230         if child is None:
231             child = RadixNode(token_ids=chunk, block_id=block_ids[index], parent=node)
232             node.children[chunk] = child
235             self.allocator.incref(block_ids[index])   # ◀ the TREE's own reference keeps the block alive
⋮
253     return [node for node in self._nodes.values()
256             if node.is_leaf and node.block_id is not None
258             and self.allocator.ref_count(node.block_id) == 1]   # ◀ evictable ⇔ leaf AND only the tree holds it
⋮
278     while freed < num_blocks:
279         candidates = self._evictable_leaves()
282         candidates.sort(key=lambda n: n.last_accessed)   # ◀ LRU first
283         for node in candidates:
286             self._remove(node)                          # ◀ detach + allocator.free (tree's ref)
287             freed += 1
```
| Name | What it is |
|---|---|
| `RadixNode` | dataclass (`:52-71`): `token_ids` (the key), `block_id`, `parent`, `children: dict[tuple, RadixNode]`, `last_accessed` |
| `PrefixMatch` | dataclass (`:74-88`): `block_ids`, `num_tokens`, `nodes` |
| `self._nodes` | every cached node, for eviction scans |

### A.6 Scheduler: running first, then admission (`scheduler.py`)
```python
322 def _schedule_running(self, output, budget):
325     while index < len(self.running):
326         request = self.running[index]
327         num_new = request.num_uncomputed_tokens            # ◀ 1 for a decode
332         needed = request.table.blocks_needed_for(num_new)   # ◀ 0 or 1 block
333         while self.allocator.num_free_blocks < needed:
344             if self._evict_for(needed):                    # ◀ 1st choice: drop unused cached blocks
345                 continue
346             victim = self._pick_victim(protect_index=index) # ◀ 2nd: preempt the NEWEST running request
347             if victim is None:
349                 self._preempt(request); output.preempted.append(request); break
352             self._preempt(victim)                          # ◀ recompute later: free blocks, computed=0
353             output.preempted.append(victim)
354         else:                                              # ◀ Python while-else: runs if no `break`
356             output.scheduled.append(self._prepare(request))
357             budget -= num_new; index += 1
⋮
383 while self.waiting:                                       # ◀ inside _schedule_waiting
386     request = self.waiting[0]                             # ◀ strict FCFS head
391     if request.num_uncomputed_tokens > budget: break
402     match = self._match_cached_prefix(request, table)     # ◀ PROBE only (record_stats=False)
403     needed = self._blocks_to_admit(request, table, match) # ◀ blocks beyond the cached prefix
414     while self.allocator.num_free_blocks - needed < self.watermark_blocks:
415         if self._evict_for(needed) == 0: break
417         match = self._match_cached_prefix(request, table) # ◀ eviction may have removed our prefix → re-price
418         needed = self._blocks_to_admit(request, table, match)
419     if self.allocator.num_free_blocks - needed < self.watermark_blocks: break
424     chunk = self._first_uncached_chunk(request, match)
425     if chunk is not None and chunk in pending: break      # ◀ same uncached prefix admitted this iteration → wait 1
430     if match is not None:
431         self._adopt_cached_prefix(request, table, match)  # ◀ COMMIT: incref, adopt_blocks, num_computed_tokens
⋮
440     output.scheduled.append(self._prepare(request))
⋮
578 num_full = request.num_computed_tokens // self.block_size  # ◀ _publish_blocks: only full, computed blocks
579 if num_full <= request.num_published_blocks: return 0
581 newly_cached = self.prefix_cache.insert(request.all_token_ids[: num_full * self.block_size],
583                                         request.table.block_ids)
⋮
626 def _preempt(self, request):
633     request.table.free()                 # ◀ refs dropped (published blocks fall to the tree's ref only)
634     request.num_computed_tokens = 0      # ◀ will re-prefill prompt + generated tokens
644     self.waiting.insert(0, request)      # ◀ front of the queue: keeps FCFS
⋮
667 for scheduled, token_id in zip(output.scheduled, sampled_token_ids):   # ◀ inside update()
670     request.num_computed_tokens = request.num_tokens   # ◀ BEFORE appending the new token (it has no KV yet)
673     self._publish_blocks(request)
677     request.output_token_ids.append(int(token_id))
681     if hit_limit or hit_eos: self._finish(request, now)   # ◀ publish, then table.free() → blocks back now
```
| Name | What it is |
|---|---|
| `request.num_uncomputed_tokens` | `num_tokens − num_computed_tokens` (prompt + output minus tokens with KV) |
| `budget` | `max_num_batched_tokens` (8,192) minus tokens already scheduled this iteration |
| `watermark_blocks` | 1% of the pool kept free |
| `pending` | set of first-uncached 16-token chunks admitted this iteration |
| `_prepare` | grows the block table, returns `ScheduledRequest(token_ids, positions, slot_mapping)` (`:599-612`) |

### A.7 One engine step: `engine.py`
```python
111 def step(self) -> list[Request]:
113     output = self.scheduler.schedule()
122     token_ids, positions, slots, query_lens = [], [], [], []
123     for scheduled in output.scheduled:          # ◀ flatten into ONE ragged batch, no padding
124         token_ids.extend(scheduled.token_ids)
125         positions.extend(scheduled.positions)
126         slots.extend(scheduled.slot_mapping)
127         query_lens.append(scheduled.num_tokens)
132     logits_indices, offset = [], 0
133     for q_len in query_lens:
134         offset += q_len
135         logits_indices.append(offset - 1)        # ◀ last token of each request only
137     with torch.inference_mode():
138         logits = self.model(input_ids=..., positions=..., cache=self.cache, slot_mapping=...,
143                             block_tables=[s.request.table.block_ids for s in output.scheduled],
144                             query_lens=query_lens,
145                             context_lens=[s.request.num_tokens for s in output.scheduled],
146                             logits_indices=self._tensor(logits_indices))
149     sampled = logits.argmax(dim=-1).tolist()     # ◀ greedy decoding
150     return self.scheduler.update(output, sampled)
```
Example: request A prefills 20 tokens and B decodes 1. Then `token_ids` has 21 entries, `query_lens = [20, 1]`, and `logits_indices = [19, 20]`.

### A.8 Llama layer over the paged cache: `model/llama.py`
```python
271 decode_meta = None
272 if self.use_fused_decode and can_use_decode_kernel(query_lens, hidden.device):   # ◀ every query_len == 1
273     decode_meta = build_decode_metadata(block_tables, context_lens, hidden.device)  # ◀ once per pass, not per layer
275 for index, layer in enumerate(self.layers):            # ◀ 16 decoder layers (HF weights reused)
280     residual = hidden
281     x = input_norm(hidden)                             # ◀ RMSNorm in fp32
283     q = attn.q_proj(x).view(total_tokens, self.num_q_heads, self.head_dim)    # ◀ [T, 32, 64]
284     k = attn.k_proj(x).view(total_tokens, self.num_kv_heads, self.head_dim)   # ◀ [T, 8, 64]
285     v = attn.v_proj(x).view(total_tokens, self.num_kv_heads, self.head_dim)
286     q, k = apply_rope(q, k, cos, sin)                  # ◀ rotary with llama3 scaling, θ = 500000
290     write_kv_to_cache(cache, index, slot_mapping, k, v)   # ◀ write BEFORE attending (seq sees its own new tokens)
292     if decode_meta is not None:
293         out = paged_attention_decode(q, cache, index, decode_meta, self.scale)   # ◀ A.4
294     else:
295         out = paged_attention(q, cache, index, block_tables, query_lens, context_lens, self.scale)  # ◀ A.3
298     out = out.reshape(total_tokens, self.num_q_heads * self.head_dim)   # ◀ [T, 2048]
299     hidden = residual + attn.o_proj(out)
301     residual = hidden
302     x = post_norm(hidden)
304     hidden = residual + mlp.down_proj(nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))  # ◀ SwiGLU MLP
308 hidden = self.norm(hidden)
310 hidden = hidden[logits_indices]                        # ◀ keep last token per request
311 return self.lm_head(hidden)                            # ◀ [num_requests, 128256]
```
