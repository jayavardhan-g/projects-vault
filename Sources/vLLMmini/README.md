# mini_vllm

A from-scratch LLM inference server with **block-based paged KV-cache memory
management**, **continuous batching**, **radix-tree prefix sharing** across
requests, and a **fused Triton attention kernel that reads directly from
non-contiguous paged memory**.

The problem it addresses is the one vLLM and SGLang exist to solve: naive
serving allocates KV cache as one dense, contiguous, padded tensor per batch.
That wastes GPU memory to internal fragmentation, caps concurrency well below
what the hardware can hold, and re-computes shared prefixes from scratch for
every request that shares them.

## Architecture

```
Client → HTTP front-end (FastAPI/asyncio)
            │
            ▼
     Continuous-Batching Scheduler
     (admission control, iteration-level scheduling, preemption)
            │
            ▼
     Paged KV-Cache Memory Manager
     (per-sequence block table → physical block pool, like OS page tables)
            │
            ▼
     Radix-Tree Prefix Cache
     (shares KV blocks across sequences with common prefixes; reference-counted
      so a shared block is never evicted while a live sequence still needs it)
            │
            ▼
     Fused Paged-Attention Kernel (Triton)
     (gathers scattered physical blocks per sequence for attention compute)
            │
            ▼
     Model Runner (Llama 3.2 1B, PyTorch)
```

## Status

| Phase | Description | State |
|---|---|---|
| 0 | Setup, repo scaffold, naive HF `generate()` baseline | ✅ done |
| 1 | Paged KV-cache memory manager (block table + allocator) | ✅ done |
| 2 | Naive paged attention (correctness vs. unpaged baseline) | ✅ done |
| 3 | Continuous-batching scheduler | ✅ done |
| 4 | Radix-tree prefix cache with refcounted eviction | ✅ done |
| 5 | Fused Triton paged-attention kernel (decode path) | ✅ done |
| 6 | Benchmarking vs. HF baseline | ✅ done |
| 6 | Real vLLM as an upper-bound reference | ✅ done |
| 6.5 | FastAPI front-end, buffered and streaming | ✅ done |

Every item in the definition of done is met: a working server, a benchmark
suite, this write-up, a prefix-cache hit rate demonstrated on a multi-turn
workload, and a measurement against the production system this is a model of.
The tree is green at **421 tests** — 349 CPU (~6.5 min) and 72 GPU (~14 min),
no skips and no xfails.

Phase 6.5 is not in the original build plan: the architecture and definition of
done both call for an HTTP server, but no phase created one. It was scheduled
after benchmarking deliberately — it is a thin async wrapper over the engine's
existing `step()` loop, and building it before Phase 5 would have meant
rewriting it once the hot path changed, which is exactly what Phase 5 then did.

### Where performance actually stands

Measured like for like in Phase 6 — one process, one GPU, identical prompts and
identical per-request output lengths, five rounds with the two arms alternating
within each round:

| | HF baseline | mini_vllm | |
|---|---|---|---|
| output throughput | 631.5 tok/s | **2415.0 tok/s** | **3.82×** |
| request throughput | 4.93 req/s | **18.87 req/s** | 3.83× |
| TTFT p50 | 2448 ms | **179 ms** | 13.7× |
| TTFT p99 | 4879 ms | **252 ms** | 19.4× |
| latency p50 | 4049 ms | **1690 ms** | 2.4× |
| latency p99 | 6486 ms | **1696 ms** | 3.8× |

**3.82× the naive baseline**, from 2.9× slower before Phase 5. This replaces a
4.4× previously quoted here, which compared a baseline forced to exactly
`--max-new-tokens` against a server running varied output lengths — part of
that ratio was a measurement of the workload rather than of the system. The
number went down when the confound came out, which is the point of removing it.

Two things this number is, and is not:

* The **cleanest isolated result remains the 7.49× kernel speedup** below:
  fused against naive with the same weights, scheduler, pool and prompts,
  differing only in which attention implementation decode calls. The 3.82×
  additionally folds in continuous batching and paged memory, which is what
  makes it the right number to quote for the *system* and the wrong one for the
  *kernel*.
* Prefill still runs the naive gather. Decode is where the time went, so this
  moves the headline; it does not mean everything is fused.

And the number in the other direction, measured the same paired way against
**real vLLM 0.26.0 over HTTP: 0.16×, i.e. vLLM is 6.2× faster than this.** That
is the honest ceiling — roughly 6× on TTFT and 6× on per-token latency alike,
which is what a fused prefill, chunked prefill and CUDA graphs are worth. See
"Real vLLM" under Phase 6.

## Phase 6.5: the HTTP front end

`src/mini_vllm/server.py`. `POST /generate`, plus `/health` and `/stats`.

```bash
CUDA_VISIBLE_DEVICES=7 python -m mini_vllm.server --port 8000

curl -s localhost:8000/generate -H 'content-type: application/json' \
     -d '{"prompt": "The capital of France is", "max_tokens": 32}'
```

```json
{"request_id": "f634...", "text": " Paris. The Eiffel Tower is located in Paris...",
 "prompt_tokens": 6, "output_tokens": 32, "finish_reason": "length",
 "ttft": 0.036, "latency": 0.71, "tpot": 0.0151,
 "num_preemptions": 0, "num_cached_tokens": 0}
```

**The engine is not made concurrent, and that is the whole design.** Every
invariant in the block pool and the radix tree assumes a single caller of
`step()`. What this layer adds is concurrent *waiting*: many HTTP handlers
suspend on their own request while one background task drives `step()` and
resolves each future as its request finishes. Handlers and that task are
coroutines on one event loop, and the loop never awaits inside a step, so
`add_request` can only ever interleave between steps — which is exactly where
the scheduler documents it as safe. No locks, no threads, no second caller.

Eight concurrent curls against the live server reach `max_running: 8` in
`/stats`, so requests genuinely share iterations rather than queueing: TTFT
30–52 ms for seven of the eight, 11.4–15.1 ms per output token, 0.59–1.11 s
end to end for 48 tokens each. `/stats` afterwards shows 4070 of 4096 blocks
free — the 26 held back are the prefix cache doing its job, not a leak.

### The first request paid 17 seconds for everyone else

Measured on the first version, which had no warmup: the first request through a
fresh server took **17.0 s** end to end — 2.72 s to its first token, then 462 ms
per token. The identical request against a warmed server takes **0.36 s**: 21.7
ms to first token, 10.9 ms per token. Same 32 tokens, same text, 47× the
latency. None of that is model work. It is CUDA context creation, cuBLAS
handles, and the Triton decode kernel compiling for its first shape, all landing
on whichever request happens to arrive first.

`warmup()` runs one throwaway 8-token request through prefill and several decode
steps at startup, then drops its cache entry and zeroes the stats so `/stats`
does not open on a lie. `--no-warmup` disables it.

### What the HTTP layer can get wrong

None of it is numerics, and all of it looks like a hang or a 500 rather than a
wrong answer — so it is tested against the hash-chain oracle from Phase 4 on
CPU, in seconds, rather than against bfloat16 on a GPU in minutes:

* **Admission rejections must be a 4xx.** A prompt over
  `max_num_batched_tokens` can never be scheduled, so it comes back **413** with
  the scheduler's own explanation, not a traceback and a 500. Retrying the same
  request id after a rejection has to work, which means nothing may be
  registered until admission has accepted.
* **A client that hangs up must have its request aborted.** Otherwise the engine
  keeps decoding for nobody, holding blocks and preempting requests that do have
  a caller.
* **A dead engine loop must fail every pending future.** Without that, the first
  failure is invisible and every request after it waits forever — the failure
  mode that reads as a hang in production. `/health` reports the stored failure.

The tests were checked by breaking the server, the same way the Phase 5 kernel
tests were — **eight mutations across the phase**, each one a defect a plausible
implementation could actually have:

| mutation | caught by the first version of the suite? |
|---|---|
| drop the per-iteration `await` (no batching) | yes |
| `cancel()` forgets to abort the request | **no** |
| resolve futures without removing them from the pending map | **no** |
| register the future before admission can reject it | **no** |
| remove the disconnect poll | **no** — and only by hanging |
| emit deltas without the split-codepoint hold-back | yes |
| drain streams *after* resolving finished requests (drops the last token) | yes |
| never resolve a future at all | only by hanging |

Five of the eight were missed or produced a hang rather than a failure. The
abort test, the pending-map test, the retry-after-rejection test and the
disconnect test all exist because of that pass, and the two hangs were rewritten
to fail in seconds instead. A suite that only fails by timing out is a suite
that will be disabled the first time CI is slow.

### Streaming, without touching the engine's contract

`{"stream": true}` on the same endpoint returns `text/event-stream`: one frame
per token, then a final frame carrying the same usage and timing fields the
buffered response returns, then `data: [DONE]`.

The obvious implementation is a per-token callback on the engine. It is the
wrong one here: `step()`'s return value is the contract every benchmark in
`results/` was measured against, and widening it to serve the front end couples
the two. It is not needed either — the loop already holds each `Request`, whose
`output_token_ids` is the list `step()`'s return value is derived from, so
after each iteration it pushes whatever is new onto that request's queue. The
engine gained nothing and lost nothing.

**A token is not a character.** One byte-pair token can be half a multi-byte
codepoint, so decoding tokens individually emits U+FFFD and then the real
character on the next token — visibly broken output for any non-ASCII text.
Deltas are therefore produced by decoding the whole output so far and emitting
the suffix, holding a frame back while the accumulated decode still ends in a
replacement character. The concatenation of the deltas is exactly the
non-streaming `text`, which is what the tests assert against the oracle.

**A hung-up client has to be polled for, not inferred.** The first version
relied on the generator being closed when the connection dropped, and its test —
which closed the generator by hand — passed. Against a live server it did not
work at all: a `curl` killed mid-stream left the request decoding for 12 s to
its token limit, for a caller that had gone. A plain streaming endpoint on the
same server *was* closed promptly, so it is not the engine starving the event
loop. `_sse` now checks `is_disconnected()` after every frame, which stops the
work within one token, and the test drives that mechanism rather than the one
that only looked right. A stream cut short — by a hang-up or by an engine
failure — says so in a final frame: an SSE body that simply stops is
indistinguishable from one that finished.

## Phase 6: benchmarking

Every earlier phase isolated one variable. This one deliberately does not: it
measures the system as deployed against the system you would otherwise have
written. That makes the comparison meaningful and the methodology fragile, so
`benchmarks/serving.py` removes three confounds explicitly.

**Identical work.** Both arms receive the same prompts *and* the same
per-request output length list. Two length modes are reported. `uniform` gives
every request the same token count: a static batch whose members all finish
together wastes nothing, so Phase 3's contribution is stripped out and what
remains is the attention and memory path. That is the conservative floor and
the number quoted above. `varied` spreads lengths 0.2–1.0×, the realistic case.

**Identical machine conditions.** Arms alternate within each round and the
median of the paired per-round ratios is reported, as Phase 5 established.

**Identical arrival semantics.** All requests arrive at t=0 in both arms. This
was a bug first time through, and it mattered: the HF arm timed each request
from the moment *its own batch* was reached, which deletes the time a request
spent queued behind earlier batches — the dominant term in its TTFT and exactly
what static batching costs you. Corrected, the baseline's TTFT p50 went from
24.6 ms to 2448 ms. The bug flattered the baseline, so fixing it moved the
result in our favour, which is the direction that deserves the most scrutiny.

### Varied lengths score *lower* than uniform, and that is not a regression

3.68× against 3.82×. Continuous batching is supposed to win *more* when lengths
vary, so this looks backwards. It is an artefact of the metric: with all 32
requests submitted at once, wall time is set by the longest request, so
spreading the lengths delivers fewer total tokens over roughly the same wall
clock and depresses tok/s in both arms about equally.

Request throughput is the metric that shows the difference — 19.35 against 5.26
req/s — and the static-batch tax is visible directly as
`static_batch_waste_frac = 0.265`: the baseline computes 3840 tokens to deliver
2824. Both are in `results/serving.json`.

### Prefix cache: 91% hit rate by turn 4

`benchmarks/multi_turn.py` drives conversations that grow — turn N+1's prompt is
turn N's prompt plus the model's own reply plus a new question. This is what the
definition of done means by multi-turn, and it is not what the Phase 4 numbers
measured: those came from the CPU oracle, over *concurrent* requests sharing one
head.

| turn | prompt tokens | reused | hit rate |
|---|---|---|---|
| 1 | 439 | 0 | 0.0% |
| 2 | 1080 | 896 | 83.0% |
| 3 | 1739 | 1536 | 88.3% |
| 4 | 2389 | 2176 | **91.1%** |

Overall 81.6% token-weighted, zero evictions.

Turn 1 reading exactly 0.0% is what makes the rest trustworthy, and it did not
at first — it read 40.1%. The per-conversation marker had been appended to the
persona text rather than prepended, so every conversation sharing a persona
also shared a ~40-token head, and those cross-conversation hits were being
counted as self-reuse. A prefix cache keys on the prefix, so isolation has to
start at token 0.

### The cache saves 91% of the tokens and 0% of the time, and why

Against a cache-off control the multi-turn workload runs at **1.00×**. That
reads as a broken cache. It is not, and `benchmarks/prefix_cache_value.py`
measures the reason rather than arguing it.

A forward pass carries a fixed cost — dispatch, kernel launches, 16 layers of
small ops — independent of how many tokens ride along:

| prompt tokens | prefill | regime |
|---|---|---|
| 128 | 15.6 ms | overhead-bound |
| 512 | 15.5 ms | overhead-bound |
| 1024 | 19.1 ms | overhead-bound |
| 2048 | 49.8 ms | compute-bound |
| 8192 | 665.6 ms | compute-bound |

Below ~2k tokens the tokens are nearly free, so skipping 85% of them skips
nothing worth having. These conversations peak at ~1.2k tokens, inside the flat
region. Past the crossover the same hit is worth a great deal — timing one
request with 90% of its prefix cached against the same request with an empty
cache:

| context | no cache | 90% cached | speedup |
|---|---|---|---|
| 512 | 15.5 ms | 15.5 ms | 1.00× |
| 1024 | 19.1 ms | 16.0 ms | 1.19× |
| 2048 | 49.8 ms | 17.0 ms | 2.92× |
| 4096 | 173.9 ms | 28.2 ms | 6.16× |
| 8192 | 665.5 ms | 89.8 ms | **7.41×** |

**A hit rate quoted without the context length it was measured at does not tell
you whether the cache earns its complexity.** On this model and card it earns it
above ~2k tokens of shared context and not below.

### Fragmentation

Reproduced from Phase 1 unchanged — the calculation is deterministic and needs
no GPU:

| workload | naive | paged | |
|---|---|---|---|
| independent | 16.8% | **3.9%** | 4.3× better, 904 slots saved |
| shared_prefix | 0.8% | 2.8% | worse by 176 slots |

The second row is not a regression, and it is the more instructive one. Paging
recovers *padding* waste, which only exists when prompt lengths differ within a
batch. `shared_prefix` prompts are near-uniform, so there is almost no padding
to recover and block rounding — up to `block_size - 1` slots per sequence —
costs more than it saves. Paging's memory win comes from length variance; on
this workload the win comes from Phase 4 sharing instead.

### Real vLLM: it is 6.2× faster than this

The brief asks for vLLM as an upper-bound reference "if feasible". It is:
vLLM 0.26.0 pins `torch==2.11.0`, matching this build exactly (0.27.0 moved to
2.13.0 and would not fit). It lives in its own `vllm-ref` env so the environment
every other number here was measured in stays untouched — see SETUP.md.

`benchmarks/vllm_reference.py`, 32 requests × 128 tokens, five paired rounds:

| | vLLM 0.26.0 | ours |
|---|---|---|
| output throughput | **6117.9 tok/s** | 1007.2 tok/s |
| request throughput | 47.80 req/s | 7.87 req/s |
| TTFT p50 | 164.9 ms | 1088.4 ms |
| TPOT p50 | 3.73 ms | 23.1 ms |
| latency p99 | 649.7 ms | 4056.6 ms |

**Median paired ratio 0.16× — vLLM is 6.2× faster.** Tight across rounds (0.15,
0.15, 0.16, 0.17, 0.19), and both arms produced exactly 4096 tokens in every
round, so this is the same work measured two ways rather than two workloads.

The measurement is deliberately structured so the number cannot flatter either
side. One asyncio client streams both servers and times both from its own
`send`, so TTFT means one thing. Both servers stay up on one card and rounds
alternate between them, which buys the paired statistic — this box drifts ~1.5×
with other users' jobs — without a model reload per round. Output length is
re-tokenized client-side rather than trusted from either server's accounting,
since both hold frames back at codepoint boundaries and need not agree on when.
A discarded warm-up round keeps first-request kernel compilation out of both.

**The gap is not the HTTP layer.** Measured on the same card minutes later, the
in-process engine ran the same workload at 866 tok/s against the server's 1007
— the front end costs nothing, and both are far below the 2415 tok/s of the
Phase 6 headline because a neighbouring job had the card at 100%. That is also
why the ratio is the number to read and the absolutes are not.

Where the 6.2× lives, in the order the numbers implicate:

* **TTFT is 6.6× worse**, which is prefill, and prefill is the part we never
  fused. It still materialises the whole `[heads, q_len, ctx_len]` score matrix
  in the naive Python gather; vLLM runs a fused prefill kernel with chunking.
* **TPOT is 6.2× worse.** Our decode kernel is fused but does no split-K, no
  CUDA graphs, and re-enters Python for every iteration of the scheduler loop.
  vLLM captures the decode step as a graph and pays none of that.
* Nothing here suggests a defect. It is the size of what is left to build, and
  it is roughly the same multiple in both places.

## Phase 5: the fused decode kernel

`paged_attention_triton.py` replaces the Python gather on the decode path. One
Triton program per (sequence, query head) walks that sequence's block table and
reduces over its context with an online softmax, so the gathered KV is never
materialised — the indirection happens in registers instead of in a loop of
tensor indexing ops.

Measured on 32 requests, five alternating rounds:

| | naive gather | fused kernel |
|---|---|---|
| output throughput | 379.1 tok/s | **2839.9 tok/s** (7.49×) |
| wall time | 5.58 s | **0.74 s** |
| p50 latency | 4.752 s | **0.564 s** |
| p99 latency | 5.575 s | **0.743 s** |
| mean TTFT | 0.0997 s | 0.0999 s |

**TTFT is unchanged, and that is the result being correct rather than a
disappointment.** Time to first token is dominated by prefill, which this phase
deliberately does not touch. A kernel that claimed to improve it would be
evidence the measurement was wrong.

### Why the reported speedup is a paired median

Absolute throughput on this box drifts by ~1.5× as other users' jobs come and
go — the naive arm alone measured 287, 378, 379, 379 and 379 tok/s across five
rounds of the same command. Taking the median of each arm independently and
dividing gave **4.77×** on one three-round run whose paired ratios were 7.61,
4.76 and 7.51. The arms had simply been measured under different machine loads.

The benchmark therefore alternates the two arms within each round and reports
the **median of the per-round ratios**, so a transient neighbour job perturbs
both arms together and cancels. Both numbers are written to the results JSON;
the unpaired one is kept only so the difference stays visible.

### Decode only, on purpose

The kernel refuses any step where a sequence contributes more than one query
token, and such steps fall back to the reference. During decode the single
query sits at absolute position `context_len - 1`, so every key in the context
is legal and causal masking reduces to a bounds check against `context_len`.
That collapse is the reason the kernel is short enough to trust. It stops
holding the moment a prefill enters the batch, which is why the dispatch
predicate is a hard gate rather than a heuristic.

There is also no split-K. At 32 sequences × 32 query heads there are already
1024 programs, which fills an H200 at these batch sizes; flash-decoding's
context partitioning would add a two-pass log-sum-exp rescale for occupancy
this workload does not lack. It is recorded in the module docstring as the
change to make if profiling ever shows the card idle on long contexts.

### The tests were checked by breaking the kernel

Three bugs in this kernel produce plausible output rather than a crash: reading
the wrong KV head under GQA, attending to the stale tail of a partly-filled
final block, and ignoring the block table. Each was introduced deliberately and
the suite re-run — they fail 19, 19 and 19 of the 28 tests respectively.

That last one is the one worth recording. It first failed only **2** tests,
because `BlockAllocator` hands a lone sequence a *contiguous* run starting at
block 0 — and at block 0 the logical and physical indices coincide, so a kernel
that never reads the block table reads exactly the right data anyway. The tests
now fragment the pool before every case and assert no sequence starts at block
0. A test suite that passes for a kernel with the paging removed is not testing
paging, and nothing but mutating the kernel would have revealed it.

### Divergence from the naive path is 12 requests in 32, and it is fine

Greedy decoding takes an argmax; the two paths reduce over the context in
different orders; in bfloat16 the top two candidates are routinely closer than
that difference. One flip changes every subsequent token, so a single coin toss
shows up as a whole diverged request.

Every divergence was graded by the model's confidence at the point it happened:
**max top-2 logit gap 0.125 against a 0.5 near-tie threshold, and six of the
twelve were exactly 0.0000** — the top two logits bitwise equal. Zero confident
disagreements. The benchmark reports that classification rather than a raw
count, because "20/32 identical" reads as a red flag when the real finding is
that nothing disagreed anywhere the model had an opinion.

## Model

Development targets **`meta-llama/Llama-3.2-1B-Instruct`**.

The choice is about KV-cache characteristics, not output quality — Qwen's small
models score higher on general benchmarks, but this project measures throughput,
latency, fragmentation, and cache hit rate, never answer quality. What matters:

| | Llama-3.2-1B | Qwen2.5-0.5B | Qwen3-0.6B |
|---|---|---|---|
| Layers | 16 | 24 | 28 |
| Q/KV heads | 32 / 8 (4:1) | 14 / 2 (7:1) | 16 / 8 (2:1) |
| head_dim | 64 | 64 | 128 |
| **KV bytes/token (fp16)** | **32 KB** | **12 KB** | **112 KB** |

At 12 KB/token a 0.5B model fits millions of tokens on an H200, so the block
pool never fills and the preemption and eviction paths — the whole point of
Phases 3 and 4 — never fire under realistic load. Llama 3.2 1B's 32 KB/token
creates real pressure, and its 4:1 GQA / head_dim 64 shape is what reference
paged-attention kernels are written against, which keeps Phase 6 numbers
checkable against published figures.

## Quickstart

Environment setup, including two host-specific gotchas, is in
[SETUP.md](SETUP.md).

```bash
# Sends build scratch and large artifacts to warm storage; required on this host
source scripts/env.sh

# CPU-only unit tests. Both halves of this matter: a bare `pytest` collects the
# gpu-marked tests too and grabs whatever CUDA_VISIBLE_DEVICES points at.
CUDA_VISIBLE_DEVICES="" pytest -m "not gpu"

# Pick a free GPU first — shared machine, and what is free changes fast
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

CUDA_VISIBLE_DEVICES=7 pytest -m gpu

# Phase 6.5: serve it. Loading weights over NFS takes a few minutes before the
# port opens; /health is the thing to poll.
CUDA_VISIBLE_DEVICES=7 python -m mini_vllm.server --port 8000
curl -s localhost:8000/generate -H 'content-type: application/json' \
     -d '{"prompt": "The capital of France is", "max_tokens": 32}'

# ...or token by token
curl -N -s localhost:8000/generate -H 'content-type: application/json' \
     -d '{"prompt": "The capital of France is", "max_tokens": 32, "stream": true}'

# Phase 0 baseline
CUDA_VISIBLE_DEVICES=7 python benchmarks/baseline_hf.py \
    --num-requests 32 --batch-size 8 --max-new-tokens 128

# Phase 6: the headline comparison, five paired rounds
CUDA_VISIBLE_DEVICES=7 python benchmarks/serving.py --repeats 5

# Phase 6: prefix-cache hit rate on growing conversations, and what a hit costs
CUDA_VISIBLE_DEVICES=7 python benchmarks/multi_turn.py --conversations 8 --turns 4
CUDA_VISIBLE_DEVICES=7 python benchmarks/prefix_cache_value.py

# Phase 6: against real vLLM. Both servers stay up on one card; see SETUP.md
python benchmarks/vllm_reference.py --repeats 5
```

### Where the numbers live

Every benchmark writes its raw output to `results/`, tracked in git, so any
claim in this file can be checked against the run that produced it without
re-running anything.

| file | produced by | holds |
|---|---|---|
| `serving.json` | `serving.py` | the 3.82× headline, both length modes, per-round ratios |
| `vllm_reference.json` | `vllm_reference.py` | the 0.16× against real vLLM, per round |
| `decode_kernel.json` | `decode_kernel.py` | the 7.49× fused-vs-naive kernel result |
| `multi_turn.json` | `multi_turn.py` | per-turn prefix-cache hit rates |
| `multi_turn_prefill_heavy.json` | `multi_turn.py --turns 6 --max-new-tokens 8` | the same, prefill-weighted |
| `prefix_cache_value.json` | `prefix_cache_value.py` | the prefill crossover, and what a hit is worth |
| `continuous_batching.json` | `continuous_batching.py` | Phase 3 static vs. continuous |
| `fragmentation.json` | `fragmentation.py` | Phase 1 padding waste vs. paged waste |
| `baseline_hf_*.json` | `baseline_hf.py` | the Phase 0 baselines |

Absolute throughput in these files drifts by ~1.5× with other users' jobs on the
shared machine they were measured on. **Every headline is a paired ratio** —
arms alternating within a round, median of the per-round ratios — for exactly
that reason.

## The Phase 0 baseline

`benchmarks/baseline_hf.py` is deliberately naive in exactly the three ways the
rest of the project fixes, so that later phases have something honest to beat:

1. **Static batching** — a batch finishes only when its slowest member does;
   short requests idle while holding memory. Phase 3 fixes this.
2. **Padded contiguous KV** — every sequence is left-padded to the batch's
   longest prompt and KV is allocated across that padded length. The script
   reports the resulting waste as `kv_padding_waste_frac`. Phase 1 fixes this.
3. **No cross-request reuse** — two requests sharing a system prompt each
   prefill it from scratch. Phase 4 fixes this.

Decoding is greedy so later phases can be compared token-for-token, not just
statistically. Metric definitions are shared (`src/mini_vllm/metrics.py`) so the
baseline and the finished server are measured identically.

### Baseline numbers

Llama-3.2-1B-Instruct, bfloat16, 1× H200, 32 requests × 128 output tokens,
static batch size 8:

| | independent | shared_prefix |
|---|---|---|
| output throughput | 647 tok/s | 656 tok/s |
| TTFT p50 | 30.1 ms | 18.6 ms |
| TPOT p50 / p99 | 11.9 / 11.9 ms | 11.6 / 11.8 ms |
| latency p50 | 1537 ms | 1506 ms |
| peak GPU memory | 2.42 GiB | 2.44 GiB |
| **KV padding waste** | **16.8%** | 0.8% |

Two things to carry into later phases:

* **16.8% of allocated KV slots on the `independent` workload hold nothing.**
  That is pure internal fragmentation from padding every sequence to its
  batch's longest prompt, and it is the number the Phase 1 paged allocator
  exists to drive toward zero. It is only 0.8% on `shared_prefix` because those
  prompts are near-uniform in length — that workload's win comes from *reuse*
  (Phase 4), not padding, and its ~4.4k prompt tokens are currently prefilled
  from scratch on every request.

* **Baseline throughput varies with machine load.** Repeat runs of the same
  command produced 647 and 1001 tok/s at host load averages of 7.7 and 6.0.
  At this model size decode is bound by host-side kernel launches, so other
  users' CPU activity moves the number by ~1.5x. Phase 6 must therefore run
  baseline and server back-to-back and report repeated runs, not a single
  measurement of each.

## Phase 1: paged memory manager

`BlockAllocator` owns a fixed pool of reference-counted physical blocks and
hands out integer ids. `BlockTable` maps one sequence's logical token positions
onto those ids — the direct analogue of an OS page table:

```
logical tokens   0 1 2 3 | 4 5 6 7 | 8 9
                    |         |        |
physical blocks    #17       #4       #23     (scattered across the pool)
```

Blocks are reference counted from the start even though nothing shares them
yet. That is the mechanism Phase 4's prefix cache needs — a block must not
return to the pool while any sequence still references it — and retrofitting it
later would mean auditing every call site for use-after-free.

`python benchmarks/fragmentation.py` compares allocation strategies on the same
workload as the Phase 0 baseline (tokenizer only, no GPU):

| workload | naive padding | paged (block 16) | |
|---|---|---|---|
| independent | 16.8% | **3.9%** | 4.3× better, 904 slots saved |
| shared_prefix | 0.8% | **2.8%** | *worse* by 176 slots |

The naive 16.8% matches what the Phase 0 baseline measured during real
generation, which cross-checks the accounting against an actual run.

**Paging loses on `shared_prefix`, and that is the expected result.** Those
prompts are 132–139 tokens — near-uniform — so pad-to-longest wastes almost
nothing and block rounding costs more than it recovers. Paging's win comes from
length *variance*, which `independent` has and `shared_prefix` does not. That
workload's payoff is Phase 4 instead: its 4363 prompt tokens are largely
duplicated across requests and are currently prefilled from scratch every time.

Internal fragmentation under paging is bounded by `block_size - 1` tokens per
sequence regardless of sequence length, because only a sequence's final block
can be partial. The baseline's waste, by contrast, grows with how uneven the
batch is.

## Phase 2: attention over scattered blocks

`paged_attention.py` computes attention where KV lives in non-contiguous
blocks, so every read goes through the block table's indirection instead of
walking a contiguous tensor. It is deliberately unoptimised — plain PyTorch
indexing, one sequence at a time — because its job is to be *obviously correct*
and give Phase 5's fused Triton kernel something trustworthy to be checked
against.

`model/llama.py` re-implements Llama's forward pass over that cache, reusing
HF's loaded weights so there is no checkpoint conversion to get wrong. Tokens
are flattened and ragged (`[total_tokens]`, not `[batch, seq_len]`), so no
padding exists anywhere — and Phase 3's scheduler gets the iteration-level
control that `generate()` does not expose.

Correctness is established at two levels, since each catches what the other
misses:

* **Tensor level** — paged output vs a dense reference on identical data, at
  near-exact tolerance. Both paths share the same attention core, so any
  difference is an indexing bug rather than float noise. The pool is
  deliberately fragmented first: with contiguous blocks, an off-by-one in the
  indirection is invisible.
* **Model level** — greedy token ids against HuggingFace's, for single
  sequences, ragged batches, and a fragmented pool.

On that second point, exact token equality turns out not to be assertable in
general, and pretending otherwise would make the suite flaky rather than
strict. With logits around 20 in bfloat16, the top two candidates are often
separated by less than the arithmetic noise between two orderings of the same
computation. A measured case (`"Once upon a time"`, token 11):

| | top-1 | top-2 | gap |
|---|---|---|---|
| HF | `' T'` 20.000 | `' the'` 19.875 | **0.125** |
| ours | `' T'` 20.125 | `' the'` 19.875 | 0.250 |

Max logit deviation across the vocabulary is 0.156 — larger than HF's own
top-2 gap — so the argmax flips depending on whether the KV was built in one
prefill or across eleven decode steps. Both answers are equally defensible.
`tests/greedy_compare.py` therefore fails on *confident* disagreement, which is
what a genuine bug looks like, and accepts only measured near-ties.

### The bug this phase actually found

`config.rope_theta` does not exist in transformers 5.x — RoPE settings moved
into `config.rope_parameters`. Reading it with a `getattr` default silently
supplied **10000 instead of Llama 3.2's 500000**, a 50x error in the base
frequency. Short prompts still produced fluent output; only a direct comparison
against HF's own rotary module caught it, at 74.8% of cos/sin values wrong.

`resolve_rope_config` now handles both config layouts and **raises rather than
defaulting**. A wrong hyperparameter yields plausible output, which is far
harder to notice than a crash — so guessing is the wrong behaviour here.

## Phase 3: continuous batching

The baseline forms a batch and runs it until its slowest member finishes. A
request that needs 10 tokens sits idle holding memory beside one that needs
500, and nothing new starts until the batch drains.

`scheduler.py` re-decides every forward pass instead: finished requests leave
immediately, new ones join as soon as blocks are free, and a prefill rides
along with other sequences' decode steps in one ragged pass. It is pure Python
— no torch — so the cases that matter (exhaustion, preemption, starvation) are
testable without a GPU, and 38 scheduler tests run in 1.2 s.

**Admission is coupled to memory layout**, which is what separates this from a
generic OS scheduler. A request does not run because a slot is free; it runs
because the block pool can supply the blocks its KV will need.

Measured against the static-batch runner, same model and same paged attention,
so only scheduling differs (32 requests, output lengths 19–96):

| | static | continuous |
|---|---|---|
| wall time | 17.18 s | **9.39 s** |
| request throughput | 1.86 req/s | **3.41 req/s** (1.83×) |
| output throughput | 123.1 tok/s | 225.3 tok/s |
| peak blocks | **77** | 175 |
| max concurrent | 8 | 32 |

The win comes from output-length *variance* — with uniform lengths static
batching wastes nothing, the same lesson the Phase 1 fragmentation numbers
taught.

**These figures predate Phase 5 and are kept as recorded.** The same naive
configuration measured 379.1 tok/s during the Phase 5 run rather than the
225.3 here — host-load drift on a shared box, not a code change, and the
reason the Phase 5 benchmark reports paired ratios instead of absolute
numbers. Compare within a table here, not across sessions.

**Peak memory goes up, not down**, and that is worth stating plainly because
the naive expectation is the opposite. Blocks are returned the instant a
request finishes, so per-request efficiency does improve — but continuous
batching keeps all 32 requests resident where static holds 8, so it buys 4×
the concurrency for roughly 2× the blocks. Trading memory for throughput is
the intended bargain, and `max_num_seqs` is the dial that sets the exchange
rate.

### Preemption is by recompute

When the pool is exhausted, the most recently admitted request is preempted:
its blocks are freed and it returns to the queue, keeping FCFS order so the
oldest requests still finish. It re-enters as a prefill of its prompt *plus
everything it had already generated*, which reproduces the same next token it
would have sampled had it never been preempted.

Two things this phase had to get right, both caught by tests rather than
reasoning:

* **A request that can never fit is rejected at submission.** Previously it was
  accepted and then failed admission every iteration with nothing to preempt to
  help it — an apparent hang rather than an error.
* **Preemption is output-neutral only up to near-ties.** It changes the
  *accumulation order*: an untouched request builds KV one decode step at a
  time, a preempted one rebuilds it in a single bulk prefill. Arithmetically
  equivalent, not bitwise identical in bfloat16, so a genuine tie can resolve
  either way. The tests assert no divergence *where the model was confident*,
  which is what a real bug would look like.

## Phase 4: radix prefix cache

The tree, reference counting, and LRU eviction pass 26 unit tests in
`test_prefix_cache.py`, plus 69 in the CPU oracle described below, including a
randomised stress run. Blocks are keyed by exact token contents, the tree holds
one reference per cached block, sequences take another on reuse, and eviction
only ever considers blocks whose count is exactly 1 — so a block in use by a
live sequence is structurally ineligible, not merely checked for.

`enable_prefix_cache` now defaults to `True`. Getting there took two changes
beyond the tree itself, and the second is the one that is easy to miss.

### Blocks are published while the request is still running

Blocks used to enter the tree only when a request *finished*. Continuous
batching admits queued requests as fast as blocks allow, so a group sharing a
prefix is normally all in flight long before the first of them completes — and
every one of them prefills the shared prefix from scratch. The measured hit
rate on the `shared_prefix` workload was **exactly zero**.

`Scheduler._publish_blocks` now runs after every forward pass. Two rules keep
it safe, and both are load-bearing:

* **Only full blocks.** A partial block still has slots to come, so its
  contents are not final and its token key does not yet identify it. A block
  that *is* full is never written again — appends only touch positions at or
  past `num_tokens` — so publishing it mid-flight publishes something immutable.
* **Only computed tokens.** `update` sets `num_computed_tokens` before
  appending the token it just sampled, so at any other moment the last token
  has no KV. Keying a block on a token whose slot was never written would let a
  later request match it and read uninitialised memory as valid KV.

Publishing mid-flight puts a *live* sequence's blocks in the tree, which is
exactly what the reference counting was built for: the tree holds one reference
and the sequence another, so eviction cannot touch them while the sequence runs.

It does change what **preemption** is worth, though, and that took a second
fix. A victim's blocks used to go straight back to the pool; now that they are
usually published too, freeing its table only drops them to the tree's
reference, which makes them *evictable* rather than *free*. Preempting a
request can therefore release no memory at all. `_schedule_running` used to
evict once before preempting and then preempt in a loop, which under that
change would discard the entire running set while reclaimable blocks sat in the
cache. It now retries eviction after every preemption, so the cheap source of
memory is always exhausted before the expensive one.

### Two requests cannot share a prefix neither of them has computed

Publishing early is necessary but **not sufficient**, which the original
diagnosis missed. Submit N requests sharing a prefix at once into a pool big
enough for all of them and continuous batching admits every one in the *first*
iteration — before any KV exists. Each still prefills the prefix independently
and the hit rate is still zero.

Letting the second request point at the first's blocks immediately is not an
option. They go into the same forward pass, so it would be reading KV slots the
first is writing *in the same kernel launch*, with no ordering between them —
a data race, not an optimisation. Deferring by one iteration is the sound
version of the same idea, and it is what `_first_uncached_chunk` does: a
request whose first not-yet-cached block is already claimed by a request
admitted this iteration waits one iteration, then reuses it instead.

Measured on the CPU oracle (block size 16, eight requests over two groups,
135-token prompts sharing a 126-token head — the shape of the `shared_prefix`
workload, with exact integer KV rather than a model):

| admission is staggered by | before | after |
|---|---|---|
| nothing — all admitted at once | 0.0% | **62.7%** |
| `max_num_seqs=2` | 62.7% | 62.7% |
| a 300-token budget | 0.0% | **62.7%** |
| a second round against a warm cache | 47.8% | **79.1%** |

The all-at-once case costs exactly the one predicted extra iteration (8 → 9)
and saves 672 tokens of duplicated prefill. The `max_num_seqs=2` row is the
control: it already staggered admission by itself, so it was already winning
and does not move.

Deferral cannot starve anything. `pending` lives for a single iteration, so a
deferred request meets an empty set next time and is admitted whether or not
the prefix it waited on actually materialised — which matters, because the
publisher may have been preempted or evicted meanwhile.

It does stop the queue rather than skipping past the deferred request, which
costs an unrelated request behind it one iteration too. That is deliberate:
FCFS-without-skip-ahead is the same rule admission already applies when a
request does not fit the block pool or the token budget, and one scheduler with
one queue discipline is easier to reason about than one with an exception.

It is also disabled outright when the cache is off. Without a cache to publish
into, the deferred request would re-prefill the prefix anyway, so holding it
back buys nothing and costs an iteration of latency. Six scheduler tests caught
that omission immediately, which is the argument for keeping `scheduler.py`
torch-free: they run in 1.2 s.

### Turning it on changed what "no leak" means

Two Phase 3 engine tests failed the moment the cache became the default, both
asserting that the pool returns to exactly full once every request finishes.
That is no longer true, and should not be: the cache *retains* blocks after a
request completes — that is the entire point of it — so 254 of 256 free, or 1
of 6 in a deliberately tiny pool, is correct behaviour rather than a leak.

The tests were not disabled or switched back to a cache-free engine. They now
assert something strictly stronger, in two parts:

* every block is accounted for as **free or cached**, so a block that is
  neither — allocated, unreachable, never coming back — still fails;
* **clearing the cache returns every one of them**, so the tree cannot be
  holding a reference it will never release.

Before changing the tests, the same invariant was checked on the CPU oracle
across 20k randomised seeds, because "this failure is benign" is exactly the
kind of claim that deserves evidence rather than an argument. Weakening a leak
test to accommodate a leak would look identical from the outside.

### The correctness bug: admission mutated requests it then refused

`_reuse_cached_prefix` took block references, seeded the block table and set
`num_computed_tokens` **before** the watermark check that could still refuse
the admission. A refused request was left half-admitted:

* the references it took were never released, so those blocks could never be
  evicted — the pool leaked capacity on every deferral;
* `num_computed_tokens` claimed a prefix that its block table did not hold, so
  its *next* admission built a fresh table and the positions `_prepare` derived
  from that stale count addressed entirely different blocks.

Which is, precisely, "the cache serving KV that does not correspond to the
tokens it is keyed on" — just reached through the scheduler rather than the
tree. Admission is now decide-then-commit: `_match_cached_prefix` probes
without recording or acquiring anything, `_blocks_to_admit` prices the
admission, and `_adopt_cached_prefix` runs only once nothing can refuse the
request. `match(record_stats=False)` already existed for exactly this purpose;
the old code simply did not use it.

Probing before committing means the match is no longer pinned when eviction
runs, so eviction can reclaim the prefix the pending request matched and
*raise* what it needs — freeing one block can cost it two. Evict-and-rematch
therefore loops until the request fits or the cache is exhausted; a single pass
wedged the scheduler into admitting nothing while requests waited.

### It was found on CPU, in under a second

The GPU test that exhibited the bug takes ten minutes, needs weights, and
answers in bfloat16 — where a wrong answer and a rounding difference look
alike. That is the wrong instrument for a memory-addressing bug, and it never
even reached this one: it runs 2 requests against 2048 blocks, so no admission
is ever refused.

`tests/test_prefix_cache_oracle.py` replaces the model with arithmetic that has
no tolerance at all, while driving the real scheduler, block table, allocator
and radix tree. Each token's KV is a hash chain over its whole prefix, and the
fake model reads `h(start-1)` back *through the block table* — the exact read a
cache hit is supposed to make valid. Serving a block that does not match its
key is then detectable exactly, at the logical position where it happens. It is
torch-free, so it runs in the normal CPU suite.

| | crashes | wrong outputs |
|---|---|---|
| 20k randomised seeds, cache off | 0 | 0 |
| 20k seeds, cache on — before | 38 | 0 |
| 20k seeds, cache on — after | **0** | **0** |
| 20k seeds, cache on — with mid-flight publishing | **0** | **0** |

The last row re-runs the sweep against the publishing and staggering changes
above; it covers 6.05M reused tokens and 79.8k evictions, so the new paths are
exercised rather than merely present. The cache-off control is re-run alongside
it every time, so pre-existing noise is never attributed to the new code. `tests/test_prefix_cache_partial_prefill.py` covers what the oracle
cannot reach — the multi-token partial prefill (`query_len` 6 over a 70-token
context) that only a cache hit produces — against a tiny random-weight Llama in
float32, on CPU.

### The recorded symptom was a near-tie after all

The README previously recorded these two runs of the same prompt and argued
that continuations diverging *completely* ruled out a bfloat16 tie:

```
run 1: [4703, 82475, 14236,   304,  5942, 527]
run 2: [5810,   374,   459, 24131,   315, 1493]
```

That inference was wrong. A single flipped argmax at output token 0 cascades
into every token after it, so "diverges completely" does not distinguish a
wrong block from a coin flip. Measured on sequence 1 of that workload:

| | |
|---|---|
| run 1 top-2 logit gap | **0.1250** |
| max \|run 1 − run 2\| across all 128256 logits | **0.1250** |
| top-5 sets | identical; only the top two swap |

The deviation between the two runs is the entire gap between the two
candidates — one bfloat16 ulp at that magnitude — because a cache hit computes
the post-prefix tokens in a 6-token pass over a 128-token cached context rather
than inside one 134-token prefill. Same lesson as Phase 3's preemption note:
accumulation order changes, arithmetic does not.

So `test_repeated_runs_are_stable` now asserts through `assert_greedy_agrees`,
like Phase 2 and Phase 3 do, and is no longer `xfail`. Exact-equality testing
lives on CPU against the oracle, where exactness is actually available.

**Note this cuts both ways.** The real defects were found by a test the project
did not have, and the test that was failing was failing for a reason unrelated
to them. A red test is evidence that something is wrong, not evidence about
what.

### One rejection this phase added

A prompt longer than `max_num_batched_tokens` used to be accepted and then
never scheduled: prefills are not chunked, so it failed the budget check every
iteration, forever, with a completely free pool. It is now rejected at
submission, the same way a request too large for the block pool already was —
the precedent Phase 3 set for exactly this apparent-hang failure mode.

Chunked prefill would lift the limit instead of rejecting, and remains an open
limitation rather than something quietly solved here. One corner is *not*
closed either: a request whose prompt fits the budget can still exceed it after
preemption. Both are written up under "Known limitations" below, including why
the narrow check is the one that ships.

## Known limitations

Collected here because they are referenced from several phase sections above
and were otherwise only findable by reading all of them.

**The server runs as one process.** `configure_torch_runtime()` must be applied
before the first forward pass — it is worth 33× on decode — so the server runs a
single in-process uvicorn worker. Multiple workers would each need their own
copy of the weights and their own block pool anyway; scaling out is a different
design, not a flag.

**A preempted request can still exceed the token budget.** The submission check
measures the *prompt*, but preemption is by recompute, so a request re-enters
needing prompt *plus everything it had already generated*. A request that was
accepted legitimately can therefore grow past `max_num_batched_tokens` and then
fail admission every iteration — the same apparent-hang failure mode that
rejecting over-long prompts was meant to close, reached by a different route.

Two things bound how much this matters. It requires preemption to fire at all,
which needs a pool exhausted by a workload whose requests generate enough to
outgrow the budget; and the narrow fix, rejecting on
`prompt + max_new_tokens`, would refuse a large number of requests that never
get preempted, including several the current tests submit.

**Deliberately not fixed.** Chunked prefill closes it properly — it is the same
missing capability behind the over-long-prompt rejection — and is the right
shape of fix rather than a second conservative rejection rule. It was deferred
until after Phase 5 because the fused kernel changes the cost of a prefill
chunk, and therefore what the chunk size should be. **That reason has now
expired**: Phase 5 landed, and the vLLM comparison since then put a number on
what the missing capability costs (TTFT 6.6× worse). It is a live candidate
rather than a resolved question. Recorded rather than papered over.

**Prefills are not chunked.** The general form of the above: a prefill is
all-or-nothing within one forward pass, so the largest schedulable prompt is
`max_num_batched_tokens` and over-long prompts are rejected at submission
instead of being split.

**Prefill still runs the naive Python gather.** Phase 5 fused decode only. A
step that mixes a prefill with other sequences' decodes falls back to the
reference in full, so a busy server admitting requests continuously spends more
of its steps on the slow path than these benchmarks' steady-state decode
suggests. Fusing prefill needs the causal mask that decode's single query token
lets the kernel skip.

**A prompt above ~16k tokens cannot be prefilled at all.** The consequence of
the above, found while measuring Phase 6 and worth stating separately because it
is a capacity ceiling rather than a slowdown: the naive path materialises the
full `[heads, q_len, ctx_len]` score matrix, which is quadratic in prompt
length. 16384 tokens prefills in 2.3 s; 32768 asks the allocator for 128 GiB and
raises `torch.OutOfMemoryError` on a 141 GiB H200. Fusing prefill — which never
materialises the score matrix, exactly as the decode kernel already does not —
removes the ceiling along with the slowness.

**There is no HTTP front-end yet** (Phase 6.5), so "server" currently means the
engine's `step()` loop driven in-process.

## Layout

```
src/mini_vllm/
    constants.py        model, dtype and block-size defaults
    runtime.py          settings that must be identical across phases
    metrics.py          TTFT / TPOT / throughput definitions, shared by all phases
    workloads.py        reproducible request generators (independent, shared-prefix, multi-turn)
    block_allocator.py  refcounted physical block pool
    block_table.py      per-sequence logical -> physical mapping
    kv_cache.py         the paged K/V tensors, plus capacity planning
    paged_attention.py  attention gathering scattered blocks (+ dense reference)
    paged_attention_triton.py  fused decode kernel; the reference above is its bar
    prefix_cache.py     radix tree over KV blocks, refcounted with LRU eviction
    scheduler.py        continuous batching policy (no torch, exhaustively tested)
    engine.py           scheduler + paged model, one iteration loop
    server.py           HTTP front end: one loop task, many waiting handlers
    model/
        llama.py        our Llama forward over the paged cache
        runner.py       greedy generation, static batch (the Phase 3 comparison point)
benchmarks/
    baseline_hf.py      Phase 0 naive HuggingFace generate() baseline
    fragmentation.py    Phase 1 naive-padding vs paged allocation
    continuous_batching.py  Phase 3 static vs continuous scheduling
    decode_kernel.py    Phase 5 fused vs naive decode, paired to cancel host load
    serving.py          Phase 6 whole server vs the naive baseline, like for like
    multi_turn.py       Phase 6 prefix-cache hit rate on growing conversations
    prefix_cache_value.py  Phase 6 what a hit is worth, against context length
    vllm_reference.py   Phase 6 this server vs real vLLM, one client, both arms
results/                every benchmark's raw output, tracked so claims are checkable
scripts/env.sh          storage layout for this host (source before running)
tests/                  pytest suite (CPU-only unless marked `gpu`)
```

## License

MIT
