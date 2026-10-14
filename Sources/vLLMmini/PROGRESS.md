# PROGRESS

Working state of mini_vllm, written so a fresh session can pick up without
re-deriving anything. Read `CLAUDE.md` (the build brief) first, then this.

**Last updated:** 2026-08-15, after closing Phase 6.5 (HTTP front end) and
measuring against real vLLM.
**Branch:** `main` at `21f969e`, working tree clean, 33 commits.
**A git remote now exists** — `origin` is
`https://github.com/jayavardhan-g/miniVLLM.git`, the user added it and pushed on
2026-08-15. The branch was `master` and is now `main`, tracking `origin/main`.
This closes the longest-running open question on the project.

**The `v1-benchmark` tag is local only** — `git ls-remote --tags origin` comes
back empty. `git push origin v1-benchmark` publishes it. Left for the user
rather than pushed unprompted.

**Start here if you are a fresh session:** `CLAUDE.md` (the brief), then this
file, then `git log -6`. Nothing else needs re-deriving. The README is the
human-facing write-up and is current; this file is the working state and is not
tracked in git.

## Read this first if you are picking up on 2026-08-15 or later

Phase 6.5 is **done** — the front end exists, buffered and streaming — and the
vLLM reference number is **measured, not pending**. Four commits:

```
221decc  Serve the engine over HTTP without making it concurrent
eb7cbb3  Stop the first caller paying for everyone's kernel compilation
186b7a6  Stream tokens over SSE, and actually stop when the client leaves
1c96fc1  Measure the ceiling: real vLLM is 6.2x faster than this
```

**Every item in the Definition of Done is now met.** Nothing is queued. The
next thing to do is a decision for the user, not a task already chosen — see
"What is left to do".

Three things from this session worth not re-deriving:

* **vLLM 0.26.0 is installed** at
  `/mnt/warm_storage/jayavardhan/tools/conda-envs/vllm-ref` (documented in
  SETUP.md). It needs **`ninja` on PATH** or flashinfer's JIT sampler dies with
  `FileNotFoundError: 'ninja'` during engine start; ninja is installed in that
  env, so put its `bin` on PATH when launching. vLLM's memory profiler also
  aborts if a neighbour frees a large block *during* its init — retry, it is
  not our bug.
* **0.16× is the ratio (vLLM 6.2× faster), and the front end is not the
  cause.** Checked: in-process 866 tok/s vs 1007 tok/s through HTTP on the same
  card minutes apart. Absolutes were ~2.8× below the README headline because a
  neighbour had the card at 100%; the paired ratio is unaffected, which is the
  whole reason the benchmarks are paired.
* **A hung-up HTTP client is not detected by the SSE generator being closed.**
  Measured, not assumed: a killed `curl` left the request decoding to its token
  limit. `_sse` polls `is_disconnected()` per frame instead. A test that closes
  the generator by hand passes against the broken version — do not trust one.

Tree is green at **421 tests: 349 CPU (~6.5 min) + 72 GPU (~14 min)**.

---

Phase 6 landed as four commits on top of `d573e50`, plus one doc fix after the
tag:

```
0df6ec5  Measure the server against the baseline like for like: 3.82x
01bb61b  Measure the prefix cache on real multi-turn conversations: 91% by turn 4
4f03389  Explain the cache's 1.00x with the crossover that causes it
dd4de62  Report Phase 6, and correct the headline it replaces   <- v1-benchmark
61e04b7  Stop SETUP.md telling the reader to run a bare pytest
```

`v1-benchmark` points at `dd4de62`, the report commit — the doc fix after it is
hygiene, not part of the benchmark, so the tag was left where it is rather than
moved.

Those commit messages carry the reasoning in full — why each change was
necessary and what was checked. **Read them before touching the benchmarks**;
several encode decisions that look arbitrary in the code.

---

## Where the project actually stands

| Phase | Description | State |
|---|---|---|
| 0 | Setup, repo scaffold, naive HF `generate()` baseline | done |
| 1 | Paged KV-cache memory manager (block table + allocator) | done |
| 2 | Naive paged attention (correctness vs unpaged baseline) | done |
| 3 | Continuous-batching scheduler | done |
| 4 | Radix-tree prefix cache | done — on by default |
| 5 | Fused Triton decode kernel | done — 7.49×, on by default |
| 6 | Benchmarking vs HF baseline | done — 3.82×, tagged `v1-benchmark` |
| 6 | Real vLLM as upper-bound reference | done — **0.16×, vLLM is 6.2× faster** |
| 6.5 | FastAPI front-end, buffered + streaming | done |
| 7 | Stretch (H2O eviction / disaggregated prefill) | not scoped |

**Tree is green at 421 tests: 349 CPU (~6.5 min) + 72 GPU (~14 min).** No xfails.
Phase 6 added no tests — it added benchmarks, which are not run in CI. Phase 6.5
added 19, all CPU, all against the hash-chain oracle.

### The number that matters, measured like for like

| | HF baseline | ours |
|---|---|---|
| output throughput | 631.5 tok/s | **2415.0 tok/s** (3.82×) |
| request throughput | 4.93 req/s | 18.87 req/s |
| TTFT p50 | 2448 ms | **179 ms** (13.7×) |
| latency p99 | 6486 ms | **1696 ms** |

**3.82×, not the 4.4× that was on file.** The old number compared a baseline
forced to exactly `--max-new-tokens` against a server running `varied_lengths()`
— different amounts of work. `benchmarks/serving.py` now hands both arms
identical prompts and identical per-request length lists, alternates the arms
within each of five rounds, and reports the median paired ratio. Uniform lengths
are the conservative floor (3.82×); varied is 3.68×.

**The airtight kernel claim is still the 7.49×** — same weights, scheduler, pool
and prompts, only the attention implementation differs. 3.82× is the right
number for the *system*, 7.49× for the *kernel*. Do not mix them.

Two Phase 6 findings that are easy to misread later:

* **Varied lengths score below uniform.** Not a regression. With all 32 requests
  submitted at once, wall time is set by the longest request, so spreading
  lengths delivers fewer tokens over the same clock and depresses tok/s in both
  arms. Request throughput is the metric that shows the scheduling win.
* **An arrival-time bug had been flattering the baseline.** The HF arm timed
  each request from when *its own batch* started, deleting its queueing time.
  Fixed; HF TTFT p50 went 24.6 ms → 2448 ms. Both arms now measure from
  submission. If a TTFT number ever looks too good for a static batcher, this
  is the first thing to check.

One reconciliation, so the numbers do not look inconsistent: the README's
Phase 3 section records continuous batching at 225.3 tok/s, but the naive arm
measured 379.1 tok/s in the Phase 5 run on the same workload settings. That is
host-load drift, not a code change. It is exactly why the benchmarks report
paired ratios rather than absolute numbers.

---

## What Phase 6.5 did (most recent)

`src/mini_vllm/server.py` plus `tests/test_server.py` (19 tests, CPU, ~6 s) and
`benchmarks/vllm_reference.py`. **No library code outside `server.py` changed** —
the engine, scheduler, kernel and cache are byte-for-byte what Phase 6 left.

The design in one line: the engine stays single-threaded and the concurrency is
in *waiting*. One background task drives `step()`; many handlers suspend on
their own `Submission`. Handlers and the loop are coroutines on one event loop
and the loop never awaits inside a step, so `add_request` can only interleave
between steps — where the scheduler already documents it as safe. No locks.

**Streaming did not need a per-token callback on the engine**, which was the
open design question. `step()`'s return value is the contract every benchmark in
`results/` was measured against; widening it to serve the front end would couple
them. Instead `_drain_streams` reads each request's own `output_token_ids` —
the list that return value is derived from anyway — and pushes what is new onto
the caller's queue. If someone later proposes adding a callback "because
streaming needs one", it does not.

Three things this phase measured that are easy to get wrong by reasoning:

1. **The first request cost 17.0 s and everyone else paid for it.** 2.72 s TTFT
   then 462 ms/token, against 0.36 s / 21.7 ms / 10.9 ms warm. Not model work —
   CUDA context, cuBLAS handles, Triton compiling for its first shape.
   `warmup()` runs one throwaway request at startup and then clears its cache
   entry and zeroes the stats, so `/stats` does not open on a lie.
2. **A hung-up client is not detected by the SSE generator being closed.** The
   first version assumed it was, and the test that closed the generator by hand
   passed. Live, a killed `curl` left the request decoding to its token limit —
   12 s of GPU work for nobody, holding blocks. A *plain* streaming endpoint on
   the same server was closed promptly, so it is not the engine starving the
   event loop; starlette 1.6 relies on `send` raising, and it did not here.
   `_sse` polls `is_disconnected()` per frame: 68 tokens instead of 1000.
3. **A token is not a character.** Decoding tokens individually emits U+FFFD
   then the real character. Deltas are the suffix of the decode-so-far, held
   back while it still ends in a replacement character. Checked live on a prompt
   mixing Japanese, accents and an emoji — streamed text identical to buffered.

**Eight mutations, five of them initially uncaught or only caught by a hang.**
The table is in the README. The lesson worth carrying: a test that reproduces a
failure *by simulation* (closing a generator by hand) can pass against a server
that never experiences that failure in production. Where a mechanism depends on
the transport, drive the mechanism you actually shipped.

### The vLLM comparison

`benchmarks/vllm_reference.py`, `results/vllm_reference.json`. **0.16× — vLLM
0.26.0 is 6.2× faster.** Rounds 0.15–0.19, both arms exactly 4096 tokens each
round.

Its structure is the part to preserve if it is ever re-run: one asyncio client
streams both servers so TTFT has one definition; both servers stay up on one
card and rounds alternate, which buys the paired statistic without a model
reload per round (an idle server holds memory but no SMs); output length is
re-tokenized client-side rather than trusted from either server, since both hold
frames back at codepoint boundaries and need not agree on when; and a discarded
warm-up round keeps first-request compilation out of both arms.

**Checked before believing it:** the gap is not the front end. Same card,
minutes later, the in-process engine ran the same workload at 866 tok/s against
the server's 1007. Both far below the 2415 tok/s Phase 6 headline because a
neighbour had the card at 100% — which is precisely why the ratio is the number
and the absolutes are not.

TTFT 6.6× worse, TPOT 6.2× worse. Those implicate the two known gaps — unfused
prefill, and a decode loop with no CUDA graphs that re-enters Python every
iteration — at roughly the same multiple, which is what "architecture right,
engineering unfinished" looks like.

---

## What Phase 6 did

Three new benchmarks plus one new workload generator, all committed and tagged.
**Phase 6 added no library code** — nothing in `src/` changed except
`workloads.py` gaining a generator, so the serving path is exactly as Phase 5
left it.

* `benchmarks/serving.py` — the like-for-like harness, and the one that produces
  the headline. Removes three confounds: identical work (same prompts, same
  per-request length list, two length modes), identical machine conditions (arms
  alternate within a round), identical arrival semantics (all requests arrive at
  t=0 in both arms).
* `benchmarks/multi_turn.py` + `multi_turn_conversations` in `workloads.py` —
  conversations that grow, where turn N+1's prompt is turn N's prompt plus the
  model's own generated reply. Hit rate 0.0 → 83.0 → 88.3 → **91.1%** across
  four turns, 81.6% overall.
* `benchmarks/prefix_cache_value.py` — what a hit is *worth* against context
  length. This is the measurement that makes the hit rate mean something.

**The most useful thing Phase 6 learned, and the thing most likely to be
misread:** the cache achieves 81.6% hit rate and saves **zero wall clock** on
this workload. That is not a broken cache. Prefill below ~2k tokens is
overhead-bound — a forward pass costs 15.6 ms at 128 tokens and 15.5 ms at 512,
because dispatch and kernel launches dominate and the tokens ride along free.
Past the crossover the same hit is worth 2.92× at 2k, 6.16× at 4k and **7.41×
at 8k**. A hit rate quoted without the context length it was measured at says
nothing about whether the cache earns its complexity.

**Two methodology traps, both of which bit during this phase.** Both are the
same class of bug — an accounting asymmetry that silently favours one arm — and
both are worth re-checking in any new benchmark:

1. *The HF arm timed each request from when its own batch started*, deleting the
   time it spent queued behind earlier batches. TTFT p50 read 24.6 ms; corrected
   it is 2448 ms. The bug flattered the **baseline**, so fixing it moved the
   result in our favour.
2. *The multi-turn generator appended the per-conversation marker* rather than
   prepending it, so conversations sharing a persona also shared a ~40-token
   head. Turn 1 read a 40.1% hit rate when it should read 0.0%, and those
   cross-conversation hits were being counted as self-reuse. This one flattered
   **us**. A prefix cache keys on the prefix; isolation has to start at token 0.

**Found while measuring, now in the README's known limitations:** a prompt above
~16k tokens cannot be prefilled at all. The naive prefill materialises the full
`[heads, q_len, ctx_len]` score matrix; 32768 tokens asks the allocator for
128 GiB and OOMs a 141 GiB H200. The unfused prefill was recorded as a slowness
— it is also a hard capacity ceiling, which nothing had written down.

---

## What Phase 5 did (still the current hot path)

`src/mini_vllm/paged_attention_triton.py`. One Triton program per (sequence,
query head), online softmax, block-table indirection resolved **per token** —
so nothing assumes `block_size` is a power of two or that a tile lands on a
block boundary. No split-K: 1024 programs already fill the card at these batch
sizes, and the module docstring records it as the change to make if profiling
ever shows the GPU idle on long contexts at small batch.

**Decode only, by a hard gate.** `can_use_decode_kernel` requires every
sequence to contribute exactly one query token, because that is what collapses
causal masking to a bounds check against `context_len`. Mixed prefill/decode
steps — which continuous batching produces routinely — fall back to the
reference in full. `paged_attention.py` remains the deliberately-naive
reference; its docstring still says not to optimise it, and that still holds.

Decode metadata (padded block tables + context lengths) is built **once per
forward pass**, not per layer. Per layer would have restored a slice of the
very overhead the kernel exists to remove.

`PagedLlama.use_fused_decode` toggles the path; the A/B tests use it to compare
both implementations on identical weights and pool.

Two lessons worth carrying into Phase 6 and beyond:

* **Mutation testing found a real hole.** Replacing the block-table load with
  the logical index — deleting paging entirely — failed only **2 of 28** tests,
  because `BlockAllocator` hands a lone sequence a contiguous run starting at
  block 0, where logical and physical indices coincide. Tests now fragment the
  pool before every case and assert no sequence starts at block 0; the same
  mutation now fails 19. Same trick on the GQA mapping and the stale-tail mask:
  19/28 each. **Keep breaking the code to check the tests.**
* **Paired measurement matters more than expected.** Taking each arm's median
  independently reported 4.77× where the paired per-round ratios were 7.61,
  4.76 and 7.51 — the arms had been measured under different host loads.
  `benchmarks/decode_kernel.py` now alternates arms within a round.

12 of 32 requests diverge from the naive path. All are bf16 near-ties: max
top-2 logit gap 0.125 against a 0.5 threshold, six exactly 0.0000, **zero
confident disagreements**. The benchmark grades this rather than counting it,
because "20/32 identical" reads as a red flag when nothing actually disagreed.

---

## What is left to do

**Nothing is queued.** The Definition of Done is met end to end: working
server (HTTP, streaming), benchmark suite, technical write-up, prefix-cache hit
rate on a multi-turn workload, and now an upper-bound reference against real
vLLM. What follows is a menu for the user to choose from, not a plan.

Ranked by what the measurements actually justify:

1. **Fuse prefill.** The vLLM comparison points here twice: TTFT is 6.6× worse
   and it is the one hot path still running the naive Python gather. It is also
   a hard 16k-token capacity ceiling (the score matrix is quadratic), and it is
   what would make the prefix cache pay on the workloads this project
   benchmarks. Highest value, and not on the stretch list.
2. **Chunked prefill.** Closes the preempted-request budget hole properly and
   removes the over-long-prompt rejection. Long-deferred; the reason for
   deferring expired when Phase 5 landed.
3. **CUDA graphs on the decode step.** TPOT is 6.2× worse than vLLM and the
   loop re-enters Python every iteration. Plausibly the cheapest large win
   after prefill, and entirely unexplored here.
4. **Phase 7 stretch** (H2O eviction, disaggregated prefill/decode). The brief
   requires asking before starting either.

### Phase 6.5 — FastAPI front-end (DONE, kept for the reasoning)

Not in the original build plan, but the architecture diagram and the Definition
of Done both call for an HTTP server and no phase has built one. It was deferred
until after Phase 5 deliberately — building it earlier would have meant
rewriting it when the hot path changed, which is exactly what Phase 5 did.

It is the **last item the Definition of Done still needs.** Everything else —
working server, benchmark suite, technical write-up, prefix-cache hit rate on a
multi-turn workload — is now done.

Concretely, what it has to be:

* A thin asyncio wrapper over `ContinuousBatchingEngine.step()`. The engine is
  **synchronous and single-threaded by design**; do not make it concurrent. The
  front end owns one background task that loops `step()` and dispatches finished
  requests back to their waiting HTTP handlers.
* `engine.add_request(request_id, prompt_token_ids, max_new_tokens)` is the
  submission point and is already safe to call between steps. `step()` returns
  the `Request` objects that finished in that iteration.
* Streaming is the part worth thinking about before writing code: `step()`
  produces one token per running sequence per iteration, so SSE falls out
  naturally, but nothing currently exposes a per-token callback. That is a
  **[DECISION]** to put to the user rather than guess.
* Reuse `metrics.py` for TTFT/TPOT rather than inventing a second definition.

Two things that will bite:

* `configure_torch_runtime()` **must** be called before the first forward pass,
  or decode runs 33× slower (see `SETUP.md`). A web framework that forks or
  spawns workers can lose this — a single-process uvicorn worker is the safe
  starting point.
* Admission currently **rejects** prompts over `max_num_batched_tokens` at
  submission. Over HTTP that has to become a 4xx with a clear message, not an
  exception traceback.

### Phase 7 — stretch

H2O-style heavy-hitter eviction, or disaggregated prefill/decode. **Ask the
user before starting either** — the brief requires it.

If the user wants a suggestion instead, **fusing prefill is the higher-value
work** and is not on the stretch list. Phase 6 showed it is not just a
performance gap but a hard 16k-token capacity ceiling, and it is what would make
the prefix cache pay on the workloads this project actually benchmarks.

### Known gaps that are deliberate, not forgotten

* **Prefill still runs the naive Python gather**, and any step mixing a prefill
  with decodes falls back entirely. A server admitting requests continuously
  spends more steps on the slow path than the steady-state decode benchmark
  suggests. Fusing prefill needs the causal mask decode lets the kernel skip.
* **A prompt above ~16k tokens cannot be prefilled at all** — the consequence of
  the above, quantified in Phase 6. The score matrix is quadratic in prompt
  length: 16384 tokens prefills in 2.3 s, 32768 raises
  `torch.OutOfMemoryError` asking for 128 GiB. Fusing prefill removes the
  ceiling along with the slowness, since a fused kernel never materialises the
  matrix — exactly as the decode kernel already does not.
* **The prefix cache is unexercised by the workloads we benchmark.** It is
  correct and it hits 91%, but every benchmark here runs contexts under ~2.5k
  tokens, where a hit is worth roughly nothing. Anyone judging the cache by
  wall clock on these workloads will conclude it does nothing; anyone running
  8k-token system prompts gets 7.4× on prefill.
* `BLOCK_N=128` and `num_warps=4` were chosen, not tuned. There may be more
  performance available for very little work.
* No split-K, so long-context/small-batch occupancy is untested.

---

## Open questions — genuinely open, waiting on the user

These were raised and are **not** resolved. Do not act on any of them
unprompted; do surface them if the user asks what is outstanding.

1. **Chunked prefill.** See the answered list below — the reason for deferring
   it (that Phase 5 would change what a chunk costs) expired when Phase 5
   landed. It is now a live candidate, and Phase 6 strengthened the case by
   showing the unfused prefill is also a hard 16k capacity ceiling. Still the
   user's call.

---

## Open questions — answered, do not re-ask

1. **A preempted request can still exceed the token budget.** **Answered:
   defer, with a reason.** Not fixed. Chunked prefill is the right fix rather
   than a second conservative rejection rule. Written up in the README's "Known
   limitations" section (commit `6ad39af`). Phase 5 has now landed, so the
   argument for waiting — that the kernel changes what a prefill chunk costs —
   has expired; this is a live candidate again, but it is **the user's call**,
   not something to start unprompted.
2. **Should `PROGRESS.md` be tracked in git?** **Answered: no.** Stays in
   `.gitignore`. Do not propose tracking it again.
3. **Commit trailers.** The user rejected a commit carrying `Co-Authored-By:` /
   `Claude-Session:`. **Do not add either**, on any commit here, regardless of
   harness defaults.
4. **Git remote.** **Answered: done, by the user, 2026-08-15.** `origin` is
   `https://github.com/jayavardhan-g/miniVLLM.git` and `main` is pushed. Stop
   raising it. The only remnant is the unpushed `v1-benchmark` tag, noted at the
   top of this file.
5. **Real vLLM as an upper-bound reference.** **Answered: run, 2026-08-15.**
   0.16×, i.e. vLLM 6.2× faster. `results/vllm_reference.json`, README under
   Phase 6. The env is built and documented; re-running it is cheap now.
6. **Streaming.** **Answered: buffered first, then SSE**, both shipped
   2026-08-15. SSE required no engine change — do not add a per-token callback
   to `step()` on the theory that streaming needs one.

---

## How to work on this efficiently

**The GPU test suite is the wrong instrument for memory/scheduling bugs.** It
takes 6–17 minutes, needs weights, and answers in bfloat16 where a wrong answer
and a rounding difference look alike.

`scheduler.py`, `prefix_cache.py`, `block_table.py` and `block_allocator.py`
are deliberately torch-free. Drive them with **system python** for sub-second
iteration (stub out `pytest` if importing a test module):

```bash
/usr/bin/python3 -c "import sys; sys.path.insert(0,'src'); import mini_vllm.scheduler"
```

`tests/test_prefix_cache_oracle.py` is the pattern worth reusing: the model is
replaced by a hash chain over the token prefix, read back *through the block
table* — the exact read a cache hit is supposed to make valid. Exact, no float
tolerance. It found both Phase 4 defects in under a second.

Seven habits that have each caught a real defect on this project:

* **Test the mechanism you shipped, not a simulation of it.** Phase 6.5's
  disconnect test closed the SSE generator by hand and passed against a server
  that, in production, never had its generator closed at all. Where behaviour
  depends on a transport, a framework or a driver, exercise the real path once
  before trusting the fast test that stands in for it.
* **Break the code to check the test.** `git stash push <file>` the fix, or sed
  a deliberate bug into the kernel, and confirm the tests actually fail. This
  caught a Phase 4 test that passed either way, and in Phase 5 caught a whole
  suite that passed with paging removed.
* Run any fuzz **with the feature disabled as a control**.
* Assert exactness only where exactness exists — float32 or an integer oracle.
  On bfloat16 use `tests/greedy_compare.py::assert_greedy_agrees`, and grade
  divergence by top-2 logit gap rather than counting it.
* **Measure before believing a diagnosis.** The recorded reason for the Phase 4
  xfail was half wrong, and a 30-line CPU probe showed it in seconds.
* **Pair A/B timings within a round** on this shared box; absolute throughput
  drifts ~1.5× with other users' jobs.
* **Read a benchmark's output for accounting asymmetries before believing its
  headline.** Phase 6 shipped two arms that were not measuring the same thing —
  once in the baseline's favour (per-batch arrival times), once in ours
  (a shared prefix in a workload built to have none). Both were caught by a
  number looking implausible rather than by a test: a static batcher cannot
  really have 24 ms TTFT, and turn 1 of a fresh cache cannot really hit 40%.
  **When a result looks too good for the arm it favours, it usually is.**

### Host gotchas

* `/mnt/warm_storage` is NFS. Skip conda's shell hook and call the interpreter
  directly: `/mnt/warm_storage/jayavardhan/tools/conda-envs/vllmmini/bin/python`
  Importing torch from it takes minutes when the box is loaded.
* **Never pipe a background run through `tail` or `grep`** — both block-buffer
  when stdout is not a terminal, so nothing appears until the process exits and
  a hang looks identical to slow. This cost time twice in the Phase 6 session
  alone. Run the script bare with `python -u`, or if you must filter, use
  `grep --line-buffered`.
* **`pytest` alone runs GPU tests too** if a card is visible, and will grab
  whatever `CUDA_VISIBLE_DEVICES` points at — including a full one, which OOMs.
  Use `-m "not gpu"` with `CUDA_VISIBLE_DEVICES=""` for the CPU suite, and pick
  a free card explicitly for `-m gpu`.
* Shared 8×H200 box, and **what is free changes fast** — GPU 6 was empty at the
  start of the Phase 5 session and had another user's 25 GB job on it an hour
  later. Re-check immediately before launching, and prefer a card at 0 MiB.
* `/data` is a 100 GB quota shared with other work and has briefly hit ENOSPC.
  If an edit fails with ENOSPC, check `df -h /data` before assuming breakage.
* **vLLM needs `ninja` on PATH** or flashinfer's JIT sampler kills engine
  start-up with a `FileNotFoundError` buried hundreds of traceback lines above
  the `RuntimeError: Engine core initialization failed` that surfaces. It is
  installed in `vllm-ref`; put that env's `bin` on PATH when launching.
* **vLLM's memory profiler aborts if a neighbour *frees* a large block during
  its init** ("Initial free memory 31.53 GiB, current free memory 130.71 GiB").
  That is this shared box being itself. Relaunch; it is not a misconfiguration.
* A server takes **~5 minutes** before its port opens: torch import and weights
  over NFS, then the startup warmup. Poll `/health`, do not guess.
* Everything else host-specific is in `SETUP.md`.

### Commands

Call the env's interpreter directly rather than activating; conda's shell hook
over NFS is slow. `PY=/mnt/warm_storage/jayavardhan/tools/conda-envs/vllmmini/bin/python`

```bash
source scripts/env.sh                                   # required on this host

CUDA_VISIBLE_DEVICES="" $PY -m pytest -m "not gpu" -q   # CPU suite, ~2.5 min, 330 tests
CUDA_VISIBLE_DEVICES=<free> $PY -m pytest -m gpu -q     # GPU suite, ~6 min, 72 tests

# Phase 6 headline: whole server vs naive baseline, five paired rounds (~8 min)
CUDA_VISIBLE_DEVICES=<free> $PY -u benchmarks/serving.py --repeats 5

# Phase 6.5: the server. ~5 min before the port opens (torch + weights over
# NFS, then the startup warmup). Poll /health; do not guess.
CUDA_VISIBLE_DEVICES=<free> PYTHONPATH=src $PY -u -m mini_vllm.server --port 8000
curl -N -s localhost:8000/generate -H 'content-type: application/json' \
     -d '{"prompt":"Name two colours:","max_tokens":12,"stream":true}'

# Phase 6: against real vLLM. Both servers up on ONE card, rounds alternate.
# ninja must be on PATH or flashinfer's JIT sampler kills vLLM at startup.
VLLM=/mnt/warm_storage/jayavardhan/tools/conda-envs/vllm-ref/bin
CUDA_VISIBLE_DEVICES=<free> PATH=$VLLM:$PATH $VLLM/python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-1B-Instruct --port 8001 \
    --gpu-memory-utilization 0.15 --max-model-len 8192 --no-enable-log-requests
$PY -u benchmarks/vllm_reference.py --mini-url http://127.0.0.1:8000 \
    --vllm-url http://127.0.0.1:8001 --repeats 5

# Phase 6: prefix-cache hit rate on growing conversations (~6 min)
CUDA_VISIBLE_DEVICES=<free> $PY -u benchmarks/multi_turn.py --conversations 8 --turns 4

# Phase 6: the crossover — what a hit is worth against context length (~8 min)
CUDA_VISIBLE_DEVICES=<free> $PY -u benchmarks/prefix_cache_value.py

# Phase 5: fused vs naive decode, paired rounds
CUDA_VISIBLE_DEVICES=<free> $PY -u benchmarks/decode_kernel.py --repeats 5

# Phase 1: fragmentation — no GPU, no weights, deterministic
CUDA_VISIBLE_DEVICES="" $PY -u benchmarks/fragmentation.py --num-requests 32 --batch-size 8
```

Every GPU script spends its first ~4 minutes loading torch and weights over NFS
before printing anything. That is normal, not a hang — check
`nvidia-smi -i <card>` for a rising memory figure if unsure.

### Where the numbers live

Every benchmark writes JSON to `results/`, all tracked in git, so a claim in the
README can be checked against the run that produced it without re-running
anything:

| file | produced by | holds |
|---|---|---|
| `serving.json` | `serving.py` | the 3.82× headline, both length modes, per-round ratios |
| `multi_turn.json` | `multi_turn.py` | per-turn prefix-cache hit rates |
| `multi_turn_prefill_heavy.json` | `multi_turn.py --turns 6 --max-new-tokens 8` | the same, prefill-weighted |
| `prefix_cache_value.json` | `prefix_cache_value.py` | prefill crossover + what a hit is worth |
| `decode_kernel.json` | `decode_kernel.py` | the 7.49× kernel result |
| `continuous_batching.json` | `continuous_batching.py` | Phase 3 static vs continuous |
| `fragmentation.json` | `fragmentation.py` | Phase 1 padding vs paged waste |
| `baseline_hf_*.json` | `baseline_hf.py` | Phase 0 baselines |
| `vllm_reference.json` | `vllm_reference.py` | the 0.16× against real vLLM, per-round |

---

## Decisions already made (do not relitigate)

| Decision | Choice | Where it is argued |
|---|---|---|
| Model | `meta-llama/Llama-3.2-1B-Instruct` | README "Model" — 32 KB/token KV so preemption and eviction actually fire |
| Block size | 16 tokens | `constants.py` |
| Preemption policy | recompute, not swap | README Phase 3 |
| Eviction policy | LRU over unreferenced **leaves** | `prefix_cache.py` docstring |
| Prefix cache keying | exact token contents, whole blocks only | `prefix_cache.py` |
| Prefix cache default | on | README Phase 4 |
| Duplicate concurrent prefills | defer one iteration, do not dedup in-pass | `scheduler._first_uncached_chunk` |
| Over-budget prompts | reject at submission, not chunked prefill | user decision, 2026-08-14 |
| Kernel language | **Triton** — user-confirmed 2026-08-14 | README, SETUP.md |
| Kernel scope | **decode only** first; prefill stays on the reference | user-confirmed 2026-08-14, README Phase 5 |
| Kernel parallelism | **one program per (sequence, query head)**, no split-K | user-confirmed 2026-08-14, `paged_attention_triton.py` docstring |
| Benchmark statistic | median of **paired** per-round ratios | README Phase 5 — unpaired drifts ~1.5× with host load |
| Phase 6 length matching | report **both** uniform (floor) and varied (realistic) | user-confirmed 2026-08-14 — either alone invites an obvious objection |
| Which number to quote | **uniform, 3.82×** | README Phase 6 — varied's accounting is a judgement call, uniform's is not |
| Multi-turn measurement | **sequential driver**, conversations that grow | user-confirmed 2026-08-14 — `shared_prefix` is concurrent sharing, not multi-turn |
| Multi-turn arm fairness | throwaway materialisation pass, both arms replay identical prompts | `multi_turn.py` docstring — otherwise one bf16 near-tie forks the conversations |
| Streaming | buffered first, **SSE second** | user-confirmed 2026-08-15 — SSE needed no engine change in the end |
| Streaming mechanism | loop reads `output_token_ids`, **no per-token callback on the engine** | `server.py` — `step()`'s return value is what every benchmark was measured against |
| Client disconnect | **poll `is_disconnected()` per frame** | `_sse` docstring — relying on the generator being closed does not work, measured |
| Server workers | **one, in-process** | `configure_torch_runtime()` is worth 33× and a spawned worker loses it |
| vLLM comparison shape | one client, **both servers up, alternating rounds** | `vllm_reference.py` docstring |

---

## Historical: what Phase 4 did (closed, for context only)

1. **Blocks are published as they fill**, not at completion
   (`Scheduler._publish_blocks`, called from `update`). Only full blocks, only
   computed tokens. `Request.num_published_blocks` keeps it to one tree walk
   per `block_size` decode steps.
2. **Requests sharing an *uncached* prefix are not admitted in the same
   iteration** (`Scheduler._first_uncached_chunk`). N requests submitted at
   once would otherwise all prefill the same prefix; pointing the second at the
   first's blocks within one forward pass is a data race. Deferring one
   iteration is the sound version.

Second-order consequence: publishing mid-flight means **preemption often frees
no memory** (a victim's blocks drop to the tree's reference, becoming evictable
rather than free). `_schedule_running` retries eviction after *every*
preemption instead of once up front.

Measured on the CPU oracle (block 16, 8 requests / 2 groups, 126-token shared
head): prefix reuse 0.0% → 62.7% when all requests arrive at once, 47.8% →
79.1% on a second warm-cache round. Cost is exactly one extra iteration,
saving 672 duplicated prefill tokens. Fuzz: 20k seeds, 0 crashes, 0 wrong
outputs, with a cache-off control alongside.

`TestPoolHygiene` was correctly broken by turning the cache on: the pool no
longer returns to exactly full, because the cache retains blocks by design.
The tests now assert `free + cached == total` and that clearing the cache
returns every block — strictly stronger than what they asserted before.
