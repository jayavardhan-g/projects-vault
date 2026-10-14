# Adaptive Speculative Decoding

An inference engine that keeps several candidate draft models resident and
switches between them at runtime, driven by live acceptance-rate statistics,
instead of committing to one fixed draft.

Standard speculative decoding has a single small draft model propose `γ` tokens
that a larger target model then verifies in one parallel forward pass. It is
exact — the output distribution is identical to running the target alone — but
its *speedup* depends entirely on how often the draft guesses right, and that
acceptance rate swings hard across domains. A draft that is excellent on code
can be mediocre on open-ended chat. This project makes the draft choice a
runtime decision rather than a deployment-time one.

## Status

**Fixed-draft baseline, correct and faster than the target, with the
statistics adaptivity will run on — and a measurement that moved where the
adaptivity belongs.** Single-draft speculative decoding, the target-only
reference it is verified against, the static-cache plus CUDA-graph machinery
that makes a draft call genuinely cheaper than a target call, and rolling
acceptance tracking per draft and depth.

Sweeping the pool across code, chat and summarisation found **no separation by
draft** — the same draft wins every domain on acceptance, and the same other
one wins every domain on throughput — but **large separation by domain**, with
the optimum *draft depth* moving sharply between them. So the controller should
adapt depth rather than switch drafts. See [Do the drafts separate by
domain?](#do-the-drafts-separate-by-domain).

**The adaptive layer was cut.** This repo ships **fixed-draft speculative
decoding**: one draft model, one fixed depth (`γ=4` by default — the best fixed
depth measured across the three domains). The sweep above is why: with no
separation by draft there is nothing for a draft-switching controller to choose,
and the remaining depth adaptivity was not worth the complexity it carried.

A depth controller exists in `src/specdec/controller.py` and passes its unit
tests, but it is **unfinished, unverified and not part of what this repo
delivers**: nothing checks that varying `γ` mid-request leaves the output
distribution alone, and it has never run on real weights. It is kept, alongside
the cross-request statistics in `stats.py`, as the record of a measurement that
changed the design. The sections below describe the engine and the measurements
behind it, not a shipped controller.

Phase 0 finished correct but *slower than the baseline* — 0.88x — for a reason
that had nothing to do with the algorithm, so fixing that came first: a
controller that switches between drafts is pointless while there is no speedup
for it to be adaptive about. It now runs at **2.25x** the target-only baseline
at `γ=4` and **2.56x** at `γ=8`, with both sides compiled. See [Making it
actually faster](#making-it-actually-faster) for the diagnosis and
[Results](#results) for the numbers.

## The correctness property

Speculative decoding must never change the output distribution — it only
changes speed. Everything here is built around keeping that true and proving it.

At a given position let `p` be the target's next-token distribution and `q` the
draft's. Draw `x ~ q`, accept it with probability `min(1, p(x)/q(x))`, and on
rejection emit a fresh draw from the normalised residual `(p - q)₊`. Then:

```
P(emit x) = q(x)·min(1, p(x)/q(x))  +  P(reject)·(p(x) - q(x))₊ / Z
          = min(p(x), q(x))         +  (p(x) - q(x))₊
          = p(x)
```

since `Z = Σ_y (p(y) - q(y))₊ = 1 - Σ_y min(p(y), q(y)) = P(reject)`.

The result holds for *any* `q`, which is why a poor draft costs throughput but
never correctness. Three things can still break it in practice. Two are under
the engine's control and are guarded in code; the third is arithmetic and is
[measured instead](#what-exactness-costs-in-bf16).

- **`p` not being the distribution the baseline would have sampled from.** A
  mismatched temperature, a missing top-p warper, or a different vocabulary
  width silently changes what you are sampling from. Every path — baseline,
  draft proposal, verification — goes through the same `logits_to_probs` and
  the same canonical vocabulary, so the two cannot drift apart.
- **Stale KV cache after a rejection.** Rejected proposals leave K/V entries for
  tokens that are not part of the sequence; they must be rolled back before the
  next iteration (`cache_utils.crop_cache`).

### The canonical vocabulary

Pool members have different embedding widths even with byte-identical
tokenizers — the 14B and 7B targets pad to 152064, the drafts to 151936,
against a real tokenizer of ~151665 tokens. The surplus rows decode to no token
but still carry finite logits, hence nonzero probability.

The engine projects every model onto the pool's **narrowest** width. Narrowing
rather than widening is forced: whatever the target emits gets fed back into
the draft on the next iteration, so if the target could sample an id past the
draft's embedding, the draft would raise `IndexError` mid-generation. This is a
real bug, not a hypothetical — it surfaced the first time the engine ran.

Truncation does alter the target's distribution, by deleting ids that decode to
nothing. It remains exact against the baseline because the canonical width is a
property of the engine and the baseline applies the identical projection
(`generate_baseline(..., vocab_size=decoder.vocab_size)`).

## Making it actually faster

Phase 0 ended correct and **0.88x** — slower than just running the target. The
algorithm was fine: acceptance sat at 0.84–0.92 and target forwards dropped
4.3x. The cost model around it was wrong.

Measured forward-pass cost at batch size 1, divided by layer count:

| model | layers | eager ms | **eager ms/layer** | compiled ms | **compiled ms/layer** |
|---|---|---|---|---|---|
| Qwen2.5-14B (target) | 48 | 39.02 | 0.813 | 10.72 | 0.223 |
| Qwen2.5-1.5B | 28 | 22.61 | 0.808 | 2.74 | 0.098 |
| Qwen2.5-0.5B | 24 | 19.68 | 0.820 | 1.69 | 0.071 |
| Qwen2.5-Coder-1.5B | 28 | 22.53 | 0.805 | 2.74 | 0.098 |

Read the eager `ms/layer` column: **0.813, 0.808, 0.820, 0.805**. Flat, across
models spanning 28x in parameter count. Latency was `n_layers × a constant`
and did not notice model *size* at all, because the GPU was idle between kernel
launches and the wall clock was Python dispatch. A 0.5B draft cost as much per
call as the 7B target it was supposed to be cheaply approximating, and
speculation cannot win when `draft_cost ≈ target_cost`.

Compiling the decode step with `torch.compile(mode="reduce-overhead")` replays
it as a single CUDA graph, collapsing hundreds of launches into one. The
compiled `ms/layer` column varies with model size — which is what it should
have done all along — and the target/draft ratio is finally real. Two
structural changes were needed to make the capture legal, and they are why
`cache_utils` looks the way it does:

- **Static K/V addresses.** A graph is bound to the tensors it recorded
  against, so the cache is a preallocated `StaticCache` reused across calls,
  not a `DynamicCache` that reallocates every step. Rollback then becomes a
  counter reset with *no data movement*: rejected K/V is left in place, which
  is sound because the next forward overwrites it and causal masking hides
  whatever is beyond. That argument is load-bearing, so
  `test_static_cache.py` checks it by poisoning the cache with out-of-range
  values and requiring bit-identical results.
- **Static shapes.** One graph per input width, so the decode loop is arranged
  to use a small fixed set: `γ+1` for verification (prefill is split out of the
  first verify to guarantee this), 1 for draft steps and the baseline, and 2
  for the draft step that catches up after a fully-accepted iteration. Prefill
  is whatever the prompt is, so it stays eager. `decode_shapes` declares the
  set and a test checks the loop actually uses it — a width used but not
  captured silently runs eager, and the speedup just fails to appear.

The draft/target depth ratio matters too, which is why the default target moved
from the 7B to the **14B**: 48 layers against the 0.5B's 24, where the 7B had
28 against 28.

### What exactness costs in bf16

Greedy speculative output should equal greedy target-only output token for
token. On the 7B it did, 27 comparisons out of 27. On the 14B it does not, and
the reason is worth stating precisely because it is *not* a flaw in the engine.

The baseline reads the target's distribution from a **width-1** forward. The
engine reads it from row `j` of a **width-`γ+1`** verification pass. Same query,
same keys, identical arithmetic on paper — only the batch dimension of the
GEMMs differs, which changes tiling and accumulation order. Measured
(`measure_width_sensitivity` in `tests/conftest.py`):

| target | logit shift from batch width alone |
|---|---|
| Qwen2.5-0.5B | **0.0** — bit-identical |
| Qwen2.5-14B | **0.25** — about two bf16 ulps |

Two ulps is small, but greedy decoding has no tolerance at all: bf16 logits sit
on a coarse grid (the smallest nonzero top-1/top-2 margin observed was 0.125,
and 1 position in 41 was an exact tie), so wherever the model's own preference
is that narrow, either token is a legitimate `argmax` — and one flip redirects
every token after it.

This is a property of running one model at two batch widths, which every
batching server does. It is not caused by speculation and no amount of engine
care removes it.

So the test suite does not assert bit-equality, and it does not retreat to
"mostly agrees" either — that would hide real bugs. It **derives** the
tolerance: measure the batch-width noise floor for whichever target is loaded,
then require any divergence to occur at a position whose top-1/top-2 margin
falls *inside* that floor. Diverging where the model had a decisive preference
still fails, which is exactly where a misindexed verification or a stale cache
would land — those are wrong by whole units, not fractions of an ulp.

What that produced on the current pool:

```
eager    : batch-width sensitivity 0.250, all divergences within it
compiled : noise floor 0.281  (shape-specialised kernels, so slightly higher)
           gamma=1: 0 of 9 runs diverged
           gamma=4: 3 of 9 diverged, margins [0.25, 0.25, 0.25]
```

Every divergence sat exactly on the floor, and none above it.

The cleanest evidence that this is arithmetic rather than the engine: **the
target-only baseline disagrees with itself.** Running it three times on one
prompt with no speculation anywhere, two runs emitted a different token at
position 118 than the third. Whatever is happening at position 118 is not
something speculative decoding did.

None of this threatens the *sampled* path. There the accept/reject rule is
exact for any `q`, and `p` is whatever distribution the target computed — the
algorithm adds no error of its own. It is specifically greedy decoding, where
`p` collapses to a point mass, that turns two ulps into a different sentence.

## Results

H200, bf16, batch 1, 128 new tokens, `γ=4`, best of three interleaved
repetitions after a discarded warm-up rep (`scripts/profile_overhead.py`).

| | baseline | 1.5B draft | 0.5B draft | Coder-1.5B draft |
|---|---|---|---|---|
| eager | 43.5 tok/s | 58.5 (1.35x) | 58.0 (1.33x) | 54.6 (1.26x) |
| **compiled** | **86.2 tok/s** | **174.8 (2.03x)** | **194.1 (2.25x)** | **164.4 (1.91x)** |

Both columns get the same treatment — a compiled speculative run measured
against an eager baseline would report a speedup that is mostly the compiler's.
Compilation alone doubles the baseline (43.5 → 86.2); speculation on top of it
roughly doubles again, for **4.5x** end to end against where Phase 0 stood.

### Draft depth

Deeper drafting amortises one verification over more tokens, but compounds the
chance an early rejection throws the rest away. Compiled, same setup:

| draft | `γ=2` | `γ=4` | `γ=8` |
|---|---|---|---|
| Qwen2.5-1.5B | 1.68x (accept 0.955) | 2.03x (0.911) | **2.29x** (0.824) |
| Qwen2.5-0.5B | 1.84x (0.913) | 2.25x (0.798) | **2.56x** (0.681) |
| Qwen2.5-Coder-1.5B | 1.61x (0.902) | 1.91x (0.825) | **2.06x** (0.770) |

Acceptance falls monotonically with depth, as it must, and throughput still
rises — so the optimum is past 8 on this prompt and the sweep has not found it
yet. Note also that the *ordering* of the drafts is stable across depths here
but the gaps are not, which is the kind of thing a per-domain controller has to
track rather than assume.

### Does the cost model explain it?

| draft | acceptance | predicted | measured |
|---|---|---|---|
| Qwen2.5-1.5B | 0.911 | 1.93x | 2.03x |
| Qwen2.5-0.5B | 0.798 | 1.89x | 2.25x |
| Qwen2.5-Coder-1.5B | 0.825 | 1.63x | 1.91x |

Close, and consistently *under* — because the prediction assumes each proposal
is accepted independently, `(1 - a^(γ+1)) / (1 - a)`. Real acceptance is bursty:
the 0.5B emitted 4.13 tokens per iteration where independence predicts 3.35.
Runs of easy tokens get accepted together, which is a free bonus the naive model
does not capture — and another reason the controller should read measured
acceptance rather than infer throughput from a formula.

### Where the rest of the time goes

The brief flags that switching overhead can silently eat the gains, so it is
itemised rather than assumed. One `γ=4` iteration, compiled:

| | ms |
|---|---|
| 4 draft forwards + 1 verification | 23.251 |
| `crop_cache`, 14B target (48 layers) | 0.169 |
| `crop_cache`, 1.5B draft (28 layers) | 0.099 |
| `verify_and_correct` (incl. host syncs) | 0.036 |
| `logits_to_probs` (5 × 151936) | 0.029 |

Everything that is not a forward pass comes to **0.33 ms, about 1.4%**. The
accept/reject kernel and the softmax over a 152k vocabulary are nowhere near
the critical path; rollback costs one counter write per layer and scales with
depth, which is visible but small.

### Caveats worth keeping

- **Warm-up dominates a cold run.** The discarded rep is not decoration: it
  reported 0.49x and 7.64x for two drafts that both settle near 2.0x. An
  earlier version of this benchmark without it reported a 2.15x that was
  entirely warm-up.
- **Eager Coder-1.5B measured 14.13 ms/forward here**, against 22.41 ms for the
  architecturally identical 1.5B and 22.53 ms for itself in a previous run.
  That cell reflects GPU clock state, not the model. Its compiled number
  (2.74 ms) matches its twin exactly, as it should.
- **The baseline disagrees with itself.** Across three identical target-only
  runs, two produced a different token at position 118 than the third. No
  speculation involved — see [the bf16
  section](#what-exactness-costs-in-bf16).

## Do the drafts separate by domain?

Everything above is measured on one prompt. The premise of adaptive draft
selection is that no single draft is best everywhere — so before building a
controller, `scripts/domain_sweep.py` sweeps three prompts per domain across
every draft at two depths, on the 14B target, greedy, 128 new tokens, median of
3 reps after a discarded warm-up.

The answer is **no separation by draft, large separation by domain**, and it
moves where the adaptivity has to live.

Acceptance / speedup, `γ=4`:

| domain | Qwen2.5-1.5B | Qwen2.5-0.5B | Qwen2.5-Coder-1.5B |
|---|---|---|---|
| code | 0.805 / 1.89x | 0.734 / **2.23x** | 0.804 / 1.89x |
| chat | 0.463 / 1.31x | 0.401 / **1.47x** | 0.391 / 1.17x |
| summarisation | 0.411 / 1.23x | 0.338 / **1.41x** | 0.355 / 1.18x |

The 1.5B has the highest acceptance in every domain. The 0.5B has the highest
throughput in every domain — it accepts less but costs less, and the cost wins.
The ordering is identical at `γ=8`. A controller choosing *which draft* has
nothing to choose.

### Specialisation bought nothing

The sharpest number here is the code column: **Coder-1.5B scores 0.804 where
the general 1.5B scores 0.805.** A dead heat on code, from the code-specialised
model, at identical throughput. The two are architecturally identical with
matching compiled forward cost, so this is a clean acceptance comparison.

There is a mechanism that explains it, and it generalises past this pool.
Acceptance measures how close `q` is to `p` — and `p` belongs to the *target*, a
general instruct model. A domain-specialised draft moves away from a general
target's distribution by roughly as much as it gains on the domain. What earns
acceptance is **distributional proximity to the target, not task competence**.
Fine-tuning a draft on your workload is not obviously a way to speed it up.

### Where the depth optimum actually is

Sweeping `γ` ∈ {4, 8, 12, 16} on the 0.5B — the draft that wins throughput
everywhere, and the one the controller fixes on:

| domain | `γ=4` | `γ=8` | `γ=12` | `γ=16` |
|---|---|---|---|---|
| code | 2.25x | **2.60x** | 2.44x | 2.31x |
| chat | **1.42x** | 1.25x | 1.03x | 0.82x |
| summarisation | **1.41x** | 1.10x | 0.94x | 0.66x |

Code peaks at `γ=8` and turns over. Chat and summarisation fall monotonically
from `γ=4`, so **their optimum is below 4 and is not yet measured** — the
mirror of the gap this sweep closed for code.

The stakes are larger than the headline speedups suggest. At `γ=16` every
domain but code is below 1.0x, bottoming at 0.48x for the Coder draft on
summarisation. A badly chosen depth is a 2x *slowdown*, not just a forfeited
gain.

What a depth controller is worth, on an even three-domain mix:

| policy | mean speedup |
|---|---|
| fixed `γ=16` | 1.26x |
| fixed `γ=8` | 1.65x |
| fixed `γ=4` — the best fixed choice | 1.69x |
| per-domain oracle | **1.81x** |

So a perfect controller beats the *best* fixed depth by about **7%**, and a
plausibly-chosen one (`γ=8`, which is what the earlier truncated sweep pointed
at while throughput was still climbing) by 10%. The 7% is the honest number and
it is modest. The stronger claim is the one behind it: "`γ=4` is the best fixed
choice" is only knowable from a full offline sweep over every domain you will
ever serve. A runtime controller reaches it without that sweep, and stays there
when the mix changes.

### Depth is the axis that actually varies

Domain separation is real and large — code accepts roughly twice what chat and
summarisation do. It just is not exploited by swapping drafts. It is exploited
by drafting deeper or shallower:

| domain | 0.5B `γ=4` → `γ=8` | Coder-1.5B `γ=4` → `γ=8` |
|---|---|---|
| code | 2.23x → **2.60x** | 1.89x → 1.97x |
| chat | 1.47x → **1.26x** | 1.17x → **0.91x** |
| summarisation | 1.41x → **1.13x** | 1.18x → **0.90x** |

Deeper drafting helps on code and hurts everywhere else. On chat and
summarisation at `γ=8` the Coder draft drops **below 1.0x** — slower than not
speculating at all. That is a stronger case for adaptivity than the original
one: getting depth wrong does not merely forfeit gains, it goes backwards.

So the controller should adapt **draft depth per domain**, with draft identity a
fixed choice (the 0.5B) rather than a per-request decision. Phase 1 needs no
rework for this: statistics are keyed by `Arm(draft, gamma)`, and depth was in
that key from the start precisely because counts at different depths are not
comparable.

Caveats: three prompts per domain, 128 tokens, greedy, one target. This is the
precursor that decides Phase 2's shape, not the Phase 3 benchmark, which uses
HumanEval / ShareGPT / CNN-DailyMail. It is consistent with the single-prompt
run above — 2.23x here against 2.25x there for the 0.5B on code.

```bash
CUDA_VISIBLE_DEVICES=<free gpu> python scripts/domain_sweep.py --gammas 4 8 --reps 3
```

## Tracking acceptance

A controller can only be as good as the signal it reads, and two choices about
that signal shape everything downstream.

### The domain tag measures, but never controls

A request may carry a `domain` tag (`code`, `chat`, `summarisation`). It is an
**evaluation key only**: it buckets exact counters for the per-domain report,
and the controller never sees it. `AcceptanceTracker` splits the two surfaces
apart — domain-keyed counters live behind `.evaluation`, and `.control` has no
method that accepts or returns a domain.

Three reasons for holding it out:

- **The tag is a proxy for the thing that matters.** What determines speedup is
  whether *this draft* is currently predicting *this target* well. "Is this
  prompt code?" only correlates with that. Measuring acceptance measures the
  real quantity directly, with no classifier error in between.
- **A tag-driven controller is a lookup table.** If the tag selects the draft,
  the result is per-dataset tuning wearing an adaptivity label. Nothing in a
  real serving path supplies the tag anyway.
- **Held out, it becomes ground truth.** Because HumanEval *is* code, the tag
  is exact and free. Keeping it away from the controller turns Phase 3 from
  "we configured it per domain" into a real question: did the controller
  rediscover the domain structure without being told it?

`test_control_is_blind_to_domain` feeds two trackers byte-identical acceptance
streams differing only in their tags and asserts every controller-visible
quantity matches. A leak here would not crash anything — it would just quietly
invalidate the headline — so it gets an explicit test.

### Accepted-per-iteration, on two timescales

The tracked quantity is the number of proposals accepted in an iteration, not
the per-token acceptance rate. An iteration emits `n_accepted + 1` tokens for
one target forward, so this *is* the speedup ceiling, and it assumes nothing
about tokens accepting independently — which they do not. The 0.5B emitted 4.13
accepted tokens per iteration where an independence model predicted 3.35;
acceptance comes in bursts. A per-token Bernoulli estimator would report a
confidently wrong variance in exactly this regime.

Domain is constant *within* a request and varies *between* them, so a single
pooled average over a mixed workload tracks the mixture rather than the request
in front of it. So there are two estimators:

| | span | read by |
|---|---|---|
| fast, per request (`RequestScope`) | ~7 iterations at `α=0.15` | in-flight decisions |
| slow, per arm (`ControlView`) | many requests at `α=0.02` | the prior a request warm-starts from |

The EMA is what does the regime detection a domain label would otherwise do: it
has no notion of a domain, only of what has been working lately.

Both carry an explicit accumulated weight rather than dividing by a fixed
constant, which makes the cold start unbiased — a plain EMA initialised at zero
reads far too low for its first several updates, precisely where a controller
makes its worst decisions — and makes warm-starting fall out of the same
arithmetic. The prior's strength is denominated in **observations**: seeding
with 2.0 leaves a request exactly where a cold estimator would sit after seeing
the prior's value twice, so the knob is one a human can reason about and
`test_prior_observations_are_a_real_count` holds it to that meaning.

Statistics are keyed by `Arm(draft, gamma)`. Depth belongs in the key because
accepted-per-iteration is bounded above by `gamma`; pooling counts taken at
different depths would estimate nothing in particular.

Tracking is opt-in and off the Phase 0 path entirely:

```python
tracker = AcceptanceTracker()
result = decoder.generate(prompt_ids, 128, cfg, tracker=tracker, domain="code")

tracker.control.estimate(Arm("qwen2.5-0.5b", 4)).tokens_per_iteration  # decisions
tracker.evaluation.table()                                            # the report
```

### What it costs

The brief's stated risk is that switching overhead silently eats the gains it
exists to produce, so this is measured rather than assumed. `generate()` calls
`observe()` once per speculative iteration, at **2.18 µs** warm (4.57 µs on the
first iteration for an arm, which pays the warm-start). Against a 23.6 ms
iteration that is **0.0092%** — three orders of magnitude below the 1.4%
already spent on non-forward work, so the throughput numbers above are
unchanged.

That rules out the *bookkeeping*. It does not yet rule out the switching
decision itself, which does not exist yet.

## Layout

```
src/specdec/
  sampling.py      canonical logits -> probs; shared by every path
  cache_utils.py   positioned forwards, KV rollback, one cache per model
  compile_utils.py capturing the fixed-shape decode step as a CUDA graph
  baseline.py      target-only decoding — the correctness reference
  speculative.py   the accept/reject kernel and the fixed-draft engine
  stats.py         rolling acceptance, split into control and evaluation views
  models.py        pool registry and loading
configs/models.yaml   the model pool
scripts/              weight download, demo, overhead profiler, per-domain sweep
tests/                see "Tests" below
```

## The model pool

| role | model | layers | note |
|---|---|---|---|
| target | `Qwen2.5-14B-Instruct` | 48 | default |
| target | `Qwen2.5-7B-Instruct` | 28 | small-target comparison |
| draft | `Qwen2.5-1.5B-Instruct` | 28 | general, higher acceptance, costlier proposals |
| draft | `Qwen2.5-0.5B-Instruct` | 24 | general, cheaper proposals, lower acceptance |
| draft | `Qwen2.5-Coder-1.5B-Instruct` | 28 | code-specialised — and never wins; see below |

The layer column is there because it turned out to matter more than parameter
count — see below.

Every draft must share the target's tokenizer exactly — the accept/reject test
compares `p(x)` and `q(x)` over a common token id space, so differing
tokenisation makes the comparison meaningless rather than merely inaccurate.
`tests/test_pool_compat.py` asserts this rather than trusting it.

The Qwen2.5 family was picked because it is the one readily available stack that
offers both axes this project needs: a size ladder (the capability/latency
tradeoff) *and* a domain-specialised sibling on the same tokenizer.

The second of those did not pay off, and the pool is worth keeping as the
record of why. The bet was that `Qwen2.5-Coder-1.5B` would win on code and lose
elsewhere, giving a controller a reason to switch. [Measured](#do-the-drafts-separate-by-domain),
it wins nothing anywhere: on code it ties its general-purpose twin to within a
thousandth (0.804 vs 0.805) and at `γ=8` it falls below 1.0x on chat and
summarisation. Acceptance turns out to track proximity to the *target's*
distribution rather than competence at the task, and both 1.5Bs are equally
close to a general instruct target.

That is a real finding rather than a failed setup — it says the size ladder is
the axis that matters and the specialisation axis is not — but it does mean
`Qwen2.5-Coder-1.5B` currently earns its ~3 GB of residency only as evidence.

## Tests

| file | what it pins down |
|---|---|
| `test_verify_kernel.py` | the accept/reject rule emits `p`, on synthetic distributions, at 60k samples |
| `test_engine_correctness.py` | the engine around it: cache indexing, rollback, vocab alignment, iteration chaining |
| `test_acceptance_stats.py` | the estimator's cold start and decay, and that the domain tag never reaches the controller |
| `test_sampling.py` | the shared warper path |
| `test_static_cache.py` | the preallocated backend is indistinguishable from the growable one, including rollback |
| `test_pool_compat.py` | every draft shares the target's tokenizer |
| `test_real_models.py` | the eager guarantees on real Qwen2.5 weights in bf16 (`-m gpu`) |
| `test_compiled_correctness.py` | what survives CUDA-graph capture, and what doesn't (`-m gpu`) |

The statistical tests run against tiny randomly-initialised models rather than
the real pool. That is deliberate: distribution equality needs tens of thousands
of samples to have any power, and a 7B model gives you a few dozen. The toy
models exercise identical engine code at a sample count where a broken
verification rule actually fails.

Greedy decoding gives a second, sharper check — with `p` a point mass the whole
scheme collapses to "accept iff the draft guessed the argmax", so speculative
output must equal target-only output *token for token*. That catches cache and
indexing bugs immediately, which the statistical tests would only see as drift.
On the tiny models and on the 0.5B that equality is exact; on the 14B it holds
up to the measured batch-width noise floor, for the reason above.

The `-m gpu` suite has one more probe worth knowing about: **the target
drafting for itself**. Under greedy decoding every proposal is then that
model's own argmax, so the acceptance rate is precisely the rate at which the
width-1 and width-`γ+1` paths agree — the numerical gap isolated from how good
any draft happens to be.

See `SETUP.md` for environment and how to run.
