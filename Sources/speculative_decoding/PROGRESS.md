# PROGRESS

Working state for the adaptive speculative decoding project. Written so a
session starting cold can pick up without re-deriving anything.

**Read `CLAUDE.md` first** — it is the project brief and defines the phases,
the working agreement, and the non-negotiable correctness property. This file
records what has actually happened against that plan.

Last updated: 2026-08-16, at commit `99ed1c2`.

`RESULTS.md` (tracked, added at `99ed1c2`) is the standalone write-up of the
fixed-draft engine — results, the 0.88x → 2.25x diagnosis, per-domain spread,
correctness. It deliberately contains **nothing** about the cancelled adaptive
work; the README still carries that history. Keep the two consistent if any
number changes.

---

## 0. Starting cold — read this first

> **SCOPE CUT — SETTLED 2026-08-16.** The project is now **fixed-draft
> speculative decoding only: one draft model, one fixed depth.** Phases 2, 3
> and 4 are **cancelled**. The adaptive code stays in the tree, parked, by
> explicit decision — see §0a. **No code has changed under this decision**; the
> tree is at `cabe822` and the cut is so far recorded only in documentation.

**The engine is done.** What Phases 0 and 0.5 built *is* the deliverable:
correct fixed-draft speculative decoding at **2.25x** (γ=4) / **2.56x** (γ=8)
over the target-only baseline, both sides compiled, correctness verified. There
is no remaining build work in scope. In order:

1. **Read `CLAUDE.md`** (the brief), especially the working agreement. It is
   gitignored, so it exists only in the working directory. **The brief describes
   a much wider project than is now in scope** — it still specifies multi-draft
   adaptive switching across four phases. §0a is authoritative; `CLAUDE.md` is
   history.
2. **Do not start Phase 2/3/4 work.** Cancelled. In particular, do not "finish"
   the depth controller, do not run the GPU sweeps in §7, and do not act on the
   `[DECISION]`s recorded in §3 for Phase 2 — they were settled for a phase that
   no longer exists.
3. **Phase 1 was never checkpointed with the user** — the brief requires one at
   the end of every phase. Now largely moot, since the phase it belongs to is
   cancelled, but it is recorded here rather than quietly dropped.
4. **The adaptive code is parked, not dead-but-hidden.** `controller.py`,
   `stats.py` and their tests remain in the tree and remain green. Do not build
   on them, and do not delete them either (§0a).
5. Confirm the tree still matches this file:

   ```bash
   cd /data/jayavardhan/projects/speculative_decoding
   git log --oneline -3      # expect 99ed1c2 on top, working tree clean
   CUDA_VISIBLE_DEVICES="" /data/jayavardhan/tools/conda-envs/speculative_decoding/bin/python \
       -m pytest -m "not gpu" -q     # expect 131 passed, 21 deselected
   ```

Three traps that have already cost time and are easy to walk back into:
**never run a bare `pytest`** (§6), **never `pkill` by pattern** (§2, fact 5),
and **never quote a speedup with only one side compiled** (§4a).

**Verified:** 131 CPU tests and 21 GPU tests green. Phase 1 has run on the 14B;
the per-domain and depth sweeps are complete (§5).

**Not verified:** the Phase 2 controller. Its 19 unit tests drive it through
its own episode interface with synthetic throughput; one further test runs it
through a real `generate()`, but for 8 tokens on a tiny model and only to check
the capture guard. **No test checks that varying γ mid-request leaves the
output distribution alone**, and it has never touched real weights or run long
enough for an episode (`dwell=16`) to complete. Treat it as unproven code that
happens to have passing unit tests.

The single most important finding: **the sweep falsified the assumption Phase 2
was to be built on.** Draft switching buys nothing on this pool; adapting draft
*depth* does, and choosing depth badly costs up to a 2x slowdown. §5 has the
numbers, §7 the plan.

---

## 0a. Scope cut — settled 2026-08-16

### The decisions, as made

Raised 2026-08-15 (*"I want to shift down to implementing just speculative
decoding"*) and settled 2026-08-16. Three decisions, all the user's:

1. **Fixed-draft only.** *"Fixed draft only — only one draft model, and we
   already have the performance improvements over no draft already I think."*
   That reading is correct: 2.25x at γ=4 and 2.56x at γ=8, both sides compiled,
   correctness verified. **Phases 2, 3 and 4 are cancelled.** The engine is the
   deliverable and it is already built, tested and committed.
2. **The adaptive code is parked, not deleted.** `controller.py` (265 lines),
   `stats.py` (442), `test_depth_controller.py` (248) and
   `test_acceptance_stats.py` (446) stay in the tree, stay exported, and stay
   green. Deletion was offered and declined. **Do not delete them** in a later
   tidy-up believing it to be obvious cleanup — it was considered and rejected.
3. **`scripts/domain_sweep.py` stays.** It produced the per-domain table in §5
   and is how the γ default would be re-picked for a different workload.
   *Consequence worth noting:* keeping `stats.py` means the sweep's imports
   still resolve, so **it needs no rewiring** — a rewire was only required under
   the delete-everything option.

### Settled by data, not preference

- **Default depth stays γ=4.** Already the default in `speculative.py:160` and
  `demo.py:38`. §5 makes it the best fixed depth on an even mix (1.69x mean, vs
  1.65x at γ=8). No change was needed, so none was made.
- **The draft is the 0.5B** — highest throughput in every domain at every depth
  measured (§5).

### Explicitly not done

The user's instruction was to record the decisions **only**: *"Don't do any
changes to the project, nor any demo etc."* So the following are **known, open,
and deliberately untouched** — they are not oversights:

- **`demo.py:37` still defaults to `--draft qwen2.5-1.5b`**, not the 0.5B the
  deliverable is built around. The demo showcases the slower draft by default.
  Raised with the user and deliberately left alone.
- **The parked code is not marked as parked in the source.** No module docstring
  in `controller.py` or `stats.py` says the phase was cancelled, and both are
  still exported from `__init__.py` whose docstring still describes Phase 1 as
  live. Only this file records it.
- **`README.md` and `CLAUDE.md` still read as a multi-draft adaptive project**
  beyond the Status note. The README's opening paragraph still describes an
  engine that "switches between them at runtime".

### Why multi-draft routing was dropped — do not re-litigate

The user asked whether keeping the pool would show different drafts winning
their own domains, since that is what would sound good on a resume.

**It would not, and the measurement already exists** (§5): the 1.5B has the
highest acceptance in every domain and the 0.5B the highest throughput in every
domain, identical ordering at γ = 2, 4, 8, 12, 16. Coder-1.5B on code scored
0.804 against the general 1.5B's 0.805.

The mechanism, which is why real datasets were unlikely to overturn it:
acceptance measures proximity of `q` to `p`, and `p` belongs to a *general*
instruct target. A specialised draft moves away from a general target's
distribution by about what it gains on the domain. The effects cancel.

**The one way it could genuinely work** — the honest answer, and a good
interview answer: distil the domain drafts **from the target's own outputs** per
domain, so proximity-to-target and specialisation stop being in tension (the
EAGLE-style insight). That means *training* drafts, not loading them. Explicitly
not recommended and out of scope.

### Resume framing — the reason the scope moved

The user's concern was which claim is strongest on a resume. What was argued,
and is worth not re-deriving:

- **"Implemented speculative decoding, got 2.25x" is the weakest true claim
  here.** Published spec-dec speedups are 2–3x, so it reads as correct, not
  notable, and it does not differentiate.
- **Strongest material, in order:**
  1. **The performance diagnosis.** A *correct* decoder that ran at **0.88x —
     slower than not speculating** — because batch-1 decode is CPU-launch-bound,
     not compute-bound (0.81 ms/layer for the 14B *and* the 0.5B, flat across a
     28x parameter span). Static cache + no-data-movement rollback + fixed
     shapes + CUDA graphs → **2.56x**. Lead with this, not the speedup alone.
  2. **The falsified hypothesis.** Designed for draft switching, measured it,
     found the axis flat, understood *why*, and cut the scope rather than
     shipping a feature the data did not support.
  3. **The correctness methodology** (§4b) — underrated. Proved bf16
     batch-width numerics alone break token equality on the 14B, with the
     target-only baseline *disagreeing with itself* as evidence, then derived
     the test tolerance from a measured noise floor instead of guessing or
     loosening to a token-error rate.
- **Do not quote the "7% over best fixed depth" number** — it belonged to the
  cancelled controller and is not a claim this project makes any more.

### The question that triggered it, and the answer given

The user asked: *if we keep the pool, do different drafts actually work best in
their own domain and collectively raise throughput — because that is what would
sound good on a resume?*

**Answer given: no, and it will not happen.** Already measured (§5): the 1.5B
has the highest acceptance in every domain and the 0.5B the highest throughput
in every domain, identical ordering at γ = 2, 4, 8, 12, 16. Coder-1.5B on code
scored 0.804 against the general 1.5B's 0.805.

The mechanism, which is why real datasets are unlikely to overturn it:
acceptance measures proximity of `q` to `p`, and `p` belongs to a *general*
instruct target. A specialised draft moves away from a general target's
distribution by about what it gains on the domain. The effects cancel.

**The one way multi-draft routing could genuinely work** — worth keeping because
it is the honest answer and a good interview answer: distil the domain drafts
**from the target's own outputs** per domain, so proximity-to-target and
specialisation stop being in tension (the EAGLE-style insight). This requires
*training* drafts, not just loading them. Explicitly **not** recommended and
beyond even the brief's Phase 4 stretch — but it is the correct answer to
"could this ever work", not "no".

### Resume framing — the reason the scope moved

The user's concern was which claim is strongest on a resume. What was argued,
and is worth not re-deriving:

- **"Implemented speculative decoding, got 2.25x" is the weakest true claim
  here.** Published spec-dec speedups are 2–3x, so it reads as correct, not
  notable, and it does not differentiate.
- **Strongest material, in order:**
  1. **The performance diagnosis.** A *correct* decoder that ran at **0.88x —
     slower than not speculating** — because batch-1 decode is CPU-launch-bound,
     not compute-bound (0.81 ms/layer for the 14B *and* the 0.5B, flat across a
     28x parameter span). Static cache + no-data-movement rollback + fixed
     shapes + CUDA graphs → **2.56x**. Lead with this, not the speedup alone.
  2. **The falsified hypothesis.** Designed for draft switching, measured it,
     found the axis flat, understood *why*, redirected to depth. The
     replacement finding is stronger than the original: depth swings 2.60x
     (code) → 0.66x (summarisation), so a wrong fixed depth is a *regression*,
     not a missed gain.
  3. **The correctness methodology** (§4b) — underrated. Proved bf16
     batch-width numerics alone break token equality on the 14B, with the
     target-only baseline *disagreeing with itself* as evidence, then derived
     the test tolerance from a measured noise floor instead of guessing or
     loosening to a token-error rate.
- **Do not write "7% faster."** If depth adaptivity survives, the defensible
  framing is that *"γ=4 is best" is only knowable from an offline sweep across
  every domain you will ever serve*, plus avoiding the 0.66x cliff.

### Consequence for the draft pool

The pool's value is **the experiment, not the feature** — cited as a defensible
negative result explaining *why* one draft ships, never as adaptive routing.
`configs/models.yaml` still declares all five models; that is a registry, not a
load, so it costs nothing at runtime. The engine uses one draft. Nothing about
the pool was changed under this decision.

---

## 1. Where things stand

**Scope cut to fixed-draft only, 2026-08-16 (§0a).** The table below is the
*original* four-phase plan, kept for context. Phases 2–4 are cancelled.

| phase | state |
|---|---|
| Phase 0 — fixed-draft baseline | **done**, checkpointed — *in scope* |
| Phase 0.5 — make it faster (unplanned, user-approved) | **done**, checkpointed — *in scope* |
| Phase 1 — online acceptance-rate tracking | **done**, checkpoint owed but moot — code parked |
| Phase 2 — depth controller | **cancelled** — code parked, unfinished, unverified |
| Phase 3 — cross-domain benchmarking | **cancelled** |
| Phase 4 — tree drafting (stretch) | **cancelled** |

**The deliverable is complete.** Phases 0 and 0.5 — the correct fixed-draft
engine, the target-only reference it is verified against, the static-cache and
CUDA-graph machinery, and the measured 2.25x / 2.56x — are built, tested and
committed. No build work remains in scope.

What the cancelled phases left behind, all still green and all still in the
tree by decision (§0a): `controller.py`, `stats.py`, and their two test files.
The Phase 2 controller was never finished — **nothing verifies that varying γ
mid-request preserves the output distribution**, and it never ran on real
weights. That is now permanently untrue-of-scope rather than a debt: it does
not matter, because the shipped engine holds γ fixed.

Working tree clean, 21 commits on `master`, nothing pushed to a remote yet.
`CLAUDE.md` is gitignored and untracked, as the brief requires.

**Headline result:** speculative decoding runs at **2.25x** the target-only
baseline at `γ=4` and **2.56x** at `γ=8`, with both sides compiled. It started
Phase 0 at 0.88x — i.e. slower than not speculating.

**Headline finding:** across code / chat / summarisation, the drafts do **not**
separate by domain — the same draft wins everywhere on acceptance and the same
other one wins everywhere on throughput — but the optimal draft *depth* moves
sharply by domain. Code peaks at γ=8 (2.60x); chat and summarisation peak at
γ=4 or below and fall to 0.66–0.82x by γ=16. Phase 2 adapts **depth**, not
draft identity (§5, §7). Realistic headroom over the best fixed depth is
**~7%** on an even mix — modest, and worth stating as such.

---

## 2. Environment

| | |
|---|---|
| conda env | `speculative_decoding` (Python 3.11.15) |
| env path | `/data/jayavardhan/tools/conda-envs/speculative_decoding` |
| python | `/data/jayavardhan/tools/conda-envs/speculative_decoding/bin/python` |
| torch / transformers | 2.13.0+cu130 / 5.15.0 |
| GPU | 1x H200, bf16, batch size 1 |
| weights | `HF_HOME=/mnt/warm_storage/jayavardhan/tools/huggingface` |

`SETUP.md` has the reproducible version. Five environment facts that cost real
time to learn:

1. **The NFS mount reads at ~26 MB/s and other users evict the page cache.**
   A cold 14B load takes **20–40 minutes**. Structure GPU work as *one*
   long-lived process that sweeps every configuration. Budget the load before
   expecting any output.
2. **`/data` enforces a quota well below its free space.** Staging weights
   locally failed at 12 GB with `ENOSPC` while `df` reported 1.7 TB free and 2%
   inodes. It also broke `git commit` on `.git/index.lock` at the same moment.
   Do not stage weights locally. Check `du -sh /data/jayavardhan`, not `df`.
3. **This is a shared box (~30 users).** Check `nvidia-smi` immediately before
   *and after* any timing run. Never touch a process you did not start.
4. **Never run two GPU jobs concurrently when one is timing-sensitive.** Batch-1
   decode is CPU-launch-bound, so a co-running job skews exactly the numbers
   being measured. Correctness runs first, then the benchmark alone.
5. **Your own uid runs other projects.** On 2026-08-14 a `pytest -m gpu` from a
   *different* project (env `.../conda-envs/vllmmini`, GPU 7, another working
   directory) was running alongside this one. So `pgrep -u $(whoami)` is **not**
   proof a process is yours to kill, and a broad `pkill -f <pattern>` can
   destroy another project's hours-long GPU run while looking like tidying up.
   In this session such a `pkill` also matched its own wrapper shell and killed
   the run it was meant to manage. Identify the exact PID with
   `ps -o pid,cmd -p <pid>` and confirm env path *and* working directory before
   signalling anything; prefer letting a stale run finish. The same co-tenancy
   inflated this project's CPU suite from 4m21s to 9m48s — check before quoting
   any wall-clock duration.

---

## 3. What was built

### Phase 0 — fixed-draft speculative decoding

Standard Leviathan/Chen accept-reject with a single fixed draft, plus the
target-only baseline it is verified against.

```
src/specdec/
  sampling.py       canonical logits -> probs; shared by EVERY path
  cache_utils.py    positioned forwards, KV rollback, CachePool
  compile_utils.py  CUDA-graph capture of the fixed-shape decode step
  baseline.py       target-only decoding — the correctness reference
  speculative.py    the accept/reject kernel and the engine
  stats.py          Phase 1: rolling acceptance, control vs. evaluation views
  controller.py     Phase 2: draft-depth selection, episodes and hysteresis
  models.py         pool registry and loading
```

**Model pool** (`configs/models.yaml`, `default_target: qwen2.5-14b`):

| role | model | layers |
|---|---|---|
| target | Qwen2.5-14B-Instruct | 48 |
| target | Qwen2.5-7B-Instruct | 28 (comparison only) |
| draft | Qwen2.5-1.5B-Instruct | 28 |
| draft | Qwen2.5-0.5B-Instruct | 24 |
| draft | Qwen2.5-Coder-1.5B-Instruct | 28 |

Decided with the user, deviating from the brief's Llama 3.1 + SmolLM2: SmolLM2's
tokenizer is unrelated to Llama's (49k GPT2-style vs 128k), which makes the
`p(x)` vs `q(x)` comparison *meaningless* rather than merely inaccurate. Qwen2.5
is the one available stack with both axes the project needs — a size ladder and
a domain-specialised sibling on the same tokenizer. **Any new draft must be
tokenizer-identical to the target**; `tests/test_pool_compat.py` enforces it.

Of those two axes, only the size ladder turned out to matter: the
domain-specialised sibling wins nothing anywhere (§5). Kept in the pool as the
evidence for that, but it is a candidate for removal — dropping it shrinks the
arm space Phase 2 has to search.

### Phase 0.5 — making it actually faster

Phase 0 was correct but **0.88x**. The algorithm was fine (acceptance 0.84–0.92,
target forwards down 4.3x); the cost model around it was wrong. The user chose
to fix both the overhead *and* the target/draft depth ratio.

What changed:

- **`StaticCache` behind a `KVCache` handle**, length tracked host-side (reading
  it back from the backend costs a device sync per step — the exact overhead
  being removed).
- **Rollback is now a counter reset with no data movement.** Rejected K/V is
  left in place; sound because the next forward overwrites it and causal
  masking hides the rest.
- **Prefill split out of the first verification**, so every verify is exactly
  `γ+1` wide and every draft step is 1 (or 2, catching up after a fully-accepted
  iteration). `SpeculativeDecoder.decode_shapes` declares the set.
- **`CachePool`** keyed by `(model, role)` — one cache per model, so switching
  drafts leaves the target's captured graphs alone. Keyed by role too, because
  one module can play both parts.
- **`torch.compile(mode="reduce-overhead")`** on those fixed shapes.
- **Default target moved 7B → 14B**, for a 48-vs-24 layer ratio against the
  0.5B where the 7B had 28-vs-28.

### Phase 1 — online acceptance-rate tracking

`src/specdec/stats.py`. Two `[DECISION]`s were put to the user and answered:

**How "domain" is detected/labelled → hybrid: the tag measures, behaviour
controls.** A request may carry a `domain` tag; it is an *evaluation key only*
and the controller never sees it. Rationale: the tag is a proxy (what matters
is whether this draft predicts this target well, not whether the prompt is
code), a tag-driven controller is just a per-dataset lookup table, and holding
it out turns Phase 3 into a real question — did the controller rediscover the
domain structure unaided? Since HumanEval *is* code, the held-out tag is exact
and free as ground truth.

**Estimator → EMA on accepted-per-iteration.** Not the per-token acceptance
rate: the per-iteration count maps linearly to throughput and assumes nothing
about independence, which §5 disproved. A Beta-Bernoulli posterior was
considered and rejected for exactly that reason — its variance would be
confidently wrong under bursty acceptance.

Structure:

| | |
|---|---|
| `Arm(draft, gamma)` | the control key. Depth is in the key because accepted-per-iteration is bounded by γ, so counts at different depths do not pool. |
| `AcceptanceEMA` | carries an accumulated weight rather than a fixed divisor, so cold start is unbiased and warm start is the same arithmetic. Tracks mean, variance, `tokens_per_iteration`. |
| `ControlView` | arm-keyed only. The controller's entire input. |
| `EvaluationView` | `(domain, arm)`-keyed exact counters. Reporting only. |
| `RequestScope` | fast per-request estimate, warm-started from the slow one. Keyed per arm so a mid-request switch does not lose the arm it left. |

Two timescales, because domain is constant *within* a request and varies
*between* them — a pooled average over a mixed workload tracks the mixture, not
the request in front of it. Fast `α=0.15` (~7 iterations), slow `α=0.02`. The
EMA is what does the regime detection a domain label would otherwise do.

Wired into `SpeculativeDecoder.generate(..., tracker=..., domain=...)`, both
optional — the Phase 0 path is untouched and a test asserts identical tokens.

The whole API, which is all Phase 2 needs:

```python
from specdec import AcceptanceTracker, Arm

tracker = AcceptanceTracker()          # alpha_fast=0.15, alpha_slow=0.02
result = decoder.generate(prompt_ids, 128, cfg, tracker=tracker, domain="code")

# One generate() == one request; the scope is created internally and exposed:
scope = result.extra["scope"]          # RequestScope, or None if untracked
arm   = result.extra["arm"]            # Arm(draft_name, gamma)

# What a controller reads (never domain-keyed):
scope.estimate(arm).tokens_per_iteration   # fast, this request
scope.estimate(arm).variance               # confidence; honest under burstiness
tracker.control.estimate(arm).mean         # slow, across requests
tracker.control.arms                       # every arm seen

# What the report reads (never a decision):
tracker.evaluation.table()                 # [(domain, arm, ArmStats), ...]
tracker.evaluation.stats("code", arm).acceptance_rate
```

To drive it manually without the engine — `tracker.new_scope(domain)` then
`scope.observe(arm, n_accepted)` per iteration.

**The prior-strength knob is denominated in observations**, not in an opaque
weight. First attempt made it a raw weight; a test caught that "0.5" then meant
roughly *four* iterations of evidence, not half of one, because an update
contributes only `α` to the accumulated weight. Now `seeded_from(...,
observations=n)` sets weight to `1-(1-α)^n`, and
`test_prior_observations_are_a_real_count` pins the unit. This matters because
Phase 2 has to tune it by hand.

---

### Phase 2 — depth controller (**in progress, not finished**)

`src/specdec/controller.py`. Three `[DECISION]`s were put to the user and
answered:

1. **Control axis → depth only, draft fixed to the 0.5B.** The brief says
   switch *drafts*; the sweep says draft choice is a flat axis and depth is
   not (§5). Agreed with the user as a deviation from the brief.
2. **Learning → explore and measure, not predict.** Acceptance at one depth
   does not give throughput at another, and the analytic cost model is already
   known wrong here (assumes independent acceptance; §5 measured 4.13
   tokens/iteration against 3.35 predicted). So the controller *tries* other
   depths.
3. **Anti-thrash → minimum dwell + switch margin.** Dwell stops fast flapping;
   the margin stops slow drift between near-equal depths, which an EMA alone
   does not.

**Episodes.** Ranking depths by throughput needs wall-clock time, but timing
each iteration means a device sync each iteration — the exact overhead Phase
0.5 removed. So the controller holds a depth for `dwell` iterations and the
loop times the *episode*: one sync per ~16 iterations, and the minimum dwell
falls out of the same structure rather than being bolted on.

**Preference vs current.** The controller separates the depth exploitation
favours (`preferred`) from what is actually running (`current`, which differs
only during a probe). `n_switches` counts preference changes only. This matters
for judging stability: counting probes as switches makes a healthy controller
look like a thrashing one at a rate set by `explore_every`.

Wired in as `generate(..., controller=...)`, which then sets γ per iteration
and ignores the `gamma` argument. `compile(depths=...)` captures every
candidate width up front — a width the controller picks but that was never
captured runs eager and shows up as an unexplained slowdown.

Defaults: `dwell=16`, `margin=0.05`, `explore_every=8`, `alpha=0.3`. Untuned
against real workloads.

**What is NOT done — Phase 2 is not finishable without these:**

- **No correctness test that varying γ mid-request preserves the output
  distribution.** The argument that it does is sound (the accept/reject rule is
  exact for any `q` and any width, and each iteration is self-contained) but
  it is currently *only* an argument. This is precisely the class of thing the
  brief says to surface rather than assume, and it is the next task.
- **Barely run end to end.** `test_static_cache.py::test_controller_depths_are_
  all_checked_before_the_first_token` drives a real `generate()` with a
  controller, so the wiring executes — but for 8 tokens on a tiny model, and
  only to check the capture guard. Nothing checks the *output*.
- **Never run on real weights**, and never with enough tokens for an episode
  (`dwell=16`) to complete inside a single request.
- **No comparison against a fixed-γ baseline**, which is the number that
  justifies the phase.
- Defaults unturned; `explore_every=8` in particular trades throughput for
  adaptability and has never been measured against a real workload.

## 4. Findings that must not be re-derived

### 4a. Eager batch-1 decode is launch-bound, not compute-bound

Forward cost per layer, eager: **0.813 / 0.808 / 0.820 / 0.805 ms** for the
14B / 1.5B / 0.5B / Coder-1.5B. Flat across a 28x span in parameter count.
Latency was `n_layers x a constant` and ignored model *width* entirely — the
GPU sat idle between kernel launches and the wall clock was Python dispatch.

That is why Phase 0 lost: with a 7B target (28 layers) and a 1.5B draft (28
layers), a draft call cost the same as a target call, and speculation needs
`draft_cost << target_cost`.

Compiled, the same column reads **0.223 / 0.098 / 0.098 / 0.071**. Latency
tracks model size again.

> **Always compile both sides before quoting a speedup.** Compilation alone
> took the baseline from 43.5 to 86.2 tok/s. A compiled speculative run
> measured against an eager baseline reports the compiler's win, not
> speculation's.

### 4b. bf16 batch-width numerics break token-for-token equality on the 14B

Row 0 of a width-2, width-5 or width-9 forward differs from a width-1 forward
at the same absolute position by exactly **0.25** on the 14B, and exactly
**0.0** on the 0.5B. Same query, same keys — only the GEMM batch dimension
differs, changing tiling and accumulation order.

The engine reads `p` from a width-`γ+1` verification; the baseline reads it from
width 1. bf16 margins are quantised (smallest nonzero top-1/top-2 gap observed:
0.125; 1 position in 41 was an exact tie), so two ulps is enough to flip an
argmax and redirect the whole sequence.

**This is not caused by speculative decoding.** It is a property of running one
model at two batch widths, which every batching server does. The cleanest proof:
the *target-only baseline disagrees with itself* — three identical runs, no
speculation anywhere, two emitted a different token at position 118 than the
third.

Consequences for how tests are written:

- Never assert bit-equality of generated tokens on the 14B.
- `tests/conftest.py::measure_width_sensitivity` measures the noise floor for
  whichever target is loaded.
- `assert_matches_or_diverges_within_noise` accepts a divergence **only** at a
  position whose top-1/top-2 margin falls inside that floor. A divergence at a
  decisive margin still fails — that is where a misindexed verification or a
  stale cache would land, since those are wrong by whole units.
- Compiled runs measure their own floor (0.281 vs eager 0.250); inductor's
  shape-specialised kernels stack on top of eager's reordering.

**Do not "fix" this by loosening to a token-error rate.** The derived-tolerance
approach is deliberate and was chosen over that.

### 4c. Things already ruled out — do not re-investigate

- **Static cache leaking stale K/V.** Poisoned the cache with ±100 (far outside
  the activation range), rolled back, compared: bit-identical in prefill, decode
  and verify at every poison scale. Rollback-by-offset is sound.
- **Prefill width changing results.** Reaching the same position via five
  different prefill splits: `max|d| = 0.000000` throughout.
- **The drafts being secretly the same model.** Confirmed distinct proposals and
  distinct `id()`.
- **Sampling being on the critical path.** It is 1.4% of an iteration (§5).

### 4d. Measurement traps that produced wrong conclusions

- **Warm-up.** The discarded first rep reported 0.49x and 7.64x for two drafts
  that both settle near 2.0x. An earlier benchmark without a warm-up rep
  reported a 2.15x that was *entirely* warm-up.
- **Round-robin.** Measure interleaved with repetitions, never one configuration
  at a time — otherwise a draft that merely ran later looks faster.
- **Missing `vocab_size`.** `generate_baseline(...)` defaults to the target's
  full 152064 embedding while the engine uses the canonical 151936. Comparing
  those proves nothing. **Every** baseline used as a reference must be passed
  `vocab_size=decoder.vocab_size`. This bit the benchmark, both GPU test files
  and the demo.

### 4e. A test had been red since a6ce931, unnoticed

`test_pool_compat.py::test_registry_has_one_target_and_several_drafts` asserted
the registry held exactly one target. Phase 0.5 added the 7B as a comparison
target and kept it, so the assertion had been false since **a6ce931** — the
same commit that moved the default to the 14B. It was found only because Phase
1 prompted a full-suite run.

Two lessons, both about how it stayed hidden:

- **`git log` on the test file does not tell you the test still passes.**
  `test_pool_compat.py` was touched *after* a6ce931 (in 99fa9bc) without the
  stale assertion being noticed, because the file was edited for another
  reason and the suite was not re-run.
- **These tests are `hub`-marked and need tokenizer files**, so a run without
  the HuggingFace cache reports them as errors rather than failures, and an
  eye scanning for "failed" slides past.

Fixed, and the neighbouring bug it exposed fixed with it: the `target_spec`
fixture resolved the target as *the first entry with role "target"* rather than
through `default_target`, so with two targets declared the tokenizer-identity
tests may have been validating drafts against the 7B while the engine loads the
14B. Both Qwen2.5 targets share a tokenizer, so nothing was actually wrong —
but the test was not checking what it claimed to. It now resolves via
`default_target_name()`, and `test_every_draft_shares_every_target_vocabulary`
checks the drafts against *every* declared target, since `default_target` is
one line of YAML away from changing.

---

### 4f. The self-drafting test measured the wrong denominator

The GPU suite was run on 2026-08-14 (GPU 7, 63 min): **20 passed, 1 failed**.
The failure was `test_self_drafting_measures_shape_agreement`, and it was a
measurement bug in the test, not a defect in the engine.

Observed on the `code` prompt: `accepted_per_iteration=[4,4,4,4,4,4,4,1,4,4,4]`
— ten perfect iterations and one argmax flip. Reported 0.932 against a 0.95
floor.

The test used `SpecStats.acceptance_rate`, i.e. `n_accepted / n_proposed`. But
`n_proposed += gamma` unconditionally, while verification stops at the first
disagreement — so the proposals after a rejection are counted in the
denominator despite never having been compared against the target. One flip at
index 1 of a `γ=4` iteration becomes **three** apparent disagreements.

- positions actually verified: `10×4 + 2 = 42`
- agreements: `41`
- honest rate: **0.976** (passes) vs reported **0.932** (fails)

Two things follow, and both are now fixed:

- **The statistic.** `conftest.shape_agreement(stats, gamma)` counts `gamma`
  for a fully-accepted iteration and `n+1` otherwise. The bias it removes grows
  with `γ`, so a fixed floor on the old statistic was implicitly a floor on
  draft depth too — `tests/test_shape_agreement_metric.py` pins this on CPU,
  including the exact failing counts as a regression anchor.
- **The loop asserted per domain**, so the failure on `code` (first in the
  dict) left `chat` and `summarisation` unevaluated. It now measures all three
  and asserts once, reporting every rate.

`acceptance_rate` itself is **not** wrong and was left alone: counting wasted
draft work is exactly what it is for, and that is the right metric for the
demo, profiler and benchmark. It was only the wrong metric for this one test's
specific claim about per-position argmax agreement.

**Re-run and confirmed:** 21/21 on GPU 6, 2026-08-14. The corrected metric
passes on real weights.

---

## 5. Current numbers

H200, bf16, batch 1, 128 new tokens, best of 3 interleaved reps after a
discarded warm-up. Reproduce with:

```bash
CUDA_VISIBLE_DEVICES=<free gpu> python scripts/profile_overhead.py \
    --gamma 4 --sweep-gammas 2 8 --reps 3
```

| | baseline | 1.5B | 0.5B | Coder-1.5B |
|---|---|---|---|---|
| eager | 43.5 tok/s | 1.35x | 1.33x | 1.26x |
| compiled | 86.2 tok/s | 2.03x | **2.25x** | 1.91x |

Draft depth, compiled:

| draft | γ=2 | γ=4 | γ=8 |
|---|---|---|---|
| 1.5B | 1.68x (accept 0.955) | 2.03x (0.911) | 2.29x (0.824) |
| 0.5B | 1.84x (0.913) | 2.25x (0.798) | **2.56x** (0.681) |
| Coder-1.5B | 1.61x (0.902) | 1.91x (0.825) | 2.06x (0.770) |

Throughput still climbing at γ=8 — **the optimum has not been found**; the sweep
should be extended (12, 16) before Phase 2 tunes depth per domain.

One γ=4 iteration, compiled — this answers the brief's "measure switching
overhead explicitly" risk:

| | ms |
|---|---|
| 4 draft forwards + 1 verification | 23.251 |
| `crop_cache`, 14B (48 layers) | 0.169 |
| `crop_cache`, 1.5B (28 layers) | 0.099 |
| `verify_and_correct` (incl. host syncs) | 0.036 |
| `logits_to_probs` (5 × 151936) | 0.029 |

Non-forward work totals **0.33 ms of 23.6 ms = 1.4%**.

The cost model under-predicts every draft (1.89x predicted vs 2.25x measured for
the 0.5B) because it assumes independent acceptance. Real acceptance is
**bursty**: the 0.5B emitted 4.13 tokens/iteration against a predicted 3.35.
The controller should read *measured* acceptance, not infer throughput from the
formula.

### Per-domain sweep — the result that reshapes Phase 2

`scripts/domain_sweep.py --gammas 4 8 --reps 3`, run 2026-08-14 on GPU 6. Three
prompts per domain, all three drafts, 14B target, greedy, 128 new tokens,
median of 3 reps after a discarded warm-up.

Acceptance / speedup, **γ=4**:

| domain | 1.5B | 0.5B | Coder-1.5B |
|---|---|---|---|
| code | 0.805 / 1.89x | 0.734 / **2.23x** | 0.804 / 1.89x |
| chat | 0.463 / 1.31x | 0.401 / **1.47x** | 0.391 / 1.17x |
| summarisation | 0.411 / 1.23x | 0.338 / **1.41x** | 0.355 / 1.18x |

**γ=8**:

| domain | 1.5B | 0.5B | Coder-1.5B |
|---|---|---|---|
| code | 0.700 / 2.08x | 0.614 / **2.60x** | 0.683 / 1.97x |
| chat | 0.299 / 1.08x | 0.249 / **1.26x** | 0.233 / **0.91x** |
| summarisation | 0.258 / 1.05x | 0.190 / **1.13x** | 0.219 / **0.90x** |

Three findings, in order of how much they change the plan:

1. **No separation by draft.** The 1.5B has the highest acceptance in every
   domain; the 0.5B has the highest throughput in every domain; identical
   ordering at both depths. A controller choosing *which draft* has nothing to
   choose. **This falsifies the assumption Phase 2 was going to be built on.**
2. **Large separation by depth.** Deeper drafting helps on code (0.5B:
   2.23x → 2.60x) and hurts on chat (1.47x → 1.26x) and summarisation
   (1.41x → 1.13x). At γ=8 the Coder draft goes **below 1.0x** on both — slower
   than not speculating. Getting depth wrong does not merely forfeit gains, it
   goes backwards, which is a *stronger* argument for adaptivity than the
   original one.
3. **Specialisation bought nothing.** On code, Coder-1.5B scores 0.804 against
   the general 1.5B's 0.805 — a dead heat, at identical throughput, from
   architecturally identical models with matching compiled forward cost.

The mechanism behind (3), which is worth keeping because it generalises:
acceptance measures how close `q` is to `p`, and `p` belongs to the *target* — a
general instruct model. A domain-specialised draft moves away from a general
target's distribution by about as much as it gains on the domain. **What earns
acceptance is distributional proximity to the target, not competence at the
task.** Fine-tuning a draft on the workload is not obviously a way to speed it
up.

Consistency check: 2.23x here for the 0.5B on code against 2.25x from the
earlier single-prompt benchmark.

Caveats: three prompts per domain, 128 tokens, greedy, one target. This is the
precursor that decides Phase 2's shape, **not** Phase 3.

### Extended depth sweep — the optimum, bracketed

`scripts/domain_sweep.py --gammas 4 8 12 16 --reps 3`, GPU 6, 2026-08-14.
Speedup for the **0.5B** (the draft Phase 2 fixes on):

| domain | γ=4 | γ=8 | γ=12 | γ=16 |
|---|---|---|---|---|
| code | 2.25x | **2.60x** | 2.44x | 2.31x |
| chat | **1.42x** | 1.25x | 1.03x | 0.82x |
| summarisation | **1.41x** | 1.10x | 0.94x | 0.66x |

**Code peaks at γ=8 and turns over** — closing the question open since Phase
0.5, where throughput was still climbing at 8.

**New gap, the mirror image:** chat and summarisation fall monotonically from
γ=4, so their optimum is **below 4 and unmeasured**. γ ∈ {1, 2, 3} has not been
swept per-domain.

At γ=16 everything but code is below 1.0x, worst 0.48x (Coder on
summarisation). Choosing depth badly is a 2x slowdown, not a missed gain.

What the controller is worth, even three-domain mix:

| policy | mean speedup |
|---|---|
| fixed γ=16 | 1.26x |
| fixed γ=8 | 1.65x |
| fixed γ=4 — best fixed | 1.69x |
| per-domain oracle | **1.81x** |

**Be honest about this number:** a perfect controller beats the best fixed
depth by ~**7%**. The defensible claim is not the 7% but what sits behind it —
"γ=4 is best" is only knowable from a full offline sweep across every domain
you will ever serve, and a runtime controller gets there without one and
survives the mix changing. Do not oversell it as a large speedup.

### Phase 1's addition to the hot loop

`generate()` now calls `scope.observe(arm, n_accepted)` once per speculative
iteration. Measured directly (CPU, steady state, 200k calls):

| | µs |
|---|---|
| `observe()`, warm arm | 2.179 |
| `new_scope()` + first `observe()` on an arm (pays the warm-start) | 4.572 |

That is **0.0092% of a 23.6 ms iteration** — three orders of magnitude below the
1.4% already spent on non-forward work, so it does not move the numbers above
and they were not re-measured. Re-check it if the loop ever gains real work:

```python
# CPU only, no weights needed.
import timeit
from specdec.stats import AcceptanceTracker, Arm
tracker = AcceptanceTracker(); arm = Arm("qwen2.5-0.5b", 4)
scope = tracker.new_scope(domain="code"); scope.observe(arm, 3)
n = 200_000
print(timeit.timeit(lambda: scope.observe(arm, 3), number=n) / n * 1e6, "us")
```

This matters because of the brief's stated risk — *switching overhead can
silently eat the gains it is supposed to produce*. Tracking is now measured and
ruled out; the switching *decision* Phase 2 adds is not yet.

---

## 6. Tests

152 total: 131 CPU, 21 GPU. **All 152 green** — CPU as of 2026-08-15 at
`00d6dd5`, GPU at **21/21** on GPU 6 (2026-08-14, 16:40) after the metric fix
in `66482df` (§4f). The GPU set has *not* been re-run since the controller
landed; it does not exercise the controller, but that is reasoning, not
evidence.

```bash
CUDA_VISIBLE_DEVICES="" pytest -m "not gpu" -q   # 131 CPU tests, ~4.5 min
CUDA_VISIBLE_DEVICES=<free> pytest -m gpu -q     # 21 GPU tests, 40–65 min (mostly loading)
pytest -m hub                                    # tokenizer compatibility across the pool
```

Log the GPU run to a file rather than trusting scrollback — inductor's DEBUG
output buries the summary, and a truncated log already cost one re-run here.

**Run the CPU set as `-m "not gpu"`, never as a bare `pytest`.** Bare `pytest`
collects the GPU tests; where no GPU is visible they skip harmlessly, which is
why the earlier "74 CPU tests" note read as safe. On this box they instead try
to load the 14B onto the default device, and against another user's job that is
sixteen minutes ending in `torch.OutOfMemoryError`.

The earlier "95 total, all green" claim in this file was **wrong on both
counts** — see §4e.

| file | what it pins down |
|---|---|
| `test_depth_controller.py` | the depth controller: dwell, margin, convergence, and the brief's "no oscillation on a mixed synthetic workload" bar |
| `test_acceptance_stats.py` | the estimator's cold start, decay and prior units; that the domain tag never reaches the controller; that tracking does not perturb generation |
| `test_shape_agreement_metric.py` | the denominator behind the self-drafting GPU test, on CPU — see §4f |
| `test_verify_kernel.py` | the accept/reject rule emits `p`, 60k samples, synthetic distributions |
| `test_engine_correctness.py` | cache indexing, rollback, vocab alignment, iteration chaining |
| `test_sampling.py` | the shared warper path |
| `test_static_cache.py` | static backend indistinguishable from dynamic, incl. poisoned rollback |
| `test_pool_compat.py` | every draft shares the tokenizer of *every* declared target, and `default_target` resolves (`-m hub`, needs tokenizer files) |
| `test_real_models.py` | the eager guarantees on real weights (`-m gpu`) |
| `test_compiled_correctness.py` | what survives graph capture, and what doesn't (`-m gpu`) |

Statistical tests deliberately use tiny randomly-initialised models: distribution
equality needs tens of thousands of samples, and a 14B gives you a few dozen.

Sharpest probes worth knowing about:
- **Self-drafting.** The target drafts for itself; under greedy decoding
  acceptance then *is* the width-1 vs width-`γ+1` agreement rate, isolated from
  draft quality.
- **Declared vs used decode shapes.** A width used but not captured silently
  runs eager and the speedup just fails to appear, so a test asserts the loop
  only uses declared widths.

`-m gpu` runs both GPU files in one session via session-scoped fixtures in
`conftest.py` — run them **together**, not separately, or you pay the load twice.
`test_compiled_correctness.py` compiles the models in place and uncompiles on
teardown; `test_real_models.py` asserts it received them uncompiled.

---

## 7. What to do next

> **CANCELLED — this whole section is out of scope as of 2026-08-16 (§0a).**
> It predates the scope cut and assumes the full adaptive project. **Do not
> execute any of it.** The GPU sweeps below exist to tune a controller that no
> longer ships; the Phase 2 and Phase 3 plans belong to cancelled phases.
> Kept as a record of what the project *would* have done, and because §5's
> numbers are referenced from elsewhere.
>
> **Nothing is required to finish the deliverable — the engine is done.**
> The only open items are documentation, all listed under "Explicitly not done"
> in §0a, and all deliberately untouched on the user's instruction.

### The next GPU session

Items 1–3 of the previous list are **done** (2026-08-14, GPU 6): the corrected
metric passes 21/21 on real weights, Phase 1 ran against the 14B, and the
per-domain spread is measured (§5). What remains:

1. **Extend the depth sweep past γ=8.** Throughput was still climbing at γ=8 on
   code (2.60x) while already falling on chat and summarisation, so the optimum
   is bracketed on two domains and unbounded on the third. The controller
   cannot be tuned against a range whose top is unknown — run
   `--gammas 4 8 12 16`.
2. **Consider whether Coder-1.5B stays in the pool.** It wins nothing anywhere
   and costs ~3 GB resident. Keeping it is defensible as evidence; dropping it
   simplifies the arm space Phase 2 searches. User's call.

Chaining two jobs in one shell is worth repeating: the sweep paid the cold load
(~13 min) and the `-m gpu` suite that followed took **16:40** instead of the
63 min it took cold, because the page cache was warm. Run the timing-sensitive
job first and the correctness job second.

Command shapes that worked, both logging to a file — the run is long and the
inductor DEBUG output buries the summary:

```bash
CUDA_VISIBLE_DEVICES=<free> python scripts/domain_sweep.py --gammas 4 8 --reps 3
CUDA_VISIBLE_DEVICES=<free> python -m pytest -m gpu -q -rf --tb=short
```

### Phase 1 — checkpoint owed

Built, committed, fully green, and exercised on the 14B. **Never checkpointed
with the user** — the brief requires one at the end of every phase, and Phase 2
began without it. That is a process slip to own rather than quietly skip. The
sweep in §5 is the substantive thing to raise, since it redirected Phase 2 away
from what the brief specified.

### Phase 2 — finish the controller. **Start here.**

All three `[DECISION]`s are settled (§3). The controller exists and its 19
tests pass. What remains, in order:

1. **Correctness: varying γ mid-request must not change the output.** Add it to
   `test_engine_correctness.py` on tiny models — greedy, controller with
   several depths, compare token-for-token against `generate_baseline`. The
   argument that it holds is sound but unverified, and this is exactly the
   failure mode the brief warns about (silently wrong output distribution).
   **Do this before anything else in Phase 2.**
2. **Run it end to end for real.** It has driven `generate()` exactly once, for
   8 tokens on a tiny model, in a test about the capture guard. With
   `dwell=16` an episode has never even completed inside a request, so
   `end_episode` has never been reached from the decode loop.
3. **Compare against fixed γ**, per domain, on real weights. This is the number
   that justifies the phase; §5 predicts ~7% over the best fixed depth on an
   even mix, and that prediction should be checked rather than quoted.
4. **Sweep γ ∈ {1,2,3}** per domain, or simply put them in the controller's
   candidate set and let it find the optimum — the latter is the better
   demonstration, and chat/summarisation optima are currently unbracketed
   below (§5).
5. **Tune the defaults** (`dwell=16`, `margin=0.05`, `explore_every=8`) against
   a real mixed workload. `explore_every` in particular trades throughput for
   adaptability and has never been measured.

Groundwork already in place:

- `CachePool` keeps one cache per model, so changing depth does not disturb
  the target's cache or its captured graphs. Changing *depth* is cheaper than
  changing draft would have been — no cache consequences at all.
- `generate()` **raises** if any candidate depth was not compiled, checked up
  front before the first token rather than partway through a request.
- All three drafts stay resident (~36 GB); the 0.5B is the one the controller
  uses, so the other two are now only needed for benchmarking.

One thing Phase 1 left open and Phase 2 has not closed:

- **Cold start is real but unquantified.** A request warm-starts from the
  cross-request prior, worth 2 observations. If domains differ sharply, a
  request's first iterations read a different regime's prior. Measure before
  adding machinery for it.

### Phase 3 — cross-domain benchmarking

HumanEval / ShareGPT / CNN-DailyMail, measured **per domain**. The adaptivity
story only lands if the fixed baseline underperforms somewhere. The single-process
constraint from §2 matters here: sweep all domains × drafts × depths in one run.

### Phase 4 — tree drafting

Stretch. Explicitly requires discussing the complexity/timeline tradeoff with
the user before starting.

### Smaller items worth picking up

- Extend the depth sweep past γ=8 to find the actual optimum on code (§5) —
  now the single most useful GPU measurement outstanding, because Phase 2 tunes
  this axis.
- The width-2 draft step (catch-up after a fully-accepted iteration) costs a
  little acceptance because a width-2 forward disagrees with a width-1 one. It
  is harmless — the accept/reject rule is exact for any `q` — but it is a small
  free win if it can be avoided cheaply.
- `scripts/demo.py --all-drafts --compile` has been reviewed but not executed
  end to end since the refactor.

---

## 8. Working agreement reminders

From `CLAUDE.md`, the ones that have actually bitten:

- **Ask before deciding on `[DECISION]` items** — present 2–3 options with
  tradeoffs and wait. Items marked `[DECISION]` are a floor, not a ceiling.
- **Checkpoint with the user at the end of every phase.** Done for Phase 0 and
  Phase 0.5. **Phase 1 is built but not yet checkpointed** — do that before
  starting Phase 2.
- **Commit every coherent working unit**, with messages explaining *why*.
- **Never commit code that doesn't run** or with failing tests unless flagged
  and agreed.
- **Surface uncertainty explicitly**, especially around verification
  correctness — a subtle bug there silently changes the output distribution,
  which defeats the whole point.
- **Keep Claude-only files out of git.** `CLAUDE.md` and this file are
  gitignored. If unsure whether a new file belongs in the repo, ask.

---

## 9. Commit history

```
99ed1c2 Give the fixed-draft engine its own results document
1f9d1ad Record that the adaptive layer was cut, not merely paused
cabe822 Stop the README promising a controller that is not finished
00d6dd5 Phase 2: adapt draft depth, and measure it rather than predict it
863f0cc Record the sweep: the drafts do not separate, but draft depth does
f0d3a86 Ask whether the drafts separate by domain, before building a controller for it
66482df Count the positions actually verified, not the ones merely proposed
94e5b17 Measure what the acceptance bookkeeping costs, and say so in the README
7fd1ff6 Phase 1: track acceptance so the tag measures but never controls
a35c413 Check the target the engine actually loads, not the first one declared
ca82059 Gitignore PROGRESS.md alongside CLAUDE.md
40c2c00 Land the numbers: 0.88x -> 2.25x, and account for where the rest goes
217b8cd Write up the batch-width finding, with the numbers that settled it
6b5278a Fix the demo's reference and stop it failing on expected numerics
6c9dbf1 Derive the correctness tolerance from a measurement, not from a guess
99fa9bc Export the cache and compile machinery, and add a draft-depth sweep
942b9e6 Hold the compiled path to a precise bar rather than a brittle one
c3104bb Test the property greedy exactness actually rests on, not just its symptom
cc77005 Compare like with like: canonical vocabulary in the benchmark and GPU tests
09e8bbe Cover the compiled path with its own correctness tests
a6ce931 Make the decode step capturable: static KV cache, fixed shapes, 14B target
b08e59d Measure the overhead risk: speculation is currently 0.88x, and why
8709c3c Add warmup to demo, plus an overhead profiler
40fca50 Resolve RNG generator/tensor device mismatch instead of requiring callers to match
e6332ce Phase 0: fixed-draft speculative decoding with verified exactness
```

The messages carry the reasoning; `git log` is worth reading before changing
anything in `cache_utils.py`, `compile_utils.py` or the test tolerances.
