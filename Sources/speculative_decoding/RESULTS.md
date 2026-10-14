# Fixed-Draft Speculative Decoding — Results

Measured results for the engine this repo ships: **one draft model, one fixed
draft depth**, verified to preserve the target model's output distribution.

This document covers only the fixed-draft engine. The adaptive/multi-draft
exploration that preceded it, and the measurements that led to it being cut,
are in [`README.md`](README.md).

---

## Summary

| | |
|---|---|
| Target | Qwen2.5-14B-Instruct (48 layers) |
| Draft | Qwen2.5-0.5B-Instruct (24 layers) |
| Draft depth | `γ=4` (default), `γ=8` for code-heavy workloads |
| Speedup vs target-only | **2.25x** (`γ=4`) / **2.56x** (`γ=8`) on the benchmark prompt |
| Range across domains | **1.4x – 2.6x**, depending on domain and depth |
| Output distribution | Unchanged vs running the target alone — verified, not assumed |
| Overhead outside forward passes | 1.4% of an iteration |

**The honest headline:** speculative decoding is worth 2.25x on the benchmark
prompt, but the gain is strongly domain-dependent — 2.25x on code, 1.42x on
chat, 1.41x on summarisation at the same depth. A single number overstates it.

---

## Setup

All numbers: 1× H200, bf16, batch size 1, 128 new tokens, greedy, best of 3
interleaved repetitions after a discarded warm-up.

**Both sides are compiled.** This matters more than it sounds — see
[Why the first working version was slower](#why-the-first-working-version-was-slower).
Compilation alone takes the target-only baseline from 43.5 to 86.2 tok/s, so a
compiled speculative run measured against an *eager* baseline reports the
compiler's win, not speculation's. Every speedup below is against the
**compiled** baseline at 86.2 tok/s.

```bash
# Headline numbers and the overhead breakdown
CUDA_VISIBLE_DEVICES=<free gpu> python scripts/profile_overhead.py \
    --gamma 4 --sweep-gammas 2 8 --reps 3

# Per-domain and per-depth
CUDA_VISIBLE_DEVICES=<free gpu> python scripts/domain_sweep.py \
    --gammas 4 8 12 16 --reps 3
```

---

## Headline results

Target-only baseline: **43.5 tok/s** eager, **86.2 tok/s** compiled.

| | baseline | speculative (0.5B draft) |
|---|---|---|
| eager | 43.5 tok/s | 1.33x |
| compiled | 86.2 tok/s | **2.25x** (≈194 tok/s implied) |

At `γ=8` the same configuration reaches **2.56x** (≈221 tok/s implied) on the
benchmark prompt.

### Effect of draft depth

Deeper drafting amortises one target verification over more tokens, but
compounds the chance that an early rejection throws the rest away.

| depth | speedup | acceptance rate |
|---|---|---|
| `γ=2` | 1.84x | 0.913 |
| `γ=4` | 2.25x | 0.798 |
| `γ=8` | **2.56x** | 0.681 |

Acceptance falls monotonically with depth, as it must. Throughput does not —
it depends on the domain, which is why `γ` is a deployment choice rather than a
constant to maximise. See [Choosing the depth](#choosing-the-depth).

---

## Why the first working version was slower

The first correct implementation ran at **0.88x — slower than not speculating
at all**. The algorithm was fine: acceptance was 0.84–0.92 and target forward
passes dropped 4.3x. The cost model around it was wrong.

**Diagnosis: batch-1 decode is CPU-launch-bound, not compute-bound.** Per-layer
forward cost, measured eager:

| model | eager | compiled |
|---|---|---|
| Qwen2.5-14B (48 layers) | 0.813 ms | 0.223 ms |
| Qwen2.5-1.5B (28 layers) | 0.808 ms | 0.098 ms |
| Qwen2.5-0.5B (24 layers) | 0.820 ms | 0.098 ms |

Eager cost is **flat across a 28x span in parameter count**. Latency was
`n_layers × a constant` and ignored model *width* entirely — the GPU sat idle
between kernel launches and the wall clock measured Python dispatch. Speculative
decoding needs `draft_cost << target_cost`; with a 7B target (28 layers) and a
1.5B draft (28 layers), a draft call cost the *same* as a target call.

**What fixed it:**

- **`StaticCache` behind a `KVCache` handle**, with length tracked host-side.
  Reading the length back from the backend costs a device sync per step — the
  exact overhead being removed.
- **Rollback became a counter reset with no data movement.** Rejected K/V is
  left in place; sound because the next forward overwrites it and causal masking
  hides the rest. Verified by poisoning the cache with values ±100 outside the
  activation range, rolling back, and confirming bit-identical results.
- **Prefill split out of the first verification**, so every verification is
  exactly `γ+1` wide and every draft step is width 1 (or 2, catching up after a
  fully-accepted iteration). Fixed shapes are what make graph capture possible.
- **`torch.compile(mode="reduce-overhead")`** over those fixed shapes.

Result: **0.88x → 2.25x**, and per-layer latency tracks model size again.

---

## Per-domain results

Three prompts each across code, chat and summarisation, 0.5B draft, 14B target,
greedy, 128 new tokens, median of 3 reps.

**Speedup:**

| domain | `γ=4` | `γ=8` | `γ=12` | `γ=16` |
|---|---|---|---|---|
| code | 2.25x | **2.60x** | 2.44x | 2.31x |
| chat | **1.42x** | 1.25x | 1.03x | 0.82x |
| summarisation | **1.41x** | 1.10x | 0.94x | 0.66x |

**Acceptance rate:**

| domain | `γ=4` | `γ=8` |
|---|---|---|
| code | 0.734 | 0.614 |
| chat | 0.401 | 0.249 |
| summarisation | 0.338 | 0.190 |

Code accepts roughly twice as often as chat or summarisation, which is what
drives the throughput gap. Code is more predictable token-to-token — boilerplate,
closing brackets, repeated identifiers — so a small draft guesses it well.

### Choosing the depth

The optimum moves sharply by domain, and **getting it wrong costs more than it
gains**. At `γ=16`, chat runs at 0.82x and summarisation at 0.66x — meaningfully
*slower* than not speculating.

| policy | mean speedup across the three domains |
|---|---|
| `γ=16` | 1.26x |
| `γ=8` | 1.65x |
| **`γ=4`** | **1.69x** |

**`γ=4` is the shipped default** — the best single depth across a mixed
workload. If you know your workload is code-heavy, `γ=8` is worth 2.60x. If it
is chat or summarisation, do not go above 4.

---

## Where the time goes

One `γ=4` iteration, compiled:

| | ms |
|---|---|
| 4 draft forwards + 1 verification | 23.251 |
| `crop_cache`, 14B target (48 layers) | 0.169 |
| `crop_cache`, 1.5B draft (28 layers) | 0.099 |
| `verify_and_correct` (incl. host syncs) | 0.036 |
| `logits_to_probs` (5 × 151936 vocab) | 0.029 |

Everything that is not a forward pass totals **0.33 ms of 23.6 ms — about
1.4%**. The accept/reject kernel and the softmax over a 152k vocabulary are
nowhere near the critical path. Rollback costs one counter write per layer.

### The analytic cost model under-predicts

| acceptance | predicted | measured |
|---|---|---|
| 0.798 | 1.89x | **2.25x** |

The standard prediction `(1 - a^(γ+1)) / (1 - a)` assumes each proposal is
accepted independently. Real acceptance is **bursty**: the 0.5B emitted 4.13
tokens per iteration where independence predicts 3.35. Runs of easy tokens get
accepted together. Measure acceptance; do not infer throughput from the formula.

---

## Correctness

Speculative decoding must not change the output distribution — it only changes
speed. The accept/reject rule is exact for any draft distribution `q` and any
depth, and this is tested rather than assumed.

**What is verified:**

- The accept/reject kernel emits samples from `p`, checked against synthetic
  distributions over 60k samples.
- Cache indexing, rollback, vocabulary alignment and iteration chaining.
- The static cache is indistinguishable from the dynamic one, including under a
  poisoned-rollback probe.
- **Self-drafting**: the target drafts for itself, so under greedy decoding
  acceptance *is* the width-1 vs width-`γ+1` agreement rate — isolating the
  verification logic from draft quality entirely.
- Compiled and eager paths each get their own correctness tests.

**152 tests: 131 CPU, 21 GPU, all green.**

```bash
CUDA_VISIBLE_DEVICES="" pytest -m "not gpu" -q   # 131 CPU tests
CUDA_VISIBLE_DEVICES=<free>  pytest -m gpu -q    # 21 GPU tests
```

### Why the tests do not assert bit-equality

On the 14B, **running the same model at two batch widths changes its logits.**
Row 0 of a width-2, width-5 or width-9 forward differs from a width-1 forward at
the same position by exactly 0.25 — same query, same keys, only the GEMM batch
dimension differs, which changes tiling and accumulation order. On the 0.5B the
difference is exactly 0.0.

This is not caused by speculative decoding; it is a property of running one model
at two batch widths, which every batching server does. The cleanest evidence:
**the target-only baseline disagrees with itself** — three identical runs, no
speculation anywhere, and two emitted a different token at position 118 than the
third.

Because bf16 margins are quantised (smallest observed nonzero top-1/top-2 gap:
0.125; 1 position in 41 was an exact tie), two ulps is enough to flip an argmax
and redirect the rest of the sequence.

So the tests **measure the noise floor for whichever target is loaded** and
accept a divergence only at a position whose top-1/top-2 margin falls inside it.
A divergence at a decisive margin still fails — which is exactly where a
misindexed verification or a stale cache would land, since those are wrong by
whole units. The compiled path measures its own floor (0.281 vs eager 0.250),
since inductor's shape-specialised kernels stack on top of eager's reordering.

This is deliberately *not* a loosened token-error-rate check, which would stop
catching real bugs.

---

## Caveats

- **Batch size 1 only.** The launch-bound analysis above is specific to it;
  batched serving changes the arithmetic substantially.
- **Three prompts per domain**, 128 new tokens, greedy. Enough to establish the
  domain gap, not a full benchmark-suite evaluation.
- **One target model.** The draft/target depth ratio (48 vs 24 layers) is what
  makes this pay; a shallower target would not.
- **Implied tok/s figures** (194, 221) are derived from the measured baseline and
  the measured ratio, not separately timed.
- Speedups are sensitive to co-tenancy on a shared GPU — batch-1 decode is
  CPU-bound, so a co-running job skews exactly these numbers.
