# Project report: Speculative decoding (Qwen2.5-0.5B drafting for Qwen2.5-14B)

> **For the LLM reading this.** This report describes a project built by G Jayavardhan, an M.Tech CSE student at IIT Bombay.
> He has to defend it in interviews as its sole author. Use it to teach him, quiz him, or answer questions. It was written from
> the repository: code, README, RESULTS.md, configs and git log.
>
> Labels:
> - **[CODE]**: what the code does, with `file:line`;
> - **[MEASURED]**: a number from the repo's docs or results;
> - **[DOC]**: a reason stated in the repo;
> - **[RECON]**: reasoning or arithmetic added by the report writer.
>
> When something is not covered here, say so rather than invent it.

---

## 1. One-paragraph summary

The project is a from-scratch **speculative decoding** engine on top of HuggingFace transformers models. A small
**draft** model (Qwen2.5-0.5B-Instruct) proposes γ tokens autoregressively. The large **target** model
(Qwen2.5-14B-Instruct) checks all of them in **one** forward pass. An accept/reject rule (Leviathan et al. 2023 /
Chen et al. 2023) keeps the output distribution **exactly** that of the target alone.

The first correct version was **slower** than the baseline (0.88×). At batch size 1, every forward pass was bound by
Python dispatch and kernel launches, so the 0.5B draft cost as much per layer as the 14B target. Three changes fixed it:
- `StaticCache` (fixed K/V addresses), with rollback as a counter reset;
- fixed input shapes (prefill split out);
- `torch.compile(mode="reduce-overhead")`, which captures CUDA graphs.

Final result: **2.25× over a compiled target-only baseline at γ=4**, and **2.60× on code prompts at γ=8**.

The original goal was to switch adaptively between several draft models. A domain sweep showed the drafts do *not*
separate by domain: the same draft wins everywhere. The optimal *depth* does change by domain, so the adaptive layer
was cut and the repo ships **fixed-draft, fixed-depth** speculative decoding.

- **Type:** self project, Summer 2026. 23 commits by Jayavardhan, 2026-08-13 → 2026-08-16.
- **Repo:** `/data/jayavardhan/projects/speculative_decoding`, branch `main` (latest `99ed1c2`). Package `src/specdec/`.
- **Hardware:** 1× NVIDIA H200, bf16, batch size 1, on a shared server.
- **Environment:** mamba env `speculative_decoding`, Python 3.11, PyTorch with `torch.compile`, HF transformers (5.x cache API).
- **Tests:** 152 (131 CPU, 21 GPU), all green.

---

## 2. Resume bullets vs the repo

| Resume bullet | Verdict | Evidence |
|---|---|---|
| "Accelerated LLM inference by 2.25× using fixed-depth speculative decoding with Qwen2.5-0.5B drafting Qwen2.5-14B" | **Supported.** | README/RESULTS "Results". Compiled baseline 86.2 tok/s, compiled speculative (0.5B, γ=4) 194.1 tok/s = **2.25×**. Setup: H200, bf16, batch 1, 128 new tokens, greedy, **best of 3** interleaved reps after a discarded warm-up, on **one prompt**: "Write a Python function that checks whether a string is a palindrome." (`scripts/profile_overhead.py:37`). Both sides are compiled. |
| "…reaching 2.60× on code workloads" | **Supported.** | Domain sweep, 0.5B draft, γ=8, median over 3 code prompts × 3 reps = 2.60×. |
| "…while preserving the target distribution" | **Supported, with nuance.** | The accept/reject rule is exact for any q (proof in §3.3). It is tested statistically at 60k samples on toy models. With **greedy** decoding on the 14B, outputs can differ from target-only at bf16 near-ties, because the target at batch width γ+1 vs width 1 shifts logits by 0.25. That is a property of batch width, not of speculation (§7.4). |
| "Performed draft model ablations and depth ablations across three draft models and a range of depths" | **Supported.** | Drafts: Qwen2.5-1.5B, Qwen2.5-0.5B, Qwen2.5-Coder-1.5B. Depths γ = 2, 4, 8 on the benchmark prompt, and γ = 4, 8, 12, 16 in the domain sweep. |
| "…identifying an optimal draft model and depth outperforming other draft models" | **Supported.** | The 0.5B wins throughput in every domain at every depth tried. γ=4 is the best *fixed* depth on a mixed workload (mean 1.69×). γ=8 is best for code. |

**Wording to know:**
- 2.25× is against a **compiled** baseline. Against the eager baseline the same run would look about 4.5×, which would be mostly the compiler's credit.
- 2.25× comes from one code prompt. On chat and summarisation the same setup gives only about 1.4×.

---

## 3. Background (teach these first)

### 3.1 Why decoding is slow
Autoregressive generation produces one token per forward pass. At batch size 1, each pass has to read all the
model's weights from memory to do very little arithmetic. Decode is therefore memory-bandwidth bound in theory.
**In this project it was bound even lower down, by Python and kernel-launch overhead** (§6.1). The GPU sits idle
while the CPU launches hundreds of small kernels.

### 3.2 The idea of speculative decoding
A cheap draft model guesses γ next tokens, one at a time. The expensive target then runs **one** forward pass over
the last committed token plus all γ guesses. That single pass gives the target's distribution at every guessed
position, plus one more (the "bonus" position). The target accepts a prefix of the guesses and emits one more token of
its own. So each target pass emits between 1 and γ+1 tokens. If guesses are usually right, the target runs far fewer times.

### 3.3 The accept/reject rule and why it is exact
At a position, let p be the target distribution and q the draft distribution. Draw x ~ q and accept it with
probability min(1, p(x)/q(x)). On rejection, sample from the residual (p − q)₊ normalised. Then:

```
P(emit x) = q(x)·min(1, p(x)/q(x)) + P(reject)·(p(x) − q(x))₊ / Z
          = min(p(x), q(x)) + (p(x) − q(x))₊        [Z = Σ(p−q)₊ = 1 − Σ min(p,q) = P(reject)]
          = p(x)
```

This holds for **any** q. A bad draft costs speed, never correctness. Across positions it chains, because position
i+1 is only reached if i was accepted, and the target's p at i+1 is conditioned on exactly the accepted prefix.

**Greedy special case:** p is a point mass at argmax, so the rule becomes "accept iff the draft's token equals the
target's argmax; otherwise emit the target's argmax" (`speculative.py:119-127`).

### 3.4 Expected tokens per iteration
If each proposal is accepted independently with probability α, the expected tokens per target pass is
**(1 − α^(γ+1)) / (1 − α)**. Deeper γ amortises one verification over more tokens. But a rejection at position i
throws away everything after it, so acceptance per drafted token falls with depth. The speedup also depends on the
**cost ratio c = draft cost / target cost**, roughly (tokens per iteration) / (γ·c + 1).

### 3.5 CUDA graphs and `torch.compile(mode="reduce-overhead")`
A CUDA graph records a sequence of kernel launches once and replays them with a single launch. This removes per-kernel
CPU launch cost. It needs **static memory addresses** (the same tensors every replay) and **static shapes** (one
graph per input shape). `torch.compile(..., mode="reduce-overhead")` fuses kernels with Inductor and wraps the step in
CUDA graphs.

### 3.6 KV-cache rollback
After a rejection, the K/V computed for rejected proposals must not affect later attention. With a growable
`DynamicCache` you crop the tensors. With a preallocated `StaticCache` you just move the write offset back. Stale
entries above it are overwritten by the next forward, and the causal mask hides anything beyond.

### 3.7 Alternatives you should know (not implemented here)
- **Medusa:** extra decoding heads on the target predict several future tokens, with tree attention to verify candidates.
- **EAGLE:** a light draft head on the target's features (feature-level autoregression).
- **Tree-based speculation (SpecInfer):** verify a tree of candidates in one pass.
- **Lookahead / prompt-lookup decoding:** no draft model; n-grams come from Jacobi iterations or from the prompt.

These need training (Medusa, EAGLE) or more complex attention masks. The project used the classic separate-draft
method because it needs no training and its exactness is easy to prove and test [RECON].

---

## 4. Goals and how they changed [DOC]

1. **Original brief: adaptive draft selection.** Keep several drafts resident and switch between them at runtime from
   live acceptance statistics. The bet was that a code-specialised draft wins on code and loses elsewhere.
2. **Phase 0:** a correct fixed-draft engine. It came out at 0.88×, slower than the baseline.
3. **Phase 0.5:** make it fast (static cache, fixed shapes, compile). Result 2.25×.
4. **Phase 1:** acceptance tracking (`stats.py`). The domain tag is held out from the controller.
5. **Domain sweep:** no separation by draft, large separation by domain on *depth*.
6. **Phase 2:** a depth controller (`controller.py`) was written and unit-tested, but **cut**. It never ran on real
   weights, and nothing verifies that changing γ mid-request preserves the distribution. A perfect per-domain depth
   oracle beats the best fixed depth by only about 7%.

**Final deliverable:** fixed-draft speculative decoding, 0.5B draft, γ=4 by default (γ=8 for code-heavy workloads).

---

## 5. Architecture and code

```
prompt ids
   │
   ├─ prefill target and draft over prompt[:-1]   (eager; variable length; runs once)
   ▼
loop until max_new_tokens or EOS:                         (speculative.py:337-418)
   1. DRAFT:  γ forwards of the draft, width 1 (width 2 after a fully accepted round)
              q_i = logits_to_probs(align_vocab(logits)) ; x_i ~ q_i
   2. VERIFY: ONE target forward over [last committed token, x_1..x_γ]  (width γ+1)
              p rows for the γ proposals + 1 bonus row
   3. ACCEPT: verify_and_correct(proposals, q, p) → (n_accepted, next_token)
   4. COMMIT: append proposals[:n_accepted] + next_token
              crop_cache(target, seq_len+n_accepted); crop_cache(draft, same)
```

**Files** (`src/specdec/`):
- `sampling.py` (140 lines): the single logits → probs path, vocabulary alignment, sampling.
- `cache_utils.py` (226): `KVCache` wrapper, `CachePool`, `crop_cache`, `forward_with_cache`.
- `compile_utils.py` (139): `compile_decode` captures fixed-shape decode steps.
- `baseline.py` (98): target-only decoding, the correctness reference.
- `speculative.py` (439): `verify_and_correct` and `SpeculativeDecoder`.
- `stats.py` (442): acceptance tracking with separate control and evaluation views.
- `controller.py` (265): the cut depth controller.
- `models.py` (118): pool registry and loading from `configs/models.yaml`.

**Scripts:**
- `scripts/profile_overhead.py`: headline numbers, per-layer costs, the iteration breakdown.
- `scripts/domain_sweep.py`: drafts × domains × depths.
- `scripts/demo.py`
- `scripts/download_models.py`

### 5.1 The accept/reject kernel: `verify_and_correct` (`speculative.py:84-144`)
- Inputs:
  - `draft_tokens (k,)`;
  - `q_probs (k, V)`, the draft distribution at each drafted position;
  - `p_probs (k+1, V)`, the target distribution at each drafted position plus the bonus row.
- For i in 0..k−1:
  - **greedy:** if `x == p_i.argmax()` continue, else return `(i, argmax)` (`:119-127`);
  - **sampled:** `acceptance_prob = min(1, p_i[x]/q_i[x])`; if `U < acceptance_prob` continue (`:129-131`).
    Otherwise `residual = clamp(p_i − q_i, min=0)`, normalise, sample, and return `(i, token)` (`:133-140`).
    If the residual sums to 0 (possible only through float noise), fall back to p.
- All accepted: return `(k, sample(p_probs[k]))`, the free bonus token (`:142-144`).
- It always emits **n_accepted + 1** tokens.
- It is a pure function of distributions, so it is tested without models on synthetic p and q at 60k samples (`tests/test_verify_kernel.py`).

### 5.2 `SpeculativeDecoder.generate` (`speculative.py:245-439`)
- **Canonical vocabulary:** `self.vocab_size = min(target.vocab_size, draft.vocab_size)` (`:178`).
  - The 14B and 7B pad their embeddings to 152,064. The drafts pad to 151,936. The real tokenizer has about 151,665 tokens.
  - The surplus rows still produce finite logits, so they carry nonzero probability.
  - **Narrowing is forced:** if the target sampled an id ≥ 151,936, it would be fed to the draft next round and the
    draft's embedding would raise `IndexError`. This bug actually happened the first time the engine ran [DOC].
  - `align_vocab` truncates wider logits, or pads narrower ones with −inf (`sampling.py:42-70`).
  - The baseline applies the same projection, so the comparison stays exact.
- **Prefill over `prompt[:-1]`** for both models (`:314-326`). Holding back the last prompt token means:
  - the target always enters verification one position behind, so it is always fed exactly γ+1 tokens (last committed + γ proposals);
  - the draft always enters with one token (or two after a fully accepted round, because the draft never ran a forward over its own last proposal).
  - These fixed widths are what make CUDA-graph capture possible (`decode_shapes`, `:180-202`).
- **Draft phase** (`:340-356`): γ draft forwards. Each samples `x ~ q` and stores the q row.
- **Verify phase** (`:358-371`): `feed = cur_ids[t_len:] + proposals`, one target forward. Rows `lo .. lo+γ`
  (with `lo = seq_len − 1 − t_len`) become `p_probs`.
- **Commit and rollback** (`:388-396`): `keep = seq_len + n_accepted`, then `crop_cache` on both caches.
  - After this the target's cache is at `seq_len + n_accepted`, which is one short of the new sequence length
    (the emitted correction or bonus token has no target K/V yet). That keeps the next verification width at exactly γ+1.
- EOS check on emitted tokens (`:400-404`).
- `SpecStats` (`:55-81`):
  - `acceptance_rate = n_accepted / n_proposed`, where `n_proposed += γ` every iteration;
  - `mean_accepted_per_iteration = (n_accepted + n_iterations) / n_iterations`, i.e. tokens per target pass.

### 5.3 One sampling path: `logits_to_probs` (`sampling.py:73-106`)
Every path goes through this one function, so p in the test is bit-for-bit what the baseline would sample from:
- the baseline;
- draft proposals;
- verification.

Behaviour:
- greedy (temperature 0) gives a one-hot at the argmax;
- otherwise it applies temperature, then top-k, then top-p (HF's order), then softmax in float32;
- top-p uses an exclusive cumulative sum and always keeps the top token.

Why it matters: a missing warper or a different temperature in one path would silently break exactness [DOC].

### 5.4 Caches (`cache_utils.py`)
- `KVCache` dataclass: `backend` (HF `DynamicCache` or `StaticCache`), `length` tracked **host-side**. Reading the length from a `StaticCache` would cost a device-to-host sync every step [DOC].
- `CachePool`: one cache per (model, role). It is reused across sequences, because a CUDA graph is bound to the K/V addresses it was captured with. Keyed by role too, so a model can draft for itself in tests.
- `crop_cache(cache, length)`:
  - **dynamic:** `backend.crop(length − cache.length)`;
  - **static:** `_rewind_static` fills each layer's `cumulative_length` counter **in place** (`counter.fill_(length)`) and moves no data. Rebinding the counter would invalidate captured graphs. It runs under `inference_mode` because the counter is an inference tensor.
- Why the static rollback is sound [DOC]: the next forward overwrites stale positions before attention reads them, and positions beyond that are excluded by the causal mask. `test_static_cache.py` checks this by **poisoning** the stale region with values ±100 outside the activation range and requiring bit-identical results.
- `forward_with_cache(model, cache, tokens, start_pos)`:
  - asserts `cache.length == start_pos`, so drift between cache and sequence raises;
  - builds `cache_position`;
  - picks the compiled forward if this width was captured;
  - clones the logits, because a CUDA-graph replay returns a view into graph-owned memory that the next replay overwrites.

### 5.5 Compilation (`compile_utils.py`)
- `compile_decode(model, shapes)` wraps `model.forward` with `torch.compile(mode="reduce-overhead", dynamic=False)` for the declared widths (`:45-92`). Prefill stays eager because its length varies.
- `SpeculativeDecoder.compile()` captures:
  - target widths {γ+1} plus {1} for the baseline (`include_baseline=True`);
  - draft widths {1, 2}.
- `generate()` **raises** if the target was compiled for different widths than the γ requested (`speculative.py:280-293`). Otherwise an uncaptured width silently runs eager and is several times slower.
- Compilation is lazy, so benchmarks discard a warm-up rep.

### 5.6 Acceptance statistics (`stats.py`)
This module was built for the controller and is now kept as the measurement layer.
- Keyed by `Arm(draft, gamma)`. Depth is in the key because accepted-per-iteration is bounded by γ, and pooling different depths estimates nothing in particular [DOC].
- **The tracked quantity is accepted per iteration**, not per-token Bernoulli acceptance [DOC].
- Two exponential moving averages:
  - fast, per request (`RequestScope`, α=0.15, about 7 iterations);
  - slow, per arm across requests (`ControlView`, α=0.02), which acts as a warm-start prior.
- The EMAs carry explicit accumulated weight, so a cold start is unbiased. Prior strength is measured in "observations".
- **The domain tag measures but never controls.** `AcceptanceTracker.evaluation` holds domain-keyed counters, and `.control` has no method that takes a domain. `test_control_is_blind_to_domain` feeds two byte-identical streams that differ only in their tags and asserts all controller-visible values match.
- Cost: `observe()` takes 2.18 µs per iteration, about 0.009% of an iteration [MEASURED].

### 5.7 The cut depth controller (`controller.py`)
- It chooses γ with the draft fixed. It measures instead of predicting, because the analytic model mispredicted.
- It works in **episodes**: about 16 iterations, with one device sync per episode for wall-clock timing.
- It is not delivered. It is unverified on real weights, and nothing tests that a γ change mid-request preserves the distribution [DOC].

---

## 6. Why the first version was slower, and the fix

### 6.1 The diagnosis [MEASURED]
Forward cost at batch 1, divided by layers:

| model | layers | eager ms | eager ms/layer | compiled ms | compiled ms/layer |
|---|---|---|---|---|---|
| Qwen2.5-14B | 48 | 39.02 | **0.813** | 10.72 | 0.223 |
| Qwen2.5-1.5B | 28 | 22.61 | **0.808** | 2.74 | 0.098 |
| Qwen2.5-0.5B | 24 | 19.68 | **0.820** | 1.69 | 0.070 |
| Qwen2.5-Coder-1.5B | 28 | 22.53 | **0.805** | 2.74 | 0.098 |

- The eager cost per layer is **flat across a 28× range of parameter counts**. Latency was layers × constant. The GPU was idle between launches, so wall time measured Python dispatch.
- With the original 7B target (28 layers) and a 1.5B draft (28 layers), a draft call cost the same as a target call. Speculation cannot win when draft cost ≈ target cost.
- In Phase 0, acceptance was 0.84–0.92 and target forwards dropped 4.3×, but throughput was still 0.88×.
- [RECON, checked] RESULTS.md lists the 0.5B's compiled per-layer cost as 0.098 ms. The README's own numbers give 1.69 ms / 24 layers = **0.070 ms**, so RESULTS.md has a copy error. Quote 0.070.

### 6.2 The fix [DOC]
1. **`StaticCache`**: fixed K/V addresses for graph replay. Length is tracked host-side, and rollback is a counter reset.
2. **Fixed shapes**: prefill is split out of the first verification, so verify is always γ+1 wide and draft steps are 1 or 2 wide.
3. **`torch.compile(mode="reduce-overhead")`** over those shapes.
4. **The target moved from 7B to 14B.** At batch 1 the cost ratio is set by **depth (layers)**, not parameter count. 14B has 48 layers vs the 0.5B's 24. The 7B had 28, the same as the 1.5B drafts.

Result: **0.88× → 2.25×**. Compiling alone doubles the baseline (43.5 → 86.2 tok/s), and speculation doubles it again (about 4.5× over the original eager baseline).

---

## 7. Results

### 7.1 Headline (benchmark prompt, γ=4, best of 3 after warm-up) [MEASURED]
| | baseline | 1.5B draft | 0.5B draft | Coder-1.5B draft |
|---|---|---|---|---|
| eager | 43.5 tok/s | 58.5 (1.35×) | 58.0 (1.33×) | 54.6 (1.26×) |
| **compiled** | **86.2 tok/s** | 174.8 (2.03×) | **194.1 (2.25×)** | 164.4 (1.91×) |

### 7.2 Depth ablation (benchmark prompt, compiled; speedup and acceptance_rate) [MEASURED]
| draft | γ=2 | γ=4 | γ=8 |
|---|---|---|---|
| Qwen2.5-1.5B | 1.68× (0.955) | 2.03× (0.911) | 2.29× (0.824) |
| **Qwen2.5-0.5B** | 1.84× (0.913) | **2.25× (0.798)** | **2.56× (0.681)** |
| Coder-1.5B | 1.61× (0.902) | 1.91× (0.825) | 2.06× (0.770) |

`acceptance_rate` falls with depth while throughput rises on this (code) prompt.

### 7.3 Domain sweep (`scripts/domain_sweep.py`; 3 prompts per domain, 14B target, greedy, 128 tokens, median of 3 reps after warm-up) [MEASURED]
**Acceptance / speedup at γ=4:**

| domain | 1.5B | 0.5B | Coder-1.5B |
|---|---|---|---|
| code | 0.805 / 1.89× | 0.734 / **2.23×** | 0.804 / 1.89× |
| chat | 0.463 / 1.31× | 0.401 / **1.47×** | 0.391 / 1.17× |
| summarisation | 0.411 / 1.23× | 0.338 / **1.41×** | 0.355 / 1.18× |

- **No separation by draft:**
  - the 1.5B has the highest acceptance in every domain;
  - the 0.5B has the highest throughput in every domain (lower acceptance, but cheaper, and cost wins);
  - the ordering is identical at γ=8.
- **Specialisation bought nothing:** Coder-1.5B scores 0.804 on code vs 0.805 for the general 1.5B.
  - [DOC] The explanation: acceptance measures how close q is to p, and p belongs to a *general instruct* target.
  - A domain-specialised draft moves away from the target's distribution about as much as it gains on the domain.
  - So **what earns acceptance is distributional proximity to the target, not task competence.**

**Depth sweep, 0.5B:**

| domain | γ=4 | γ=8 | γ=12 | γ=16 |
|---|---|---|---|---|
| code | 2.25× | **2.60×** | 2.44× | 2.31× |
| chat | **1.42×** | 1.25× | 1.03× | 0.82× |
| summarisation | **1.41×** | 1.10× | 0.94× | 0.66× |

- Code peaks at γ=8. For chat and summarisation the optimum is **below 4 and was not measured**.
- A wrong depth is a slowdown: γ=16 gives 0.66× on summarisation, and the Coder draft reached 0.48×.

**Policies on an even three-domain mix:**

| policy | mean speedup |
|---|---|
| fixed γ=16 | 1.26× |
| fixed γ=8 | 1.65× |
| **fixed γ=4 (best fixed)** | **1.69×** |
| per-domain oracle | 1.81× (+7%) |

Why the controller was cut: a perfect oracle gains 7% over the best fixed depth.

### 7.4 Overhead outside forward passes (one compiled γ=4 iteration) [MEASURED]
| | ms |
|---|---|
| 4 draft forwards + 1 verification | 23.251 |
| `crop_cache`, 14B (48 layers) | 0.169 |
| `crop_cache`, 1.5B draft (28 layers) | 0.099 |
| `verify_and_correct` (incl. host syncs) | 0.036 |
| `logits_to_probs` (5 × 151,936) | 0.029 |

Non-forward work totals **0.33 ms of 23.6 ms (≈1.4%)**.

[RECON] The table lists the 1.5B draft's `crop_cache`, and 4×2.74 + about 10.7 ms ≈ 23 ms matches 1.5B drafts. So this breakdown was measured with the **1.5B** draft, not the 0.5B.

### 7.5 The analytic model "under-predicts", and a correction [MEASURED + RECON]
- The README predicts speedup using `(1 − a^(γ+1))/(1 − a)` with `a = acceptance_rate`. For the 0.5B at γ=4 this predicts 3.35 tokens/iteration (1.89×), against **2.25× measured**. The README attributes the gap to "bursty" acceptance (runs of easy tokens accepted together).
- **[RECON, checked arithmetic] That explanation is probably wrong.** In the code, `acceptance_rate = n_accepted / n_proposed` with `n_proposed += γ` per iteration (`speculative.py:72-74, 377`). That is accepted ÷ *drafted*, **not** the per-token acceptance probability α the formula needs.
  - Under independence, expected accepted per iteration = α(1−α^γ)/(1−α), so `acceptance_rate` = that ÷ γ. That is smaller than α, because positions after a rejection are drafted but never verified.
  - Solving for α from the measured rates gives α ≈ **0.941 / 0.912 / 0.914** for γ = 2 / 4 / 8.
  - With α = 0.912, γ = 4, the formula predicts (1 − 0.912⁵)/(1 − 0.912) ≈ **4.19 tokens/iteration**. The measured value is 0.798 × 4 + 1 = 4.19.
  - So the data is **consistent with independent acceptance**. The "under-prediction" came from plugging the wrong quantity into the formula. (The README quotes 4.13 measured; that is likely another run.)
- Use this in interviews: it shows you understand the metric. Don't repeat the "bursty" story.

### 7.6 Exactness in bf16 [MEASURED]
- On the 7B target, greedy speculative output equalled greedy target-only output in 27 of 27 comparisons.
- On the 14B it does not always match. The baseline reads p from a **width-1** forward. The engine reads p from row j of a **width-(γ+1)** forward. Only the GEMM batch dimension differs, which changes tiling and accumulation order.

  | target | logit shift from batch width alone |
  |---|---|
  | 0.5B | 0.0 (bit-identical) |
  | 14B | 0.25 (about 2 bf16 ulps) |

  - bf16 logits sit on a coarse grid: the smallest nonzero top-1/top-2 margin observed was 0.125, and 1 position in 41 was an exact tie.
  - So 0.25 can flip an argmax, and one flip changes every token after it.
- **Proof that this isn't the engine:** the **target-only baseline disagrees with itself.** Across 3 identical runs, two emitted a different token at position 118 than the third.
- **How the tests handle it:** they don't use bit equality, and they don't use a loose "mostly agrees". They **measure the noise floor** for the loaded target (eager 0.250, compiled 0.281). A divergence is allowed only at a position whose top-1/top-2 margin is inside that floor. Compiled results: γ=1, 0 of 9 runs diverged; γ=4, 3 of 9 diverged, all at margin 0.25. Divergence at a decisive margin still fails, which is where a misindexed verification or stale cache would show up.
- **Self-drafting probe:** the target drafts for itself. Under greedy, acceptance then equals exactly the width-1 vs width-(γ+1) agreement rate, which isolates numerics from draft quality.
- The **sampled** path is unaffected: the rule is exact for any q, and p is whatever the target computed.

### 7.7 Warm-up matters [MEASURED]
- A cold first rep reported 0.49× and 7.64× for two drafts that both settle near 2.0×.
- An earlier benchmark without warm-up reported 2.15×, which was entirely warm-up.
- An eager Coder-1.5B cell once read 14.13 ms/forward against about 22.5 ms normally. That was GPU clock state. Its compiled number matched its twin exactly.

---

## 8. Hardest problems (real, from the repo)

1. **Correct but slower than the baseline (0.88×).**
   - Diagnosed with per-layer timing: flat ms/layer means launch-bound.
   - Fixed with static cache, fixed shapes, `reduce-overhead` compile, and a deeper target.
   - Lesson: speculation needs draft_cost ≪ target_cost, and at batch 1 cost tracks *depth*.
2. **Vocabulary width mismatch crashed the draft** (`IndexError` when the target sampled a padding id). Fixed with the narrowest canonical vocabulary, applied in the baseline too.
3. **Greedy outputs diverging on the 14B.**
   - Traced to batch-width numerics (0.25 logit shift), not to the engine.
   - The tolerance is derived from the measured noise floor, not guessed.
   - Backed by the baseline disagreeing with itself and by the self-drafting probe.
4. **Making rollback compatible with CUDA graphs.**
   - The static cache counter must be mutated in place (`fill_`) under `inference_mode`, or captured graphs break.
   - Verified by poisoning the stale region.
5. **The research pivot.** The sweep showed the premise (drafts separate by domain) was false. The controller was cut instead of shipping an unverified one.

---

## 9. Design decisions

| Decision | Alternatives | Why | Source |
|---|---|---|---|
| Separate small draft model (Leviathan/Chen) | Medusa, EAGLE, lookahead, prompt lookup | no training; simple exactness proof and tests | [RECON] |
| Qwen2.5 family | Llama | size ladder + Coder sibling on the *same* tokenizer (required: p and q compared over one id space; `test_pool_compat.py` asserts byte-identical tokenizers) | [DOC] `configs/models.yaml` |
| 14B target (48 layers) | 7B (28 layers) | cost ratio at batch 1 is set by depth | [DOC] |
| 0.5B draft | 1.5B, Coder-1.5B | best throughput in every domain | [MEASURED] |
| γ=4 default | γ=8 | best fixed depth over the mixed workload (1.69× mean); γ=8 only for code | [MEASURED] |
| Narrowest canonical vocab | widest | target ids are fed back to the draft | [DOC] |
| StaticCache + counter rollback | DynamicCache + crop | graph capture needs fixed addresses; rollback becomes free | [DOC] |
| Host-side length tracking | read from backend | avoids a device sync per step | [DOC] |
| Prefill `prompt[:-1]` | prefill the full prompt | makes every verify exactly γ+1 wide | [DOC] |
| Compile both arms | compile only speculation | otherwise the speedup is the compiler's | [DOC] |
| Best-of-3 interleaved + discarded warm-up | single run | lazy compile, shared-GPU noise | [DOC] |
| Accepted-per-iteration as the signal | per-token rate | it *is* the tokens-per-pass ceiling | [DOC] |
| Domain tag for evaluation only | tag-driven controller | a tag is a proxy; a tag-driven controller is a lookup table | [DOC] |
| Cut the controller | ship it | +7% at best, unverified distribution safety | [DOC] |
| Batch size 1 | batched | Phase 0 simplicity | [DOC] |

---

## 10. Limitations

- **Batch size 1 only.** With batching, decode becomes more compute-bound and the economics of speculation change, usually for the worse at high batch.
- **Small domain sweep:** three prompts per domain, 128 tokens, greedy, one target. HumanEval/ShareGPT/CNN-DM was planned (Phase 3) and not run.
- The **headline is one prompt** (code). Code accepts about 2× more often than chat or summarisation.
- The depth optimum for chat and summarisation (below 4) was not measured. γ=1–3 was never swept there.
- Implied tok/s values (194, 221) are derived from ratios, not separately timed.
- Speedups are sensitive to co-tenants on the shared GPU (batch-1 decode is CPU-bound).
- The controller is unfinished, and changing depth mid-request is not distribution-verified.

---

## 11. Likely interview questions

1. *Prove speculative decoding preserves the distribution.* → §3.3.
2. *What happens at γ=4 when tokens 1–2 are accepted and token 3 is rejected?*
   - `verify_and_correct` returns `(2, residual_sample)`, which emits 3 tokens.
   - `keep = seq_len + 2`, and both caches are cropped to `seq_len+2`.
   - K/V for proposals 3–4 (and the draft's) are discarded by moving the counter.
   - The target has K/V for positions up to the last accepted proposal. The emitted correction token has none yet; it is fed first in the next verification.
3. *Why was it slower than the baseline at first?* → §6.1, flat ms/layer.
4. *Why compile the baseline too?* → otherwise the measured speedup is the compiler's.
5. *Why the 0.5B and not the 1.5B, which accepts more?* → cost wins: 1.69 ms vs 2.74 ms per forward.
6. *Why did the code-specialised draft not win on code?* → proximity to p, not competence.
7. *Why does acceptance fall with γ?* → each position is reached only if all earlier ones were accepted. Also, here `acceptance_rate` counts all drafted positions, including those after a rejection.
8. *Your formula under-predicted. Why?* → §7.5: `acceptance_rate` ≠ α. Solve for α and it fits.
9. *Why don't greedy outputs match exactly on the 14B?* → batch-width numerics, §7.6.
10. *How is rollback free?* → static cache counter. Stale K/V is overwritten or masked.
11. *What breaks with batching?* [RECON] Different sequences accept different numbers of tokens, so caches go ragged. Verification costs more at high batch because the target is no longer idle. vLLM handles this with per-sequence lookahead slots and paged KV.
12. *Why cut the adaptive controller?* → 7% upside, unverified exactness, nothing to choose between drafts.
13. *What would you do next?* [RECON] Measure γ < 4 for chat; try EAGLE/Medusa-style heads; tree verification; batched speculation; real benchmarks (HumanEval, ShareGPT).
14. *What is the cost of the accept/reject step?* → 0.036 ms, about 0.15% of an iteration.
15. *How do you test a probabilistic algorithm?* → toy models at 60k samples (statistical power), plus greedy token-equality for indexing bugs, plus self-drafting.

---

## 12. Numbers cheat sheet

| Number | Meaning |
|---|---|
| 0.88× | first correct version (eager, 7B target) |
| 43.5 → 86.2 tok/s | target-only baseline, eager → compiled |
| **2.25×** | 0.5B → 14B, γ=4, compiled, palindrome code prompt (194.1 tok/s) |
| 2.56× | same prompt, γ=8 |
| **2.60×** | code domain (3 prompts), γ=8 |
| 1.42× / 1.41× | chat / summarisation at γ=4 |
| 1.69× vs 1.81× | best fixed γ (4) vs per-domain oracle, mixed workload |
| 0.813 / 0.808 / 0.820 ms | eager ms/layer for 14B / 1.5B / 0.5B (flat = launch-bound) |
| 0.223 / 0.098 / 0.070 ms | compiled ms/layer |
| 48 vs 24 layers | 14B target vs 0.5B draft |
| 0.33 ms (1.4%) | non-forward overhead per iteration |
| 0.25 | 14B logit shift from batch width alone (2 bf16 ulps) |
| 0.804 vs 0.805 | Coder-1.5B vs 1.5B acceptance on code |
| α ≈ 0.912 | per-token acceptance implied by acceptance_rate 0.798 at γ=4 [RECON] |
| 152 tests | 131 CPU + 21 GPU |
