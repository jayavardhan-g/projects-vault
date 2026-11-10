---
type: study-day
day: 4
date: 2026-11-11
tags: [interview, day, speculative-decoding]
---
# Day 4 (Wed Nov 11): speculative decoding

Part of [[Interview plan (Nov 2026)]]. Repo `speculative_decoding` (branch `main`). About 7 h. Prep pack:
[[Speculative decoding - prep]] (diagram, numbers sheet, 60 questions).

**Code walkthrough:** [[Day 04 - code walkthrough]]: one small figure per pipeline, the exact code to read in order, and what each piece does.

**Resume bullets:**
- "Accelerated LLM inference by **2.25×** using fixed-depth speculative decoding with Qwen2.5-0.5B drafting Qwen2.5-14B, reaching **2.60×** on code workloads while preserving the target distribution."
- "Performed draft-model and depth ablations across three draft models and a range of depths…"

**By tonight you can:**
- prove that speculative sampling preserves the target distribution;
- derive the expected tokens per step and the speedup formula;
- explain the 0.88× → 2.25× story with the code;
- answer "why not Medusa or EAGLE".

## Schedule

| # | Time | Kind | What |
|---|---|---|---|
| 0 | 15 min | recall | Days 1–3: online softmax, the 7.41× context story, the roofline |
| 1 | 30 min | read | The prep pack's pitch and diagram; RESULTS.md headline table |
| 2 | 75 min | theory + maths | The algorithm, the proof, the speedup formula (A1, A2) |
| 3 | 45 min | code | `sampling.py`, `verify_and_correct` (B1) |
| 4 | 75 min | rebuild | Speculative sampling in NumPy, with a Monte-Carlo proof (C) |
| — | lunch | | |
| 5 | 60 min | code | `SpeculativeDecoder.generate`, KV rollback (B2) |
| 6 | 30 min | theory | Why batch-1 decode is launch-bound; StaticCache, torch.compile, CUDA graphs (A3) |
| 7 | 40 min | code | `cache_utils.py`, `compile_utils.py` (B3) |
| 8 | 30 min | theory | Medusa, EAGLE and tree drafting: why not (A4) |
| 9 | 20 min | skim | `stats.py`, `controller.py`, `models.py`, `baseline.py` |
| 10 | 20 min | trace | The KV-rollback trace (D) |
| 11 | 40 min | practice | The questions (E), plus the prep pack's hostile follow-ups |

## A. Theory and maths

### A1. The algorithm (read: Leviathan et al. 2023, §2–3 and Appendix A; Chen et al. 2023, Algorithm 2)

**Optional extras:**
- Jay Mody's blog post *"Speculative Sampling"* (jaykmody.com): a short NumPy implementation, perfect before block 4;
- a 20-minute explainer video: search YouTube for *"Speculative Decoding: When Two LLMs are Faster than One"* (Efficient NLP channel).

One iteration, with draft distribution `q` and target distribution `p`:
1. The draft model generates γ tokens autoregressively: `x₁ … x_γ`, each from `q(·|prefix)`.
2. The target model scores **all γ+1 positions in one forward pass**, giving `p₁ … p_{γ+1}`.
3. For i = 1 … γ: accept `x_i` with probability `min(1, p_i(x_i) / q_i(x_i))`. On the first rejection, sample a
   replacement from the **residual** `norm(max(0, p_i − q_i))` and stop.
4. If all γ are accepted, sample one **bonus** token from `p_{γ+1}`.

So each iteration yields between 1 and γ+1 tokens, for one target forward pass.

### A2. The maths (75 min with A1; do it on paper)

**Proof that the output is distributed exactly as `p`** (one position):
- P(output = x) = P(draft x and accept) + P(reject) · P(residual gives x)
- = `q(x)·min(1, p(x)/q(x))` + `(1 − β) · max(0, p(x) − q(x)) / Σ_y max(0, p(y) − q(y))`
- where β = Σ_y min(p(y), q(y)) is the total acceptance probability.
- Key identity: `Σ_y max(0, p − q) = 1 − Σ_y min(p, q) = 1 − β`. So the second term is just `max(0, p(x) − q(x))`.
- Total: `min(p(x), q(x)) + max(0, p(x) − q(x)) = p(x)`. ∎

**Acceptance rate:** α = E[β] = 1 − TV(p, q), where TV is the total-variation distance. The closer the draft is to the target, the higher α.

**Expected tokens per iteration** (if each position is accepted independently with probability α):
`E[tokens] = 1 + α + α² + … + α^γ = (1 − α^{γ+1}) / (1 − α)`.

**Speedup** with cost ratio `c = (draft forward time) / (target forward time)`:
`speedup = E[tokens] / (γ·c + 1)`. You pay γ draft passes plus one target pass per iteration.

**Worked with your numbers** (per-layer times from `RESULTS.md`, around lines 88–104; take α = 0.8, γ = 4):
- E = (1 − 0.8⁵)/0.2 = **3.36** tokens per iteration.
- **Eager:** draft 24 layers × 0.82 ms = 19.7 ms; target 48 × 0.81 = 39 ms; c ≈ 0.5. Speedup = 3.36 / (4·0.5 + 1) = **1.12×**.
  With your first 7B target (28 layers) and 1.5B draft (28 layers), c ≈ 1, so speedup = 3.36 / 5 ≈ **0.67×**. That is your 0.88× story.
- **Compiled:** draft 24 × 0.098 = 2.35 ms; target 48 × 0.223 = 10.7 ms; c ≈ 0.22. Speedup = 3.36 / 1.88 ≈ **1.79×**.
- You measured **2.25×**, above the formula (1.89× with your real α). Acceptance is **bursty**, not independent: easy
  stretches (code) accept long runs (RESULTS.md).
- **Best γ:** maximise `E(γ)/(γc + 1)`. A higher α (code) favours a larger γ (8 → 2.60×). A lower α (chat) favours a
  small γ; at γ = 16, chat runs at 0.82×.

### A3. Why decode was launch-bound (block 6, 30 min)

**Read:** the PyTorch blog *"Accelerating Generative AI with PyTorch II: GPT, Fast"* (gpt-fast). It covers the static KV cache,
`torch.compile(mode="reduce-overhead")` and speculative decoding, the same three ideas as your fix.

**Concepts:**
- **Eager mode** launches each small kernel from Python. At batch 1, the GPU finishes each kernel faster than the CPU can launch the next, so time ≈ layers × per-layer launch cost, whatever the model width (0.81 ms per layer for 0.5B *and* 14B).
- **A StaticCache** gives fixed tensor shapes, so a compiled graph can be reused.
- **`reduce-overhead`** captures **CUDA graphs**: the whole kernel sequence is replayed with one launch.
- Because of this, the target moved to 14B, with **48 layers vs the draft's 24**: speculation pays only when `c` is small.

### A4. Why not Medusa, EAGLE or tree drafting (block 8, 30 min)

Read the abstract and first figure of each:
- **Medusa** (Cai et al., 2024): extra decoding heads on the target; needs training.
- **EAGLE** (Li et al., 2024): a small draft head on the target's features; needs training.
- **Tree drafting / SpecInfer** (Miao et al., 2024): verify several candidate branches at once with a tree attention mask.

**Your honest answer:** these need training, or bigger engine changes for tree masks. A separate off-the-shelf draft
from the same family (shared tokenizer) needs neither, and it isolates the systems question. Tree drafting is the
listed next step (PROGRESS §7).

## B. Code

### B1. Sampling and verification (block 3, 45 min, deep)

- `sampling.py`:
  - 73 `logits_to_probs` (temperature, top-p);
  - **42 `align_vocab`**: Qwen 0.5B and 14B have different vocab sizes. What does it pad, and why must `p` and `q` share an index space?
  - 113 `sample_from_probs`, 131 `uniform_scalar`.
- `speculative.py` **84 `verify_and_correct`**: map it line by line to A1.
  - The greedy special case (around line 120): `p` is a point mass, so accept iff the draft equals the argmax.
  - `acceptance_prob = min(1, p/q)` (around line 129).
  - The residual is `clamp(p − q, min=0)` (around line 133), and its fallback when the residual sums to 0.

### B2. The decode loop (block 5, 60 min, deep)

`speculative.py`:
- 147 `SpeculativeDecoder`, 156 `__init__`;
- 181 `decode_shapes`: why the shapes are fixed;
- 204 `compile`;
- **246 `generate`**: the draft loop, the single verify pass over γ+1 positions, how many tokens are emitted, and the bonus token.

Tests to read alongside: `tests/test_engine_correctness.py` and `tests/test_sampling.py`. They show how "preserves the
distribution" is *tested* despite bf16 noise (the noise-floor method; see the prep pack).

### B3. Cache and compile (block 7, 40 min, deep)

- `cache_utils.py`:
  - 34 `KVCache`, 70 `CachePool`;
  - **124 `crop_cache`**, 150 `_rewind_static`: rolling back rejected draft positions;
  - 170 `forward_with_cache`.
- `compile_utils.py`: 45 `compile_decode`, 110 `warmup_decode`. Why warm-up matters for honest timing.

## C. Rebuild it yourself (block 4, 75 min)

In NumPy, about 60 lines:
1. Make a random target distribution `p` and draft `q` over a vocab of 50 (for example, softmaxes of correlated logits).
2. Implement `verify(p, q, x)` exactly as in A1, with the residual.
3. **Monte-Carlo proof:** run 10⁶ single-token rounds (draft x ~ q, then verify). Show the histogram of outputs matches `p`
   (TV distance < 0.005). Then break it: use `max(p, q)` instead of the residual, and watch the TV grow.
4. Simulate full iterations with γ = 1…16 and a fixed per-position α. Plot the empirical tokens per iteration against
   `(1 − α^{γ+1})/(1 − α)`.
5. Plug in c = 0.22 and c = 0.5 and plot speedup against γ. Find the best γ for α = 0.6 and for α = 0.85.

## D. Trace (block 10, 20 min)

γ = 4, prefix length 100. Tokens 1–2 are accepted, token 3 is rejected and resampled. Write down:
- the length of the target KV cache and of the draft KV cache right after the verify pass;
- what `crop_cache` cuts each one to;
- which tokens still need their KV computed next iteration.

## E. Questions

1. Prove that speculative sampling doesn't change the output distribution.
2. Derive the expected accepted tokens. Why did you measure more than the formula predicts?
3. Your first version was 0.88×, slower than no speculation. Why, and how did you fix it?
4. Why a 14B target and not 7B? Why the 0.5B draft and not 1.5B?
5. Why does the speedup depend on the domain (code 2.25–2.60×, chat 1.42×)?
6. How do you choose γ? What happens at γ = 16 for chat?
7. Greedy outputs aren't bit-identical to the baseline. Is your implementation wrong? (*No: bf16 batch-width noise changes logits by about 0.25. Tests measure the noise floor.*)
8. Why did you cut the adaptive controller? (*Drafts didn't separate by domain, so there was nothing to switch between. Depth does matter, so a depth controller is the follow-up.*)
9. Draft and target vocab sizes differ. How do you handle it?
10. Why not Medusa or EAGLE?
11. How does this interact with batching? (*At large batch, decode becomes compute-bound and the spare compute that speculation uses disappears.*)
12. Hostile: "2.25× on one prompt set, at batch 1. Is that a real-world number?"

## My notes and fumbles

- 
