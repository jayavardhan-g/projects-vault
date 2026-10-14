---
type: project
folder: projects/speculative_decoding
repo: github.com/jayavardhan-g/speculative_decoding
dates: 2026-08-13 → 2026-08-16
status: done — adaptive layer cut
kind: ML systems — inference
stack: [Qwen2.5-14B-Instruct, Qwen2.5-0.5B/1.5B, torch.compile, StaticCache, H200]
tags: [project, ml-systems, inference]
---
# Speculative decoding — fixed-draft engine with verified exactness

**What.** It started as *adaptive* speculative decoding, switching draft models at runtime from live acceptance stats,
and ended as a **fixed-draft engine** (one draft, γ=4) whose output distribution is verified to match the target's.

**Key results** (H200, batch 1, both sides compiled):
- **2.25×** at γ=4 and **2.56×** at γ=8 over the compiled 14B baseline (86.2 tok/s).
- Strongly domain-dependent: at γ=4, **code 2.25×, chat 1.42×, summarisation 1.41×**. At γ=16, chat and
  summarisation run *slower* than no speculation (0.82× and 0.66×). γ=4 is the best single depth (mean 1.69×).
- The analytic formula under-predicts (1.89× predicted vs 2.25× measured) because acceptance is bursty.
- Non-forward overhead is 1.4% of an iteration.

**What changed the design.**
- The first correct version ran at **0.88×**. Eager batch-1 decode is *launch-bound*: ~0.81 ms/layer whatever the
  model size, so a 0.5B draft cost as much as the target. The fix was a StaticCache, fixed shapes and `torch.compile(reduce-overhead)`, and
  the target moved to 14B (48 layers vs the draft's 24).
- **Drafts don't separate by domain**, but the best depth does. So draft switching had nothing to choose between, and the
  adaptive controller was **cut** (`controller.py` exists, is unverified and not shipped).
- **bf16 batch width changes the 14B's logits by 0.25** (two ulps), so greedy runs can't be bit-equal. Tests measure
  this noise floor and fail any divergence at a decisive margin. The baseline even disagrees with itself across runs.

**Tests.** 152 (131 CPU and 21 GPU), all green.

**Lessons** → [[Measurement traps]] (warm-up, round-robin, vocab-size mismatch, wrong denominator).

**State.** Done. The env `speculative_decoding` was deleted and has no registry record. Options left open in PROGRESS §7: tree drafting, a cross-domain benchmark.

**Source.** `RESULTS.md` (fixed-draft numbers), `README.md` (full story incl. the adaptive exploration), `PROGRESS.md` §4 (findings not to re-derive).
