---
type: lesson
tags: [lesson, benchmarking]
---
# Measurement traps (performance numbers)

These traps produced wrong numbers in more than one project. Check them before believing any speedup.

**Warm-up contaminates whichever arm runs first.**
- Spec-dec: a discarded first rep read 0.49× and 7.64× for two drafts that both settle near 2.0×. An earlier "2.15×" was
  entirely warm-up. → [[Speculative decoding - fixed-draft engine]]
- LongContext RAG: in grouped blocks the first block read 196 ms vs 77 ms for an identical prompt. One run made S=2048 look
  faster than S=1024, which is impossible. → [[LongContext RAG - retrieval-head-guided compression]]
- mini-vLLM: the first HTTP request paid 17 s of kernel compilation for everyone else. → [[mini-vLLM - paged-attention inference server]]

**Fix: interleave the arms, discard warm-up and report the paired median ratio.** This is a shared GPU and throughput
drifts about 1.5× with other users' jobs. The ratio stays stable while the absolute numbers do not.

**Compare like with like.**
- Compiled vs eager: compiling alone doubles the spec-dec baseline (43.5 → 86.2 tok/s). A compiled speculative run against an eager
  baseline measures the compiler.
- Spec-dec: a baseline that ran with a different `vocab_size` (152064 vs 151936) proves nothing. This bit three files.
- mini-vLLM: the baseline was forced to fixed output lengths while the server ran varied ones (the old 4.4× became 3.82×). The HF arm
  also timed requests from when *its batch* started rather than t=0, which hid the queueing time. That bug made the baseline look better
  (TTFT 24.6 ms → 2448 ms once fixed).

**Check whether the cost you are cutting matters at this size.**
- Batch-1 eager decode is launch-bound (~0.81 ms/layer for every model size), so a small draft is not cheaper (spec-dec).
- Prefill under ~2k tokens is overhead-bound, so a 91% prefix-cache hit saves 0% of the time (mini-vLLM).
- Small-N "wins" against PyTorch measure launch overhead (CS794 flash-attention, N ≤ 128).

**Use the right denominator.** Spec-dec `acceptance_rate` counted proposals after the first rejection, so 0.976
read as 0.932. The LCRAG `|g_h|/|k|` score has a ceiling of 0.371 on long needles, and this made a head look like it "failed on rare names".
An impossible value such as BountyBrain's nDCG@10 = 1.113 is a gift, because it exposes the bug.
