---
type: lesson
tags: [lesson, evaluation]
---
# Evaluation honesty patterns

These are practices that recurred across projects and made the results trustworthy. Reuse them.

- **Set a kill criterion before the run.** LCRAG's head signal had to beat dense R@5 0.770. It didn't (0.672), so
  the "replacement" framing was dropped and the complement framing was found. → [[LongContext RAG - retrieval-head-guided compression]]
- **Use placebo controls at the same weight.** LCRAG fused length, random and BM25 signals in place of the head signal. All three hurt, which rules out
  "fusion just helps". A same-shortlist dense-only control isolated the head signal's contribution.
- **Beat the trivial baseline, not just random.** BountyBrain's planner had to beat a global-frequency prior. Its first
  version lost (0.280 vs 0.325). → [[BountyBrain - bug bounty knowledge graph]]
- **Hold out evidence when the gold label comes from the same graph.** The planner excluded the held-out templates from its evidence pool.
- **Look for confounds in who sees what vocabulary.** BountyBrain's keyword baseline saw curated aliases that the
  LLM arm never got. A "negative result" turned out to be *unmeasured*.
- **Keep negative results in the README.** These were all kept: the LLM classifier vs the alias table, real vLLM being 6.2× faster, paging doing worse
  on uniform prompts, spec-dec being slower than baseline at γ=16 on chat, and the head signal failing alone.
- **Derive tolerances from a measurement.** Spec-dec measures the bf16 batch-width noise floor and fails only on
  divergences at a decisive margin. A looser token-error rate would hide real bugs.
- **Break the code to test the tests.** mini-vLLM checked its kernel tests by deliberately breaking the kernel.
- **Cut scope when the data says so.** Spec-dec's drafts didn't separate by domain, so the adaptive controller was cut.
