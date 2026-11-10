---
type: index
tags: [interview, code]
---
# Code walkthroughs: the code to read each day

One note per study day. Each has **one small figure per pipeline** (the functions involved and how they hand off to
each other), then the **exact code excerpts** in reading order, then what each piece does and how it connects to the
next. Most sections end with a "check yourself" question with the answer in brackets.

The excerpts are copied from the repos at the commits named in each note, so line numbers match the source. Read each
note **with the files open**: the note says what to look for, and the file shows the surrounding code.

| Day | Walkthrough | Repo · ref |
|---|---|---|
| 1 | [[Day 01 - code walkthrough]]: allocator, block table, KV cache, one decoder layer | `vLLMmini` · `main` |
| 2 | [[Day 02 - code walkthrough]]: schedule, admission, preemption, radix cache | `vLLMmini` · `main` |
| 3 | [[Day 03 - code walkthrough]]: Triton decode kernel, paired benchmark, server loop | `vLLMmini` · `main` |
| 4 | [[Day 04 - code walkthrough]]: draft/verify loop, accept/reject, rollback, CUDA graphs | `speculative_decoding` · `main` |
| 5 | [[Day 05 - code walkthrough]]: naive → coalesced → tiled → register-blocked matmul | `sysml` |
| 6 | [[Day 06 - code walkthrough]]: stream splits, checkpoint arena, FlashAttention | `sysml` |
| 7 | [[Day 07 - code walkthrough]]: speech pipelines; illustrative VQ/RVQ/FSQ (no repo code) | — |
| 8 | [[Day 08 - code walkthrough]]: pretokenize, LoRA injection, the LM loss, train loop, WER | `cosyvoice_rnd` · `main` |
| 9 | [[Day 09 - code walkthrough]]: retrieval-head scoring (worked example), pruning pipeline | `LongContext_RAG` · `main` |
| 10 | [[Day 10 - code walkthrough]]: hybrid parser, SQL prefilter, CLIP + FAISS | `photo_seek` · `upgrade-2026-10` |
| 11 | [[Day 11 - code walkthrough]]: margin sampling, constrained refine; steering + guard | `photo_seek` · `upgrade-2026-10`; `emotion_steering` |
| 12 | [[Day 12 - code walkthrough]]: weakness lookup, BM25 + dense + RRF + rerank, grounding | `bountybrain` · `upgrade-2026-10` |
| 13 | [[Day 13 - code walkthrough]]: LRU, pool, cache race; SPSC ring, matching, WAL | `decs`; `microexchange` · `upgrade-2026-10` |
| 14 | [[Day 14 - code walkthrough]]: voting, commit rule, AppendEntries, durable writes | `raftkv` · `upgrade-2026-10` / `2a19135` |

Day 15 (revision) uses the first figure of each note as the "redraw from memory" check.

**Corrections found while writing these** (also fixed in the day notes):
- CosyVoice: 1.36% is the adapter's share of the **whole** CosyVoice 2 (LLM + flow + vocoder); against the 0.5B Qwen it's ~1.8%. The LLM's loss never reads the speaker embedding. The reported WER is the **mean of per-clip WERs**.
- Emotion steering: the guard accepts a NaN perplexity ratio.
