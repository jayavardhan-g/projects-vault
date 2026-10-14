---
type: open-questions
tags: [meta]
---
# Open threads

Work that was started or planned and left unfinished, roughly from cheapest to most expensive.

**Housekeeping (minutes)**
- ~~Push bountybrain and LongContext_RAG~~ — done 2026-10-14. → [[Repositories and git state]]
- Pick one of the duplicate Raft repos. Scrub the Postgres password from the DECS repo.

**Finish existing projects**
- **BountyBrain:** rebuild the graph, `rm data/corpus/cwe_candidates.npz`, then re-run the classification eval. That gives the first fair
  keyword-vs-LLM comparison. After that: shortlist recall, a faithfulness eval and the first end-to-end UI run. → [[BountyBrain - bug bounty knowledge graph]]
- **CS794 PA1:** the KV-cache and radix benchmark is implemented but was never run. → [[CUDA kernels - CS794 sysml]]
- **Photo-Seek:** every model-backed component (CLIP, FAISS, ArcFace, clustering, active learning). → [[Photo-Seek]]
- **mini-vLLM:** chunked and fused prefill. It closes the >16k OOM, the preemption budget hole and most of the 6.6× TTFT gap to vLLM.
- **Speculative decoding:** tree drafting, or finishing and verifying the depth controller.
- **LongContext RAG:** test generalisation beyond 32k and HotpotQA, and ablate the head-selection method.
- **raftkv:** dynamic resharding (the original goal), log compaction, pre-vote.

**Recommended but never started** (from [[Project strategy and resume plan]])
- Agentic GPU-kernel optimisation (rated top flagship).
- Real-time streaming inference for speech-LLMs.
- Long-horizon agent memory and context compaction.
- Interpretability of speech-LLM representations. This is also the closest fit to the thesis.
