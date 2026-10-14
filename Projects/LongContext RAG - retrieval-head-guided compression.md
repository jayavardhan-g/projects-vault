---
type: project
folder: projects/LongContext_RAG
repo: github.com/jayavardhan-g/LongContext_RAG
dates: 2026-08-10 → 2026-08-13
status: done
kind: research — retrieval, interpretability
stack: [Llama-3.1-8B-Instruct, bge-m3, bge-reranker-v2-m3, HotpotQA, PyTorch]
tags: [project, llm, retrieval, interpretability]
---
# LongContext RAG — retrieval-head-guided context compression

**What.** At the same context budget, can the generator's own **retrieval heads** pick context better than a
dense retriever? Answer: **not as a replacement, but yes as a complement.** This was #4 on the
[[Project strategy and resume plan]] shortlist.

**Built, in three stages.**
1. **Retrieval-head identification** on Llama-3.1-8B. Reimplements Wu et al. with needle-in-a-haystack over Paul Graham essays.
   **25 of 1024 heads** score above 0.1, and the top head is L15H30 at 0.865 copy-precision. Verified causally:
   masking 5 identified heads costs 29 points of needle recall, while masking 5 random heads costs 0.
2. **Benchmark.** HotpotQA padded to 4k/16k/32k with distractor paragraphs, sentence-level retrieval units,
   matched token budgets through one shared packer. Baselines: naive long-context, truncation, BM25, dense, hybrid, oracle.
3. **Head-guided compression.** Head attention is read at one decode step (SDPA prefill, then eager for the
   single query token), then fused with dense scores and run as a **dense-shortlist → head-rescore cascade**.

**Key results** (n=300, 32k context, 256-token budget):
- Dense retrieval beats stuffing the window: naive F1 is 0.480 and dense is 0.605, at **15× lower latency**. The oracle reaches 0.747.
- The head signal alone **fails** its preset kill criterion: R@5 0.672 vs dense 0.770. Retrieval heads give a
  *copying* signal, and multi-hop bridge facts are not copied.
- The head signal is nearly uncorrelated with dense on gold sentences (ρ=0.07). Fused, it reaches **R@5 0.844**.
  Placebos (length, random, BM25) at the same weight all make dense worse.
- **Cascade F1 0.661 vs dense 0.607 (+0.054, 95% CI [+0.028, +0.085], p<0.001).** The same-shortlist dense-only
  control gives +0.053. The cascade's head pass costs 105 ms instead of 1497 ms.

**Caveats.** Heads were identified on essays, not QA. Only 32k was tuned. Fusion is *not* significantly better
than the cross-encoder (p=0.14).

**Lessons** → [[Measurement traps]] (block-order warm-up, normaliser artefact), [[Infrastructure traps]] (NFS mmap model loading).

**State.** Done. 2 local commits are not pushed. The env `lcrag` was path-based on warm storage and no longer exists, so recreate it by name ([[Environments]]).
The local model mirrors in `/data/jayavardhan/lcrag_models/` are **gone** as of 2026-10-14, so re-copy them before any rerun.

**Source.** `README.md` (full write-up), `PROGRESS.md` (45 KB session log), `CHECKPOINTS.md`, `results/*/*.md`.
