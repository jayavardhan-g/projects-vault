---
type: project
folder: projects/bountybrain
repo: github.com/jayavardhan-g/bountybrain
dates: 2026-08-16 → 2026-08-18
status: done — one evaluation needs a re-run
kind: AI application — knowledge graph, RAG, evaluation
stack: [SQLite, Qdrant, bge-small-en-v1.5, bge-reranker-base, Qwen2.5-14B via vLLM, FastAPI, Streamlit]
tags: [project, retrieval, rag, knowledge-graph]
---
# BountyBrain — knowledge graph and retrieval for learning bug bounty hunting

**What.** It answers "explain this vulnerability class" with cited prose, "how does this weakness relate to others" with a typed
graph, and "I ran recon, what should I test next" with a ranked plan. It **never scans**: it only parses recon output the user already has.
All 8 sources are bulk, openly licensed downloads (CWE, CAPEC, nuclei-templates, Exploit-DB,
pentester.land metadata, PayloadsAllTheThings, OWASP WSTG, Bugcrowd VRT).

**Key results** (`RESULTS.md`, 2026-08-16):
- Graph: **111,896 nodes / 96,805 edges**. The core without isolated CVE and exploit pairs is 35k nodes at mean degree 3.6.
  1,389 aliases map **93.1%** of writeup labels onto CWE (up from 76.5%).
- Retrieval, 200 held-out queries: hybrid plus cross-encoder rerank reaches **R@5 0.965, MRR 0.861**. Fusion alone adds little.
  The reranker is what pays (+0.125 R@1) at about 3× the latency.
- Recon planner, 400 held-out cases: **P@1 0.427 vs prior 0.325**. The first version *lost* to the prior because its
  applicability term saturated. Smoothing P(weakness|product) fixed it.
- **CWE classification: the keyword alias table beat the RAG-LLM** (0.398 vs 0.368 on nuclei). **This comparison is confounded**:
  the curated aliases were never persisted, so only the keyword arm saw them. The fix has landed but the eval has not been re-run.

**Defects the evals caught** (10, listed in RESULTS §5): nDCG 1.113, per-character CVE parsing, merged hosts, and CAPEC links that are only stated as `CanFollow`.

**Not done.** The re-run in PROGRESS step 0. You must `rm data/corpus/cwe_candidates.npz`, or the fix silently does nothing.
There is no faithfulness/citation evaluation, and the Streamlit UI has never been run end-to-end.

**Lessons** → [[Evaluation honesty patterns]].

**State.** All commits are pushed (2026-10-14). The env `bountybrain` (Python 3.11.15) was deleted and has no registry record. SETUP.md lists four vLLM traps.

**Docs in this vault** (copied 2026-10-14 from `/data/jayavardhan/projects/bountybrain/`):
- [[Sources/bountybrain/README|README]]
- [[Sources/bountybrain/RESULTS|RESULTS]]
- [[Sources/bountybrain/PROGRESS|PROGRESS]]
- [[Sources/bountybrain/SETUP|SETUP]]
- [[Sources/bountybrain/ATTRIBUTION|ATTRIBUTION]]
- [[Sources/bountybrain/examples/README|examples/README]]
- [[Sources/bountybrain/examples/sample_plan|examples/sample_plan]]
- [[Sources/bountybrain/prompts/classify_weakness|prompts/classify_weakness]]
- [[Sources/bountybrain/prompts/generate_query|prompts/generate_query]]
- [[Sources/bountybrain/prompts/tutorial|prompts/tutorial]]

**Source.** `README.md`, `RESULTS.md`, `PROGRESS.md` (hand-over and priorities), `SETUP.md`, `ATTRIBUTION.md`.
