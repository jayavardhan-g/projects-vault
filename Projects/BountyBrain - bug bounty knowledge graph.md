---
type: project
folder: projects/bountybrain
repo: github.com/jayavardhan-g/bountybrain
dates: 2026-08-16 → 2026-08-18; upgraded 2026-10-22
status: done — upgrade on branch upgrade-2026-10 (pushed)
kind: AI application — knowledge graph, RAG, evaluation
stack: [SQLite, Qdrant, bge-small-en-v1.5, bge-reranker-base, Qwen2.5-14B via vLLM, FastAPI, Streamlit]
tags: [project, retrieval, rag, knowledge-graph]
---
# BountyBrain — knowledge graph and retrieval for learning bug bounty hunting

**What.** It answers "where do I start?" with a graph-ordered learning path, "explain this vulnerability class" with cited prose, "how does this weakness relate to others" with a typed
graph, and "I ran recon, what should I test next" with a ranked plan. It **never scans**: it only parses recon output the user already has.
All 8 sources are bulk, openly licensed downloads (CWE, CAPEC, nuclei-templates, Exploit-DB,
pentester.land metadata, PayloadsAllTheThings, OWASP WSTG, Bugcrowd VRT).

**Key results** (`RESULTS.md`; graph/retrieval/planner 2026-08-16, reproduced and extended 2026-10-22):
- Graph: **111,896 nodes / 96,805 edges**; ontology core **35,278 nodes / 63,340 edges**, mean degree 3.59 (now in
  `graph_metrics.json` → `core_structure`). 1,389 aliases map **93.1%** of writeup labels onto CWE.
- Retrieval, 200 held-out queries: hybrid + cross-encoder rerank **R@5 0.965, MRR 0.861** (reproduced on torch 2.13).
- Recon planner, 400 held-out cases: **P@1 0.427 vs prior 0.325** (reproduced exactly).
- CWE classification, **fair re-run** after persisting the curated aliases (same 500 items/set): keyword still beats the
  RAG-LLM — nuclei 0.398 vs 0.374, titles 0.402 vs 0.364 (titles was 0.280 under the confound). Embedding **shortlist
  recall@12 is only 0.45 / 0.53**; when gold is on it the LLM is right 83% / 69%. keyword→LLM routing is best (0.404 / 0.474).
- Tutorial faithfulness (new): baseline prompt cited 13.2% of sentences; **5.1% → 32.6%** of sentences cited *and*
  supported (hand-calibrated) after numbering the graph facts as a source and moving the rules after the sources (tuned on
  10 dev topics, 30 test topics run once). LLM judge vs 40 blind hand labels: κ 0.42 (baseline claims), 0.71 (current).
- Learning path (new, graph-ordered weakness → CAPEC → writeups → labs → next): complete for **25/29** learner topics.
- End to end (new `e2e.py`): API 13/13 over HTTP, Streamlit pages 7/7 with the real model, live server 2/2.

**Defects the 2026-10-22 run caught**: topic resolver sent "SQL injection" to the Hibernate variant (every page);
vLLM 0.27 silently ignores `guided_json` (effect ≤0.002); Metrics page crashed on committed results; first `/search`
paid the whole model load; the tutorial prompt's unnumbered graph facts leaked uncited (a $288,500 bounty misattached).
All in RESULTS §8.

**Not done.** 45% of tutorial sentences still cite nothing and the Learn page does not flag them (PROGRESS step 1);
candidate retrieval caps classification; NoSQL injection does not resolve (CWE-943 alias stored as one string).

**Lessons** → [[Evaluation honesty patterns]]. New here: a citation prompt's failure was invisible until sentences were
counted; the judge was calibrated against blind hand labels rather than trusted (it is stricter than a reader).

**State.** Branch `upgrade-2026-10` pushed (last: docs refresh). Env `bountybrain` (Python 3.11.17, torch 2.13.0+cu130,
vllm 0.27.1, 8.8 GB) recorded in env-registry with rebuild notes; flashinfer patch still needed on 3.11. Weights and
`HF_HOME` at `/mnt/warm_storage/jayavardhan/bountybrain/hf` (28 GB).

**Docs in this vault** (copied 2026-10-14 — stale; the repo's README/RESULTS/PROGRESS are current as of 2026-10-22):
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
