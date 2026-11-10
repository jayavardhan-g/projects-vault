---
type: study-day
day: 12
date: 2026-11-19
tags: [interview, day, bountybrain, rag, knowledge-graph]
---
# Day 12 (Thu Nov 19): BountyBrain (full day)

Part of [[Interview plan (Nov 2026)]]. Repo `bountybrain`, **branch `upgrade-2026-10`** (HEAD `4267980`). Read it with
`git show upgrade-2026-10:<path>`; don't switch branches. About 7 h. Prep pack: [[BountyBrain - prep]] (architecture §2,
23 design decisions §3, 10 stories §4, numbers §5).

**Code walkthrough:** [[Day 12 - code walkthrough]]: one small figure per pipeline, the exact code to read in order, and what each piece does.

**Resume bullets (resume 4):**
- "Built an interactive cybersecurity learning application connecting vulnerability classes, attack patterns and real-world writeups, with **cited explanations** and resources for hands-on practice."
- "Built a knowledge graph of **30k+ nodes across exploits** enabling contextual vulnerability exploration."

**Three things that can hurt you** (prep pack §0). Settle them first, in block 1:
1. **"30k+ nodes across exploits" is wrong as worded.** 30k+ is the *core* subgraph (35,278 nodes), and the core is defined as everything *except* CVE and exploit nodes (`kg/metrics.py:165`, `CORE_KINDS`). The full graph has **111,896 nodes, including 46,647 exploits**. Honest wording: "a 112k-node knowledge graph linking 969 CWE weakness classes, 615 CAPEC attack patterns, 13.6k detection signatures, 46.6k public exploits and 6.4k disclosed writeups".
2. **Authorship.** The first commit (`4c8c10b`, 2026-08-16) is authored by **Tarunsai**, not you. It contains the scaffold, including RRF k = 60 and the bge-small / bge-reranker choice. Later commits carry a `Co-Authored-By: Claude` trailer. The faithfulness hand labels were made by Claude (RESULTS §5). **Decide tonight how you describe this.** Interviewers can open the repo. Don't claim sole authorship of the first commit's choices.
3. **Several "why" questions have no recorded answer:** the embedder, the reranker, RRF k = 60, the 400-token chunks, Qwen2.5-14B. Say what the code shows and what you'd test. Don't invent a selection process.

Also: "cited explanations" means **54.5%** of sentences carry a citation and **32.6%** are cited *and* supported. The UI caption "Every claim is cited" (`ui/pages/1_Learn.py:16`) is false.

**By tonight you can:**
- explain the graph's schema and why SQLite, not a graph database;
- explain BM25, dense retrieval, RRF and cross-encoder reranking, with your ablation numbers;
- explain R@k, MRR, nDCG, and how faithfulness was measured;
- tell stories 4.2 (planner vs baseline), 4.4 (the confounded comparison) and 4.5 (constrained decoding that was never on).

## Schedule

| # | Time | Kind | What |
|---|---|---|---|
| 0 | 15 min | recall | Day 11: BCubed, margin sampling, steering vectors |
| 1 | 30 min | read | Prep pack §0, §1 pitch, §2 diagrams; settle the three risks |
| 2 | 40 min | theory | Knowledge graphs: CWE, CAPEC; BM25 vs dense; bi- vs cross-encoders; RRF (A1) |
| 3 | 60 min | code | `kg/build.py`, skim `kg/models.py`, `kg/queries.py` (B1) |
| 4 | 60 min | code | `rag/index.py`, `rag/hybrid.py`, `rag/reranker.py`, `ingestion/chunker.py` (B2) |
| — | lunch | | |
| 5 | 30 min | theory | R@k, MRR, nDCG; faithfulness; LLM judges and Cohen's κ (A2) |
| 6 | 45 min | code | The tutorial path and the recon planner (B3) |
| 7 | 30 min | skim | `clients/llm.py`, `agent/path.py`, `agent/labs.py`, `api/app.py`, the eval scripts (B4) |
| 8 | 60 min | rebuild | BM25 + dense + RRF + rerank on 200 documents (C) |
| 9 | 20 min | trace | `POST /learn {"topic": "IDOR"}` by hand (D) |
| 10 | 40 min | practice | The questions (E) |

## A. Theory

### A1. Graph and retrieval (block 2, 40 min)

**Read:**
- MITRE's CWE and CAPEC "About" pages (10 min): a CWE is a weakness *class* (with a parent/child hierarchy); a CAPEC is an attack *pattern* that exploits CWEs; CVEs are concrete instances.
- Cormack et al., *"Reciprocal Rank Fusion outperforms Condorcet…"* (2009): the 2-page paper. 10 min.
- The Sentence-Transformers docs page *"Retrieve & Re-Rank"* (bi-encoder vs cross-encoder). 15 min.

**Concepts:**
- **BM25:** a lexical score from term frequency (saturated by k1), inverse document frequency and length normalisation (b). It's good at exact identifiers ("CWE-639", "SSRF").
- **Dense:** a bi-encoder embeds the query and the passages separately (bge-small-en-v1.5; the query gets an instruction prefix, passages don't). It's good at paraphrase.
- **RRF:** `score(d) = Σ_runs 1/(k + rank)`, k = 60. It uses ranks, not raw scores, so BM25 and cosine scales never need calibrating.
- **Cross-encoder rerank:** reads (query, passage) together. More accurate, but O(candidates) model calls, so only the top 40 fused candidates are reranked.
- **Why SQLite and not Neo4j** (decision 3.2): typed relational tables, simple joins, one file, no server. The graph queries you need are 1–3 hops.

### A2. Evaluation (block 5, 30 min)

- **R@k:** the gold chunk is in the top k. **MRR:** the mean of 1/rank of the first relevant hit. **nDCG@k:** a graded, position-discounted gain over the ideal (story 4.3: your first nDCG was 1.113, which is impossible).
- **Your retrieval eval:** 200 LLM-generated known-item queries. hybrid+rerank **R@5 0.965, MRR 0.861**; dense 0.885/0.759; BM25 0.850/0.735; hybrid 0.910/0.773. Caveat: the source chunk *stays* in the index, and queries can reuse its words.
- **Faithfulness:** split the answer into sentences; for each cited sentence an LLM judge checks support; reweight by hand labels. **Cohen's κ** measures judge–label agreement beyond chance. Caveats: the judge is the same model as the generator, and the "blind hand labels" were made by Claude.

## B. Code

### B1. The graph (block 3, 60 min, deep)

`bountybrain/kg/build.py`:
- **470 `build`**; **79 `GraphBuilder`**;
- 107 `load_weaknesses` (curated aliases persisted via **440 `_merged_aliases`**: the story 4.4 fix), 151 `load_techniques`, 211 `ensure_cves`, 223 `ensure_products`, 231 `load_signatures`, 278 `load_exploits`, 328 `load_documents`.
- `kg/queries.py`: **78 `find_weakness`** (number → `CWE-…` → exact name → alias, 103), 139 `weakness_hierarchy`, 173 `techniques_for_weakness`, 185 `attack_chain`, 244 `documents_for_weakness`.
- `kg/metrics.py`: **165 `CORE_KINDS`** (why the 30k+ has no exploits), 168 `core_subgraph`, 410 `compute_all`.

### B2. Retrieval (block 4, 60 min, deep)

- `ingestion/chunker.py` **59 `chunk_text`**: structure-aware; code fences kept whole.
- `rag/index.py` **115 `build_index`**: 70 `iter_chunks`, 97 the chunk cache (11,355 chunks; BM25 reads this file).
- `rag/hybrid.py`: **99 `_bm25_search`** (in memory), **137 `_dense_search`** (embedded Qdrant), **155 `reciprocal_rank_fusion`**, **176 `search`** (mode `hybrid_rerank`, 40 candidates; 200 the rerank call).
- `rag/reranker.py` **25 `rerank`** (bge-reranker-base).

### B3. Tutorial and planner (block 6, 45 min, deep)

- `agent/tools.py`: 27 `local_rag_search`; 102 `tutorial_sources` (passages cut to 1,200 chars; the graph summary is the last numbered source); **137 `generate_tutorial`** (temperature 0; 174 a deterministic fallback).
- `recon/planner.py`: **649 `plan_from_findings`** → **233 `rank_weaknesses`**; **299–307** P(weakness | product) shrunk toward the global prior with a pseudo-count of 25 (story 4.2).

### B4. Skim (block 7, 30 min)

`clients/llm.py` (117 `chat`; `response_format` json_schema, story 4.5), `agent/path.py` (59 `learning_path`), `agent/labs.py` (144 `find_practice_labs`; 29 CWE classes), `api/app.py` (33 warm-up at startup; 106 `/learn`), `eval/classification.py` (the confound), `eval/faithfulness.py`.

## C. Rebuild it yourself (block 8, 60 min)

About 90 lines, CPU only (`rank_bm25`, `sentence-transformers`):
1. Take 200 short documents (for example, CWE descriptions or any Wikipedia paragraphs). Chunk them.
2. BM25 top 20, dense (any small embedder) top 20, then `rrf([bm25, dense], k=60)`.
3. Rerank the fused top 20 with a small cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
4. Write 20 queries with known gold documents. Report R@5 and MRR for BM25, dense, RRF and RRF + rerank. Does your ordering match BountyBrain's?

## D. Trace (block 9, 20 min)

From memory, then check against prep pack §2 "One request, end to end": `POST /learn {"topic": "IDOR"}`.
Which step resolves "IDOR" to CWE-639? How many candidates come from each retriever? What is fused, what is reranked, how many sources reach the prompt, and what happens if vLLM is down?

## E. Questions

1. Give the 60-second pitch.
2. What's in the graph? Why is CWE the spine?
3. "30k+ nodes across exploits": walk me through the counts. (*Correct it first.*)
4. Why SQLite and not a graph database?
5. Why hybrid retrieval? Show the ablation.
6. Why RRF and not a weighted sum of scores?
7. Why rerank only the top 40?
8. How did you measure faithfulness? What are its weaknesses? (*Same model judges; Claude hand labels; one generation per topic.*)
9. The LLM classifier didn't beat keyword matching. Why? (*Story 4.4: the shortlist was the ceiling; gold in the top 12 only 45–53% of the time.*)
10. Your planner lost to a baseline at first. What was wrong? (*Story 4.2.*)
11. How do you know constrained decoding is on? (*Story 4.5.*)
12. Who wrote the first commit? Which design choices were yours?
13. Hostile: "A cybersecurity tool that tells me how to exploit things: how do you keep it from being misused?" (*It never scans; learning content only; decision 3.23.*)

## My notes and fumbles

- 
