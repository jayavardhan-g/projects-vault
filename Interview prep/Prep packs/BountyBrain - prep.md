> Prep pack for **BountyBrain**, written 2026-11-05 from `bountybrain/INTERVIEW_PREP.md`. The vault copy is the one to study and edit.

# BountyBrain — Interview Prep

Written 2026-11-05 from the `upgrade-2026-10` branch (HEAD `4267980`).
Sources read: README.md, RESULTS.md, PROGRESS.md, SETUP.md, ATTRIBUTION.md,
the full git log (28 commits), the source code, and the JSON in `artifacts/`.
There is no project CLAUDE.md.

What I checked myself, without a GPU:
- `pytest` in the `bountybrain` env: **84 passed**, 0 skipped (CPU only).
- A read-only look at `data/bountybrain.sqlite`: the table counts match
  `artifacts/graph_metrics.json` exactly.
- Every headline number below matches its JSON file in `artifacts/`.

What I did **not** do: re-run any GPU or LLM evaluation. "Reproduces" below
means "the committed artifact exists and matches the docs". The project's own
2026-10-22 re-run (commit `11a9e1d`) reproduced retrieval and planner numbers
from August.

Line numbers are for HEAD `4267980`.

---

## 0. Read this first: three things that can hurt you

1. **Your bullet "30k+ nodes across exploits" is wrong as worded.** The 30k+
   figure is the *core* subgraph (35,278 nodes). The core is defined as
   everything *except* CVE and exploit nodes (`bountybrain/kg/metrics.py:165`,
   `CORE_KINDS`). So the 30k+ core contains **zero** exploits. The 46,647
   exploit nodes sit in the full 111,896-node graph, mostly in tiny
   CVE↔exploit islands. See §5 for honest wording.

2. **"Sole author" does not match the git history.** The first commit
   (`4c8c10b`, 2026-08-16) is authored by `Tarunsai <dtarunsai08@gmail.com>`,
   not by `Jayavardhan`. It holds the original scaffold: LangGraph agent, API,
   Streamlit pages, Qdrant client, hybrid search with RRF k=60, and the
   bge-small / bge-reranker choice. All 27 later commits carry a
   `Co-Authored-By: Claude` trailer. The faithfulness hand labels were made by
   Claude, not you (RESULTS §5 says so). An interviewer who opens the repo
   will see this. Decide how you describe it before they ask (see §8).

3. **Several "why" questions have no recorded answer.** The embedder, the
   reranker, RRF k=60, the 400-token chunk size and Qwen2.5-14B are never
   justified in any doc or commit. bge-small, bge-reranker-base and RRF all
   arrived in the first commit. Do not invent a selection process. Say what
   the code shows and what you would test (§3).

---

## 1. The pitch

### 60 seconds

BountyBrain is a learning tool for bug bounty hunting. It answers four
questions a beginner has. Where do I start on a vulnerability class? Explain
it to me, with sources. How does it relate to other classes? And, given my
recon output, what should I test next?

Under it is a knowledge graph built from eight openly licensed bulk datasets.
These are MITRE CWE and CAPEC, nuclei detection templates, Exploit-DB,
pentester.land writeup metadata, and three more. The graph has about 112
thousand nodes, with a connected core of 35 thousand.

On top sits hybrid retrieval. Dense embeddings and BM25 are fused with
reciprocal rank fusion, then a cross-encoder reranks. That reaches recall@5
of 0.965 on 200 generated queries. A local Qwen 14B model on vLLM writes the
cited tutorials.

The part I care most about is the evaluation. Every component has a baseline,
and several results came out badly. The LLM classifier did not beat a keyword
table. The planner's first version lost to a frequency baseline. Only a third
of tutorial sentences were cited and supported. I kept those results and
fixed what I could. And it never scans anything.

### 3 minutes

**The problem.** Bug bounty knowledge is scattered. MITRE says "Authorization
Bypass Through User-Controlled Key". Hunters say "IDOR". Detection templates,
public exploits and disclosed reports each use their own labels. A learner
cannot join them up.

**The data.** I only used sources published for bulk download under open
licences. That is a deliberate constraint. It means anyone can run three
commands and get my numbers. Scraped data decays and cannot be shared. From
pentester.land I took only the metadata feed. The 6,421 linked articles live
on about 500 blogs, and I never crawled them.

**The graph.** CWE is the spine. Every other source resolves onto CWE ids.
There are seven node types and typed edge tables in SQLite, one table per
relationship. Joining sources needed entity resolution. I built an alias
table in three layers: official names, MITRE's alternate terms plus short
names hidden in parentheses, and 109 hand-curated community terms like IDOR
and SSTI. That maps 93.1% of labelled writeups onto CWE, up from 76.5%.

**Retrieval.** Documents are chunked so code blocks are never split, because
a half payload is useless. Each chunk is embedded with bge-small and stored in
embedded Qdrant. BM25 runs over the same chunks. The two ranked lists are
fused with RRF, and a cross-encoder reranks the top 40. In a four-way
ablation, fusion alone barely helps. The reranker is what pays: recall@1
jumps from 0.675 to 0.800.

**Generation.** The Learn page resolves the topic in the graph, retrieves six
passages, and asks Qwen2.5-14B to cite each sentence by number. I measured
this. At first only 13% of sentences cited anything. I moved the citation
rules after the sources and made the graph summary a numbered source. Tuned
on 10 dev topics, then run once on 30 test topics, cited-and-supported
sentences went from 5.1% to 32.6%.

**The recon planner.** You give it files from your own recon tools. It parses
them and ranks weakness classes per host. Scoring blends applicability to the
fingerprinted product, EPSS, public exploits, and bounty economics. The first
version lost to a "most common class" baseline. The applicability term
saturated for popular products. I replaced it with a smoothed conditional
probability. P@1 went from 0.280 to 0.427, against 0.325 for the baseline.

**What I learned.** A negative result I can explain is worth more than a
positive one I cannot. The classification comparison was confounded: only
the keyword arm could see the curated aliases. After fixing it, keyword still
won. The real ceiling was the candidate shortlist, which held the right class
only about half the time.

---

## 2. Architecture

### Offline build (no GPU needed for steps 1–2)

```
 fetch_data.py ──► data/raw/  (CWE XML, CAPEC XML, nuclei repo, Exploit-DB CSV,
                               pentester.land JSON, PATT, WSTG, VRT)
        │
        ▼
 build_graph.py:25 main
        └─► bountybrain/kg/build.py:470  build()
              ├─ datasets/cwe.py:169 load, :362 alias_map   (names + alt terms + parentheticals + curated)
              ├─ GraphBuilder (build.py:79)
              │    load_weaknesses :107  (persists curated aliases via _merged_aliases :440)
              │    load_techniques :151  (CAPEC, CAPEC->CAPEC, CAPEC->CWE)
              │    ensure_cves :211, ensure_products :223
              │    load_signatures :231  (nuclei -> CWE / CVE / product; _useful_tech_keys :458)
              │    load_exploits :278    (EDB -> CVE; denormalises CVE.exploit_count)
              │    load_documents :328   (articles, writeups, CWE pages, CAPEC pages)
              └─► data/bountybrain.sqlite  (kg/models.py, kg/db.py:29 WAL pragmas)
        │
        └─► kg/metrics.py:410 compute_all ──► artifacts/graph_metrics.json
                (build_networkx :57, core_subgraph :168)

 index_corpus.py:24 main
        └─► rag/index.py:115 build_index
              ├─ iter_chunks :70 ──► ingestion/chunker.py:59 chunk_text (fences kept whole)
              ├─ save_chunk_cache :97 ──► data/corpus/chunks.jsonl  (11,355 chunks; BM25 reads this)
              └─ clients/embeddings.py:59 embed_passages (bge-small, no prefix)
                   ──► clients/qdrant.py:96 upsert_points ──► data/qdrant (embedded, cosine)
```

### Online (query time)

```
                          ┌──────────────────────────────┐
  Streamlit pages ───────►│ bountybrain/agent/tools.py   │◄────── FastAPI  api/app.py
  ui/pages/*.py           │ plain functions, JSON out    │        13 routes
                          └──────┬───────────┬───────────┘
                                 │           │
          ┌──────────────────────┘           └─────────────────────┐
          ▼                                                        ▼
  kg/queries.py  (SQLite joins)                         rag/hybrid.py:176 search
  find_weakness :78                                     ├─ _dense_search :137 ─► Qdrant
  weakness_hierarchy :139                               ├─ _bm25_search  :99  ─► rank_bm25 (in memory)
  techniques_for_weakness :173                          ├─ reciprocal_rank_fusion :155 (k=60)
  attack_chain :185                                     └─ rag/reranker.py:25 rerank (bge-reranker-base)
  documents_for_weakness :244
          │                                                        │
          └──────────────────┐                    ┌────────────────┘
                             ▼                    ▼
                  tools.generate_tutorial :137 ──► clients/llm.py:117 chat
                                                   ──► vLLM (OpenAI API, localhost:8010)
                                                       Qwen2.5-14B-Instruct, serve_llm.py

  Recon:  api/app.py:160 recon_plan ─► tools.plan_recon :83
          ─► recon/planner.py:649 plan_from_findings ─► rank_weaknesses :233
  Path:   api/app.py:183 ─► agent/path.py:59 learning_path
  Labs:   agent/labs.py:144 find_practice_labs (hand-curated, keyed by CWE id)
```

### One request, end to end: `POST /learn {"topic": "IDOR"}`

1. **Startup, once.** `api/app.py:33 _startup` calls `init_db()` and runs a
   throwaway `search("warm-up")`. That loads the embedder, the reranker and
   BM25 before the first user arrives (commit `e3f81dc`).
2. **Route.** `api/app.py:106 learn` calls four tools in order.
3. **Resolve in the graph.** `agent/tools.py:53 kg_context` →
   `kg/queries.py:78 find_weakness("IDOR")`.
   - Not a number, not `CWE-…`, not an exact name.
   - Exact alias step (`queries.py:103`):
     `aliases_json ILIKE '%"IDOR"%'`. CWE-639's stored aliases include
     `"idor"` (curated layer, persisted since `6187c2a`). It matches.
   - Then `weakness_hierarchy` (:139), `techniques_for_weakness` (:173), and
     up to 8 writeups via `documents_for_weakness` (:244).
4. **Retrieve.** `tools.py:27 local_rag_search("IDOR", top_k=6)` →
   `rag/hybrid.py:176 search`, mode `hybrid_rerank`, `candidates=40`.
   - Dense: `embeddings.py:73 embed_query` adds the BGE instruction prefix
     (`embeddings.py:25`), then `qdrant.py:111 search` returns the top 40 by
     cosine.
   - Lexical: `hybrid.py:99 _bm25_search` scores all 11,355 chunks with
     `BM25Okapi`, top 40.
   - Fuse: `hybrid.py:155` sums `1/(60 + rank)` per (document, chunk).
   - Rerank: `hybrid.py:200` sends `fused[:40]` to `reranker.py:25`, which
     scores 40 (query, chunk) pairs with the cross-encoder and keeps 6.
5. **Number the sources.** `tools.py:102 tutorial_sources` truncates each
   passage to 1,200 chars (`SOURCE_CHARS`, :98). It appends the graph summary
   (`_context_brief`, :155) as the last numbered source.
6. **Generate.** `tools.py:137 generate_tutorial` → `tutorial_prompt` (:128)
   fills `prompts/tutorial.md` → `clients/llm.py:117 chat`, temperature 0,
   max 2,048 tokens, to vLLM. If anything throws, it falls back to a
   deterministic template (`tools.py:174`).
7. **Labs.** `agent/labs.py:144 find_practice_labs` resolves the topic through
   the graph again and walks up the CWE tree if a class has no lab.
8. **Response.** Topic, weakness, techniques, numbered sources, tutorial,
   labs. In the e2e run this took 9.1 s (`artifacts/e2e.json`).

---

## 3. Design decisions

Each entry: alternatives → choice → why → cost. Tag: **documented** (with
source) or **reconstructed** (inferred from code; no recorded reason).

### 3.1 Bulk-downloaded open datasets, never scraping
- **Alternatives:** scrape HackerOne Hacktivity, crawl writeup blogs.
- **Choice:** only git repos or single archives under open licences.
- **Why:** anyone can reproduce the numbers; scraped data decays and is not
  redistributable. HackerOne was explicitly ruled out.
- **Cost:** writeups are metadata only (title, labels, bounty, link). No
  article bodies. Tutorials can only cite a writeup's title and labels.
- **Documented:** README "Why the data is bulk-downloaded", PROGRESS
  "Design decisions worth not re-litigating", commit `40196d6`.

### 3.2 Typed relational tables in SQLite, not a graph database
- **Alternatives:** Neo4j or another property-graph DB; one generic
  `edges(src, dst, type)` table; Postgres.
- **Choice:** SQLAlchemy models, one table per edge type
  (`kg/models.py:224–321`), SQLite by default, Postgres via `DATABASE_URL`.
- **Why (documented):** no server, no port to collide with on a shared
  machine; `git clone && build_graph.py` reproduces the metrics; typed tables
  make "which patterns follow this one" a join, not a string filter
  (`models.py:1–8`, README "Design decisions", PROGRESS).
- **Why not Neo4j (reconstructed — no doc discusses Neo4j as an option):**
  the queries are shallow. Most are one or two joins. The deepest is the
  attack-chain walk (depth 3, done in Python, `queries.py:185`). The graph is
  38 MB on disk. Cypher's strength is variable-length traversal at scale,
  which this workload barely needs. Neo4j appears only as an export target
  (`export_graph.py`).
- **Cost:** multi-hop queries are hand-written Python loops. Graph metrics
  need a full NetworkX copy in memory (`metrics.py:57`). Only one writer at a
  time. And the metric graph silently drops one populated edge table (see
  §8.4).

### 3.3 Qdrant embedded, not a vector server
- **Alternatives:** Qdrant server, FAISS, pgvector.
- **Choice:** `qdrant-client` in local mode against `./data/qdrant`.
- **Why:** no server, no port, on a shared box. Set `QDRANT_URL` to switch.
- **Cost:** it takes an exclusive lock. The API and an eval run cannot both
  hold it; `e2e.py` has to stop the API before the UI pass.
- **Documented:** README "Design decisions", SETUP "Vector store".

### 3.4 BM25 built in memory from the same chunk cache
- **Alternatives:** Elasticsearch / OpenSearch; Qdrant sparse vectors.
- **Choice:** `rank_bm25.BM25Okapi` over `chunks.jsonl` (`hybrid.py:66`).
- **Why:** one chunking pass, so dense and lexical see an identical corpus,
  which the ablation needs.
- **Cost:** `get_scores` is a linear scan per query in Python. Fine at 11k
  chunks (16 ms median). It would not be at millions.
- **Documented:** `rag/index.py:6–10` docstring.

### 3.5 RRF, not score interpolation
- **Alternatives:** weighted sum of normalised cosine and BM25 scores;
  learned fusion.
- **Choice:** RRF with k=60 (`hybrid.py:28`, `:155`).
- **Why:** cosine and BM25 live on different scales; normalising needs
  per-query statistics; RRF uses only ranks and needs no tuning.
- **Cost:** it throws away score margins. A query where dense is very
  confident gets no extra weight.
- **Documented:** `hybrid.py:10–13` docstring. **But:** the comment "results
  are insensitive to it [k]" was never tested. No sweep exists in the repo.
  RRF and k=60 were already in the first commit `4c8c10b`.

### 3.6 bge-small-en-v1.5 embedder and bge-reranker-base
- **Alternatives:** bge-base/large, e5, gte, OpenAI embeddings; ms-marco
  MiniLM cross-encoders, bge-reranker-large or v2-m3.
- **Choice:** `config.py:38–39`.
- **Why — reconstructed. No doc or commit explains it.** Both defaults are
  already in the first commit `4c8c10b`. What the code and docs do support:
  bge-small is 384-dimensional and small, so 11k chunks embed in about a
  minute and query embedding is cheap; it shares a GPU with a 14B LLM; BGE
  v1.5 is a strong small English retriever. bge-reranker-base is from the
  same family. The one documented detail is the asymmetric query prefix
  (`embeddings.py:1–12`, README).
- **Cost:** no comparison against any other model was run. You cannot claim
  bge-small was the best choice, only that it worked.

### 3.7 Rerank the top 40 fused candidates
- **Choice:** `candidates=40` per arm, rerank `fused[:40]`, keep top k
  (`hybrid.py:180,188,200`).
- **Why (documented):** the win is in ordering, not finding. Hybrid already
  has R@10 0.955; rerank lifts R@1 by 0.125 (RESULTS §2).
- **Cost:** ~3× latency (154 ms vs 56 ms median). Depth 40 is not justified
  anywhere (reconstructed).

### 3.8 Structure-aware chunking
- **Choice:** split into prose and fenced-code blocks; pack to ~1,600 chars;
  never cut a fence; block-level overlap capped at a quarter of target
  (`chunker.py:59`).
- **Why:** in this corpus a payload is the answer; half a payload is noise.
- **Cost:** oversized code blocks become one huge chunk; the 4-chars-per-token
  rule is rough. The 400-token target has no recorded reason.
- **Documented:** `chunker.py` docstring, README, PROGRESS.

### 3.9 LLM picks a CWE from a 12-item shortlist
- **Alternatives:** let the LLM name any CWE; fine-tune a classifier.
- **Choice:** embed the text, take the 12 nearest CWE classes, constrain the
  LLM to that list, reject anything off it (`classifier.py:194`).
- **Why:** an LLM cannot recall ~970 numeric ids; the shortlist makes
  invented ids impossible.
- **Cost:** recall@12 of the shortlist caps accuracy (45% / 53%).
- **Documented:** `classifier.py:14–18`, README, RESULTS §3.

### 3.10 Keyword as the default classifier; keyword→LLM routing
- **Choice:** use the alias table; ask the LLM only when keyword abstains.
- **Why:** keyword is free, fast, and at least as accurate.
- **Cost:** keyword abstains on 35% (nuclei) and 52% (titles).
- **Documented:** RESULTS §3, commit `ae21e83`.

### 3.11 Planner: hand-set weights, smoothed applicability
- **Choice:** `ScoringWeights` 0.55 / 0.15 / 0.05 / 0.25 (`planner.py:75`);
  applicability = `(n + 25·prior) / (N + 25)`, scaled so the top class is 1
  (`planner.py:299–319`).
- **Why:** applicability is the only term conditioned on the stack, so it
  gets the most weight. Smoothing makes a product with few templates fall
  back to the global prior.
- **Cost:** weights and α=25 were never swept (PROGRESS §5).
- **Documented:** `planner.py:58–107`, README, RESULTS §4.

### 3.12 Value-weighted vs likelihood-only ranking, both shown
- **Why:** "what will this stack turn up" and "what is worth my afternoon"
  are different questions.
- **Cost:** value weighting is slightly worse at prediction (P@3 0.562 vs
  0.570).
- **Documented:** README, RESULTS §4.

### 3.13 Hold out 20% of templates and hide them from the evidence pool
- **Why:** gold is a signature→CWE edge and the planner scores with those
  edges. Without hiding them the eval grades the planner on what it was given.
- **Cost:** a residual leak through the global prior, stated (~5% of edges).
- **Documented:** `eval/planner.py:7–35`, RESULTS §4, commit `63ff739`.

### 3.14 Faithfulness: deterministic splitting + LLM judge + blind hand labels
- **Choice:** split sentences with regex (`faithfulness.py:141`); check
  citation numbers exactly; judge cited sentences with the same Qwen model
  against the exact excerpts; calibrate on 40 blind hand labels per prompt.
- **Why:** an LLM judge alone is not trustworthy; the generator judging
  itself is biased.
- **Cost:** one judge model, one hand labeller (Claude), 40 labels, one
  sample per topic.
- **Documented:** `faithfulness.py:1–27`, RESULTS §5, commits `f78e2a6`,
  `462c8af`.

### 3.15 Graph summary as the last numbered source; rules after sources
- **Why:** unnumbered "framing" facts were used uncited and misattached (the
  $288,500 SSRF bounty pinned to an unrelated example); rules above ~7k chars
  of sources were ignored.
- **Documented:** `tools.py:102–113`, commit `462c8af`, RESULTS §5.

### 3.16 Local vLLM serving Qwen2.5-14B-Instruct at 40% GPU memory
- **Alternatives:** Ollama (the first commit used `qwen2.5:7b` via Ollama),
  a hosted API.
- **Choice:** `serve_llm.py` picks the freest GPU, asks for 40% of it.
- **Why (documented):** nothing leaves the machine; continuous batching makes
  bulk classification practical (`llm.py:3–6`); vLLM's 0.9 default would
  reserve ~130 GB of a 143 GB card on a shared server (SETUP).
- **Why Qwen2.5-14B — reconstructed.** No doc compares models. It fits on one
  GPU next to the embedder and reranker and follows JSON instructions.
- **Cost:** the judge is the same model; output is not bit-stable under
  batching (RESULTS §5).

### 3.17 `response_format` json_schema for structured output
- **Why:** vLLM 0.27 silently ignores the old `guided_json` field.
- **Documented:** `llm.py:14–22`, SETUP, commit `83090cd`.

### 3.18 No web-search tool
- **Why:** the old Brave tool returned nothing without a key, so the agent
  claimed to have searched and grounded answers in nothing.
- **Documented:** `tools.py:7–12`, README, PROGRESS, commit `40196d6`.

### 3.19 Learning path borrows from neighbours, and says so
- **Choice:** if a stage is empty, take patterns from a parent, writeups from
  a child then a parent, labs from a parent; set `inherited_from`.
- **Documented:** `agent/path.py:14–17`, commit `f78e2a6`.

### 3.20 Labs keyed by CWE id, resolved through the graph
- **Why:** string keys meant "IDOR", "Insecure Direct Object Reference" and
  "CWE-639" were three lookups.
- **Documented:** `labs.py:1–16`, commit `eeae180`.

### 3.21 Artifacts committed; README tables generated from them
- **Why:** so the numbers can be checked, not trusted; tables cannot drift.
- **Documented:** commit `de660d5`, `report.py`, RESULTS header.

### 3.22 LangGraph optional
- **Choice:** a 4-node linear graph (`agent/graph.py`); if LangGraph is
  missing, the same nodes run in sequence.
- **Honest note (reconstructed):** the API and UI call `tools.py` directly.
  The LangGraph flow is not on the request path. Do not sell it as an
  "agent".

### 3.23 Never scans
- **Choice:** parse files the user produced; no network at query time;
  labs point only at training platforms; the tutorial prompt has a fixed
  "test safely" section.
- **Documented:** README, `app.py:5–6`, `planner.py:26–27`, `labs.py:13–16`,
  `prompts/tutorial.md`.

---

## 4. Hardest problems, as STAR stories

### 4.1 Replacing a corpus that measured nothing (`40196d6`)
- **S:** The first version ran on 500 templated reports from 15 hand-written
  templates, 60 ingested. The graph had 15 vulnerability nodes and no CVEs or
  techniques: a star, not a graph. Its recall@5 of 0.87 came from 15 fixtures
  that counted a hit if the URL contained "ssrf-".
- **T:** Make the graph real and the metrics mean something.
- **A:** Rebuilt on eight bulk, licensed datasets with CWE as the spine.
  Wrote the alias table. Made each retrieval stage runnable alone so the
  ablation could fail. Deleted the synthetic corpus.
- **R:** 111,896 nodes, 96,805 edges, reproducible with three commands.
- **Careful:** that first version is the commit authored by Tarunsai. Tell
  this as "the project started from a scaffold with a synthetic corpus" unless
  you wrote it.

### 4.2 The planner lost to a trivial baseline
- **S:** First planner scored applicability as a log-scaled count of matching
  templates. P@1 0.280 vs 0.325 for "rank by global frequency".
- **T:** Find out why a model with more information lost.
- **A:** The log count saturated at 1.0 for any popular product. Every
  candidate got the same applicability, so stack-independent terms decided
  the ranking. Replaced it with P(weakness | product), shrunk toward the
  global prior with a pseudo-count of 25 (`planner.py:299–307`).
- **R:** P@1 0.427 vs 0.325 prior, on 400 held-out cases across 147
  products. Reproduced to four places on 2026-10-22 (`11a9e1d`).
- **Caveat:** the 0.280 run is documented (README since `40196d6`) but no
  artifact or code for it exists in git. The fix predates the first real
  commit.

### 4.3 An impossible nDCG (`3b92ad2`)
- **S:** The first retrieval run reported nDCG@10 = 1.113.
- **T:** nDCG cannot exceed 1. Find the bug.
- **A:** The ideal DCG assumed one relevant item. But the gold document is
  chunked, so several hits can be relevant. Normalised against all relevant
  hits found (`eval/retrieval.py:176–182`).
- **R:** 0.873 for the best arm. The impossible value exposed the bug.
- **Follow-up to expect:** "Isn't normalising by relevant hits *found* still
  wrong?" Yes, a little. The textbook ideal uses all relevant items in the
  collection, found or not. This version cannot penalise missing chunks of the
  gold document. It is bounded by 1, but optimistic.

### 4.4 The confounded classification comparison (`6187c2a`, `ae21e83`)
- **S:** In August, the LLM classifier lost to keyword on writeup titles,
  0.280 vs 0.402.
- **T:** Decide whether that was the model or the setup.
- **A:** "IDOR" resolved to nothing in the UI, though it is the most common
  label in the corpus. The cause: the curated aliases were folded in at query
  time from the catalogue on disk, but never stored in the graph. The keyword
  arm read the disk table and saw them. The embedding and LLM arms built their
  shortlist from stored `aliases_json` and did not. So the comparison was
  unfair. Fixed the builder to persist them (`build.py:440`), deleted the
  cached candidate index, re-ran on the same 500 items per set. Keyword
  scores came out bit-identical, which confirmed the pairing.
- **R:** Titles: LLM 0.280 → 0.364, embedding 0.188 → 0.330. Nuclei barely
  moved (0.368 → 0.374). Keyword still won. Then I measured the shortlist:
  gold was in the top 12 only 45% / 53% of the time. When it was, the LLM was
  right 83% / 69%. The retriever, not the model, was the ceiling.
- **Lesson:** a cache (`cwe_candidates.npz`) can turn a fix into a silent
  no-op. PROGRESS now warns about it.

### 4.5 Constrained decoding that was never on (`83090cd`)
- **S:** The client sent `extra_body={"guided_json": schema}`.
- **T:** Check the "constrained" claim in the docstring.
- **A:** Sent a haiku prompt with it; got a haiku back. vLLM 0.27.1 drops the
  field with no error or warning, so the fallback for a rejected parameter
  never fired. Switched to the OpenAI-standard `response_format` json_schema,
  which vLLM enforces. Also closed the async client inside its own loop, which
  removed an "Event loop is closed" traceback after every batch.
- **R:** Measured effect on classification: 0.372 → 0.374 nuclei, 0.364 →
  0.364 titles, 0 model failures. The prompt alone had been producing valid
  JSON, so earlier numbers were not distorted. But the docstring had been
  false.

### 4.6 Tutorials that did not cite (`f78e2a6`, `462c8af`)
- **S:** The Learn page promised every claim cited. Nobody had checked.
- **T:** Measure it, then fix it without tuning on the test set.
- **A:** Built a faithfulness suite on 30 fixed topics. Baseline: 13.2% of
  sentences cited, 5.1% cited and supported. Two causes were visible: graph
  facts were unnumbered, and the rules sat above ~7k chars of sources. Made
  the graph a numbered source, moved the rules after the sources with an
  example. Iterated on 10 disjoint dev topics (v1 → v2 → v3), froze v3, ran
  the test set once. Calibrated the judge on blind hand labels.
- **R:** 54.5% cited; 32.6% cited and supported (hand-calibrated); judge
  agreement κ 0.71 (3-way), 0.75 (binary). Still 45% uncited.

### 4.7 vLLM on a shared GPU server (SETUP.md; `9110924`, `a88eeb1`, `9f9c201`)
- **S:** Each failure came *after* a ~10-minute 14B weight load from network
  storage, so each looked like a model problem.
- **T:** Get a reliable launch.
- **A:** Found four separate causes. (1) flashinfer uses `array.array[int]`,
  which only works on Python 3.12+; the compile pass imports it →
  `--enforce-eager`. (2) Kernel warmup imports it anyway → one-line
  `from __future__ import annotations` patch. (3) Triton cannot create its
  cache on GlusterFS (`Errno 61`) → caches on local disk. (4) JIT shells out
  to `ninja`, missing from PATH in bare launches → prepend the env's `bin/`.
  Also asked for 40% of the card, not 0.9. `serve_llm.py` now does 1, 3, 4
  and the memory cap itself.
- **R:** Documented launch path works; the 3.11 patch is still manual.
  Python 3.12 avoids the first two.

### 4.8 "SQL injection" resolved to the Hibernate variant (`5cc8d85`)
- **S:** The learning-path eval exposed it. `find_weakness` went from exact
  name straight to a substring search ranked most-specific first.
- **A:** Added an exact, case-insensitive alias match before the substring
  fallback (`queries.py:100–110`), with tests.
- **R:** "SQL injection" → CWE-89, "path traversal" → CWE-22 family, "race
  condition" off CWE-1223. Learning-path completeness rose to 86.2%.

### 4.9 Recon parsing that corrupted input quietly (`3d4bd23`)
- **S:** nuclei writes `cve-id` as a string for one CVE and a list for many.
- **A:** Iterating the string gave one finding per character: "CVE-2021-22205"
  became 14 findings. Also every parser grouped by registrable domain, so
  GitLab and Jira hosts merged into one plan. Normalised to a list; carried
  `host`; planned per host; normalised host spellings at every site.
- **R:** 2 CVEs instead of 28 on the samples; 6 plans instead of 1.

### 4.10 First search took 174 seconds (`e3f81dc`)
- **S:** The first `/search` after startup loaded three models lazily.
- **A:** Warm them in the startup hook.
- **R:** First `/search` 160 ms; API healthy after ~21 s. (These one-off
  timings are in the commit message only, not in any artifact.)

---

## 5. Numbers sheet

Hardware for all GPU numbers: one "H100-class" GPU (exact model not
recorded), shared by vLLM (Qwen2.5-14B-Instruct, 40% memory,
`--enforce-eager`), the embedder and the reranker. Env `bountybrain`,
Python 3.11.17, torch 2.13.0+cu130, vllm 0.27.1.

### 5.1 Numbers on your resume

| claim | what the repo shows | supported? |
|---|---|---|
| "30k+ nodes across exploits" | Full graph 111,896 nodes (46,647 exploits). Core 35,278 nodes = weakness + technique + signature + product + document; **no exploit or CVE nodes** (`metrics.py:165`, `graph_metrics.json → core_structure.node_kinds`) | **No, as worded.** |

**Honest wording options:**
- "Built a 112k-node knowledge graph linking 969 CWE weakness classes, 615
  CAPEC attack patterns, 13.6k detection signatures, 46.6k public exploits and
  6.4k disclosed writeups."
- If you want "30k+": "...with a 35k-node connected core of weaknesses,
  attack patterns, detection signatures, products and documents." Do not
  attach "exploits" to the 35k.

Bullet 1 has no numbers, but check its words:
- "real-world writeups": metadata records (title, labels, bounty, link), not
  article text. Say "disclosed writeup records" if pressed.
- "cited explanations": 54.5% of sentences carry a citation; 32.6% are cited
  *and* supported. The UI caption still says "Every claim is cited… Nothing is
  answered from memory" (`ui/pages/1_Learn.py:16`). That caption is false.
- "resources for hands-on practice": a hand-curated lab table covering 29
  CWE classes (`agent/labs.py:29`).

### 5.2 Results you will quote

| number | how measured | n / baseline | caveats | reproduces? |
|---|---|---|---|---|
| 111,896 nodes, 96,805 edges | `build_graph.py --rebuild` → `metrics.py:410` | — | Counts 8 edge kinds. `signature_references_cve` (4,196 rows) is populated but not added to the metric graph (§8.4). Tool table has 0 rows. | Yes: SQLite counts match the JSON (I checked) |
| core 35,278 nodes / 63,340 edges, mean degree 3.59 | `core_subgraph` drops CVE + exploit nodes | whole graph mean degree 1.73 | Core excludes exploits by definition | Yes (`b049272` put it in `graph_metrics.json`) |
| 1,389 aliases; 93.1% writeup labels resolved (was 76.5%) | alias table applied to 671 distinct pentester.land labels | 22 non-vuln labels excluded from denominator | 76.5% is from a pre-commit state; no artifact for it | 93.1%: yes (build log). 76.5%: docs only |
| hybrid+rerank R@5 0.965, MRR 0.861 | 200 LLM-generated queries, 50 per doc type, known-item: hit = chunk from the source document; median latency per query on the GPU | dense 0.885 / 0.759; BM25 0.850 / 0.735; hybrid 0.910 / 0.773 | Source chunk **stays in the index** ("held-out" means sampled, not removed). Queries often keep rare tokens ("sharp printer … pierre kim 2024"), flattering BM25. Recall is a lower bound. One query set, one seed | Yes: `eval_retrieval.json`; re-run 10-22 matched August except dense R@5 0.880→0.885 |
| R@1 0.675 → 0.800 from rerank | same | +25 queries of 200 | Hybrid vs dense R@1 is +2 queries: noise | Yes |
| keyword 0.398 vs LLM 0.374 (nuclei) | 500 templates with exactly one CWE, seed 17; LLM picks from top-12 shortlist | embedding 0.210; keyword→llm 0.404 | 12 items of 500 apart, ~1 standard error. **Say "the LLM did not beat keyword", not "keyword beats LLM".** Only 51 distinct gold classes. Template names often contain the class verbatim | Yes: `eval_classification.json` |
| titles: keyword 0.402, LLM 0.364, keyword→llm 0.474 | 500 writeup titles | — | Gold made with the same alias table keyword uses → keyword gets free wins. Nuclei is the fair set | Yes |
| shortlist R@12 0.45 / 0.53 | gold in top-12 embedding candidates | R@50 only 0.622 | — | Yes |
| LLM 83.1% / 68.7% when gold is on the shortlist | split of the same run | n = 225 / 265 | — | Yes |
| planner P@1 0.427 vs prior 0.325, random 0.095 | 400 held-out templates over 147 products, 20% per product hidden from evidence, seed 31 | likelihood-only 0.422 | "P@k" is really **hit@k** (one gold). Baselines reorder the **planner's own top-10**, so P@10 is 0.720 for all arms by construction; 112 of 400 cases never rank gold. Small prior leak, stated | Yes: `eval_planner.json`, identical to four places on 10-22 |
| 0.280 → 0.427 (smoothing fix) | — | — | 0.280 has no artifact or code in git | Docs only |
| cited-and-supported 5.1% → 32.6% | 30 fixed topics, 1 generation each at T=0; regex sentence split; Qwen judge on cited sentences; reweighted by 40 blind hand labels per prompt | baseline prompt 530 sentences, current 424 | Hand labels by **Claude**, not a human; judge = generator; vLLM output not bit-stable; κ from a stratified sample | Yes: `eval_faithfulness*.json`, `bountybrain/fixtures/faithfulness_handcheck*.json` |
| 54.5% of sentences cited (was 13.2%) | same | — | 45% still uncited | Yes |
| learning path complete 86.2% (25/29 topics); 16.2% of all 969 classes | `eval.py --learning-path` | — | 69.0% without borrowing; NoSQL injection unresolved | Yes: `eval_learning_path.json` |
| API 13/13, UI 7/7, live server 2/2 | `e2e.py` against the real graph, index, reranker and Qwen | — | One run. AppTest cannot drive the file uploader, so the Recon page is checked for rendering only. Checks prove "works", not "fast" | Yes: `artifacts/e2e.json` |
| 84 tests | `pytest`, no network or GPU | — | CI workflow has never run on GitHub (PROGRESS) | **Yes — I ran it: 84 passed** |
| `/learn` 9.1 s, `/search` 159 ms | e2e single calls | — | One sample each | Yes (`e2e.json`) |
| 174 s → 160 ms first search | one-off timing | — | Commit message only | Not in an artifact |

---

## 6. Prerequisite concepts

**BM25.** A lexical ranking function. It scores a document by query-term
frequency, damped so repeats matter less, and weighted by inverse document
frequency so rare words count more. It normalises for document length. Strong
on exact names, CVE ids and product strings.
*Source:* Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25
and Beyond" (2009).

**Dense retrieval / bi-encoders.** Encode query and passage separately into
vectors; rank by cosine similarity. Passages are embedded once, offline, so
search is fast. Good at paraphrase, weaker on rare exact tokens.
*Sources:* Karpukhin et al., "Dense Passage Retrieval" (2020); Reimers &
Gurevych, "Sentence-BERT" (2019).

**BGE and instruction prefixes.** BGE models were trained with a short
instruction on the query side only. Leaving it off, or putting it on passages,
hurts retrieval. That is why `embed_query` and `embed_passages` are separate.
*Source:* Xiao et al., "C-Pack: Packaged Resources to Advance General Chinese
Embedding" (2023), the BGE paper.

**Cross-encoder reranking.** Feed query and passage together through one
transformer, which outputs a relevance score. Much more accurate than a
bi-encoder because tokens attend across both texts. Too slow for the whole
corpus, so you rerank a shortlist.
*Source:* Nogueira & Cho, "Passage Re-ranking with BERT" (2019).

**Reciprocal Rank Fusion.** Each document gets Σ 1/(k + rank) over the lists
it appears in. Only ranks matter, so different score scales do not. k (often
60) damps the influence of the very top ranks.
*Source:* Cormack, Clarke & Büttcher, "Reciprocal Rank Fusion Outperforms
Condorcet and Individual Rank Learning Methods" (SIGIR 2009).

**Recall@k, MRR, nDCG.** Recall@k: is a relevant item in the top k? MRR:
mean of 1/rank of the first relevant item. nDCG: discounted gain by position,
divided by the best possible, so it lies in [0, 1].
*Source:* Järvelin & Kekäläinen, "Cumulated Gain-Based Evaluation of IR
Techniques" (2002); Manning, Raghavan & Schütze, *Introduction to IR*, ch. 8.

**Synthetic query generation for eval.** Ask an LLM to write a query a passage
answers; the passage becomes the known item. Cheap, but queries can copy
words from the passage and flatter lexical search.
*Sources:* Bonifacio et al., "InPars" (2022); Dai et al., "Promptagator"
(2022).

**RAG.** Retrieve passages, put them in the prompt, generate. Grounding
reduces made-up facts only if the model actually uses and cites the
passages, which is why faithfulness must be measured.
*Source:* Lewis et al., "Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks" (2020).

**Attribution / citation faithfulness.** A sentence is attributable if a
reader of the cited source would agree it says that. Measure citation rate
and support separately; a citation that points at an unrelated passage is
worse than none.
*Sources:* Rashkin et al., "Measuring Attribution in Natural Language
Generation Models" (2021); Gao et al., "Enabling LLMs to Generate Text with
Citations" (ALCE, 2023).

**LLM-as-judge and calibration.** An LLM can grade outputs, but it has biases,
including favouring its own generations. Check it against human labels and
report agreement, not just the judge's score.
*Source:* Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot
Arena" (2023).

**Cohen's κ.** Agreement between two labellers, corrected for agreement
expected by chance. 0 = chance, 1 = perfect; 0.6–0.8 is usually called
substantial.
*Source:* Cohen, "A Coefficient of Agreement for Nominal Scales" (1960).

**Instruction placement in long prompts.** Models use the start and end of a
long context better than the middle. Rules buried above 7k chars of sources
were ignored; moving them after the sources helped.
*Source:* Liu et al., "Lost in the Middle" (2023).

**Constrained / guided decoding.** At each step, mask tokens that would break
a JSON schema or grammar, so the output always parses. vLLM uses xgrammar.
*Sources:* Willard & Louf, "Efficient Guided Generation for LLMs" (2023);
Dong et al., "XGrammar" (2024).

**vLLM and PagedAttention.** vLLM stores the KV cache in fixed-size pages,
like virtual memory, which cuts waste and allows continuous batching. That is
why batch classification is fast.
*Source:* Kwon et al., "Efficient Memory Management for LLM Serving with
PagedAttention" (2023).

**Additive smoothing / Bayesian shrinkage.** Estimate P(class | product) as
(count + α·prior) / (total + α). With few observations you get the prior;
with many you get the data. α is a pseudo-count.
*Sources:* Zhai & Lafferty, "A Study of Smoothing Methods for Language
Models…" (2001); Manning et al., *Introduction to IR*, ch. 13.

**Data leakage in evaluation.** If the gold answer is also an input feature,
the score measures nothing. Hold it out of the evidence, not just the test
list.
*Source:* Kaufman et al., "Leakage in Data Mining" (KDD 2011 / TKDD 2012).

**Entity resolution / alias normalisation.** Map many surface forms ("IDOR",
"BOLA", "Insecure Direct Object Reference") onto one canonical entity.
Longest match first, so a specific class beats its parent.
*Source:* Christen, *Data Matching* (2012).

**Knowledge graphs vs relational storage.** A graph is a model, not a
database product. Typed relational tables can store one fine when traversals
are shallow; graph databases pay off on deep or variable-length paths.
*Source:* Hogan et al., "Knowledge Graphs" (ACM Computing Surveys, 2021).

**CWE, CAPEC, EPSS, CVSS.** CWE: weakness classes in a hierarchy (Pillar →
Class → Base → Variant). CAPEC: attack patterns, mapped to CWEs, with
execution flows. CVSS: severity. EPSS: probability a CVE is exploited in the
next 30 days.
*Sources:* cwe.mitre.org, capec.mitre.org; Jacobs et al., "Exploit
Prediction Scoring System (EPSS)" (2021).

---

## 7. Fifty likely questions

### Architecture and storage

1. **Why SQLite and not Neo4j?**
   The queries are shallow joins; the deepest walk is depth 3. The DB is
   38 MB, needs no server, and anyone can rebuild it. On a shared machine a
   server is one more port and process to manage. I never benchmarked Neo4j;
   that comparison is reasoning, not measurement.

2. *Hostile:* **"So it's not really a knowledge graph."**
   It is a typed graph: seven node types, typed edges, real traversals
   (taxonomy, attack chains, learning paths). The storage engine is
   relational. A graph is a data model. I also export GraphML and CSV for
   Gephi or Neo4j.

3. **Why one table per edge type instead of a generic edge table?**
   Each relation has its own semantics and columns, like provenance and
   confidence on document→weakness. Foreign keys keep edges valid. A query is
   a join on the right table, not a filter on a type string.

4. **What breaks first at 10× the data?**
   BM25: it is a Python linear scan per query over an in-memory list. Then
   the embedded Qdrant, which is meant for small local use. Then the NetworkX
   metrics, which copy the whole graph into memory. Move BM25 to a real index
   (OpenSearch or Qdrant sparse vectors) and Qdrant to a server.

5. **Why embedded Qdrant?**
   No server and no port on a shared box. The cost is an exclusive file lock:
   the API and an eval run cannot both open it.

6. **How do the API and UI share logic?**
   Both call plain functions in `agent/tools.py` that return JSON-able data.
   No translation layer.

7. **Is there an agent?**
   A small LangGraph flow exists (resolve → retrieve → labs → write), but the
   API and UI call the tools directly. It is a fixed pipeline, not an
   autonomous agent.

### Retrieval

8. **Walk me through a search.**
   Embed the query with the BGE prefix; top 40 from Qdrant. BM25 top 40 over
   the same chunks. Fuse with RRF k=60. Cross-encoder rescored the top 40;
   return the top k.

9. **Why hybrid?**
   Dense handles paraphrase; BM25 handles exact strings like product names
   and CVE ids. In this eval they mostly agree, so fusion added little:
   R@1 0.675 vs 0.665 dense, two queries out of 200.

10. *Hostile:* **"Fusion didn't help. Why keep it?"**
    R@5 went 0.885 → 0.910 and R@10 0.925 → 0.955, so it widens the candidate
    pool the reranker sees. It costs ~17 ms. But I agree the case is thin on
    this query set. Real user queries with CVE ids would probably favour BM25
    more; I have not measured that.

11. **Why is the reranker what pays?**
    Hybrid R@10 is already 0.955, so the right passage is usually in the
    pool. It is just not first. A cross-encoder reads query and passage
    together and orders them better. R@1 went 0.675 → 0.800, MRR 0.773 →
    0.861.

12. **Why RRF and not a weighted score sum?**
    Cosine and BM25 scores are on different scales, and BM25's scale shifts
    with the query and corpus. RRF uses ranks only and has no weight to tune.

13. **Why k=60? Did you tune it?**
    No. It is the standard value from the original paper, and it was in the
    first commit. The code comment says results are insensitive to it; I
    never tested that. With a reranker on top, I would expect little effect,
    because k mostly changes order inside the 40 the reranker re-sorts anyway.

14. **Why bge-small and bge-reranker-base?**
    Honestly, they were the defaults from the start, and I did not run a
    model comparison. They are small enough to share one GPU with a 14B LLM,
    and BGE v1.5 is a strong small English retriever. The one detail I did
    handle is BGE's query-only instruction prefix. Next step would be to try
    bge-base and a larger reranker on the same 200 queries.

15. **Why rerank 40?**
    Enough depth that the gold chunk is almost always present (R@10 is
    already 0.955). Not tuned. Latency grows roughly linearly with depth.

16. **How did you build the eval queries?**
    Sampled 50 chunks of ≥400 chars per document type, asked Qwen to write a
    practitioner query without copying distinctive words. Hit = any chunk from
    the source document.

17. *Hostile:* **"Your queries leak the answer."**
    Partly. The instruction is not checked mechanically, and I can see
    queries like "sharp printer vulnerabilities pierre kim 2024". That
    flatters BM25. Also the source chunk stays in the index. But all four arms
    see the same queries, so the comparison between arms stands. Absolute
    numbers are not what a real user would see.

18. **Why is recall a lower bound?**
    Only the source document counts. A different document that also answers
    the query is scored as a miss.

19. **Tell me about the nDCG bug.**
    The first run gave 1.113. The ideal assumed one relevant hit, but a
    document has several chunks, so several hits can be relevant. I normalise
    by all relevant hits found. Strictly, the ideal should count all relevant
    chunks in the collection, so mine is still slightly optimistic.

20. **Why never split code blocks when chunking?**
    In this corpus a payload or request trace is often the answer. Half a
    payload retrieves badly and reads as garbage when cited.

### Classification

21. **Why not let the LLM pick any CWE?**
    It cannot reliably recall ~970 numeric ids and will invent some. With a
    12-item shortlist, anything off-list is rejected, so invented ids cannot
    get through.

22. *Hostile:* **"A keyword table beat your LLM. Why use an LLM at all?"**
    On the fair nuclei set the LLM did not beat keyword: 0.374 vs 0.398,
    12 items of 500, within noise. Keyword is my default for bulk labelling.
    The LLM earns its place writing cited prose. Routing keyword first and
    the LLM only on abstentions was the best arm.

23. **What was confounded, and how did you catch it?**
    Only the keyword arm saw the 109 curated aliases. The graph never stored
    them, and the other arms shortlist from the stored aliases. I noticed when
    "IDOR" resolved to nothing in the UI. After persisting them and rebuilding
    the cached index, the LLM gained 0.084 on titles and 0.006 on nuclei.

24. **How do you know the re-run was paired?**
    Same seed, same 500 items, and the keyword arm's scores were
    bit-identical to August.

25. **Where is the ceiling?**
    The shortlist. Gold is in the top 12 only 45% / 53% of the time. When it
    is, the LLM is right 83% / 69%. Even top 50 holds gold only 62% of the
    time, so a bigger k alone won't fix it.

26. **How would you fix candidate retrieval?**
    Cheapest first: show the model each candidate's description, not just its
    name; rerank the top 50 classes with the cross-encoder already in the
    stack; embed class texts with aliases weighted up. Measure shortlist
    recall each time.

27. **Why report abstention and macro F1?**
    A classifier that abstains a lot can look accurate on what it answers.
    Keyword abstains on 35–52%. Macro F1 shows the LLM concentrates on common
    classes (0.105 vs keyword 0.216 on nuclei).

28. **What does "hierarchy ±2" mean exactly?**
    Correct if the prediction is the gold class or an ancestor or descendant
    within two ChildOf hops. Siblings do not count; the code follows directed
    paths only.

### Planner

29. **How does the planner score?**
    For each candidate CWE on the fingerprinted product: applicability (0.55),
    exploitability from EPSS or CWE likelihood (0.15), public exploits (0.05),
    bounty economics (0.25). Then it attaches CAPEC Explore/Experiment steps,
    detection templates to run, and matching disclosures.

30. **Explain the smoothing fix.**
    The old term was a log-scaled template count. It hit 1.0 for every class
    on a popular product, so the stack stopped mattering. Now it is
    (n + 25·prior) / (N + 25): a sparse product falls back to the global
    prior; a well-covered product is dominated by its own counts. P@1 went
    0.280 → 0.427.

31. *Hostile:* **"So your planner just counts templates per product."**
    In likelihood-only mode, yes: it ranks by smoothed per-product frequency.
    That is the point. It is a Bayesian estimate of what this product's next
    finding will be, and it beats the global prior by 0.10 P@1. The value
    terms are deliberately not about prediction.

32. **Why does the eval need a hold-out?**
    Gold is a signature→CWE edge, and the planner scores with those edges.
    Without hiding the gold template, it is graded on the fact it was given.
    I hide all 400 held-out templates from every case, so gold for product A
    cannot leak as evidence for product B.

33. *Hostile:* **"Your baselines only reorder your planner's top 10."**
    Correct. That isolates ordering, but it means P@10 is identical for all
    arms by construction, and the prior baseline benefits from the planner's
    candidate set. P@1 and MRR are the honest comparisons. Also, my "P@k" is
    really hit@k, since there is one gold class.

34. **Are the weights learned?**
    No, hand-set and never swept. The eval shows they beat the baselines; it
    does not show they are optimal. Learning them from the held-out split is
    an obvious next step, with a separate test split.

35. **Is predicting a held-out nuclei template the real task?**
    It is a proxy. A template is a known detection, not a bug a hunter will
    find. It is the best labelled signal available offline.

### Generation and faithfulness

36. **How do you know tutorials are faithful?**
    I measured it: deterministic sentence split, exact check of citation
    numbers, an LLM judge reading only the cited excerpts, calibrated against
    blind hand labels. 32.6% of sentences are cited and supported; 45% cite
    nothing.

37. *Hostile:* **"The judge is the same model. That's grading your own
    homework."**
    Yes, which is why I did not trust it alone. Its "supported" and
    "unsupported" calls held on every hand-checked case (20/20, 6/6). Its
    "partial" bucket was a mixed bag, so the population estimate reweights
    that bucket by the hand-checked share. A second judge model is on the
    to-do list.

38. *Hostile:* **"Who made the hand labels?"**
    Claude made them, blind to the judge's verdicts, against the excerpts
    only. I did not label them myself. They are committed with a one-line
    reason each, so anyone can audit them. A human second labeller would make
    the number much stronger.

39. **Did you tune the prompt on the test set?**
    No. Three prompt versions on 10 dev topics, disjoint from the 30 test
    topics. Froze v3, ran test once.

40. **What changed in the prompt?**
    The graph summary became the last numbered source, so its facts can be
    cited and checked. The citation rules moved after the sources, with an
    example, plus "leave the safety section uncited" and a reread step.

41. **Why is 45% still uncited?**
    Section structure invites generic sentences; the model fills them. Half
    of the hand-unsupported sentences are safety reminders it still tags with
    a source number. Next: mark uncited sentences in the UI, and regenerate
    once when a tutorial has no citations.

42. **Is generation deterministic?**
    Temperature 0, but vLLM output changes under batching. The same request
    produced a cited tutorial once and an uncited one the next time under the
    old prompt. Each topic is one sample.

### Systems and failures

43. **Tell me about a silent failure.**
    vLLM 0.27 ignored `guided_json` with no error. Every "constrained" call
    ran unconstrained. I found it by sending a haiku prompt with a schema and
    getting a haiku. Switched to `response_format`. The measured effect on
    accuracy was ≤0.002, but the claim in the docstring had been false.

44. **What were the vLLM traps?**
    On Python 3.11, flashinfer uses a 3.12-only annotation: needs
    `--enforce-eager` and a one-line patch. Triton cannot make its cache on
    GlusterFS. `ninja` must be on PATH. All of these failed after a
    10-minute weight load, so they looked like model problems. Also cap GPU
    memory at 40% on a shared card.

45. **What happens if the LLM is down?**
    `generate_tutorial` falls back to a deterministic summary of the sources.
    The graph, planner, Explore and Metrics pages need no LLM at all.

46. **How is the app tested?**
    84 unit and smoke tests with no network or GPU; the smoke tests render
    every Streamlit page and call every route on a fixture graph. `e2e.py`
    runs the real stack: 13/13 routes, 7/7 pages, live server 2/2. CI is
    configured but has never actually run on GitHub.

### Safety, scope and next steps

47. **Isn't this a hacking tool?**
    It never scans or sends a packet to a target. It reads recon files you
    made with your own tools on targets you are authorised to test. Its
    content is public MITRE, nuclei and Exploit-DB metadata. Labs point only
    at training platforms. The tutorial has a fixed "test safely, in scope"
    section.

48. *Hostile:* **"Recon parsing plus exploit counts still helps an attacker."**
    It ranks weakness classes and links public CAPEC steps and templates. It
    does not generate exploits or run anything. The same information is what
    bug bounty programs and nuclei already publish. The line I drew is: no
    network egress, no scanning, no target interaction.

49. **What would you do differently?**
    Measure before building: the faithfulness eval came months after the
    "every claim is cited" promise. Store aliases in one place from day one.
    Use Python 3.12 to dodge the vLLM traps. Add a real human labeller.
    Compare at least two embedders and rerankers.

50. **What comes next?**
    In order: flag uncited sentences in the Learn page; improve CWE candidate
    retrieval; fill learning-path gaps (labs for subdomain takeover,
    clickjacking, CRLF; split "NoSQL Injection, NoSQLi"); add a second
    signature source so the planner covers more stacks.

---

## 8. "It didn't work" / "I didn't do that"

How to say it: state the fact, give the number, say what you learned or what
you would do. Do not apologise and do not over-explain.

### 8.1 The LLM classifier
"It didn't beat a keyword table. On the fair set it's 0.374 against 0.398,
which is within noise. I dug in and found the shortlist only contains the
right class about half the time. So the fix is retrieval of candidates, not a
bigger model."

### 8.2 Tutorial faithfulness
"About a third of sentences are cited and supported. Forty-five percent cite
nothing. I raised it from five percent with a prompt fix tuned on a separate
dev set. The next step is to show the reader which sentences are uncited."

Also: the Learn page caption (`ui/pages/1_Learn.py:16`) still says "Every
claim is cited… Nothing is answered from memory." The README was corrected,
the caption was not. Fix it before anyone demos the app, or own it if asked.

### 8.3 Model choices
"I didn't compare embedders or rerankers. bge-small and bge-reranker-base
were the starting defaults and they worked. The ablation shows reranking
matters; it doesn't show these are the best models."

### 8.4 Graph statistics
- The metric graph adds 8 edge kinds and skips `signature_references_cve`
  (4,196 populated rows; `metrics.py:57–151`). The planner uses those edges
  (`planner.py:487`, `:552`), but the node/edge counts and component counts
  ignore them. The README explains the CVE↔exploit islands as "no nuclei
  template references them", but some islands exist only because this edge
  type was left out. Honest line: "The published counts cover eight of nine
  populated edge types; adding signature→CVE would add 4,196 edges and merge
  some of the small components."
- The `tools` table and four document→X edge tables are in the schema but
  empty. Don't claim tools are in the graph.
- `config.py` says the graph is ~150 MB; README says ~40 MB; the file is
  38 MB.

### 8.5 Evaluation shortcuts
- "Held-out passages" in retrieval means sampled; the source chunk is still
  indexed.
- Planner "precision@k" is hit@k.
- The 0.280 pre-fix planner number has no artifact.
- No significance tests anywhere. The nuclei keyword-vs-LLM gap and the
  hybrid-vs-dense R@1 gap are both within noise.
- An old commit message (`3b92ad2`) says errors were "near-tied once sibling
  classes count". The hierarchy metric does not count siblings, only
  ancestors and descendants.

### 8.6 Things not built
- The Learn page's "Retrieval mode" selector does nothing: `mode` is never
  passed to `local_rag_search` (`ui/pages/1_Learn.py:21–25, 49`).
- CI has never executed on GitHub.
- Planner weights and α were never swept.
- No screenshots, no user study, no real users.
- Writeup bodies were never fetched.

### 8.7 Authorship
The repo shows a first commit by another author (Tarunsai) and a Claude
co-author trailer on every later commit. If an interviewer asks "did you
write all of this?", a safe, true shape is: "I own the design and every
decision in it, and I can explain any line. The first scaffold commit came
from [whoever it was]. I built on it with Claude Code as a pair programmer,
and the commits say so." Adjust to what is actually true for you. Do not
claim you hand-labelled the faithfulness set.

---

## 9. Self-test exercises

Predict the effect first. Then read the answer. Do not commit any of these.

### Exercise 1 — RRF k
In `bountybrain/rag/hybrid.py:28`, change `RRF_K = 60` to `RRF_K = 1`.
What happens to `hybrid` and to `hybrid_rerank` on the 200-query eval?

<details><summary>Answer</summary>

With k=1, rank 1 in either list scores 0.5 and rank 2 scores 0.33, so the
top of each list dominates. `hybrid` R@1 will move toward whichever arm wins
each query's first slot; it could go up or down a little. `hybrid_rerank`
should barely change. The reranker re-sorts `fused[:40]`, and with two lists
of 40 the union is at most 80, so only the tail membership changes. This is
also the experiment the "insensitive to k" comment never ran.
</details>

### Exercise 2 — smoothing strength
In `bountybrain/recon/planner.py:71`, set `SMOOTHING_ALPHA` to `0`, then to
`1e9`. Predict the likelihood-only P@1 in each case.

<details><summary>Answer</summary>

Applicability ranks by n_c + α·prior_c (the denominator is shared, and the
top class is scaled to 1). With α=0 it is raw per-product counts: products
with few templates get noisy ties, so P@1 probably drops a little below
0.422. With α→∞ the prior swamps the counts, so the ranking becomes the
global-frequency order over the same candidates. Likelihood-only P@1 should
converge to about the `prior` baseline, 0.325. That shows what the per-product
evidence is worth.
</details>

### Exercise 3 — bring back the old bug
In `rank_weaknesses`, replace the applicability line (`planner.py:319`) with
`applicability = _log_scale(len(entry["signatures"]), 10)`.
Predict value-weighted P@1.

<details><summary>Answer</summary>

It should fall to around the documented 0.280, below the prior's 0.325.
Popular products have more than 10 templates for several classes, so they
all saturate at 1.0. The ranking is then decided by EPSS, exploits and
bounties, which do not depend on the stack. The exact figure will differ,
because the original saturation constant is not recorded.
</details>

### Exercise 4 — a bigger shortlist
In `bountybrain/ingestion/classifier.py:197`, change `k: int = 12` to `50`.
Predict the nuclei LLM accuracy. What else must you change?

<details><summary>Answer</summary>

The ceiling rises from 0.45 to 0.622. But 50 candidates make the choice
harder and the prompt longer, so accuracy when gold is present will drop
below 83%. Net effect is uncertain; RESULTS argues a bigger k alone won't
fix it. You must also change `LLM_SHORTLIST_K` in
`bountybrain/eval/classification.py:230`. Otherwise the "gold on shortlist"
split is computed at 12 while the model chose from 50, and that table
silently becomes wrong.
</details>

### Exercise 5 — split joined alternate terms
In `bountybrain/datasets/cwe.py:121–124`, split each alternate term on `", "`
and `" / "` before appending. Rebuild. Predict three effects.

<details><summary>Answer</summary>

1. "NoSQL injection" resolves to CWE-943, so all 30 learner topics resolve
   (was 29).
2. Terms like "BOLA" and "Non-Persistent XSS" start resolving, because CWE-639
   and CWE-79 store them as one joined string today. The alias count rises
   above 1,389.
3. The keyword classifier may fire more often, so its abstention drops and
   its numbers change. The embedding/LLM arms change **only if** you delete
   `data/corpus/cwe_candidates.npz`; otherwise `build_candidate_index()`
   returns the stale cache and the change is a silent no-op for them. That is
   the same trap that hid the alias confound.
</details>

---

## Appendix: commit map

| date | commit | what |
|---|---|---|
| 08-16 | `4c8c10b` | First commit (author: Tarunsai). Synthetic corpus, scaffold |
| 08-16 | `40196d6` | Real graph from 8 datasets; planner; hybrid ablation; no web search |
| 08-16 | `3d4bd23` | Per-character CVEs; per-host plans |
| 08-16 | `63ff739` | Guided-decoding fallback; eval caveats; planner leak stated |
| 08-16 | `3b92ad2` | nDCG fix; LLM arms run; RESULTS and PROGRESS |
| 08-16 | `a88eeb1`, `9f9c201` | flashinfer patch; ninja on PATH |
| 08-18 | `6187c2a` | Persist curated aliases (the confound fix) |
| 08-18 | `eeae180` | CanFollow chains; related reports; labs by CWE id |
| 10-22 | `9110924` | `--enforce-eager` by default below 3.12 |
| 10-22 | `b049272` | Core structure into `graph_metrics.json` |
| 10-22 | `ae21e83` | Fair classification re-run; shortlist recall; routing |
| 10-22 | `5cc8d85` | Exact alias before substring |
| 10-22 | `83090cd` | `response_format` instead of ignored `guided_json` |
| 10-22 | `f78e2a6` | Learning path; faithfulness eval (baseline) |
| 10-22 | `462c8af` | Citing prompt: 5.1% → 32.6% |
| 10-22 | `e3f81dc` | e2e run; startup warm-up; Metrics page fix |
| 10-22 | `11a9e1d` | Retrieval and planner reproduced; summary merge |
| 10-22 | `4267980` | Docs refresh; CI installs smoke-test deps |
