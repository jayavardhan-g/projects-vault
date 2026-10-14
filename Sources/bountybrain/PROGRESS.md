# Progress and handover

Status as of **2026-08-16**. Written so a fresh session can pick this up
without re-deriving context. Metrics live in [RESULTS.md](RESULTS.md);
environment gotchas in [SETUP.md](SETUP.md).

---

## Where the project stands

**Working end to end.** Corpus → knowledge graph → retrieval index →
evaluation → UI. Every headline number in the README is regenerated from
`artifacts/` by `report.py`, so the docs cannot drift from the code.

| area | state |
|---|---|
| Dataset fetchers (8 sources) | done, licensed, tested |
| Knowledge graph (112k nodes / 97k edges) | done, metrics computed |
| Entity resolution (1,389 aliases, 93.1% coverage) | done, guarded by a test |
| Chunking + hybrid retrieval (4 modes) | done, ablated |
| Recon planner + per-host plans | done, beats baseline |
| Retrieval evaluation | done — 200 queries |
| Classification evaluation | done — negative result, see below |
| Planner evaluation | done — 400 held-out cases |
| Streamlit UI (5 pages) | written, **not yet run end-to-end** |
| FastAPI | 8 endpoints smoke-tested via TestClient, all 200/404 as expected |
| Tests | 60 written (44 + 16 new graph-query tests), no network/GPU needed |
| Lint | ruff clean |
| CI | GitHub Actions workflow added, never executed |

**Not done:** faithfulness/citation evaluation; the UI and API have never been
launched against a browser or client.

---

## Git

15 commits on `main`, 14 **not pushed** — there are no push credentials on this
machine (no SSH key, no credential helper), and pushing is the user's job.

```bash
cd /data/jayavardhan/projects/bountybrain
git log --oneline        # review
git push -u origin main  # remote already set to jayavardhan-g/bountybrain
```

Remote is already `https://github.com/jayavardhan-g/bountybrain.git` and the
repo is currently empty, so the first push will need `-u origin main`.

Commit identity is configured as `Jayavardhan <jayavardhan.gollapalli@gmail.com>`.

---

## Rebuilding from scratch

Nothing derived is committed except the small JSON metrics, so a clean machine
needs the full pipeline. Steps 1–3 need **no GPU and no LLM**.

```bash
conda activate bountybrain                 # python 3.11.15
export TMPDIR=/path/you/own/tmp            # root fs is full on this box
python fetch_data.py                       # ~1 GB, ~5 min
python build_graph.py --rebuild            # ~2 min, writes artifacts/graph_metrics.json
python index_corpus.py                     # ~1 min on GPU
python serve_llm.py                        # ~15 min to first token (see SETUP.md)
python eval.py --all
python report.py
```

`data/` is gitignored in full (raw datasets, SQLite graph, Qdrant store, chunk
cache). `artifacts/*.json` is committed; `artifacts/*.graphml` and the node/edge
CSVs are not.

---

## What to do next, in priority order

### 0. Re-run the pipeline — REQUIRED, code has changed

Three fixes landed that change stored data. Until this runs, the graph on disk
predates them and RESULTS.md §3 carries a warning banner.

```bash
export TMPDIR=/path/you/own/tmp
python build_graph.py --rebuild                 # persists the curated aliases
rm -f data/corpus/cwe_candidates.npz            # SEE BELOW — easy to miss
python index_corpus.py
python eval.py --classification                 # the arm the fix affects
python report.py
pytest -q                                       # 60 tests, no GPU/network
```

**The `rm` is not optional.** `build_candidate_index()` returns the cached
`.npz` unless `force=True`, and that cache was built from the pre-fix text, so
skipping it makes the whole fix a silent no-op that looks like it worked.

### 1. Diagnose the classification result — now that the comparison is fair

The previous run was **confounded**, not merely negative: `classify_keyword`
loads the alias table from the CWE catalogue on disk and therefore saw all 109
curated community terms, while the embedding and LLM arms shortlist from
`build_candidate_index()`, which embeds each class from the *stored*
`aliases_json` — and the builder never persisted the curated layer. The
baseline was handed vocabulary the other two arms were denied.

The builder now persists it, so step 0 produces the first honest comparison.
Only then are these worth doing:

- Check whether the embedding shortlist (k=12) even *contains* the gold class.
  If shortlist recall is low, the ceiling is the retriever's fault, not the
  model's. Measure this first — it is cheap and it decides everything
  downstream.
- Try feeding the model the CWE *descriptions* of the shortlisted candidates.
  It currently chooses from bare `CWE-{id}: {name}` lines, which is the same
  impoverishment one level up.
- Dump the LLM's disagreements with gold and read 50 by hand — the
  hierarchy-aware score was nearly tied on nuclei (0.440 vs 0.444), suggesting
  sibling confusions rather than category errors.
- Consider an ensemble: keyword when it fires, LLM only on abstentions.
  Keyword abstains 35–52% of the time and is 0.62–0.85 accurate when it does
  answer, so that routing could plausibly beat both.

### 2. Build the faithfulness evaluation

The tutorial prompt demands a citation per claim, and nothing verifies it.
Suggested: split generated tutorials into claims, and for each check whether
the cited passage entails it (LLM-judge, plus a cheap lexical-overlap floor).
Report citation-support rate and hallucinated-citation rate. This is the last
gap in the "grounded, never from memory" claim.

### 3. Actually run the UI

The API has now been exercised (`/health`, `/stats`, `/weakness/{id}`,
`/weakness?q=`, `/product/signatures`, `/recon/parse`, `/chains` all return
real data via `fastapi.testclient`). The Streamlit UI still has never been
launched against a browser.

```bash
streamlit run bountybrain/ui/streamlit_app.py     # then click all 5 pages
uvicorn bountybrain.api.app:app --reload          # then hit /docs
```

Worth checking specifically, because they are new and only unit-tested:
Learn → "Real disclosures" expanders (report-to-report links), Learn → "Where
to practise this", Attack Chains on SSRF and XSS (was empty for every topic
before), and Recon Planner → "Disclosed reports matching this recon".

Note: embedded Qdrant takes an **exclusive lock** on `data/qdrant`, so the UI
and an eval run cannot both hold it. Run one at a time, or set `QDRANT_URL`.

Screenshots of the Learn and Recon Planner pages would materially help the
README — it currently describes the UI without showing it.

### 4. Improve coverage where the planner is thin

Only 17.1% of weakness classes have a detection signature, so the planner is
strong on well-represented stacks (WordPress, GitLab) and silent elsewhere.
Options: map CAPEC prerequisites onto recon-observable conditions so classes
without nuclei templates can still be proposed; or add a second signature
source.

### 5. Optional polish

- Tune `SMOOTHING_ALPHA` (currently 25) and the planner weights against the
  held-out evaluation — they are hand-set and never swept.
- Populate `technique_variant_of` and the patch table (modelled, unused).
- Add Louvain community detection over the co-occurrence graph; `python-louvain`
  is already installed and the metrics module has a natural slot for it.

---

## Design decisions worth not re-litigating

Recorded so they are not accidentally reversed:

- **Bulk-download only, never scrape.** The point is that anyone can reproduce
  the metrics. HackerOne was explicitly ruled out by the user.
- **pentester.land metadata only.** The 6,421 linked articles are on ~500
  third-party blogs and are deliberately not crawled.
- **SQLite + embedded Qdrant by default.** No servers, no ports to collide with
  on a shared machine; `git clone && build` reproduces the numbers anywhere.
- **No web-search tool.** An earlier Brave-backed version returned nothing
  silently without an API key, so the agent would claim to have searched and
  ground answers in nothing. Removed rather than left as a stub. There is no
  Brave key available.
- **Two GPUs maximum** (user constraint): one vLLM, one embeddings/reranker.
  `serve_llm.py` requests 40% of a card, not vLLM's 0.9 default.
- **Code fences are never split when chunking.** In this corpus a payload *is*
  the answer.
- **Negative results stay in the README.** The planner's earlier loss to the
  prior, and the LLM classifier's current loss to keyword, are both documented.

---

## Known environment traps

Full detail in [SETUP.md](SETUP.md); the short version, because each of these
cost real time:

1. Root filesystem is full — export `TMPDIR` somewhere you own before any pip
   install or archive extraction.
2. vLLM on Python 3.11 needs `--enforce-eager` **and** a one-line flashinfer
   patch (`array.array[int]` is 3.12-only). Python 3.12 avoids both.
3. Triton/Inductor caches cannot be created on GlusterFS — point them at local
   disk. `serve_llm.py` now does this automatically.
4. `ninja` must be on `PATH` or JIT compilation fails *after* the ten-minute
   weight load. `serve_llm.py` now handles this too.
5. Loading a 14B checkpoint from network storage takes ~11 minutes at ~83s per
   shard. It is not hung.

Only `/data/jayavardhan` and `/mnt/warm_storage/jayavardhan` belong to the
user on this shared machine. Nothing else may be written to.
