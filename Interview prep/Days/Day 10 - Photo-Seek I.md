---
type: study-day
day: 10
date: 2026-11-17
tags: [interview, day, photo-seek, retrieval]
---
# Day 10 (Tue Nov 17): Photo-Seek I: retrieval and the query parser

Part of [[Interview plan (Nov 2026)]]. Repo `photo_seek`, **branch `upgrade-2026-10`** (HEAD `aa5d476`; `main` is the old scaffold).
Read it with `git show upgrade-2026-10:<path>` or a separate worktree; don't switch the repo's branch. About 7 h.
Prep pack: [[Photo-Seek - prep]] (diagram and Trace 1 in §2, stories 4.4–4.5, numbers §5).

**Code walkthrough:** [[Day 10 - code walkthrough]]: one small figure per pipeline, the exact code to read in order, and what each piece does.

**Resume bullet (part 2 of 2, both resumes):** "Engineered an on-device multimodal retrieval engine combining **FAISS vector search, CLIP embeddings**, and
**LLM-based query routing** to translate natural-language queries into structured, **fully offline** photo search."
Faces (part 1) are tomorrow ([[Day 11 - Photo-Seek II and emotion steering]]).

**Wording to fix** (prep pack §5.1, §5.4):
- "**on-device**": everything was measured on the shared H200 server. Say "local" or "self-hosted".
- "**LLM-based query routing**": the LLM *parses* the query into filters; it doesn't route between backends. Say "LLM query parsing".
- "fully offline" holds at the process level (an audit hook saw 6 socket events, all to 127.0.0.1), not at the OS level.
- **84% counts queries, not fields:** on 84 of 100 hand-written queries all four filters (people, date_from, date_to, place) were right at once.
- **The env is broken** (`import numpy` fails with `libmkl_gf_lp64.so.3`), so no number can be re-run today. Last recorded tests: 156 passed (the README says 157).

**By tonight you can:**
- explain CLIP's contrastive training and why text and images share one space;
- choose between FAISS Flat, IVF, HNSW and PQ, and defend Flat here;
- trace "photos of mom at the beach last summer" from the parser to the results;
- defend R@1 39.4 and 84% vs 58%, with their setups.

## Schedule

| # | Time | Kind | What |
|---|---|---|---|
| 0 | 15 min | recall | Day 9: retrieval heads, Recall@k, IMP |
| 1 | 30 min | read | Prep pack "Read these first", §1 pitch, §2 diagram |
| 2 | 50 min | theory | CLIP (A1) |
| 3 | 40 min | code | `embed/clip.py`, `embed/base.py` (B1) |
| 4 | 40 min | theory | FAISS index types (A2) |
| 5 | 30 min | code | `index/faiss_store.py` (B2) |
| — | lunch | | |
| 6 | 25 min | theory | Structured LLM output; Ollama (A3) |
| 7 | 90 min | code | `search/query.py`, `search/router.py`, `search/engine.py` (B3) |
| 8 | 30 min | skim | Ingest, catalog, config, CLI, eval scripts (B4) |
| 9 | 45 min | rebuild | A tiny CLIP + FAISS search with a SQL prefilter (C) |
| 10 | 20 min | trace | Trace 1, by hand (D) |
| 11 | 40 min | practice | The questions (E) |

## A. Theory

### A1. CLIP (block 2, 50 min)

**Read:** Radford et al., *CLIP* (2021), §2.1–2.3 and Figure 1. About 35 min. **Optional:** OpenAI's CLIP blog post.
- Two encoders (image ViT, text transformer). For a batch of N pairs, the N×N cosine-similarity matrix is trained with a **symmetric cross-entropy**: each image should pick its own caption and vice versa. That InfoNCE loss pulls matching pairs together in one shared space.
- **Zero-shot retrieval:** embed the query text, take cosine similarity against all image embeddings, rank.
- Your model: **open_clip ViT-B-32, `laion2b_s34b_b79k`**, 512-d. You didn't train it; R@1 39.4 on COCO-5k is off-the-shelf CLIP and should sit near open_clip's published number.

### A2. FAISS (block 4, 40 min)

**Read:** the FAISS wiki page *"Guidelines to choose an index"* (github.com/facebookresearch/faiss/wiki). About 20 min. Optional: the Pinecone *"Faiss: The Missing Manual"* chapters on IVF and HNSW.
- **Flat (exact):** brute-force inner product. O(N·d) per query. At 5k–100k vectors it takes milliseconds.
- **IVF:** k-means into lists; probe `nprobe` lists. Approximate; needs training.
- **HNSW:** a layered proximity graph. Fast and high recall; more memory.
- **PQ:** compress vectors into codes. Saves memory, loses accuracy.
- **Your choice:** `IndexIDMap2(IndexFlatIP)`, inner product on unit vectors = cosine. Exact, needs no training, supports deletes by ID, and is fast enough for one person's library. **Prefiltering:** `IDSelectorBatch(candidates)` makes FAISS score only the IDs the SQL filters allow.

### A3. Structured output (block 6, 25 min)

**Read:** the Ollama docs on **structured outputs** (`format` = a JSON schema). 10 min.
- The schema constrains decoding, so the model can only emit valid JSON of that shape. Values can still be wrong (story 4.5: dates).
- Ollama is a local server (`127.0.0.1:11534` here) running `qwen2.5:7b` (4.4 GB quantised). Temperature 0, 200 tokens max.
- **Why hybrid:** the LLM is better at people (97 vs 83) and places (95 vs 85); the rules are better at dates (87 vs 82). So the rules' dates win whenever the rules match a date phrase.

## B. Code

### B1. CLIP wrapper (block 3, 40 min, deep)

- `src/photoseek/embed/clip.py`: **64 `embed_images`** (batching, normalisation), **100 `embed_texts`**.
- `embed/base.py`: the `Protocol` pattern. Why a Protocol and not an abstract base class? (*Structural typing: tests can pass a fake embedder without inheriting.*)

### B2. The index (block 5, 30 min, deep)

`index/faiss_store.py` (184 lines): **81 `add`**, **111 `search`** (the `IDSelectorBatch` prefilter), 147 `save` (write a temp file, then rename, for atomicity).

### B3. Parser and engine (block 7, 90 min, deep)

- `search/query.py`: `StructuredQuery`; 82 `from_dict` (bad dates become `None`).
- `search/router.py` (452 lines):
  - **81 `RuleRouter`**: 95 `route`, **114 `_extract_dates`**, 175 `_season_year` ("last summer" in November = this year);
  - **250 `OllamaRouter`**: 261 the JSON schema, 314 `route`, 332 the fallback to rules, 339 `_generate` (today's date goes into the prompt);
  - **366 `HybridRouter`**: 383 `route`, **386** rule dates override the LLM's;
  - 391 `build_router`; 424 `_extract_json`.
- `search/engine.py` (225 lines):
  - **78 `run`**; 84–90: an unknown place moves into the CLIP text (story 4.4);
  - **109 `candidate_ids`**: the SQL; 127–130: `date_to` + 1 day with `<`, because `taken_at` has a time part;
  - **150 `resolve_people`**: names → identity IDs, following `merged_into`; one `EXISTS` per person (all must appear);
  - 178 `_ensure_models` (lazy load); 192 `_place_known`; 207 `_hydrate`.

### B4. Skim (block 8, 30 min)

`config.py` (119 `Config.load`: YAML merged key by key), `catalog/db.py` (57 `Catalog`, SQLite WAL at 83; tables: photos, faces, identities, embeddings, corrections), `ingest/scanner.py` (89 `scan`, 50 BLAKE2b hash, 119 skip by path + mtime, 112 `min_bytes`), `ingest/exif.py` (53), `cli.py` (315 `main`).
Eval scripts, metric code only: `eval/retrieval.py`, `eval/router.py`, `eval/offline.py`.

## C. Rebuild it yourself (block 9, 45 min)

About 70 lines; a CPU is enough:
1. Load `open_clip` ViT-B-32 (or any small CLIP from Hugging Face). Embed 50 local images and 5 text queries; L2-normalise.
2. Build `faiss.IndexIDMap2(faiss.IndexFlatIP(512))`, add the images with your own integer IDs.
3. Put fake metadata in SQLite (date, place). Write `search(text, date_from, date_to)` that gets the candidate IDs by SQL, then searches FAISS with `IDSelectorBatch` via `SearchParameters`.
4. Check that an empty candidate set returns nothing, and that cosine scores match `img @ txt.T`.

## D. Trace (block 10, 20 min)

Do prep pack §2 Trace 1 from memory on paper: `photoseek search "photos of mom at the beach last summer" --router hybrid`, today = 2026-11-05.
Write: what the LLM returns; what the rules return for "last summer"; which dates win; what happens to "beach"; the SQL; what happens if nobody is named "mom".

## E. Questions

1. Explain CLIP's training objective. Why can text search images at all?
2. Why FAISS Flat and not HNSW or IVF? At what library size would you switch?
3. How do metadata filters combine with vector search? Pre- or post-filtering, and why?
4. What does the LLM do in your system? Why isn't "routing" the right word?
5. Why a hybrid parser? Give the per-field numbers.
6. How did you measure 84%? What is the 95% confidence interval at n = 100? (*About ±7–9 points.*)
7. Did you design the hybrid after seeing the gold-set results? (*Answer truthfully; the hybrid commit predates the eval commit, but no doc says what you looked at.*)
8. How do you know it's fully offline? (*Audit hook plus `ss` sampling; process level.*)
9. R@1 39.4 on COCO: is that your contribution? (*No: zero-shot CLIP. Your contribution is the system around it.*)
10. "A dog on the beach" once returned nothing. Why? (*Story 4.4.*)
11. Hostile: "On-device, but every number is from an H200?"

## My notes and fumbles

- 
