# Project report: Photo-Seek, an offline photo organiser (face clustering with active learning, plus natural-language search)

> **For the LLM reading this.** This report describes a self project by G Jayavardhan (M.Tech CSE, IIT Bombay), Summer
> 2026, upgraded in October 2026. He must defend it in interviews. It was written from the repository
> `/data/jayavardhan/projects/photo_seek`, **branch `upgrade-2026-10`** (HEAD `aa5d476`; GitHub `main` still shows the
> August scaffold until that branch is pushed): source, README results, eval scripts and the JSON results.
>
> Labels:
> - **[CODE]**: `file:line` or function;
> - **[MEASURED]**: from `eval/results/*.json` and the README results block;
> - **[DOC]**: a reason written in the repo;
> - **[RECON]**: the report writer's reasoning.

---

## 1. Summary

Photo-Seek organises a personal photo library **entirely on the user's machine**. It has two pipelines over one SQLite catalog.

1. **Faces → people.**
   - **SCRFD** detects faces with 5 landmarks; they are aligned and embedded with **ArcFace** (insightface `buffalo_l`, 512-d).
   - Embeddings are clustered into identities (HDBSCAN or average-linkage agglomerative on cosine distance).
   - An **active-learning loop** asks the user about the faces it is least sure of: uncertainty or margin sampling, with random as the baseline.
   - Corrections (confirm, reject, merge, split, name) are recorded append-only and turned into **constraints**: pinned faces and rejected pairs. An incremental `refine()` updates the clustering.
   - A small **adapter head** (a 512×512 linear map on the *frozen* ArcFace embeddings, trained with a supervised contrastive loss and pulled toward the identity) is trained from the corrections. **Measured, it adds nothing.**
2. **Natural-language search.**
   - Photos are embedded once with **CLIP** (open_clip ViT-B-32, `laion2b_s34b_b79k`) into an exact **FAISS** `IndexFlatIP`.
   - A **query router** turns "photos of mom at the beach last summer" into a `StructuredQuery` (text, people, date_from/date_to, place, limit). Three routers: rule-based, local LLM (**qwen2.5:7b via Ollama**, JSON-schema-constrained), and a **hybrid** (LLM for people/place/text, rules for dates).
   - Structured fields become **SQL filters first**; CLIP then ranks only the surviving photos.

Other pieces: EXIF ingest (dates, camera, GPS, orientation), content-hash dedupe, a CLI (`photoseek ingest | index | faces | label | search | ui`), a local FastAPI web UI, and an **offline proof** (socket audit hook). 157 tests (model-backed parts behind `Protocol`s with fakes).

Project facts:
- Code: about 4,200 lines of Python in `src/photoseek/`. 31 commits by Jayavardhan: 2026-08-18 scaffold, 2026-10-22 implementation and evaluation, 2026-11-04 docs.
- Environment: mamba env `photo-seek`, Python 3.11.17, pytorch 2.13.0, open-clip-torch 3.3.0, faiss 1.14.3 (CPU), onnxruntime 1.30.0, insightface 1.0.1, scikit-learn 1.9.1, Ollama 0.21.0.

---

## 2. Resume bullets vs evidence

| Bullet | Verdict | Evidence |
|---|---|---|
| "Built an active-learning face-clustering pipeline for offline photo organization, **fine-tuning ArcFace embeddings on user correction signals** to progressively refine identity clusters across 1,000+ photos" | **Partly wrong, so reword it.** | (a) **ArcFace is never fine-tuned**: the backbone is frozen by design [DOC `faces/head.py`]. (b) The trained **adapter head adds nothing** [MEASURED]: +0.00 BCubed F1 on LFW after 300 reviews, **−0.07** on CPLFW after 1,000 (uncertainty with vs without the head). (c) What works is the **sampler and the constrained refine**: on CPLFW, 1,000 margin-sampled reviews cut clustering error 4.09% → 2.43% (**−41%**), against −7% for random reviews. (d) "1,000+ photos": evaluation used LFW (5,985 photos) and CPLFW (11,435 photos). **Honest wording:** "active-learning loop whose uncertainty/margin sampling finds misassigned faces, cutting clustering error 41% after 1,000 reviews vs 7% for random; a contrastive adapter head on frozen ArcFace added no accuracy." |
| "Engineered an on-device multimodal retrieval engine combining FAISS vector search, CLIP embeddings, and LLM-based query routing to translate natural-language queries into structured, fully offline photo search" | **Supported.** | CLIP + exact FAISS: COCO-5k R@1 **39.4** (pretrained CLIP's own zero-shot quality, not trained by him). Hybrid router: **84%** of 100 gold queries get every filter right, vs 58% for rules and 73% for the LLM alone. Offline: audit hook saw 6 socket events over 3 queries, all to 127.0.0.1. |

---

## 3. Background (teach first)

### 3.1 CLIP (Radford et al., 2021)
- An image encoder and a text encoder trained contrastively on (image, caption) pairs, so matching pairs have high cosine similarity in a **shared embedding space** (here 512-d).
- Text-to-image search = embed the query text, then find the nearest image vectors.
- **ViT-B-32** is a Vision Transformer, Base size, with 32×32 patches. `laion2b_s34b_b79k` is the OpenCLIP checkpoint trained on LAION-2B (34B samples seen).
- CLIP cannot represent "last summer" or "mom", which is why structured filters exist.

### 3.2 FAISS index types
- **Flat (exact):** brute-force inner product or L2. O(N·d) per query; at personal-library scale it takes under 1 ms.
- **IVF:** k-means coarse quantiser; search only `nprobe` lists. Approximate; needs training.
- **HNSW:** a navigable small-world graph. Fast and high-recall, more memory.
- **PQ:** product quantisation compresses vectors into codes; lossy.
- **Inner product on L2-normalised vectors = cosine similarity.**
- `IndexIDMap2` maps FAISS rows to your own ids (here, catalog row ids).

### 3.3 Face recognition
- **SCRFD** (Sample and Computation Redistribution for Face Detection): an efficient detector that outputs boxes and 5 landmarks (eyes, nose, mouth corners).
- **Alignment:** a similarity transform of the 5 landmarks onto a canonical template gives a 112×112 crop (`face_align.norm_crop`).
- **ArcFace** (Deng et al., 2019): trained with an **additive angular margin** loss, i.e. the target logit is s·cos(θ + m). Same-person embeddings are tight on the hypersphere, different people are separated by an angular margin. Output is 512-d. `buffalo_l` = SCRFD-10G + ArcFace ResNet-50 (WebFace600K).

### 3.4 Clustering without knowing the number of people
- **k-means** needs k. Unknown here, so it is ruled out.
- **Agglomerative (average linkage):** merge the two clusters whose average pairwise distance is smallest; stop when it exceeds a threshold.
- **HDBSCAN:** density-based; finds clusters of varying density; labels low-density points as **noise** (blurry background faces stay unassigned instead of polluting someone).

### 3.5 Clustering metrics
- **BCubed precision/recall:** for each item, the fraction of its predicted cluster that shares its true label (precision), and the fraction of its true class that is in its predicted cluster (recall), averaged over items. F1 combines them.
- **Pairwise F1:** precision and recall over same-cluster pairs.
- **NMI:** normalised mutual information between predicted and true labelings.

### 3.6 Active learning
Ask labels for the most informative items:
- **uncertainty:** far from the assigned centroid;
- **margin:** nearly equidistant between the two nearest clusters;
- **random:** the baseline.

### 3.7 Supervised contrastive loss (Khosla et al., 2020)
For each anchor, pull together all positives (same label) and push away the others, using softmax over cosine similarity divided by a temperature.

### 3.8 Structured LLM output
Ollama's `/api/generate` accepts a JSON-schema `format`, so decoding is **constrained** to valid JSON matching the schema. The result is still validated in code, with a rule fallback.

---

## 4. Architecture

```
photos on disk
   │ ingest/scanner.py   walk roots, filter extensions/min size, SHA content hash → dedupe
   │ ingest/exif.py      taken_at, camera, GPS, orientation
   ▼
catalog/db.py  SQLite (WAL, foreign keys): photos, faces, identities(merged_into, name), corrections (append-only), embeddings
   ├──────────────► embed/clip.py (open_clip ViT-B-32, L2-normalised) → index/faiss_store.py (IndexIDMap2(IndexFlatIP))
   └──────────────► faces/detect.py (SCRFD) → embed/arcface.py (norm_crop + ArcFace 512-d)
                          → faces/cluster.py (seeded + HDBSCAN/agglomerative; incremental refine)
                          ↔ faces/active.py (sampling, record corrections, refine) ↔ faces/head.py (adapter)
search/router.py   rule | ollama | hybrid  →  search/query.py StructuredQuery
search/engine.py   place known? → SQL candidate ids (dates, place LIKE, EXISTS face of identity) → CLIP text vector → FAISS search(subset)
cli.py, ui/app.py  (FastAPI on 127.0.0.1:8765)
```

---

## 5. Search pipeline (code walk)

### 5.1 `StructuredQuery` (`search/query.py`)
- Fields: `text`, `people`, `date_from`, `date_to` (inclusive), `place`, `limit = 40`, `raw`.
- **It is the contract between query understanding and search.** Rule and LLM routers produce the same object, so they are interchangeable [DOC].
- `from_dict` drops unknown keys and turns malformed dates into None, because LLMs occasionally return both.
- Why split fields at all [DOC]: handing the whole sentence to CLIP lets "last summer", which it cannot represent, pull the embedding around.

### 5.2 `RuleRouter` (`search/router.py`)
- **Dates** (checked first):
  - "last/this/past <season>", using meteorological seasons. Winter Y = Dec Y−1 to Feb Y. "Last summer" = the most recent *completed* summer (`_season_year`).
  - "<season> YYYY", "June 2024", "last/this year", "last month", "last week" (the previous Monday–Sunday), "yesterday", "today", "in/from/during YYYY".
- **People:** quoted names; "of/with/featuring/showing X (and Y)" where X is a kinship word or **Capitalised** (so "photos of sunsets" doesn't become a person); bare kinship words ("mom", "baby"…).
- **Place:** "at/in/near/around/on (the) X".
- Whatever is left becomes `text`, after `_clean` removes stopwords and dangling connectives.
- **Design rule [DOC]:** "Deliberately conservative." A wrong filter silently removes the right photo, while a missed filter only leaves the phrase for CLIP.

### 5.3 `OllamaRouter`
- One `urllib` POST to the project's own Ollama server (`127.0.0.1:11534`). No client library.
- The **schema** requires text, people[], date_from, date_to, place.
- The **prompt** gives today's date and weekday, rules for each field (seasons, "last week", "since/before", single days), and 3 few-shot examples dated to a different "today".
- The output is still parsed (`_extract_json` scans for the outermost braces) and validated. **Any failure falls back to `RuleRouter`**, and `last_fallback`/`last_error` are recorded.

### 5.4 `HybridRouter`
Runs both routers. If the rules found a date, **their dates replace the LLM's**. Otherwise the LLM's dates stand, which covers holidays the rules can't parse. People, place and text come from the LLM.
**Why [DOC + MEASURED]:** the 7B model's weak spot is date arithmetic:
- it reads single days as half-open ranges (Christmas → 25–26 Dec);
- it puts "last summer" a year early;
- it invents dates for undated events.

### 5.5 `SearchEngine.run` (`search/engine.py`): filter, then rank
1. If the query is empty, return nothing.
2. **Place handling:** there is no reverse geocoding yet, so no photo has a place label. If no photo's `place` matches (`_place_known`), the place is **folded back into `text`** for CLIP and `place_as_text` is recorded. "At the beach" becomes visual text instead of a filter that would remove everything.
3. `candidate_ids`: build SQL. `taken_at >= date_from`, `taken_at < date_to + 1 day` (taken_at includes a time), `place LIKE`, and one `EXISTS (face of identity)` clause per person, so multiple people intersect.
   - **None vs empty set:** no filters → None (search everything); filters matched nothing → `set()` (return nothing).
   - An **unknown person name returns nothing** rather than being dropped, so typos are visible.
   - `resolve_people` follows the `merged_into` chain with cycle protection.
4. A pure-filter query (no text) returns photos by date and skips CLIP.
5. Otherwise: embed the text with CLIP and call `FaissStore.search(vector, limit, subset=candidates)`.
6. `_hydrate` attaches path, date and named people. It skips ids the catalog no longer has, because the index is derived and can lag.
- **Why filter first [DOC]:** a global top-k ranked before filtering can be filled entirely by photos that fail the filters, leaving nothing.

### 5.6 `FaissStore` (`index/faiss_store.py`)
- `IndexIDMap2(IndexFlatIP(512))`. **Exact search**, because at a few thousand vectors it takes under a millisecond. IVF-PQ only pays off past about 10⁶ vectors [DOC].
- `add` removes existing ids first, so re-embedding leaves no duplicates.
- **Filtered search:** `faiss.SearchParameters(sel=IDSelectorBatch(subset))`, i.e. FAISS's native ID selector. k is capped at the subset size.
- `save`: write to a temp file, then `os.replace`, so the update is atomic.
- `rebuild_from_catalog`: the index is a **derived artifact**; vectors live in the catalog, so a FAISS upgrade is a rebuild, not a migration.

---

## 6. Face pipeline (code walk)

### 6.1 Detection and embedding
- `faces/detect.py`: insightface SCRFD (`scrfd_10g_bnkps`). Drop detections with score < 0.6 or a short side < 40 px (too blurry to embed well).
- onnxruntime CPU threads are capped (bug fix `4417f35`: it used one thread per machine core on a 224-core shared box).
- `embed/arcface.py`: `face_align.norm_crop` on the 5 landmarks, then ArcFace, giving a 512-d vector.

### 6.2 `IdentityClusterer.cluster` (`faces/cluster.py`)
- L2-normalise. Cosine distance = 1 − x·y.
- **Constraints:**
  - `pinned` (face → confirmed identity): those faces never move;
  - `rejected` (face, identity) pairs: their distance is set to ∞.
- Steps:
  1. Each confirmed identity seeds a cluster from its pinned faces.
  2. Unpinned faces join the nearest seed centroid within `distance_threshold`.
  3. The rest are clustered unsupervised (`_unsupervised`: a precomputed cosine-distance matrix into `AgglomerativeClustering(linkage="average", distance_threshold)` or HDBSCAN). Groups smaller than `min_cluster_size` become noise.
- Each `Cluster` stores the centroid and a **dispersion** (mean member-to-centroid distance; a loose cluster suggests a merge of two people).

### 6.3 `refine()`: the incremental update (the important fix)
- Takes the *current* identities as they stand. For each iteration (2):
  - **distance from a face to an identity = 1 − x·mean(members)** with the mean *not* normalised. That is the **average-linkage distance itself**;
  - pinned faces stay put;
  - an assigned face moves only if another identity is **strictly closer** or the user rejected its current one;
  - an unassigned face joins the nearest allowed identity within the threshold.
- Leftovers are clustered among themselves.
- **Two bugs found by evaluation [DOC, commits `b097f2f`, `034ce8c`]:**
  1. The first `refine()` **re-seeded from pinned faces only**. Uncertainty sampling confirms *outliers*, so each person's seed sat at the edge and the rest of the person split off. On LFW, BCubed recall fell 99.73 → 96.88 after 20 reviews, and F1 ended *below* the start (99.83 → 99.58). Fix: an incremental refine.
  2. The incremental refine first compared faces to **normalised centroids** against a threshold calibrated as an *average-linkage* cut. A normalised centroid sits closer than the members do, so noise faces were pulled in and every arm lost precision; random sampling even ended below its start. Fix: use the average-linkage distance. A **no-answers control** (refine with zero corrections) now leaves the clustering unchanged.

### 6.4 `ActiveLearner` (`faces/active.py`)
- `next_batch`:
  - transform embeddings with the current head (identity if untrained);
  - compute distance to each identity centroid;
  - d1 = distance to the face's own or proposed identity, d2 = distance to the next nearest;
  - **uncertainty** ranks by largest d1, **margin** by smallest d2 − d1, **random** by a seeded permutation;
  - spread across identities (at most 2 questions per identity per batch);
  - skip pinned faces.
- `record`: append to the `corrections` table (training history is never overwritten), **and apply the effect immediately**:
  - ASSIGN → pin the face;
  - REJECT → unassign;
  - MERGE → `merge_identities` (sets `merged_into`);
  - NAME → rename, and if another identity already has that name, treat it as a MERGE;
  - SPLIT → move faces to a new identity and pin them.
- `answer(face, proposed, action)` is the vocabulary shared by the CLI and the UI: confirm, reject, name, move.
- Refinement trains the head when there are at least 20 labels across at least 3 identities, re-embeds, then calls `refine()` with the pinned and rejected constraints.

### 6.5 The adapter head (`faces/head.py`)
- `x → normalize(W·x)`, with W 512×512 **initialised to the identity**.
- Loss = supervised contrastive (temperature 0.1) over the labelled faces, plus `identity_reg · ‖W − I‖²` (weight 1.0). Adam, lr 1e-3, 200 steps.
- Skipped with fewer than 20 labels or 3 identities.
- Why not fine-tune ArcFace [DOC]: with tens to hundreds of labels, tuning about 25M backbone parameters overfits immediately. A linear map near the identity "cannot make the base embeddings much worse".
- **Measured: it doesn't make them better either** (§7.2).

---

## 7. Evaluation and results [MEASURED]

### 7.1 Text-to-image retrieval (`eval/retrieval.py`)
Every caption is a query. R@k = the right image is in the top k.

| set | images / queries | R@1 | R@5 | R@10 | through `photoseek search` (rule router) R@1/5/10 | latency |
|---|---|---|---|---|---|---|
| COCO 5k (Karpathy test) | 5,000 / 25,010 | **39.4** | 65.4 | 75.6 | 35.9 / 61.0 / 71.4 | 2.8 ms |
| Flickr30k 1k | 1,000 / 5,000 | **66.8** | 88.3 | 93.1 | 62.3 / 84.0 / 89.3 | 2.4 ms |

- These are **pretrained CLIP's** numbers through his index (validating the plumbing). They are not a model he trained.
- Going through the rule router costs 3.5–4.4 R@1, because words like "baby" or "mother" become people filters, and an unnamed person matches no photo.
- Latency: one caption embedded and searched, median of 3 × 500 after a 200-query warm-up, on a shared H200. An earlier run measured about 4.5 ms.

### 7.2 Faces
**Protocol:**
- One labelled face per photo. Identities split in half; settings chosen on the **dev** half, numbers reported on the **test** half.
- A simulated user answers 20 questions per round via `ActiveLearner.record`. Each round ends with `refine()`.
- Arms: uncertainty, margin, uncertainty with the head off, a no-answers control, and random (3 seeds).

**LFW** (people with ≥ 5 images: 423 people, 5,985 photos; test 212 people / 2,975 faces):
- Initial clustering is nearly perfect: HDBSCAN BCubed F1 **99.83**, 214 clusters, 1 unassigned. There is little room to improve.
- After 300 reviews, error 0.17% → 0.06% with margin sampling (−61%), 0.14% with uncertainty, 0.17% with random. Head effect +0.00.

**CPLFW (cross-pose)** (people with ≥ 3 images: 3,812 people, 11,435 photos; test 1,906 people / 5,606 faces):
- Initial clustering:
  - shipped default HDBSCAN (min size 3): F1 91.75, 544 unassigned;
  - **chosen on dev** (agglomerative, threshold 0.75, min size 2): F1 **95.91**, P 98.13, R 93.79, pairwise F1 93.18, NMI 0.9922.

| faces reviewed | 0 | 100 | 300 | 500 | 1,000 |
|---|---|---|---|---|---|
| uncertainty | 95.91 | 96.02 | 97.06 | 97.22 | 97.23 |
| **margin** | 95.91 | 96.14 | 97.07 | 97.30 | **97.57** |
| uncertainty, head off | 95.91 | 96.04 | 97.07 | 97.26 | 97.29 |
| control: refine, no answers | 95.91 | 95.91 | 95.91 | 95.91 | 95.91 |
| random (median of 3) | 95.91 | 95.93 | 95.96 | 96.04 | 96.19 |

- After 1,000 reviews (18% of faces): error 4.09% → **2.43% with margin (−41%)**, 2.77% with uncertainty (−32%), 3.81% with random (−7%).
- **Propagation to faces nobody reviewed is tiny:** at most **+0.09** BCubed F1 on the untouched faces, compared on the *same subset*. (A sampler that asks about hard faces leaves an easier remainder, so only a same-subset comparison is fair.)
  **So the loop's value is *finding* the wrong assignments, not generalising the corrections.**
- **Adapter head: −0.07** (uncertainty with head vs head off). The gain comes from the corrections plus the constrained update.

### 7.3 Query routing (`eval/router.py`)
100 hand-written queries with gold `StructuredQuery` (`eval/router_gold.jsonl`), today fixed at 2026-10-22, 3 interleaved reps. "All filters" = people, both dates and place all correct at once.

| router | people | date from | date to | place | text token-F1 | **all filters** | latency median (p90) |
|---|---|---|---|---|---|---|---|
| rule | 83 | 87 | 87 | 85 | 0.74 | **58** | 0.19 ms (0.29) |
| LLM (qwen2.5:7b) | 97 | 82 | 82 | 95 | 0.80 | **73** | 1.52 s (2.19) |
| **hybrid** | 97 | 94 | 92 | 95 | 0.79 | **84** | 1.56 s (2.19) |

The LLM fell back to the rules on 1 of 100 queries.
Trade-off: the hybrid costs about 1.5 s per query vs 0.2 ms, for +26 points of all-filters accuracy.

### 7.4 Offline proof (`eval/offline.py`)
- Runs `photoseek search` with the hybrid router, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, under a **Python audit hook** (`sys.addaudithook`) that records every `socket.connect`, `sendto` and `getaddrinfo`. It also samples the Ollama processes' TCP peers.
- Result: **pass**. 6 socket events across 3 queries, all to 127.0.0.1, and Ollama's only peers were 127.0.0.1.
- Caveat [DOC]: this is process-level evidence, not a network namespace.

---

## 8. Hardest problems (real)
1. **refine() re-seeding made things worse** (§6.3, bug 1): uncertainty sampling picks outliers, so seeding from them split people.
2. **Centroid vs average-linkage distance mismatch** (§6.3, bug 2). The no-answers control row exists to prove `refine()` alone is neutral.
3. **The LLM's date arithmetic is unreliable**, which led to the hybrid router (§5.4). The +11 points on all-filters accuracy over the LLM alone comes mostly from dates.
4. **No reverse geocoding**, so place filters would empty every result. Places are folded into CLIP text when no label matches (commit `1637f63`).
5. **Honest evaluation design:** dev/test identity split, a same-subset propagation measure, a no-answers control, random with 3 seeds.
6. **Shared-server hygiene:** onnxruntime thread cap; model caches on warm storage; the project's own Ollama server on a non-default port.

---

## 9. Design decisions
| Decision | Alternatives | Why | Source |
|---|---|---|---|
| Frozen ArcFace + linear adapter | fine-tune the backbone | few labels → overfitting | [DOC] |
| HDBSCAN / agglomerative | k-means | unknown number of people; noise label | [DOC] |
| Uncertainty / margin sampling | random | find wrong assignments fast (−41% vs −7% error) | [MEASURED] |
| Corrections as hard constraints (pins, rejected pairs) | soft labels only | the loop must never undo the user | [DOC] |
| Append-only corrections log | overwrite | training history; auditing | [DOC] |
| Filter in SQL, then rank in FAISS | rank then filter | the top-k could be all filtered out | [DOC] |
| Exact `IndexFlatIP` | IVF / HNSW / PQ | small N; exact recall; < 1 ms | [DOC] |
| Inner product on normalised vectors | L2 | equals cosine | [DOC] |
| `StructuredQuery` contract | free-text to CLIP | CLIP can't do dates or people | [DOC] |
| Conservative rule router | aggressive extraction | a wrong filter hides the right photo | [DOC] |
| Schema-constrained Ollama + validation + fallback | trust the LLM | robustness; offline | [DOC] |
| Hybrid router (rule dates) | LLM only | the LLM's date errors | [MEASURED] |
| Unknown person → empty result | ignore the name | typos visible | [DOC] |
| Index derived from the catalog | index as source of truth | rebuild instead of migrate | [DOC] |
| Protocols + fakes in tests | real models in tests | 157 fast tests without weights | [DOC] |

---

## 10. Limitations
- **The adapter head doesn't help, and corrections don't propagate** to unreviewed faces. Both are open problems.
- No reverse geocoding, so place filters rarely apply.
- LFW clustering is already about 99.8%, so it is a weak testbed for active learning. CPLFW is the real test.
- Simulated user (oracle labels), not a user study.
- LLM routing costs about 1.5 s per query.
- Exact search won't scale to millions of photos (IVF/HNSW would be next).
- Offline proof is process-level, not a network sandbox.

---

## 11. Likely interview questions
1. *Trace "photos of mom at the beach last summer".* → the rule router gives people=[mom], dates = 2026-06-01..2026-08-31 (asked in Oct 2026, summer 2026 has fully ended, so `_season_year` keeps 2026; asked in July 2026 it would give summer 2025), place=beach, text="". No photo has place "beach", so place moves into text → "beach". SQL: date range + EXISTS a face of identity "mom" (unknown name → no results). CLIP embeds "beach" and FAISS ranks within the subset.
2. *Why not fine-tune ArcFace? Did your fine-tuning help?* → no backbone fine-tuning; the head added +0.00 / −0.07. Be honest about the resume wording.
3. *Uncertainty vs margin sampling: which won and why?* → margin (−41%): faces on a boundary between two people are the likely errors. Uncertainty also picks faces that are merely unusual (pose, blur) but correctly assigned.
4. *What is BCubed, and why that metric?* → per-item precision/recall; works with unequal cluster sizes and noise.
5. *Why HDBSCAN vs agglomerative, and why did dev pick agglomerative on CPLFW?* → HDBSCAN marked 544 faces as noise (hurting recall) on hard cross-pose data; average linkage at 0.75 kept more.
6. *How does filtered FAISS search work?* → an `IDSelectorBatch` restricts the scan to the candidate ids.
7. *When would you switch from a flat index?* → about 10⁶ vectors: IVF-PQ or HNSW; consider recall/latency/memory.
8. *Why a hybrid router?* → §5.4 and §7.3.
9. *How did you prove it is offline?* → §7.4.
10. *What was your hardest bug?* → refine() seeding from outliers (§6.3).
11. *Why does routing captions hurt R@1 on COCO?* → kinship words become person filters that match nothing.
12. *What's ArcFace's loss?* → the additive angular margin, cos(θ + m), scaled by s.
13. *What would make corrections propagate?* [RECON] → metric learning with more labels; transductive label propagation over a kNN graph; re-clustering with must-link/cannot-link constraints (COP-KMeans-style); learning per-library thresholds.

---

## 12. Numbers cheat sheet
| Number | Meaning |
|---|---|
| 39.4 / 66.8 | CLIP ViT-B-32 R@1 on COCO-5k / Flickr-1k through his FAISS index |
| 2.8 ms | search latency (embed + exact search) |
| 99.83 | LFW initial BCubed F1 |
| 95.91 → 97.57 | CPLFW BCubed F1 after 1,000 margin reviews |
| −41% / −32% / −7% | error reduction: margin / uncertainty / random (CPLFW, 1,000 reviews) |
| +0.09 max | propagation to unreviewed faces |
| −0.07 | adapter head effect (CPLFW) |
| 58 / 73 / 84 | all-filters accuracy: rule / LLM / hybrid (100 gold queries) |
| 0.19 ms vs 1.52 s | rule vs LLM routing latency |
| 6 events, all loopback | offline proof |
| 157 | tests |
| 512 | CLIP and ArcFace embedding dims; adapter W is 512×512 |
