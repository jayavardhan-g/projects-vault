> Prep pack for **Photo-Seek**, written 2026-11-05 from `photo_seek/INTERVIEW_PREP.md`. The vault copy is the one to study and edit.

# Photo-Seek — Interview Prep

Prepared 2026-11-05 from branch `upgrade-2026-10` (HEAD `aa5d476`). Sources: README.md, SETUP.md,
eval/README.md, CLAUDE.md, the git log (`main` scaffold of 2026-08-18 plus 19 commits), every file
under `src/` and `eval/`, the result JSONs in `eval/results/`, and the local run logs in
`.claude/run/` (git-ignored). No code was changed. This file is excluded from git.

**Read these first. They are the facts that will decide the interview.**

1. **The ArcFace backbone is never fine-tuned.** A 512×512 linear adapter is trained on frozen
   ArcFace vectors. It changes BCubed F1 by −0.07 points on CPLFW and +0.00 on LFW. All of the
   gain comes from the corrections, the constraints, and *which* faces the sampler asks about.
2. **"1,000+ photos" matches nothing in the repo.** The face eval clusters **5,606 held-out faces**
   (1,906 people) from an **11,435-photo** CPLFW set. "1,000" is the number of **simulated
   reviews**. No real personal library was ever processed: `data/catalog.db` has 0 photos.
3. **95.9 → 97.6 needs two qualifiers.** 95.9 is the clustering setting chosen on the dev half
   (agglomerative, 0.75, min size 2). The shipped default (HDBSCAN, min size 3) scores 91.75.
   97.6 is **margin** sampling. The shipped default strategy is uncertainty, which ends at 97.2.
4. **"84%" counts queries, not fields.** On 84 of 100 hand-written queries, all four filters
   (people, date_from, date_to, place) were correct at once. The text field is not part of it.
5. **The environment is broken right now.** In `photo-seek`, `import numpy` dies with
   `Cannot load libmkl_gf_lp64.so.3`. So `pytest` exits with code 2 and no number can be
   re-run today. The last test run on record is "156 passed" (`.claude/run/pytest.txt`,
   22 Oct). The README says 157. I did not repair the env, as instructed.

---

## 1. The pitch

### 60 seconds

Photo-Seek organises and searches a personal photo library, and it runs fully offline.
It has two pipelines.

The first one groups faces into people. SCRFD finds the faces. ArcFace turns each face into
a vector. Agglomerative clustering groups the vectors. Clustering is never fully right, so
the system asks the user about the faces it is least sure of. Each answer becomes a hard
constraint for the next round. On a hard cross-pose benchmark, 1,000 reviews raised BCubed F1
from 95.9 to 97.6. That cuts the error by 41%. Reviewing random faces cut it by only 7%.

The second one is search. CLIP embeds every photo into an exact FAISS index. A local LLM,
qwen2.5 7B on Ollama, turns a query like "mom at the beach last summer" into structured
filters. Rules supply the dates. SQL applies the filters, then CLIP ranks what is left.
CLIP gets 39.4 R@1 on COCO-5k. The hybrid parser gets every filter right on 84 of 100 test
queries; rules alone get 58. An audit hook showed the search made only loopback connections.

### 3 minutes

**The problem.** People have tens of thousands of photos and cannot find any of them. Cloud
tools solve this by uploading your faces. I wanted the same features with nothing leaving
the machine. That one rule drove every choice: local models, SQLite, a local LLM server,
and a rule-based fallback for when the LLM is down.

**Faces.** Ingest hashes each file with BLAKE2b, so duplicates are skipped, and reads EXIF
for the date and GPS. SCRFD-10G detects faces and gives five landmarks. I warp each face to
the 112×112 ArcFace crop with those landmarks, because ArcFace expects aligned input. ArcFace
R50 from insightface's `buffalo_l` pack gives a 512-d unit vector. I cluster with average-linkage
agglomerative clustering at a cosine-distance cut. I don't use k-means, because the number of
people is unknown. Small groups become "unassigned" rather than being forced into someone.

**Active learning.** The sampler ranks faces by uncertainty, meaning distance to their
cluster's centroid, or by margin, meaning the gap between the nearest and second-nearest
identity. It caps the batch at two questions per person. The user confirms, rejects, names, or
moves a face. Every answer goes into an append-only corrections table and takes effect at once.
A confirm pins the face. A reject becomes a cannot-link. Then `refine()` updates the clusters
step by step. Pinned faces stay put. Other faces move only if another identity is strictly
closer by average-linkage distance. I also train a small linear adapter on the confirmed faces.
It turned out to add nothing, and I report that.

**Two bugs taught me the most.** My first `refine()` rebuilt each person from their confirmed
faces only. But the uncertainty sampler confirms the outliers, so the rest of each person split
off. On LFW, F1 ended below where it started. The second bug: I compared faces to a normalised
centroid using a threshold meant for average linkage. That pulled 100–170 noise faces into
identities in the first round. A "no answers" control run caught it, because the score moved
even with zero corrections.

**Search.** The key object is `StructuredQuery`: text, people, date range, place, limit. Three
parsers produce it. A rule parser is deterministic and conservative. An Ollama parser uses
qwen2.5:7b with JSON-schema-constrained decoding and falls back to the rules on any failure.
A hybrid takes the LLM's people, place, and text, and the rules' dates when the rules match.
The LLM was bad at date arithmetic; it put "last summer" a year early. Filters run in SQL
first, then FAISS ranks only the survivors through an ID selector. That way a top-k cannot
fill up with photos that fail the filters.

**Evaluation.** Every README number comes from a script in `eval/`, and `report.py` writes the
tables, so they cannot drift. Face settings are chosen on half the identities and reported on
the other half. I also report what did not work: corrections barely spread to unreviewed faces,
and the adapter does nothing. I built reverse geocoding? No. That is the main missing piece,
so place words fall back to CLIP text.

> Practise this last paragraph without the joke line. Say plainly: "Reverse geocoding is not
> built, so places currently match visually."

---

## 2. Architecture

```
                         photos on disk
                               │
  cli.py:315 main ─► cmd_ingest cli.py:80
                               │
  ingest/scanner.py:89 scan ── hash_file :50 (BLAKE2b-128), skip by path+mtime :119, min_bytes :112
  ingest/exif.py:53 read_metadata (DateTimeOriginal, GPS, orientation)
                               │
  catalog/db.py:57 Catalog  (SQLite, WAL :83; schema.sql: photos, faces, identities,
                               │                 embeddings, corrections[append-only])
            ┌──────────────────┴─────────────────────────┐
            │ cmd_index cli.py:112                       │ cmd_faces cli.py:153
  embed/clip.py:64 embed_images                faces/pipeline.py:22 detect_and_embed
  (open_clip ViT-B-32 laion2b_s34b_b79k)         faces/detect.py:127 SCRFD detect_array
            │                                    embed/arcface.py:72 align (5-pt, 112x112)
  catalog put_embedding db.py:209                embed/arcface.py:85 embed_faces (512-d, L2)
  index/faiss_store.py:81 add                    catalog add_face db.py:232
  (IndexIDMap2(IndexFlatIP))                               │
  faiss_store.py:147 save (tmp + rename)       faces/pipeline.py:66 cluster_library
            │                                    faces/cluster.py:76 cluster  (full)
            │                                    faces/cluster.py:314 assign_new_faces (incremental)
            │                                    faces/cluster.py:286 _unsupervised (agglom | HDBSCAN)
            │                                    faces/cluster.py:341 persist (identity ids survive)
            │                                              │
            │                         cmd_label cli.py:176 / ui/app.py:110,140,151
            │                                    faces/active.py:97  next_batch (uncertainty|margin|random)
            │                                    faces/active.py:239 answer → :173 record
            │                                    faces/active.py:290 refine
            │                                      ├─ faces/head.py:60 ProjectionHead.fit (SupCon, W≈I)
            │                                      └─ faces/cluster.py:174 IdentityClusterer.refine
            │                                              │
            └───────────────► search/engine.py:78 SearchEngine.run ◄──┘ (faces.identity_id)
                                       ▲
                  search/router.py:391 build_router
                    RuleRouter :81  |  OllamaRouter :250  |  HybridRouter :366
                                       ▲
                     cmd_search cli.py:226   /   ui/app.py:59 /api/search
```

### Trace 1: one search query, end to end

Command: `photoseek search "photos of mom at the beach last summer" --router hybrid`, run on
2026-11-05.

1. `cli.py:315 main` parses args. `Config.load` (`config.py:119`) merges your YAML over
   `configs/default.yaml` key by key. It dispatches to `cmd_search` (`cli.py:226`).
2. `--router hybrid` overrides `config.search.router` (`cli.py:229`). `build_router`
   (`router.py:391`) returns a `HybridRouter`.
3. `HybridRouter.route` (`router.py:383`) first calls `OllamaRouter.route` (`router.py:314`).
   - `_generate` (`router.py:339`) POSTs to `http://127.0.0.1:11534/api/generate`. The prompt
     includes today's date and weekday. `format` is the JSON schema (`router.py:261`).
     `temperature` is 0 and `num_predict` is 200.
   - `_extract_json` (`router.py:424`) takes the outermost `{...}`. `StructuredQuery.from_dict`
     (`query.py:82`) drops bad dates to `None`.
   - An empty result, bad JSON, timeout (30 s), or dead server raises. The `except` at
     `router.py:332` falls back to `RuleRouter` and sets `last_fallback`.
   - Likely output: `people=["mom"]`, `place="beach"`, `text=""`, plus LLM dates.
4. Then `RuleRouter.route` (`router.py:95`) runs on the same text.
   - `_extract_dates` (`router.py:114`) matches "last summer". `_season_year` (`router.py:175`):
     November is after August, so the year stays 2026. The range is 2026-06-01 to 2026-08-31.
   - Because the rules found a date, the hybrid overwrites the LLM's dates (`router.py:386`).
5. Back in `cmd_search`, `SearchEngine.run` (`engine.py:78`) gets the `StructuredQuery`.
   - `_place_known("beach")` (`engine.py:192`) runs a `LIKE` on `photos.place`. Nothing is
     reverse-geocoded, so no row matches. The place is moved into the text: `text="beach"`,
     `place=None` (`engine.py:84-90`).
   - `candidate_ids` (`engine.py:109`) builds SQL. `date_to` gets +1 day with `<`, because
     `taken_at` has a time part (`engine.py:127-130`).
   - `resolve_people` (`engine.py:150`) looks for an identity named "mom", case-insensitive,
     following `merged_into`. **If nobody is named "mom", it returns an empty set**, and the
     CLI prints "unknown people ['mom'] (name them with `photoseek label`)".
   - Otherwise it adds one `EXISTS (... faces f WHERE f.identity_id = ?)` per person. So
     several people means all of them must be in the photo.
6. `_ensure_models` (`engine.py:178`) loads CLIP and the FAISS file on first use only.
7. `embed_texts(["beach"])` (`clip.py:100`) gives a 512-d unit vector.
8. `FaissStore.search` (`faiss_store.py:111`) passes `IDSelectorBatch(candidates)` in
   `SearchParameters`. FAISS scores only those IDs, by inner product (cosine).
9. `_hydrate` (`engine.py:207`) joins each hit back to `photos` and the named people. The
   CLI prints rank, score, path, and names.

### Trace 2: one correction, end to end

Command: `photoseek label`, the user types `y` for one face.

1. `cmd_label` (`cli.py:176`) builds an `ActiveLearner` (`active.py:69`) and loads the saved
   head (`head.py:121`), or the identity map if none exists.
2. `next_batch` (`active.py:97`) loads all face vectors (`active.py:360`) and projects them
   through the head. It computes unit centroids per identity (`active.py:385`).
   - Uncertainty: sort by distance to own centroid, largest first (`active.py:137-140`).
     Unassigned faces use their nearest centroid, so they are usually far and asked first.
   - Margin: sort by `d2 − d1`, smallest first (`active.py:141-144`).
   - `_spread` (`active.py:401`) allows at most 2 questions per proposed identity.
3. `y` maps to `answer(..., "confirm")` (`active.py:239`), which calls `record(ASSIGN)`
   (`active.py:173`). That appends a row to `corrections` (`db.py:323`) and pins the face
   (`db.py:255`, `is_pinned=1`). It commits at once.
4. When pending corrections reach `retrain_after` (25 by default), `refine()` (`active.py:290`)
   runs:
   - It trains the head on every pinned face, if there are ≥20 labels across ≥3 identities
     (`head.py:27,73`). Otherwise it keeps the old head.
   - It projects all faces, reads every `reject` as a cannot-link (`active.py:319-325`), and
     calls `IdentityClusterer.refine` (`cluster.py:174`).
   - In `refine`, each identity's distance is `1 − x·mean(members)`, the average-linkage
     distance (`cluster.py:227-233`). Pinned faces stay. An assigned face moves only to a
     strictly closer identity, or if its own was rejected. An unassigned face joins the
     nearest one within the threshold (`cluster.py:243-249`). Leftovers are clustered among
     themselves (`cluster.py:261-273`).
   - `persist` (`cluster.py:341`) keeps identity ids stable. The largest cluster picks first
     and takes the old id most of its faces had. Pinned faces are never written.
   - Only then are the corrections marked consumed (`active.py:333`). A crash mid-retrain
     loses nothing.

---

## 3. Design decisions

Legend: **[doc]** = stated in a doc or commit (cited). **[code]** = reconstructed from the code;
no doc gives the reason. Nothing here is invented history.

| # | Decision | Alternatives | Why | Cost | Source |
|---|---|---|---|---|---|
| 1 | Fully offline at runtime; weights fetched once | Cloud vision / cloud LLM APIs | Privacy is the product | Smaller models; you host Ollama; no cloud quality | [doc] CLAUDE.md "Design constraint", README intro |
| 2 | SQLite catalog is the source of truth; FAISS file is derived | Vector DB as primary store; FAISS only | Index can be deleted and rebuilt; a FAISS upgrade is a rebuild, not a migration | Vectors stored twice (blob + index) | [doc] commit 4fbb0c4; `schema.sql:3-5`; `faiss_store.py:3-5` |
| 3 | Every model sits behind a `Protocol`; heavy imports inside functions | Concrete classes; top-level imports | Logic is tested with fakes, no multi-GB weights; `import photoseek` stays cheap | Some indirection; the real models are only tested by the eval scripts | [doc] commit 64e7393; CLAUDE.md |
| 4 | All vectors L2-normalised; inner product = cosine | L2 distance; raw vectors | One meaning for "distance" in CLIP and ArcFace; `IndexFlatIP` works | Must remember to normalise every new vector | [doc] commit 64e7393; `embed/base.py:7-9` |
| 5 | FAISS `IndexFlatIP` in `IndexIDMap2`, exact search | IVF-PQ, HNSW | A personal library is thousands of vectors; exact search is fast enough and loses no recall | O(N) per query; an ID-filtered search still scans every vector | [doc] `faiss_store.py:55-61` |
| 6 | Filter in SQL first, then rank inside the survivors | Rank all, then filter | A top-k over everything can be filled by photos that fail the filters | Needs `IDSelectorBatch`; the selector check costs time on huge subsets | [doc] `engine.py:3-9`; `faiss_store.py:116-119` |
| 7 | CLIP ViT-B-32 `laion2b_s34b_b79k` (open_clip) | OpenAI ViT-B/32, ViT-L-14, SigLIP | Set in the 18 Aug scaffold config. No rationale written. Likely: small (578 MB), fast, runs on CPU, LAION weights beat OpenAI B/32 on retrieval | Not the most accurate; no ablation was run | Choice [doc] `main:configs/default.yaml:25-26`; reason [code] |
| 8 | `StructuredQuery` is the contract between parsing and search | Pass raw text to CLIP | Routers become swappable; "last summer" no longer pollutes the CLIP embedding | Anything not captured by 5 fields is lost to filtering | [doc] commit dad00c6; `query.py:1-12` |
| 9 | Rule router is conservative; capital letters mark names | Aggressive extraction; NER model | A wrong filter silently removes the right photo; a missed one only falls back to CLIP | Misses lowercase names, holidays, paraphrases (58% on gold) | [doc] commit dad00c6; `router.py:84-88, 219-230` |
| 10 | LLM parser on a local Ollama server, qwen2.5:7b | llama.cpp in-process; transformers; a bigger or smaller model; a hosted API | Local = offline. Ollama gives schema-constrained output. Model name is in the 18 Aug scaffold; no doc says why qwen2.5:7b. Likely: strong JSON/instruction following at 7B, fits a 4.4 GB download | 1.5 s median per query vs 0.19 ms for rules; one more server to run | Ollama/offline [doc] commit dad00c6; schema [doc] f0abb5f; model choice [code] |
| 11 | JSON-schema constrained decoding, temperature 0, still validated, rule fallback | Free-text prompt + regex; function calling | A broken parse must degrade search, not break it | 1 of 100 queries still fell back | [doc] commit f0abb5f; `router.py:250-259` |
| 12 | Hybrid: LLM people/place/text, rule dates when the rules match | LLM only; rules only; fix the prompt | The 7B model fails date arithmetic, which rules do exactly | Two parses per query; rules' gaps (holidays) still go to the LLM | [doc] commit fc80eb8; `router.py:367-372`; README "Query routing" |
| 13 | A place no photo has is folded into the CLIP text | Hard filter always | No reverse geocoding yet, so "beach" filtered the whole library away | A real place name is "matched" visually | [doc] commit 1637f63; `engine.py:84-90` |
| 14 | An unknown person returns nothing, not everything | Drop the unknown name | A typo should be visible, not silently widen the search | "kids" or "family" from the LLM gives an empty result | [doc] `query.py:29-31`; `engine.py:112-116` |
| 15 | SCRFD-10G detector, 640×640, min score 0.6, min face 40 px | MTCNN, RetinaFace, Haar cascades | Better on group shots and profiles; tiny or weak faces give noisy vectors that bridge clusters | Misses faces under 40 px | [doc] `detect.py:94-96, 128-133` |
| 16 | Use the detector's 5 landmarks for a similarity warp to 112×112 | Feed raw boxes | ArcFace is trained on aligned crops; raw boxes degrade embeddings | None measured in repo | [doc] `arcface.py:73-77` |
| 17 | ArcFace frozen; train only a 512×512 linear adapter (SupCon, starts at I, pulled to I) | Fine-tune the backbone; no head | Tens to hundreds of labels would overfit a big backbone at once | The head added nothing measurable (−0.07 / +0.00) | [doc] commit 4722ee7; `head.py:1-14`; README |
| 18 | Agglomerative average-linkage or HDBSCAN; never k-means | k-means, DBSCAN, Chinese whispers, GCN clustering | Number of people unknown; outliers may stay unassigned | Full pairwise distance matrix: O(N²) memory | [doc] commit 4722ee7; `cluster.py:1-11` |
| 19 | Threshold and algorithm picked on a dev half of identities | Tune on test; use defaults | Honest numbers | Shipped default (HDBSCAN 0.55 / min 3) was not updated to the CPLFW winner | [doc] eval/README "Protocols"; `faces.py:308-324` |
| 20 | Uncertainty, margin, and random samplers; max 2 questions per identity | Only one sampler; core-set; expected-error reduction | Random is the baseline to beat; the cap stops 20 questions about one person | Cap may skip the very worst faces | [doc] `active.py:40-43, 97-107` |
| 21 | Confirmed faces are hard constraints; rejects are cannot-links | Treat answers as soft hints | If the loop moves a face the user labelled, it undoes its own training signal | A wrong user answer is never corrected by the model | [doc] commit 4722ee7; `cluster.py:86-90` |
| 22 | `refine()` is incremental; a face moves only on evidence | Re-cluster from scratch seeded by pinned faces | The seeded version split people (LFW recall 99.73 → 96.88) | Few faces move at all, so little propagation | [doc] commit b097f2f; `cluster.py:184-205` |
| 23 | `refine()` judges moves by average-linkage distance | Normalised centroid distance | The threshold is an average-linkage cut; centroids sit closer than members | None known | [doc] commit 034ce8c; `cluster.py:227-233` |
| 24 | Append-only corrections; merges tombstone via `merged_into` | Overwrite rows; delete merged identity | Corrections are training data and history; old references must resolve | Table grows forever | [doc] commit 4fbb0c4; `schema.sql:84-101` |
| 25 | BLAKE2b-128 content hash; skip by path+mtime | SHA-256; perceptual hash | Fast, collision-safe enough; rescan is one `stat` per file | Re-encoded copies are not seen as duplicates | [doc] commit 4f8e878 |
| 26 | `argparse`, `urllib` instead of typer / an Ollama client | Typer, `ollama` package | Fewer dependencies; core runs with only PyYAML | More boilerplate | [doc] commit d00b664; `router.py:253` |
| 27 | Project's own Ollama binary on port 11534 | System Ollama on 11434 | System binary could not reach the GPU (unreadable `cuda_v12` dir); avoid other users' instance | Extra setup step | [doc] SETUP.md "Local LLM"; commits f0abb5f, cf18911 |
| 28 | onnxruntime sessions rebuilt with 8 intra-op threads | insightface default | Default spawned 232 threads on a 216-core shared box | Slower on a dedicated big machine | [doc] commit 4417f35; `detect.py:62-77` |
| 29 | `report.py` writes the README results from the JSONs | Hand-copied tables | Numbers cannot drift from the runs | README section is generated; edit JSON, not prose | [doc] eval/README intro |
| 30 | Offline proof via a Python audit hook + `ss` | Network namespace; firewall | No root on the shared machine | Process-level evidence only | [doc] `eval/offline.py:12-15`; README "Offline" |

---

## 4. Hardest problems, as STAR stories

### 4.1 Corrections made the clustering worse

- **Situation.** First LFW active-learning run. Uncertainty sampling, 20 reviews per round.
- **Task.** Corrections should raise accuracy, never lower it.
- **Action.** I saw BCubed recall fall from 99.73 to 96.88 after 20 reviews and to 97.74 after
  60. I traced it to `refine()`: it re-clustered from scratch, and each identity was seeded
  only by its pinned faces. The uncertainty sampler pins *outliers*. So each seed sat at the
  edge of the person, and the rest of the person split off. I rewrote `refine()` to keep the
  current identities and move a face only on evidence: a strictly closer identity, or a
  rejection. I kept the failed run as a JSON file.
- **Result.** F1 no longer drops below the start. The old run ended at 99.58 (below 99.83). It
  is kept in `eval/results/faces_lfw_seeded_refine.json`.
- **Source.** Commit b097f2f; README "A negative result kept on record"; `cluster.py:201-204`.
- **Lesson to say aloud.** "The sampler and the update rule interact. The faces you ask about
  are the least typical, so you cannot rebuild a person from them."

### 4.2 Refine changed the clusters with zero answers

- **Situation.** CPLFW run. Every arm lost about 2 points of BCubed precision in its first
  `refine()` round. Random sampling ended below its start.
- **Task.** Find out why a refine with no new information moves faces.
- **Action.** I added a control arm: same rounds, no answers. It should be flat. It was not.
  The cause: the threshold 0.75 is an average-linkage cut. But `refine()` compared it with the
  distance to a **normalised centroid**. A normalised centroid sits closer to every face than
  the members do on average. So 100–170 noise faces fell under the threshold and joined
  identities. I switched to the true average-linkage distance, `1 − x·mean(members)` with the
  mean left unnormalised.
- **Result.** The no-answers control is now flat at 95.91 for all 50 rounds. The broken run
  was overwritten; the commit message is the record.
- **Source.** Commit 034ce8c; README "A second bug surfaced on CPLFW".
- **Lesson.** "A null control is cheap and catches bugs that a treatment-vs-baseline plot hides."

### 4.3 The detector found 21 faces in 10,326 photos

- **Situation.** First CPLFW ingest. CPLFW ships tight, aligned face crops.
- **Task.** Get the benchmark through the *real* pipeline, not a shortcut.
- **Action.** The local log shows `ingest: 10326 new ... 1110 skipped` and
  `detect+embed: 21 faces in 10326 photos`. Two causes. The 4 KB thumbnail filter skipped
  1,110 small files. And SCRFD had no context around a face that fills the frame. I probed 40
  images: 1 detected unpadded, 37 with a 50% black border. The eval now sets `min_bytes = 0`
  and pads each image before detection. Detection, alignment, and ArcFace stay unchanged. I
  also made `detect_and_embed` accept a custom image loader.
- **Result.** 11,215 faces from 11,435 photos.
- **Source.** eval/README "Protocols"; `eval/faces.py:62-69, 88-97`; commit 034ce8c (loader);
  local log `.claude/run/faces_cplfw.log`.

### 4.4 "A dog on the beach" returned nothing

- **Situation.** Early search tests on Flickr images.
- **Task.** Place words should help, not hurt.
- **Action.** The router turned "on the beach" into `place="beach"`, and the engine filtered on
  `photos.place`. That column is empty without reverse geocoding, so the filter removed every
  photo. I changed the engine: a place filters only if some photo's label matches it.
  Otherwise the place goes back into the CLIP text. Later I stopped it appending a place
  already in the text.
- **Result.** "a dog running on the beach" now ranks on "dog running beach" (see
  `eval/results/offline.json`).
- **Source.** Commits 1637f63 and 31367c6; `engine.py:84-90`.

### 4.5 The 7B model could not do date arithmetic

- **Situation.** The LLM router was much better than rules on people (97 vs 83) and place
  (95 vs 85), but worse on dates (82 vs 87).
- **Task.** Keep the LLM's language skill without its date errors.
- **Action.** I read the failures. The model read single days as half-open ranges (Christmas
  → 25–26 Dec), put "last summer" a year early, and invented dates for undated events. I
  had already put explicit season/week/month rules and examples in the prompt. Instead of more
  prompting, I built a hybrid: the rules' dates win whenever the rules match a date phrase.
- **Result.** All-filters accuracy: rules 58, LLM 73, hybrid 84. Dates went from 82 to 94/92.
  Paired on the same 100 queries, hybrid beat LLM-only on 11 queries and lost on 0.
- **Source.** Commits f0abb5f, fc80eb8; README "Query routing"; `router_predictions.jsonl`.
- **Caveat to own.** The hybrid commit (fc80eb8) comes before the eval commit (86969d4), but
  no doc says whether you looked at gold-set results before designing the hybrid. Answer this
  truthfully if asked.

### 4.6 232 threads on a shared server

- **Situation.** Face detection on a 216-core shared machine.
- **Task.** Be a good citizen; don't take the whole box.
- **Action.** insightface creates ONNX sessions without `SessionOptions`, so onnxruntime made
  one thread per core and logged `pthread_setaffinity_np failed` errors. I rebuild each session
  with `intra_op_num_threads` from `OMP_NUM_THREADS` (default 8) and `inter_op_num_threads=1`.
- **Result.** Bounded thread pool. `scripts/env.sh` also caps BLAS/OpenMP at 8.
- **Source.** Commit 4417f35; `detect.py:62-77`; `.claude/run/faces_eval.log`.

### 4.7 The system Ollama silently ran on CPU

- **Situation.** LLM router latency was far too slow.
- **Action.** The system `/usr/local/bin/ollama` could not read its `cuda_v12` library folder as
  a normal user, so it fell back to CPU without an error. No sudo. I unpacked the official
  v0.21.0 release tarball on warm storage, checked its sha256, and ran it on port 11534.
- **Result.** GPU inference; median 1.52 s per query. The permission fix needs an admin.
- **Source.** SETUP.md "Local LLM"; commit cf18911.

### 4.8 The first latency rep was 1.7× slower

- **Situation.** Retrieval latency over three reps of 500 queries.
- **Action.** With 20 warm-up queries the first rep was ~1.7× slower than the rest. I raised
  warm-up to 200 and report the median with the spread across reps.
- **Result.** Reps now agree: 2.77 / 2.77 / 2.77 ms on COCO. Recall unchanged.
- **Source.** Commit 69ea5b1.

### 4.9 LFW could not show anything

- **Situation.** LFW clustering starts at BCubed F1 99.83 before any correction.
- **Task.** Measure the active-learning loop on data where it can matter.
- **Action.** Added CPLFW (cross-pose LFW: 3 photos per person, different poses). It starts at
  95.91. I kept both sets, split by identity, tuned on dev, reported on test. I dropped a CelebA
  mirror unseen because its metadata showed similarity pre-filtering, and skipped AgeDB because
  of its licence.
- **Result.** A measurable effect: 41% error cut with margin vs 7% random.
- **Source.** Commit 78f285c; eval/README "Caveats".

---

## 5. Numbers sheet

### 5.1 The resume bullets as written

| Claim | What the repo shows | Verdict |
|---|---|---|
| "fine-tuning ArcFace embeddings on user correction signals" | ArcFace is frozen (`head.py:1-5`). A linear adapter is trained on corrections. It adds −0.07 (CPLFW) / +0.00 (LFW) BCubed F1 | **Not supported.** Remove. |
| "progressively refine identity clusters" | `refine()` after each round; F1 rises over 50 rounds | Supported, on a benchmark with a simulated user |
| "1,000+ photos" | Face eval: 5,606 test faces from 11,435 CPLFW photos (LFW: 2,975 test faces from 5,985 photos). "1,000" = simulated reviews. Flickr-1k demo index = 1,000 photos, but faces were never run on it. No real library was processed (`data/catalog.db`: 0 photos) | **Not supported as worded.** Replace with the real counts. |
| "on-device" | Runs locally. All numbers measured on a shared server with an NVIDIA H200 | Say "local" or "self-hosted". Never measured on a laptop or phone |
| "FAISS vector search, CLIP embeddings" | `IndexFlatIP` + `IndexIDMap2`; open_clip ViT-B-32 laion2b | Supported |
| "LLM-based query routing" | The LLM *parses* the query into filters. It does not route between backends | Wording is loose. Say "LLM query parsing" |
| "fully offline photo search" | Audit hook: 6 socket events, all 127.0.0.1 | Supported at process level, not OS level |

### 5.2 Every number

| Number | How it was measured | Caveats | Reproducible from repo? |
|---|---|---|---|
| **COCO-5k text→image R@1 39.4** (R@5 65.4, R@10 75.6) | `eval/retrieval.py coco_karpathy`. 5,000 Karpathy test images, all 25,010 captions as queries. Hit if the caption's own image is in the top K. Raw CLIP + exact FAISS. H200 GPU | Off-the-shelf zero-shot CLIP. You did not train it. It should sit close to open_clip's own published number for this checkpoint — check their table before quoting. Std. error ≈ ±0.3 pt (n=25,010; captions share images) | Script + JSON yes. Rerun blocked today by the broken env |
| Flickr-1k R@1 66.8 (88.3 / 93.1) | Same script, 1,000 images, 5,000 captions | Same | Same |
| Through `photoseek search` with rules: COCO R@1 35.9, Flickr 62.3 | `engine` arm: each caption through `RuleRouter` + `SearchEngine` | The rules turn "baby", "mother" into people filters, and an unnamed person matches nothing. 410 COCO captions got people/date filters. Costs 3.5–4.4 R@1 | Same |
| Query latency 2.8 ms (COCO), 2.4 ms (Flickr) | One caption: CLIP text encode on H200 + FAISS flat search over 5k/1k vectors on CPU. 200 warm-up, 3 reps × 500, median | Shared GPU; an earlier run measured ~4.5 ms. Excludes the LLM | Same |
| Image embedding 115 img/s (COCO), 141 img/s (Flickr) | Wall time of `embed_images`, batch 64, H200 | Includes NFS image reads | Same |
| **Router: all filters right 84 (hybrid) / 73 (LLM) / 58 (rules)** | `eval/router.py`. 100 hand-written queries with gold `StructuredQuery`, today pinned to 2026-10-22. "All filters" = people, date_from, date_to, place all correct. Scored on rep 1 of 3 | n=100 → 95% CI about ±7–9 pts (84: ~76–90; 58: ~48–67). Paired: hybrid beat rules on 29 queries, lost 3; beat LLM on 11, lost 0. Gold set written by the prompt author. 14 tags; several have n=2 | Script, gold, predictions yes. Needs the Ollama server |
| Per field (hybrid): people 97, date_from 94, date_to 92, place 95, text token-F1 0.79 | Same | Rules: 83/87/87/85/0.74. LLM: 97/82/82/95/0.80 | Same |
| LLM latency 1.52 s median, p90 2.19 s; rules 0.19 ms | 300 calls per arm, interleaved | GPU server; default Ollama quant of qwen2.5:7b (4.4 GB) | Same |
| LLM fallback 1 of 100; 3 queries changed across reps (hybrid: 1) | Same | Greedy decoding is still not perfectly stable | Same |
| **CPLFW BCubed F1 95.91 → 97.57 after 1,000 reviews** | `eval/faces.py --set cplfw --rounds 50`. 3,812 people, 11,435 photos, 11,215 faces. Split by identity: 1,906 dev / 1,906 test people. Settings picked on dev. 5,606 test faces. Simulated user answers 20 per round from gold labels; `refine()` after each round. **Margin** sampling | 95.91 = dev-chosen agglomerative 0.75 / min 2. Shipped default HDBSCAN = 91.75. Uncertainty sampling (shipped default) ends at 97.23. Simulated user is perfect. 3 photos per person — not like a family library. One run per arm (random: 3 seeds) | Script + JSON + plot yes. Rerun blocked by env |
| Error cut 41% (margin), 32% (uncertainty), 7% (random) | 1 − F1: 4.09% → 2.43% / 2.77% / 3.81% | Random median of 3 seeds, range 96.17–96.29 | Same |
| 1,000 reviews = 18% of test faces; 1,539 correction rows | `distinct_reviewed` = 1,000; a "name + confirm" writes 2 rows | — | Same |
| Propagation ≤ +0.09 pts on unreviewed faces | BCubed F1 on faces never asked about, vs the start clustering on the same faces | This is the honest "it barely spreads" number | Same |
| Adapter head: −0.07 (CPLFW), +0.00 (LFW) | Uncertainty with head vs head off, after 1,000 / 300 reviews | Head did train in all 50 rounds | Same |
| LFW 99.83 → 99.94 (margin, 300 reviews) | `--set lfw`, 423 people ≥5 photos, 2,975 test faces, 15 rounds | Saturated; ArcFace may have seen LFW people | Same |
| Seeded-refine bug: LFW recall 99.73 → 96.88 after 20 reviews; F1 99.83 → 99.58 at end | `faces_lfw_seeded_refine.json` | Old code; kept as a negative result | JSON yes; old code is in git history before b097f2f |
| Offline: 6 socket events, all 127.0.0.1; Ollama peers only 127.0.0.1 | `eval/offline.py`, 3 queries, `HF_HUB_OFFLINE=1`, audit hook + `ss` sampling | Process-level, not a network namespace | Script + JSON yes |
| Tests: 157 (README) | `pytest -q`, no weights | Last recorded run: 156 passed (22 Oct). 150 `def test_` functions. **Today: env broken, pytest exits 2** | No, until the env is fixed |
| Model sizes: CLIP 578 MB, buffalo_l 326 MB, qwen2.5:7b 4.4 GB | SETUP.md "Model cache" | — | Yes (files on warm storage) |

### 5.3 Two small doc errors to know about

- `faces/active.py:23` and `faces/head.py:4` say the ArcFace backbone has "25M parameters".
  The `w600k_r50.onnx` file is 174 MB. At 4 bytes per weight that is about **43M**. Say "about
  43 million" or "tens of millions" if asked.
- The 3-minute pitch above is yours to rehearse; nothing in it goes beyond the docs.

### 5.4 Proposed resume bullets (honest)

**Option A — full detail**

- "Built an offline face-clustering pipeline with active learning (SCRFD detection, frozen
  ArcFace embeddings, average-linkage clustering). Margin-based review sampling turns user
  corrections into hard clustering constraints. On 5,606 held-out cross-pose faces (CPLFW),
  1,000 simulated reviews raised BCubed F1 from 95.9 to 97.6, a 41% error cut vs 7% for
  random review."
- "Engineered a fully offline multimodal photo search engine: CLIP ViT-B/32 embeddings in an
  exact FAISS index (COCO-5k text→image R@1 39.4), SQL metadata filters, and a hybrid query
  parser (local qwen2.5-7B with schema-constrained JSON, plus rule-based dates) that got every
  filter right on 84% of 100 test queries, vs 58% for rules alone."

**Option B — shorter**

- "Built an offline active-learning face-clustering pipeline (SCRFD + frozen ArcFace); smart
  review sampling cut clustering error 41% after 1,000 reviews on 5.6k held-out faces, vs 7%
  for random review."
- "Built a fully offline photo search engine (CLIP + FAISS + local LLM query parser); hybrid
  LLM+rules parsing got all filters right on 84% of test queries vs 58% for rules."

Words to avoid: "fine-tuning ArcFace", "1,000+ photos", "on-device", "routing", "progressively
learns identities from corrections" (propagation was ≤ 0.1 point).

---

## 6. Prerequisite concepts

**CLIP and contrastive image-text learning.** Two encoders, one for images and one for text,
are trained so matching pairs have high cosine similarity and mismatched pairs low. The loss is
a symmetric InfoNCE over a batch. After training, text and images share one space, so a text
query can rank images with no task-specific training. That is "zero-shot retrieval".
Sources: Radford et al. 2021, *Learning Transferable Visual Models From Natural Language
Supervision*; Cherti et al. 2023, *Reproducible Scaling Laws for Contrastive Language-Image
Learning* (OpenCLIP/LAION).

**ViT-B/32 vs ViT-L/14.** "B" and "L" are model sizes (base, large). "/32" is the patch size in
pixels. Bigger patches mean fewer tokens, so B/32 is much cheaper than L/14 but less accurate.
`laion2b_s34b_b79k` means trained on LAION-2B for 34B samples with batch 79k.
Source: Dosovitskiy et al. 2021, *An Image is Worth 16x16 Words*.

**Recall@K for retrieval.** For each query, find the rank of the one correct item. R@K is the
share of queries where it is in the top K. COCO-5k (Karpathy split) and Flickr30k-1k are the
standard test sets; each image has 5 captions.
Source: Karpathy & Fei-Fei 2015, *Deep Visual-Semantic Alignments*.

**ArcFace.** A face-recognition loss. It adds an angular margin to the target class's angle
before the softmax. This packs each person into a tight cone on the unit sphere, so cosine
distance works as a same-person test. `buffalo_l` uses an IResNet-50 trained on WebFace600K.
Sources: Deng et al. 2019, *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*;
Zhu et al. 2021, *WebFace260M*.

**SCRFD and face alignment.** SCRFD is a single-stage face detector. It spends compute on the
scales that matter. It outputs boxes and five landmarks (eyes, nose, mouth corners). A
similarity transform maps those landmarks to fixed points in a 112×112 crop, which is what
ArcFace was trained on.
Source: Guo et al. 2022, *Sample and Computation Redistribution for Efficient Face Detection*.

**FAISS index types.** `IndexFlatIP` compares the query to every vector: exact, O(N).
IVF splits vectors into clusters and searches only a few: faster, approximate. PQ compresses
vectors into short codes: less memory, less accuracy. HNSW is a layered graph: fast and
accurate, but more memory and no deletes. `IndexIDMap2` maps your own IDs to rows.
Sources: Johnson, Douze & Jégou 2019, *Billion-scale similarity search with GPUs*;
Malkov & Yashunin 2018 (HNSW); Douze et al. 2024, *The Faiss library*.

**Agglomerative clustering and linkage.** Start with every point alone. Repeatedly merge the
two closest clusters. "Average linkage" defines cluster distance as the mean pairwise distance.
Cutting at a distance threshold decides the number of clusters for you. Cost is O(N²) memory.
Source: Müllner 2011, *Modern hierarchical, agglomerative clustering algorithms*.

**HDBSCAN.** A density-based method. It finds clusters as dense regions and labels sparse
points as noise. Its main knob is `min_cluster_size`. It does not use a distance threshold.
Sources: Campello, Moulavi & Sander 2013; McInnes, Healy & Astels 2017 (`hdbscan` library).

**BCubed, pairwise F1, NMI.** BCubed precision for one item is the share of its predicted
cluster that has its true label; recall is the share of its true group in its cluster. Average
over items. Pairwise F1 counts same-cluster pairs instead, so big clusters dominate it. NMI is
mutual information between labelings, normalised.
Sources: Bagga & Baldwin 1998; Amigó et al. 2009, *A comparison of extrinsic clustering
evaluation metrics*.

**Active learning: uncertainty and margin sampling.** Ask a human to label the items the model
is least sure of, instead of random ones. Uncertainty = lowest confidence (here: far from the
centroid). Margin = smallest gap between the top two choices. Random is the baseline.
Sources: Settles 2009, *Active Learning Literature Survey*; Scheffer et al. 2001 (margin).

**Constrained clustering.** Labels become constraints: must-link (same person), cannot-link
(not this person). Here pinned faces are fixed and rejects are cannot-links.
Source: Wagstaff et al. 2001, *Constrained K-means Clustering with Background Knowledge*.

**Supervised contrastive loss and adapters.** SupCon pulls together all items with the same
label and pushes others apart, using a softmax over similarities with a temperature. An
"adapter" is a small trainable layer on top of a frozen model. Here it is a 512×512 matrix
started at the identity and penalised by ‖W − I‖².
Sources: Khosla et al. 2020, *Supervised Contrastive Learning*; Houlsby et al. 2019 (adapters).

**Constrained decoding for LLM JSON.** At each step the decoder masks out tokens that would
break a grammar or schema. So the output is always valid JSON with the right keys. It does not
make the *values* right.
Sources: Willard & Louf 2023, *Efficient Guided Generation for LLMs* (Outlines); llama.cpp
GBNF grammars (Ollama uses llama.cpp).

**Quantised local LLMs.** Ollama serves GGUF weights, usually 4-bit. A 7B model shrinks to
~4–5 GB and runs on one consumer GPU or a fast CPU, at some cost in quality.
Sources: Qwen Team 2024, *Qwen2.5 Technical Report*; Frantar et al. 2023 (GPTQ) for background.

**Small-sample evaluation.** With n=100, a 95% interval is about ±8–10 points around 50–80%.
For two systems on the same queries, use a paired test like McNemar, which looks only at
queries where they disagree.
Source: McNemar 1947; Dietterich 1998, *Approximate Statistical Tests for Comparing
Supervised Classification Learning Algorithms*.

**SQLite WAL mode.** Write-ahead logging lets readers read while one writer writes. It relies
on shared memory and file locking, which NFS handles badly. That is why catalogs live on
`/data`, not warm storage.
Source: sqlite.org, "Write-Ahead Logging".

**Python audit hooks.** `sys.addaudithook` gets called on events like `socket.connect` and
`socket.getaddrinfo`. Useful to log every network attempt from your own process. It cannot
see other processes.
Source: PEP 578, *Python Runtime Audit Hooks*.

---

## 7. Fifty likely questions

### Faces: models and clustering

1. **Why ArcFace and not FaceNet or a CLIP image embedding?**
   ArcFace is trained for identity, with an angular margin. Same-person faces form tight cones,
   so one cosine threshold works. CLIP embeds the whole scene and is not trained for identity.
   FaceNet is older and weaker on current benchmarks. insightface also ships detector and
   recogniser in one pack.

2. **Did you fine-tune ArcFace?**
   No. I froze it. With tens to hundreds of labels, tuning ~43M parameters would overfit. I
   trained a small linear adapter on top instead. It added nothing measurable.

3. *Hostile:* **Your resume said you fine-tuned ArcFace. Which is it?**
   "I trained an adapter on the corrections, on top of frozen ArcFace. My own ablation showed
   it adds nothing, so I no longer credit it. The gain comes from choosing which faces to ask
   about and using the answers as hard constraints."

4. **Why is the adapter useless?**
   I have hypotheses, not proof. On CPLFW each person has at most 3 faces, so few positive
   pairs. The ‖W − I‖² penalty with weight 1.0 and 200 Adam steps keeps W near identity. And a
   linear map learned on ~1,000 faces of ~600 people may not generalise across poses. I would
   test with more labels per person and a weaker penalty.

5. **Why not k-means?**
   k-means needs the number of people up front. It forces every face into a cluster, including
   blurry background faces. Agglomerative with a threshold and HDBSCAN both infer the count
   and can leave faces unassigned.

6. **Why agglomerative on CPLFW but HDBSCAN as the shipped default?**
   The default (HDBSCAN, min size 3) was set before any eval. On the CPLFW dev half,
   agglomerative at 0.75 with min size 2 won (95.99 dev). On LFW, HDBSCAN won. CPLFW has
   exactly 3 photos per person, which punishes min size 3. I did not update the default. A
   real library has many photos per person, so I would re-tune on real data first.

7. **What does the 0.75 threshold mean?**
   Average cosine distance between two clusters' members. 0.75 distance = average cosine
   similarity 0.25. It sounds loose, but it is an average over all pairs, and cross-pose pairs
   of the same person are far apart. It was the best of 0.50–0.80 on dev.

8. **Does HDBSCAN use that threshold?**
   No. HDBSCAN uses only `min_cluster_size` (`cluster.py:309`). The threshold is used for
   seeding, refine, and placing new faces.

9. **What is the complexity of clustering?**
   `_unsupervised` builds a full N×N float64 distance matrix (`cluster.py:293`). For 5,606
   faces that is ~250 MB. At 100k faces it is 80 GB, so it would not scale. I would switch to
   a kNN graph (FAISS) and cluster on that.

10. **How do you handle a new photo after clustering?**
    `cluster_library` without `--recluster` places only unassigned faces at the nearest
    centroid within the threshold (`pipeline.py:102-118`). It never renumbers identities.

11. **How do identity IDs survive a recluster?**
    `persist` lets the largest cluster pick first. It takes the old identity most of its faces
    had, if nobody claimed it. So "Mom" stays "Mom" (`cluster.py:341-395`).

12. **Why align faces on landmarks?**
    ArcFace was trained on 112×112 crops aligned to five fixed points. Raw boxes shift and
    rotate the face, which degrades the embedding. The detector already gives landmarks, so it
    is free.

13. **Why drop faces under 40 px or score under 0.6?**
    Those embeddings are mostly noise. One noisy vector can sit between two people and bridge
    their clusters.

### Active learning and corrections

14. **How do corrections propagate?**
    Two steps. First, `record()` applies the answer to the catalog at once: confirm pins the
    face, reject unassigns it and stores a cannot-link, name renames or merges, move splits.
    Second, `refine()` uses them: pinned faces fixed, rejects forbidden, and other faces move
    only if another identity is strictly closer by average linkage. A pin changes an identity's
    member set, which changes distances for its neighbours. That is the only spreading path.

15. **How much did they propagate?**
    Very little. On CPLFW, faces never reviewed gained at most +0.09 points. Almost all the gain
    is the reviewed faces themselves being fixed. The win is *finding* the wrong ones fast.

16. *Hostile:* **So it is just a labelling tool with a sort order?**
    "Largely, yes, and the eval says so. But the sort order matters a lot: the same 1,000
    reviews cut error 41% with margin and 7% with random. In a real app that is the difference
    between users fixing their library and giving up."

17. **Uncertainty vs margin — which is better and why?**
    Margin won on CPLFW (97.57 vs 97.23) and LFW. Uncertainty asks about outliers far from
    their own centroid; many are just bad photos of the right person. Margin asks about faces
    between two people, which is where errors live.

18. **Where does the gain come from in the curve?**
    In the first ~300 reviews, the samplers mostly ask about the 251 unassigned faces; they drop
    to 0 by review 300. Placing them raises recall. After that, they fix misassigned outliers,
    and precision rises from 98.1 to 99.2–99.6. From 300 to 1,000 the gain is small.

19. **Why the per-identity cap of 2?**
    Without it, one messy cluster could take all 20 questions. The cap spreads effort across
    people, and the batch is filled from skipped items if short (`active.py:401-415`).

20. **Is the simulated user realistic?**
    No. It answers perfectly from gold labels and never skips. Real users make mistakes, and
    pinned mistakes are never undone by the model. I would add a noisy-user arm and a "show me
    the photo again" path.

21. **What happens if a user confirms a wrong face?**
    It is pinned and becomes a hard constraint. The model never moves it. It also pulls that
    identity's mean toward the wrong person. The user can fix it with another correction, but
    nothing flags it.

22. **Why a no-answers control?**
    It measures what `refine()` alone does. It should be flat. It caught the centroid-vs-linkage
    bug, where refine moved 100–170 faces with zero new information.

23. **Why split by identity, not by photo?**
    If one person's photos are in both halves, settings tuned on dev leak into test. Splitting
    people keeps test identities unseen.

24. **Why BCubed F1 as the headline?**
    It scores each face, so big and small clusters count fairly. Pairwise F1 is dominated by
    large clusters. Unassigned faces count as singletons, which is how a user sees them.

### Retrieval

25. **Why CLIP, and why that variant?**
    CLIP puts text and images in one space, which is exactly what text search over photos needs.
    ViT-B-32 with LAION-2B weights was set in the scaffold config. It is small (578 MB), fast,
    and runs on CPU. I did not ablate other variants; that is a gap.

26. **Is 39.4 R@1 good?**
    It is what this off-the-shelf checkpoint gets zero-shot. I did not train CLIP. Larger
    models like ViT-L/14 or SigLIP score clearly higher. The number shows the pipeline is
    wired correctly, not that I improved retrieval.

27. *Hostile:* **So the 39.4 is not your work?**
    "Correct. The model's quality is not mine. My work is the system around it: filter-then-rank,
    the parser, the offline packaging, and measuring the cost of parsing — the rule parser
    costs 3.5 points of R@1 on captions, and I can explain why."

28. **Why a flat index, not HNSW or IVF-PQ?**
    A personal library is thousands to tens of thousands of photos. Flat search is exact and
    takes about a millisecond. IVF needs training and loses recall. HNSW uses more memory and
    does not support deletes, which I need when a photo is re-embedded.

29. **When would you switch?**
    Past roughly a million vectors, or when latency matters on weak hardware. I would try IVF
    with a few thousand lists first, then PQ if memory is tight. The index is rebuildable from
    the catalog, so switching is a rebuild.

30. **How do filters combine with vector search?**
    SQL returns the candidate photo IDs. FAISS searches with `IDSelectorBatch` over those IDs.
    Empty candidates return empty. No filters means search everything (`faiss_store.py:131-137`).

31. **Why filter first instead of post-filter?**
    Post-filtering a top-40 can leave zero results when the filter is strict, such as one person
    in one month. Pre-filtering guarantees the top-k all satisfy the filters.

32. **What's the latency breakdown?**
    2.8 ms is CLIP text encoding on an H200 plus flat search over 5k vectors. The LLM adds ~1.5 s.
    Rules add 0.2 ms. On a laptop CPU, text encoding would be slower; I did not measure that.

33. **Why does the rule parser lower R@1 on COCO?**
    Captions like "a mother holding a baby" trigger kinship words. They become people filters.
    In the eval nobody is named "mother", so the query returns nothing. 410 of 25,010 COCO
    captions got people or date filters.

### Query parsing and the LLM

34. **Why Ollama?**
    It runs local models behind one HTTP call, supports JSON-schema output, and uses the GPU
    when it can. It keeps the core free of a heavy LLM dependency. One `urllib` POST is all
    the client code.

35. **Why qwen2.5:7b?**
    It was chosen in the scaffold, and no doc records why. What I can defend: it is strong at
    instruction following and JSON at 7B, and its quantised download is 4.4 GB. I did not
    compare it with other models; that is something I would do next.

36. **What does schema-constrained decoding buy you?**
    The output always parses and has the right keys. It does not make values correct. The model
    still wrote wrong dates in valid JSON.

37. **Why temperature 0?**
    Parsing should be repeatable. Even so, 3 of 100 queries changed across reps for the LLM arm,
    so greedy decoding on a GPU is not perfectly stable.

38. **Why a hybrid instead of a better prompt?**
    The prompt already states the date conventions with examples. The model still failed date
    arithmetic. Rules do dates exactly when the phrase is known. So each part does what it is
    good at. Paired, the hybrid beat the LLM-only on 11 queries and lost on 0.

39. **What are the remaining failures?**
    16 of 100. Nine are dates: holidays (Christmas read as 25–26 Dec, Halloween as all October,
    New Year's Eve off by a year), "two years ago", "last weekend", and invented dates for
    birthdays. Five are places: missed "Paris", "Taj Mahal", "garden", "zoo"; one false "beach".
    Three are fake people: "children", "dog", "family".

40. **How would you fix the fake people?**
    Check each LLM-extracted person against known identity names. If unknown and not a kinship
    word, fold it back into the CLIP text, like unknown places. That fixes "kids", "dog",
    "family" with no model change.

41. **How trustworthy is the 84%?**
    It is 100 queries I wrote myself, so the CI is about ±7 points, and there is author bias.
    I excluded the probe queries I used while writing the prompt and did not change the prompt
    after scoring. The 58 → 84 gap is large and paired (29 wins, 3 losses), so the direction
    is solid. The exact value is not.

42. *Hostile:* **Did you design the hybrid after seeing the test set?**
    Answer truthfully. The hybrid commit comes before the eval commit, but the docs do not
    settle what you saw first. If you looked at gold results, say so and say the 84 is
    therefore optimistic.

43. **Is "router" the right word?**
    Not really. It parses a query into slots. It does not route to different backends. I would
    call it a query parser now.

### Systems, scale, offline

44. **How do you prove it is offline?**
    I ran the real CLI under a Python audit hook that logs every connect, sendto, and DNS lookup.
    In parallel I sampled the Ollama server's sockets with `ss`. All 6 events went to 127.0.0.1.
    It is process-level evidence. A network namespace would be stronger, but I had no root.

45. **What if Ollama is down?**
    The LLM call fails, and the router falls back to the rules. Search still works, with lower
    parsing quality. `last_fallback` records it.

46. **Why SQLite, and why are vectors in it?**
    One file, no server, transactions, and it's in the stdlib. The catalog is the truth, so
    vectors live there as float32 blobs. The FAISS file can be deleted and rebuilt.

47. **What breaks at 1 million photos?**
    Face clustering first: the N×N matrix. Then the per-row Python loops in `refine()`. Then
    `IDSelectorBatch` with huge candidate sets. CLIP indexing time is linear and fine. I would
    use a kNN graph for faces and IVF for CLIP.

48. **How is the code tested without weights?**
    Every model sits behind a Protocol. Tests use fake embedders and stores. 150 test functions
    cover routing, catalog, clustering logic, the loop, metrics, and the UI. The real models are
    exercised only by the eval scripts.

### Reflection

49. **What would you do differently?**
    Test on a real family library with a few real users early. Make the full recluster path use
    the same fixed logic as `refine()`. Drop the adapter until I can show it helps. Ablate CLIP
    and LLM sizes. Build reverse geocoding before any place logic.

50. **What comes next?**
    Offline reverse geocoding (GeoNames + a k-d tree) so places can filter. Validate LLM people
    against known names. A holiday calendar in the rules. A noisy-user eval. Label propagation
    over a kNN graph so a correction reaches near-duplicate faces. A smaller LLM, or caching, to
    cut the 1.5 s.

---

## 8. Where the honest answer is "it didn't work" or "I didn't do that"

How to say it well: state the fact first, give the number, say what you learned, say what you
would try. Don't apologise twice. Don't hedge the fact.

| Question | Honest answer | How to say it |
|---|---|---|
| Did you fine-tune ArcFace? | No. Frozen backbone, linear adapter on top | "I chose not to. With so few labels it would overfit. I tried an adapter instead, and my ablation showed it adds nothing." |
| Does the adapter head help? | No: −0.07 / +0.00 | "No. I kept it in the report as a negative result. The constraints do the work." |
| Do corrections spread to other faces? | Barely: ≤ +0.09 pts | "Very little. The loop's value is finding wrong faces fast, not spreading fixes. Spreading is my next problem." |
| Did you run it on a real photo library? | No. Only benchmarks; the default catalog is empty | "No. I evaluated on LFW and CPLFW, plus a 1,000-image Flickr set for the search demo. A real family library is the next test." |
| What is "1,000+ photos"? | Not a real count. 1,000 = reviews. Test set = 5,606 faces | "That was loose wording. The real numbers are 5,606 test faces and 1,000 simulated reviews." |
| Was it measured on-device? | No. Shared server, H200 GPU | "It runs locally, but I measured on a server GPU. I haven't measured a laptop." |
| Did you test with real users? | No. A perfect simulated user | "No. The user is simulated from gold labels. Real users make errors, which my design pins as hard constraints — a known risk." |
| Does place search work? | Only visually. No reverse geocoding | "Not as a filter yet. Without geocoding, place words go to CLIP. I built the fallback after the filter wiped out every result." |
| Did you compare CLIP models or LLMs? | No | "No. Both came from my first config. An ablation is cheap with my eval scripts, and I'd run it next." |
| Is the router fast? | LLM path 1.5 s vs 0.2 ms rules | "The LLM path is slow for interactive search. A smaller model, or skipping the LLM when rules cover the query, would fix most of it." |
| Is the shipped default the best setting? | No. Default HDBSCAN = 91.75 on CPLFW; tuned = 95.91 | "No. I tuned on benchmarks but didn't change the default, because CPLFW's 3 photos per person is unlike a real library." |
| Are the two refine bugs fully fixed? | In `refine()` yes. **Not** in `cluster()` and `assign_new_faces()` | See below. |
| Can you reproduce the numbers today? | Not right now. Env can't import numpy | "The scripts and JSONs are there, but my env broke on an MKL library. Rebuilding from the recorded spec should fix it." |
| Is the offline proof airtight? | Process-level only | "It covers my process and the Ollama server. A network namespace would be airtight; I had no root on that machine." |
| Is the 84% unbiased? | Self-written gold set, n=100 | "It's my own test set, so treat it as ±7 points with author bias. The paired gap over rules is large." |

**Latent bugs you should know before an interviewer finds them.**

- `IdentityClusterer.cluster` (`cluster.py:117-142`) still seeds identities from pinned faces
  only, and compares faces to a *normalised* centroid against the linkage threshold. Those are
  the two bugs fixed in `refine()`. `photoseek faces --recluster` calls this path with pinned
  faces (`pipeline.py:92-100`). So a user who labels faces and then reclusters can trigger the
  old split-people behaviour.
- `assign_new_faces` (`cluster.py:314-339`) also uses a normalised centroid against the linkage
  threshold. So newly added photos may over-assign noise faces, as in bug 4.2.
- `RuleRouter._extract_people` loops over the `_KINSHIP` *set* (`router.py:208`). Set order of
  strings changes with Python's hash seed, so the order of kinship people can vary between runs.
  The filter result is the same; only the order differs.
- Say it this way: "I fixed it where my eval measured it. I found two other paths with the same
  pattern and would apply the same fix and add a test."

**Commit history.** The `main` scaffold is dated 2026-08-18. All 18 feature, fix, and eval
commits on this branch are dated 2026-10-22; one docs commit is 2026-11-04. An interviewer
reading GitHub may ask how so much landed in one day. Explain your actual workflow truthfully,
including any AI coding tools you used. Being able to explain every line is what matters.

---

## 9. Self-test exercises

Predict the effect before you read the answer. Answers are at the bottom.

**E1. Re-introduce the centroid bug.** In `cluster.py:232-233`, normalise the means:
`means = _normalize(np.stack([X[m].mean(axis=0) for m in members]))`. Re-run
`eval/faces.py --set cplfw`. What happens to the no-answers control row, and to precision in
round 1 of every arm?

**E2. Drop the rule dates from the hybrid.** Delete `router.py:386-387` (the
`if ruled.date_from or ruled.date_to:` override). Re-run `eval/router.py`. What is the hybrid's
all-filters score, and its date_from accuracy?

**E3. Let lowercase words be people.** In `router.py:230`, change
`return term[0].isupper() and term.isalpha()` to `return term.isalpha()`. What does
`photoseek search "photos of sunsets"` return? What happens to the COCO `engine` arm R@1, and
to the gold set's "visual" tag?

**E4. Swap the index to HNSW.** In `faiss_store.py:79`, use
`faiss.IndexIDMap2(faiss.IndexHNSWFlat(self._dim, 32))`. Run `photoseek index` twice on a
library where one photo was re-embedded. What breaks, and what happens to `SearchHit.score`?

**E5. Stop asking about unassigned faces.** In `active.py:115`, exclude faces whose `assigned`
is `None` from `candidates`. Re-run the CPLFW uncertainty arm. What happens to the curve
between 100 and 300 reviews, and to the final unassigned count?

### Answers

**E1.** The control row stops being flat. Per commit 034ce8c, the first refine pulls 100–170
noise faces into identities and every arm, random included, loses about 2 points of BCubed
precision in round 1. Unassigned count drops with no answers. Random sampling ends below its
start. Reason: a normalised centroid is closer to every face than the members are on average,
so more faces pass the 0.75 linkage cut.

**E2.** The hybrid becomes the plain LLM router: same people, place, text, and dates. Expect
all-filters ≈ 73 and date_from ≈ 82, the `ollama` row. Small differences can come from the 3
queries that change across reps.

**E3.** "photos of sunsets" gives `people=["sunsets"]`. No identity has that name, so the engine
returns nothing and prints "unknown people ['sunsets']". On COCO, many more captions
("a plate of food", "a group of people") turn into people filters and return nothing, so the
engine arm R@1 drops well below 35.9. On the gold set, "visual" queries with "of"/"with" fail.
The router tests that guard lowercase nouns (commit f8d7dbb) should fail.

**E4.** Two things. HNSW does not support `remove_ids`, and `FaissStore.add` removes existing
IDs before adding (`faiss_store.py:97-98`). So re-adding a known ID raises a RuntimeError.
Also, `IndexHNSWFlat(d, 32)` defaults to L2. On unit vectors the ranking is the same, but the
score becomes squared L2 distance, 2 − 2·cos, where lower is better. The `score` field no
longer means cosine similarity, and any test that checks score values fails. Filtered search
also gets approximate.

**E5.** This is a prediction from the curves, not a measured result. In the real run, the
uncertainty arm cleared about 20 unassigned faces per round, from 251 down to 0 by review 300.
Without them, the recall gain from placing those faces disappears. The final unassigned count
stays near 251, minus what `refine()` places on its own (the control shows refine alone places
none). The sampler goes straight to misassigned outliers, so precision may rise sooner. Overall
F1 at 1,000 reviews should end lower than 97.23.

---

*Environment note.* `photo-seek` (Python 3.11.17) is recorded in
`/data/jayavardhan/env-registry/photo-seek/` (archived 2026-11-04) but still exists, and
`import numpy` fails on a missing `libmkl_gf_lp64.so.3`. Rebuild from `environment.yml` or
`spec-exact.txt` before any rerun. Nothing was installed or changed for this document.
