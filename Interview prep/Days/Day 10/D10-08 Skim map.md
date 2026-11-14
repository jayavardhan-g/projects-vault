---
type: study-block
day: 10
block: 8
tags: [interview, day10, skim]
---
# D10-08 · Skim map: one line per function (30 min)

Hub: [[Day 10 - Photo-Seek I]] · Previous: [[D10-07b Search engine code]] · Next: [[D10-09 Rebuild]]

**How to skim.** For each file read only: the module docstring, the signatures, the docstrings. Do not read bodies. If you can say each line below from memory, the file is done. Line numbers are real (`upgrade-2026-10`). Descriptions come from the docstrings and signatures; bodies were read only for `cmd_search` and `cmd_index`.

## config.py: typed settings over a YAML file

| line | name | responsibility |
|---|---|---|
| 22–104 | `LibraryConfig`, `StorageConfig`, `EmbeddingConfig`, `ClusteringConfig`, `FacesConfig`, `ActiveLearningConfig`, `SearchConfig`, `LoggingConfig` | one dataclass per section of the YAML, each with defaults |
| 41–57 | `StorageConfig.*_path` | build full paths: `catalog.db`, `clip.faiss`, `faces.faiss`, `face_head.npy` under `storage.root` |
| 109 | `Config` | the root object holding all sections |
| 119 | `Config.load` | read `configs/default.yaml`, merge the user's YAML over it, build the dataclasses |
| 153 | `deep_merge` | merge two dicts key by key; nested dicts merge, everything else (lists too) is replaced whole |
| 166 | `_build` | dict → nested dataclasses; **unknown keys are ignored**, so an older build tolerates a newer config |
| 182 | `_field_type` | resolve a string type annotation to the real class |
| 193 | `_unbuild` | dataclasses → plain dict (for `to_dict`) |

Values to know: `embedding.batch_size = 32`, `normalize = true`; `search.router = "rule"`, `ollama_host = http://127.0.0.1:11534`, `ollama_timeout_s = 30`, `top_k = 40`.

## catalog/db.py: the SQLite source of truth

| line | name | responsibility |
|---|---|---|
| 30, 48 | `PhotoRecord`, `FaceRecord` | plain records passed into the catalog |
| 57 | `Catalog` | a connection wrapper with only the queries the pipelines need; a context manager |
| 73 | `open` | connect, turn on foreign keys, **WAL mode** (a long index job can read while the CLI writes), apply the schema |
| 90 | `migrate` | run `schema.sql`; every statement is `CREATE IF NOT EXISTS`, so it is safe to repeat |
| 128 | `add_photo` | insert a photo; returns its id, or `None` if the path is already there (re-ingest is a no-op) |
| 155 | `add_photos` | bulk insert in one transaction |
| 164, 167 | `get_photo`, `photo_by_path` | fetch one row by id or by path |
| 170 | `has_hash` | is this exact file content already catalogued under any path? |
| 180 | `iter_photos` | stream photos, optionally only those with no CLIP embedding yet |
| 187 | `mark_indexed` | stamp `indexed_at` once a photo has an embedding |
| 209 | `put_embedding` | upsert a photo's vector (model name, dim, float32 blob) |
| 219, 225 | `get_embedding`, `iter_embeddings` | read one vector; stream all of them (for an index rebuild) |
| 232–311 | `add_face`, `assign_identity`, `create_identity`, `rename_identity`, `merge_identities`, … | the face side (tomorrow). `merge_identities` moves faces and leaves a `merged_into` tombstone |
| 323–358 | `add_correction`, `pending_corrections`, `consume_corrections` | the append-only correction log (tomorrow) |
| 369, 375 | `pack_vector`, `unpack_vector` | list of floats ↔ little-endian float32 bytes |

Tables [code: `schema.sql`]: `photos`, `faces`, `identities`, `embeddings`, `corrections`, `meta`. Indexes exist on `photos.taken_at` and `faces.photo_id`, which the search filters use.

## ingest/scanner.py: find the photos worth cataloguing

| line | name | responsibility |
|---|---|---|
| 30 | `ScanResult` | what a scan found: new photos, skipped, errors |
| 50 | `hash_file` | **BLAKE2b, 128-bit** content hash, read in 1 MiB chunks |
| 60 | `iter_image_paths` | walk the roots; keep known image extensions; prune excluded folders during the walk |
| 89 | `scan` | for each path: skip tiny files (`min_bytes`), skip unchanged files (same path + mtime), hash, skip duplicate content; **writes nothing** |
| 151 | `_enrich` | add EXIF fields to a record; a metadata failure is never fatal |

## ingest/exif.py: metadata out of the image file

| line | name | responsibility |
|---|---|---|
| 20 | `PhotoMetadata` | the EXIF fields the catalog keeps; every field optional |
| 53 | `read_metadata` | size (corrected for orientation), `taken_at` (DateTimeOriginal → DateTime → file mtime), GPS, camera |
| 109 | `to_decimal_degrees` | degrees/minutes/seconds → signed decimal degrees |
| 119 | `_parse_exif_datetime` | parse `"YYYY:MM:DD HH:MM:SS"`; unset clocks (zeros) count as missing |
| 145 | `_register_heif` | enable HEIC reading if `pillow-heif` is installed |

Link to today: `taken_at` is what the date filter compares. **If EXIF is missing, the file's mtime is used** [code: docstring line 58–59], so a copied or downloaded photo can carry the wrong date and be filtered out wrongly. GPS is stored, but nothing turns it into `place`.

## cli.py: the commands

| line | name | responsibility |
|---|---|---|
| 28 | `build_parser` | define the sub-commands and flags (argparse) |
| 80 | `cmd_ingest` | scan folders and add new photos to the catalog |
| 112 | `cmd_index` | embed un-indexed photos with CLIP in chunks of 256; write each vector to SQLite **and** FAISS; save the index |
| 153 | `cmd_faces` | detect, embed and cluster faces (tomorrow) |
| 176 | `cmd_label` | the review loop in the terminal (tomorrow) |
| 226 | `cmd_search` | build the parser, parse, then `--json` / `--dry-run` / run the engine and print |
| 268 | `cmd_status` | print counts |
| 296 | `cmd_ui` | start the web UI |
| 315 | `main` | parse args, load the config, dispatch; map errors to exit codes 2, 3, 130 |

## eval/retrieval.py: R@1 / R@5 / R@10

| line | name | responsibility |
|---|---|---|
| 31 | `recall_at` | from the 0-based rank of each query's right image, the share with rank < K |
| 36 | `main` | embed the images, build a `FaissStore`, run every caption through two arms (`clip`: straight to CLIP + FAISS; `engine`: through `RuleRouter` + `SearchEngine`), time them, write JSON |

`recall_at` in full, since it defines 39.4:

```python
return {f"R@{k}": sum(1 for r in ranks if r is not None and r < k) / len(ranks) for k in ks}
```

Ranks `[0, 2, 0, 11]` → R@1 = 2/4, R@5 = 3/4, R@10 = 3/4. A rank of `None` (image not returned) counts as a miss.

## eval/router.py: parser accuracy

| line | name | responsibility |
|---|---|---|
| 32–33 | `TODAY`, `FIELDS` | today pinned to 2026-10-22; the four filter fields |
| 41, 51 | `norm_people`, `norm_place` | normalise before comparing: case, "my ", "'s", leading "the " |
| 58, 62 | `tokens`, `token_f1` | content-word overlap F1 for the `text` field |
| 75 | `score` | per-field 0/1 plus `filters` = **all four fields right at once** |
| 88 | `main` | run rules / LLM / hybrid on 100 gold queries, interleaved, 3 reps; write JSON and per-query predictions |

The line that defines "84": `s["filters"] = float(all(s[f] for f in FIELDS))` (line 84).

## eval/offline.py: the network audit

| line | name | responsibility |
|---|---|---|
| 37 | `_hook` | a Python **audit hook**: records every `socket.connect`, `socket.sendto` and DNS lookup |
| 48, 58 | `_is_loopback`, `_event_ok` | is this destination 127.0.0.1 / localhost / a Unix socket? |
| 66 | `sample_ollama` | every 0.2 s, list the Ollama processes' TCP peers with `ss` |
| 80 | `main` | run real searches (hybrid) under the hook; pass only if every destination is loopback |

## Check

1. Which function decides what `taken_at` is when a photo has no EXIF?
2. Where is the single line that makes "84" a per-query number?
3. `cmd_index` writes each vector to two places. Which, and which one is the source of truth?
4. What does the offline audit hook see, and what can it not see?

> [!success]- Answers
> 1. `read_metadata` in `exif.py`: it falls back to the file's mtime.
> 2. `eval/router.py:84`, `all(s[f] for f in FIELDS)`.
> 3. SQLite (`put_embedding`) and FAISS (`store.add`); SQLite is the source of truth.
> 4. Socket connects, sends and DNS lookups made by this Python process (plus Ollama's peers via `ss`). It cannot see other processes or enforce anything at the OS level.
