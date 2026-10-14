# Photo-Seek

Offline-first photo organization: cluster the people in your library, and search
it in plain English. Everything runs on your own machine — no photo, embedding,
or query ever leaves it.

```
$ photoseek search "photos of mom at the beach last summer"
```

That query is parsed into structured filters (`people=["mom"]`,
`place≈"beach"`, `date_from=2025-06-01`), the catalog narrows to the matching
photos, and a CLIP vector search ranks what's left by how well it matches
"at the beach".

## Two pipelines

**Face clustering with active learning.** Faces are detected, embedded with
ArcFace, and clustered into candidate identities. Clustering a real library is
never right the first time — so the interesting part is the correction loop: the
system surfaces the assignments it is least confident about, you merge, split,
or name them, and those corrections become training signal that refines both the
cluster assignments and a lightweight head over the embeddings. Accuracy climbs
with the handful of labels you actually give it, instead of demanding hundreds
up front.

**Multimodal retrieval.** Photos are embedded once with CLIP and indexed in
FAISS. A local LLM acts as a query router, translating free-form questions into
a `StructuredQuery` — semantic text, people, date range, place, limit. Metadata
filters cut the candidate set down; the vector search ranks within it. The
router sits behind an interface with a deterministic rule-based fallback, so
search works even with no model server running.

## Status

Early. The repository is scaffolded: configuration, the SQLite catalog, the
filesystem scanner, structured-query types, and the rule-based router are
implemented and tested. The model-backed pieces — CLIP, ArcFace, FAISS, the
Ollama router, the clustering and active-learning loop — exist as typed
interfaces awaiting implementation.

| Component | State |
|---|---|
| Config, logging | implemented |
| SQLite catalog + DAO | implemented |
| Filesystem scan, hashing, dedupe | implemented |
| EXIF extraction | interface |
| CLIP / ArcFace embedders | interface |
| FAISS vector store | interface |
| Face detection, clustering | interface |
| Active-learning loop | correction store implemented, sampler interface |
| Query router | rule-based implemented, Ollama backend written |
| Search orchestration | interface |

## Architecture

```
photos on disk
      │
   ingest/         scan, hash, dedupe, EXIF
      │
   catalog/        SQLite: photos, faces, identities, corrections
      ├──────────────────────────────┐
   embed/clip                    embed/arcface
      │                              │
   index/faiss                   faces/cluster
      │                              │
      │                          faces/active     ← your corrections
      │                              │
      └──────► search/engine ◄───────┘
                    ▲
              search/router          natural language → StructuredQuery
```

## Usage

```bash
photoseek ingest ~/Pictures      # scan and catalog
photoseek index                  # compute CLIP embeddings, build FAISS index
photoseek faces                  # detect, embed, cluster faces
photoseek label                  # review the most uncertain assignments
photoseek search "..."           # natural-language search
```

Add `--dry-run` to `search` to see the parsed `StructuredQuery` without touching
the index.

## Install

See [SETUP.md](SETUP.md).

## License

MIT
