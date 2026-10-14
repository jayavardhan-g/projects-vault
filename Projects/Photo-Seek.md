---
type: project
folder: projects/photo_seek
repo: github.com/jayavardhan-g/Photo-Seek
dates: 2026-08-18 → 2026-08-25
status: scaffold — model-backed parts not implemented
kind: application — offline photo search and face clustering
stack: [Python 3.11, SQLite, CLIP, ArcFace/insightface, FAISS, Ollama qwen2.5:7b]
tags: [project, app, retrieval]
---
# Photo-Seek — offline photo organisation and natural-language search

**What.** It has two fully offline pipelines over a personal photo library:
- **Faces.** Detect faces, embed them with ArcFace, cluster them into identities, then **active learning**: the system surfaces the least-confident
  assignments, and the user's merge/split/name corrections refine the clusters and a small embedding head.
- **Retrieval.** Embed photos with CLIP into a FAISS index. A local LLM router turns "mom at the beach last summer" into a
  `StructuredQuery` (people, date range, place, semantic text). Metadata filters narrow the set, and vector search ranks what is left.
  A rule-based router is the fallback.

**State: scaffold.** Implemented and tested: config, SQLite catalog, filesystem scan with hashing and dedupe, query types, the rule-based router and the
correction store. **Interfaces only:** EXIF, CLIP/ArcFace, FAISS, detection, clustering, the active-learning sampler and search
orchestration. There are 11 commits, all from 2026-08-18. The last session (2026-08-25) only did env setup, and its checkpoint is in `.claude/RESUME.md`.

**Run it.** The env `photo-seek` (Python 3.11.16) was **archived and removed**. Its record is in `/data/jayavardhan/env-registry/photo-seek/`.
Recreate it with `mamba env create --name photo-seek --file environment.yml`, then `pytest -q` and
`photoseek search "..." --dry-run`. The model tier (`requirements-models.txt`) was never installed.

**House rules** (from its `CLAUDE.md`): no Claude co-author trailers, never commit `CLAUDE*.md` or `.claude/`, and never push unless asked.

**Next step.** Implement the CLIP embedder and FAISS store first, which makes the search path work end to end.

**Docs in this vault** (copied 2026-10-14 from `/data/jayavardhan/projects/photo_seek/`):
- [[Sources/photo_seek/README|README]]
- [[Sources/photo_seek/SETUP|SETUP]]

**Source.** `README.md`, `SETUP.md`, `CLAUDE.md`.
