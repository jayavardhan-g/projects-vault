---
type: project
folder: projects/photo_seek
repo: github.com/jayavardhan-g/Photo-Seek
dates: 2026-08-18 → 2026-08-25; upgrade 2026-10-22 → 2026-11-04
status: implemented and evaluated — branch upgrade-2026-10, local only (repo forbids pushing)
kind: application — offline photo search and face clustering
stack: [Python 3.11, SQLite, open_clip ViT-B-32, FAISS, insightface SCRFD + ArcFace buffalo_l, scikit-learn, Ollama qwen2.5:7b, FastAPI]
tags: [project, app, retrieval]
---
# Photo-Seek — offline photo organisation and natural-language search

**What.** It has two fully offline pipelines over a personal photo library:
- **Faces.** SCRFD detection, 5-point alignment and ArcFace, clustered into identities (agglomerative or HDBSCAN), then **active learning**:
  the system asks about the least-confident faces (uncertainty / margin / random), applies confirm/reject/merge/split/name corrections, fits a
  512×512 **adapter head on frozen ArcFace** (supervised contrastive, regularised to identity), and updates the clustering under the corrections.
- **Retrieval.** CLIP embeddings in an exact FAISS index. A router turns the query into a `StructuredQuery`. There are three routers: rules,
  qwen2.5:7b (JSON schema-constrained, with a rule fallback), and a hybrid (LLM fields + rule dates). SQL filters run first, then vector ranking.
- A local FastAPI page covers search and face review; `photoseek label` is the terminal version.

**State (2026-11-04).** Everything is implemented, with 157 tests and none needing weights. There are 20 commits on `upgrade-2026-10` (last
`aa5d476`). They are **not pushed**, because the repo's CLAUDE.md forbids pushing. No reverse geocoding yet, so a place in a query matches
visually unless a photo label matches.

**Numbers** (every one from `eval/*.py` → `eval/results/*.json`, rendered into the README by `eval/report.py`):
- **Retrieval**, text→image R@1/5/10: COCO 5k **39.4 / 65.4 / 75.6**; Flickr 1k **66.8 / 88.3 / 93.1**. Going through `photoseek search`
  with the rule router costs 3.5–4.4 R@1, because words like "baby" become people filters. 2.4–2.8 ms/query on an H200.
- **Router**, 100 hand-written gold queries, all filter fields right: rule **58%**, LLM **73%**, hybrid **84%**. Latency: 0.19 ms rule,
  1.52 s LLM (p90 2.19 s). The LLM is worse than the rules on dates (82% vs 87%): it uses half-open single days, gets "last summer" a year
  early, and invents dates.
- **Faces, LFW** (≥5 photos, 212 test people): BCubed F1 **99.83** before any correction. It is saturated, so it cannot show learning.
- **Faces, CPLFW** (cross-pose, 1,906 test people, 5,606 faces): default HDBSCAN 91.75, dev-chosen agglomerative **95.91**. After
  1,000 reviewed faces: margin **97.57** (error −41%), uncertainty 97.23 (−32%), random 96.19 (−7%). A no-answers control stays at 95.91.
- **Negative results kept:** corrections barely reach unreviewed faces (≤ +0.09 points on the same subset), and the **adapter head adds nothing**
  (−0.07). The gain comes from the sampler *finding* the misassigned faces. Two `refine()` bugs were found by the eval and fixed:
  seeding identities from pinned outliers dropped LFW recall 99.73 → 96.88, and centroid-vs-linkage distance absorbed noise faces.
- **Offline:** the hybrid search under a socket audit hook, with HF offline variables set, made only loopback connections. This is process-level
  evidence; a network namespace was not allowed.

**Resume check.** The "1,000+ photos" claim holds (5,985 LFW + 11,435 CPLFW). "Fine-tuning ArcFace embeddings" overstates it: only an adapter
head is trained, and it gave no measurable gain. "Progressively refine identity clusters" holds only as *the corrections fix what the sampler
finds* (95.91 → 97.57 on CPLFW). The retrieval-engine bullet holds, with the router numbers above.

**Run it.** Env `photo-seek` (Python 3.11.17, 8.5 GB, record in `env-registry/photo-seek/`). `source scripts/env.sh` puts the caches on
warm storage and caps threads at 8. Ollama is the user-space v0.21.0 at `/mnt/warm_storage/jayavardhan/photo_seek/ollama/dist`, port 11534.
The system `/usr/local/bin/ollama` cannot reach the GPU because `lib/ollama/cuda_v12` is unreadable to users; fixing that needs an admin.
Datasets are in `/mnt/warm_storage/jayavardhan/photo_seek/datasets/` (LFW, CPLFW, COCO 5k, Flickr 1k).

**House rules** (from its `CLAUDE.md`): no Claude co-author trailers, never commit `CLAUDE*.md` or `.claude/`, and never push unless asked.

**Next step.** Make corrections propagate, for example with label propagation over the kNN graph, or a threshold learned from the confirmed
pairs. Then decide whether to keep the head. Add offline reverse geocoding so `place` can filter. Push the branch once the user approves.

**Docs in this vault** (copied 2026-10-14 from `/data/jayavardhan/projects/photo_seek/`, so older than the upgrade):
- [[Sources/photo_seek/README|README]]
- [[Sources/photo_seek/SETUP|SETUP]]

**Source.** `README.md`, `SETUP.md`, `eval/README.md`, `eval/results/`, `CLAUDE.md`.
