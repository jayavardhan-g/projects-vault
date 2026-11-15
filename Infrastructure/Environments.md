---
type: infrastructure
tags: [infrastructure, env]
---
# Environments

Checked on 2026-10-14 with `mamba env list` and `/data/jayavardhan/env-registry/`. **No project env from this folder is
live.** Only `photo-seek` has a rebuild record.

| project | env | Python | state | rebuild from |
|---|---|---|---|---|
| [[Photo-Seek]] | `photo-seek` | 3.11.16 | archived and removed | `env-registry/photo-seek/environment.yml` |
| [[BountyBrain - bug bounty knowledge graph]] | `bountybrain` | 3.11.15 | gone, no record | `SETUP.md` |
| [[Speculative decoding - fixed-draft engine]] | `speculative_decoding` | 3.11 | gone, no record | `SETUP.md` |
| [[CUDA kernels - CS794 sysml]] | `sysml` | 3.11 (torch 2.13 cu130) | gone, no record | `SETUP.md` |
| [[LongContext RAG - retrieval-head-guided compression]] | `lcrag` (was `-p` on warm storage) | 3.11 | gone | `README.md` § Setup |
| [[mini-vLLM - paged-attention inference server]] | `vllmmini`, `vllm-ref` (were `-p` on warm storage) | 3.11, torch 2.11 | gone | `SETUP.md` |
| Interview prep animations (`Days/figures/_gen/manim_d06.py`) | `manim` | 3.12.16 | **live** (created 2026-11-15, 1.2 GB); recorded in `env-registry/manim/` | `mamba create --name manim -c conda-forge python=3.12 manim` → manim 0.22.1 (Manim Community, not 3b1b's ManimGL), manimpango 0.7.1, pycairo 1.29.2, cairo 1.18.6, pango 1.58.2, ffmpeg 9.0.2, numpy 2.5.4. **No LaTeX**: scenes use `Text` only (no MathTex / DecimalNumber). |
| C++ projects (microexchange, raftkv, decs) | none | — | not needed | a C++20 compiler; decs needs libpqxx and Postgres |

**Rules** (from the global CLAUDE.md): create envs by **name** with mamba (`mamba create --name <project> python=3.11`).
The `-p` commands in LCRAG's and mini-vLLM's docs are stale. Record an env with
`/data/jayavardhan/env-registry/archive-env.sh <env>` before deleting it.
