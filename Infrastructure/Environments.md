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
| C++ projects (microexchange, raftkv, decs) | none | — | not needed | a C++20 compiler; decs needs libpqxx and Postgres |

**Rules** (from the global CLAUDE.md): create envs by **name** with mamba (`mamba create --name <project> python=3.11`).
The `-p` commands in LCRAG's and mini-vLLM's docs are stale. Record an env with
`/data/jayavardhan/env-registry/archive-env.sh <env>` before deleting it.
