---
type: infrastructure
tags: [infrastructure, git]
---
# Repositories and git state (2026-10-14)

| folder | GitHub repo (jayavardhan-g/…) | commits | unpushed |
|---|---|--:|--:|
| `bountybrain` | bountybrain | 18 | 0 |
| `LongContext_RAG` | LongContext_RAG | 6 | 0 (remote is HTTPS; push over SSH) |
| `vLLMmini` | mini-vLLM | 34 | 0 |
| `speculative_decoding` | speculative_decoding | 25 | 0 |
| `photo_seek` | Photo-Seek | 11 | 0 |
| `sysml` | CUDA-Kernel-Optimization-Efficient-ML-Inference | 1 | 0 |
| `microexchange` | Micro-Exchange | 1 | 0 |
| `raftkv` | Fault-Tolerant-Distributed-System | 1 | 0 |
| `RaftKV` | raftkv | 1 | 0 |
| `decs` | High-Performance-HTTP-Key-Value-Server | 1 | 0 |
| `emotion_steering` | Inference-Time-Emotion-Steering | 1 | 0 |
| `init` | — (not a repo) | — | — |

The single-commit repos were imported with their full history squashed into one "first commit" between 2026-08-14 and 08-19.

**Things to tidy:**
- ~~Push bountybrain and LongContext_RAG~~ — done 2026-10-14. LongContext_RAG still has uncommitted README edits and results files.
- `raftkv` and `RaftKV/RAFT` hold the same code in two public repos. → [[raftkv - Raft key-value store]]
- `decs/project/server.cpp` has a Postgres password in a public repo. → [[DECS - HTTP key-value server]]
- `microexchange/_to_delete/` holds leftover build files.
- Several repos keep `CLAUDE.md` and `PROGRESS.md` git-ignored on purpose, as local hand-over notes only.
