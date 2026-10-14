---
type: project
folder: projects/init
dates: 2026-08-10 → 2026-08-11
status: reference
kind: planning — project selection, resumes, placement data
tags: [project, planning, career]
---
# Project strategy and resume plan (`init/`)

**What.** These are the planning documents that the August projects came out of:
- `Intro.md` is the brief given to an LLM. It covers background (coursework, the speech R&D in [[CosyVoice Indic LoRA (R&D)]], the MTP),
  constraints, hard rules (e.g. no plain CosyVoice-LoRA repeat, no from-scratch FlashAttention) and a weighted scoring rubric.
- `Seniors_Resumes.md` holds seven senior resumes used for the overlap analysis.
- `Project_Recommendation_Resume_Strategy.md` (86 KB) is the output: a 20-project shortlist, a comparison table, a top 10 in
  full detail, the best 5, **five resume portfolios** (ML/AI, Systems/Infra, ML Systems/Inference, Research Engineer,
  Speech+Systems) and a final build order.
- `job_extraction/` contains `extract_jobs.py`, which parses a 318 MB placement-portal PDF (`allJobs.pdf`) into
  `jobs.csv`, `jobs_by_branch.csv` and `eligibility_long.csv`. `HOW_TO_FILTER.md` explains the spreadsheets. This folder accounts for most of the 469 MB.

**Recommended build order vs what was built:**

| rank | recommendation | built? |
|---|---|---|
| 1 | Agentic GPU kernel optimisation | no (only [[CUDA kernels - CS794 sysml]] coursework) |
| 2 | Retrieval-head-guided long-context RAG | **yes** → [[LongContext RAG - retrieval-head-guided compression]] |
| 3 | Real-time streaming inference for speech-LLMs | no |
| 4 | Long-horizon agent memory and context compaction | no |
| shortlist #2/#8 | disaggregated prefill/decode serving, KV-cache compression | related: [[mini-vLLM - paged-attention inference server]] |
| shortlist #10 | Raft KV + resharding | **partly** → [[raftkv - Raft key-value store]] (no resharding) |
| shortlist #11 | tree-based speculative decoding | **related** → [[Speculative decoding - fixed-draft engine]] (fixed-draft, no tree) |

Not from the list: [[microexchange - C++ order matching engine]], [[BountyBrain - bug bounty knowledge graph]], [[Photo-Seek]].

**Still open from the plan:** agentic GPU-kernel optimisation (rated the top flagship), streaming speech-LLM serving, agent
memory and compaction, eBPF observability for LLM serving, speech-LLM interpretability (rated highest research potential, 9/10). See [[Open threads]].

**Source.** `/data/jayavardhan/projects/init/`.
