---
type: home
tags: [meta]
---
# Side projects — home

**What this covers.** The projects in `/data/jayavardhan/projects/` that are not the M.Tech thesis. The thesis has
its own vault at `/data/jayavardhan/mtp/vault`. How this vault is organised: [[How this vault works]].

**State (2026-10-14).** 10 projects plus the planning folder `init/`. Nothing is running. All but
[[Photo-Seek]] are finished or parked. Almost every project env has been deleted; see [[Environments]].

## Projects

| project | kind | built | state | headline |
|---|---|---|---|---|
| [[LongContext RAG - retrieval-head-guided compression]] | research, LLM | 2026-08-10 → 08-13 | done | dense→head **cascade +0.054 F1** over bge-m3 at 256 tokens (p<0.001), 7× faster than naive 32k |
| [[mini-vLLM - paged-attention inference server]] | ML systems | 08-12 → 08-15 | done | **3.82×** HF baseline; real vLLM still **6.2×** faster; 91% prefix-cache hits by turn 4 |
| [[Speculative decoding - fixed-draft engine]] | ML systems | 08-13 → 08-16 | done (adaptive part cut) | **2.25×** at γ=4 (14B target, 0.5B draft); 1.4×–2.6× depending on domain |
| [[CUDA kernels - CS794 sysml]] | coursework, GPU | 08-15 | done (PA1 not run) | matmul **48% of cuBLAS**; flash-attention kernel sped up 21× |
| [[BountyBrain - bug bounty knowledge graph]] | AI app + eval | 08-16 → 08-18; upgraded 10-22 | done (branch upgrade-2026-10, pushed) | 112k-node graph (35k core); hybrid+rerank **R@5 0.965**; fair re-run: keyword still beats LLM (0.398 vs 0.374), shortlist R@12 0.45 is the ceiling; tutorials cited+supported **5.1% → 32.6%**; API 13/13 + UI 7/7 end to end |
| [[microexchange - C++ order matching engine]] | systems, C++ | ≤08-14, upgraded 10-22 | done (branch unpushed) | server re-run: **14–18× vs naive at 2M** (laptop 50× not reproduced), 3.4M ops/s; gateway RTT p50 13 µs; exact kill -9 recovery |
| [[raftkv - Raft key-value store]] | distributed systems | ≤08-18, upgraded 10-22 | upgraded (branch upgrade-2026-10, unpushed) | Raft in C++20 + snapshots, pre-vote, fault injection; linearizable in 50/50 chaos runs; live resharding across 3 groups |
| [[DECS - HTTP key-value server]] | coursework, systems | before 08 | done | C++ KV server + Postgres; peak ~3.6k req/s GET, 11.2k on hot keys |
| [[Emotion steering - Llama activation steering]] | coursework, interp | before 08 | done | best axis (Joy+) flips emotion **36%** of the time; layers 6–9 work best |
| [[Photo-Seek]] | app + eval | 08-18 → 08-25; upgrade 10-22 → 11-04 | done (branch not pushed) | COCO 5k **R@1 39.4**; router all-fields **84%** hybrid vs 58% rules; CPLFW faces 95.9 → **97.6** BCubed after 1k reviews; adapter head adds nothing |

**Interview prep (Nov 2026):** [[Interview plan (Nov 2026)]] (15-day plan and tracker) · [[Claude instructions - interview prep]] (how a Claude session continues it).

Planning and background: [[Project strategy and resume plan]] (the `init/` folder), [[CosyVoice Indic LoRA (R&D)]].

## By theme

- **LLM inference and serving:** [[mini-vLLM - paged-attention inference server]], [[Speculative decoding - fixed-draft engine]], [[CUDA kernels - CS794 sysml]]
- **Retrieval and RAG:** [[LongContext RAG - retrieval-head-guided compression]], [[BountyBrain - bug bounty knowledge graph]], [[Photo-Seek]]
- **Model internals and interpretability:** [[LongContext RAG - retrieval-head-guided compression]] (retrieval heads), [[Emotion steering - Llama activation steering]]
- **Systems in C++:** [[microexchange - C++ order matching engine]], [[raftkv - Raft key-value store]], [[DECS - HTTP key-value server]]

## Navigate

- Chronology: [[Timeline]]
- Lessons that recur across projects: [[Measurement traps]], [[Evaluation honesty patterns]], [[Infrastructure traps]]
- Running things: [[Environments]], [[Repositories and git state]]
- Not finished anywhere: [[Open threads]]
