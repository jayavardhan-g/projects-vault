---
type: plan
dates: 2026-11-05 → 2026-11-18
status: active
tags: [interview, plan]
---
# Interview plan: 14 days, 14 projects (Nov 2026)

**Goal.** Interviews start around 2026-11-19. By then, defend every project on the five resumes in depth, as their
sole author. That means answering "why this, why not that", "why this model or module", "what was the hardest
problem" and "what did you trade off", and knowing the code and the prerequisites well enough to answer follow-ups.
The thesis is **not** in this plan: it is active work and stays fresh on its own.

**This note is the master tracker.** Tick the boxes as you go, and add weak topics at the bottom. Claude sessions
that continue this work follow [[Claude instructions - interview prep]]. A plain copy of the first version is at
`/data/jayavardhan/projects/init/INTERVIEW_PLAN.md`. The resumes are in `/data/jayavardhan/projects/resumes/`
(`25M0807-1.pdf` … `-5.pdf`).

**Start moved to Nov 5** (one day late), so the plan is now 14 days: Mock 1 became a 2-hour evening block on Day 5.

**Budget.** About 7 focused hours a day, so about 95 hours. If you have less, cut Tier C to a 30-minute skim each,
and shorten code reading before you shorten Q&A practice.

## Projects, by priority

| Tier | Project | On resumes | Hours | Note |
|---|---|---|--:|---|
| A | mini-vLLM | 1 2 3 4 5 | 10 | [[mini-vLLM - paged-attention inference server]] |
| A | Photo-Seek | 1 2 3 4 5 | 10 | [[Photo-Seek]] |
| A | CosyVoice Indic LoRA (CS692) | 1 3 4 5 | 7 | [[CosyVoice Indic LoRA (R&D)]] |
| B | CUDA kernels (CS794) | 2 4 5 | 7 | [[CUDA kernels - CS794 sysml]] |
| B | Tool-retrieval heads (CS728) | 1 3 4 | 4 | no repo here; theory in [[LongContext RAG - retrieval-head-guided compression]] |
| B | Seminar: speech in LLMs | 3 4 5 | 4 | `vaults/personal/Seminar/`, `mtp/qwen/report/Seminar Report.docx` |
| B | Raft | 3 5 | 7 | [[raftkv - Raft key-value store]] |
| B | Speculative decoding | 4 5 | 6 | [[Speculative decoding - fixed-draft engine]] |
| B | Micro Exchange | 3 5 | 6 | [[microexchange - C++ order matching engine]] |
| B | HTTP KV server (CS744) | 2 5 | 5 | [[DECS - HTTP key-value server]] |
| C | BountyBrain | 4 | 4 | [[BountyBrain - bug bounty knowledge graph]] |
| C | LeNet-5 compression (CS725) | 3 4 | 3 | no repo here |
| C | Emotion steering (CS772) | 4 | 2 | [[Emotion steering - Llama activation steering]] |
| C | Welfare scheme platform (CS699) | 3 | 2 | no repo here |

If you know which resume goes to which company, skip the projects that are not on it.

## Day 1: do these first

- [x] **Stop all upgrade work.** Wave 2 (mini-vLLM, speculative decoding, CUDA, emotion steering) and DECS never started. Leave them unstarted.
- [ ] **Push Photo-Seek.** GitHub `main` is still the scaffold, and the real work is about 20 commits on `upgrade-2026-10`. Push and merge it yourself; the repo's rules forbid Claude from pushing.
- [ ] Decide what to do with the unpushed upgrade branches of microexchange and raftkv. Push them, or don't mention the upgrades.
- [ ] **Fix the resume wording**, or prepare an honest answer, for each claim the re-runs contradict:
  - Photo-Seek, "fine-tuning ArcFace embeddings": the adapter head **added nothing**. The gain comes from applying the corrections (CPLFW BCubed 95.9 → 97.6 after 1k reviews). Numbers that hold: COCO-5k R@1 **39.4**; LLM+rules router **84%** all-fields correct vs 58% for rules.
  - Micro Exchange, "10.8M ops/s, 50×": measured on a laptop. On the shared server: **3.44M ops/s, 18.1×** at 2M commands. A single thread with no queue reaches 18.1M/s, so the cross-core ring handoff is the bottleneck there.
  - DECS, "CPU saturation at 11K TPS": **resolved 11-05, the claim holds.** The 26% is vmstat's whole-machine average. The CSV's `cpu` column (probably the pinned server core) reaches 100% at 750–1000 VUs while TPS plateaus at about 11.1k. Writes plateau at about 3.1k with that core at 75%, so the database limits writes (likely commit/fsync). Disk util reads low because it is averaged over all devices.
  - Emotion steering, "up to 90%": true for the single best setting (Joy+, layer 9, strength 5, 18/20). The mean for that emotion is **36%**.
  - CosyVoice, "WER 1.99 → 0.11": a WER above 1 means insertions or babbling made the errors outnumber the reference words. Be ready to explain this.
- [ ] Collect the CS728, LeNet-5 and welfare reports and code from your laptop or Drive into `/data/jayavardhan/projects/interview_materials/<name>/`.
- [x] Start the prep-pack sessions (prompt below). Started 11-05 as background agents from one session; packs not yet ticked are still being written. They only read, need no GPU, and can all run in parallel.

### Prep packs (one `INTERVIEW_PREP.md` per project, untracked)

- [ ] mini-vLLM  - [ ] Photo-Seek  - [ ] CosyVoice  - [ ] CUDA (sysml)  - [ ] speculative decoding
- [ ] raftkv  - [ ] microexchange  - [ ] DECS  - [ ] BountyBrain  - [ ] emotion steering
- [ ] seminar  - [ ] CS728  - [ ] LeNet-5  - [ ] welfare

## Daily routine

Code comes first. Theory is there to explain what the code does.

1. **20 min, orient.** Read the prep pack's pitch and architecture diagram only, so you know the map before you read the code.
2. **4 h, read the code** in the order the code-reading track below gives for that day: about 3 h on the deep files and at most 1 h skimming. For each deep file:
   - **First pass (skim):** class and function names, the data structures, who calls what. Write a one-line purpose per function.
   - **Second pass (trace):** follow the day's trace exercise through the code with real values. Note every line you can't explain.
   - **Explain it back:** close the editor, redraw the diagram and say the trace out loud. Reopen only for the gaps.
3. **1 h, theory, driven by the code.** Look up exactly the concepts behind the lines you couldn't explain, using the day's prerequisites list. Then re-read those lines.
4. **1.5 h, out loud:** the pitch, "why X, not Y", the hardest bug. Use the prep pack's question list. Write down every answer you fumble (bottom of this note).
5. **15 min the next morning:** yesterday's numbers and diagram, from memory.

**Rule:** if a resume bullet names a mechanism (radix tree, ring buffer, LRU, LoRA), you must be able to point to the exact function that implements it.

## Day by day

- [ ] **Day 1 (Nov 5)**: the Day-1 actions; LLM inference basics: transformer forward pass, KV cache, prefill vs decode (compute- vs memory-bound), roofline, TTFT/TPOT.
- [ ] **Day 2 (Nov 6)**: **mini-vLLM I**: block manager, paged attention, scheduler. *Prerequisites:* PagedAttention paper, internal vs external fragmentation, continuous batching (Orca), recompute vs swap preemption. *Read:* `vLLMmini/README.md`, `PROGRESS.md`, source.
- [ ] **Day 3 (Nov 7)**: **mini-vLLM II**: radix prefix cache, Triton decode kernel, benchmarks, why real vLLM is 6.2× faster. *Prerequisites:* RadixAttention (SGLang), the Triton programming model, CUDA graphs, paired benchmarking.
- [ ] **Day 4 (Nov 8)**: **speculative decoding**. *Prerequisites:* why rejection sampling (Leviathan, Chen) preserves the target distribution; expected tokens per step (1−α^{γ+1})/(1−α); why batch-1 decode is limited by kernel launches; torch.compile and StaticCache; Medusa, EAGLE and tree drafting, and why not. *Read:* `speculative_decoding/README.md`, `RESULTS.md`, `PROGRESS.md` §4.
- [ ] **Day 5 (Nov 9)**: **CUDA kernels**. *Prerequisites:* SMs, warps, memory hierarchy, coalescing, bank conflicts, occupancy, arithmetic intensity, the maths of tiling and register blocking, streams, gradient checkpointing, FlashAttention's online softmax. *Read:* `sysml/RESULTS.md`, kernels. **Evening: Mock 1** (2 h: mini-vLLM, speculative decoding, CUDA).
- [ ] **Day 6 (Nov 10)**: **speech basics and the seminar**: log-mel, Whisper encoder, discrete vs continuous speech tokens, VQ/RVQ/FSQ, WavTokenizer, token rates (12.5/25/50 Hz), SpeechGPT, cascaded vs end-to-end systems.
- [ ] **Day 7 (Nov 11)**: **CosyVoice**, a full day. *Prerequisites:* the CosyVoice 2 pipeline (Qwen2 LM → 25 Hz FSQ speech tokens → flow matching → HiFi-GAN); LoRA maths (rank, alpha, target modules, and how 9M = 1.36%); BF16; DDP; catastrophic forgetting; why WER can exceed 1. *Read:* `/data/jayavardhan/cosyvoice_rnd` (scripts `phase2`–`phase5`), your R&D slides.
- [ ] **Day 8 (Nov 12)**: **CS728 tool retrieval** (morning): retrieval heads (Wu et al.), lost-in-the-middle, Recall@k, Hit Count head selection. **LeNet-5** (afternoon): magnitude pruning (lottery ticket), global vs layer-wise pruning, MNIST-C, PGD attacks, INT8 quantisation.
- [ ] **Day 9 (Nov 13)**: **Photo-Seek I**: retrieval, FAISS and the router. *Prerequisites:* CLIP contrastive training, FAISS index types (Flat, IVF, HNSW, PQ) and when to use each, structured LLM output, Ollama.
- [ ] **Day 10 (Nov 14)**: **Photo-Seek II**: faces and active learning. *Prerequisites:* ArcFace's additive angular margin, SCRFD detection, agglomerative clustering vs HDBSCAN, BCubed, uncertainty sampling. Then **emotion steering**: the residual stream and difference-of-means steering.
- [ ] **Day 11 (Nov 15)**: **BountyBrain** (morning): knowledge-graph design, BM25 + dense + RRF, cross-encoder reranking, MRR/nDCG, RAG faithfulness, LLM judges. **Mock 2** (afternoon): CosyVoice, CS728, LeNet, Photo-Seek, BountyBrain.
- [ ] **Day 12 (Nov 16)**: **DECS and Micro Exchange**. *Prerequisites:* thread pools, LRU (hash map + doubly linked list), Little's law, the utilisation law, closed-loop load, connection pooling; order books, SPSC ring buffers, acquire/release memory ordering, false sharing, object pools, WAL/fsync/mmap, tail latency.
- [ ] **Day 13 (Nov 17)**: **Raft**: election, log matching, the commit rule, Figure 8, safety, persistence, snapshots, pre-vote, linearizability, CAP, Raft vs Paxos. Plus a **welfare** skim: Flask, auth, the scraper and rule engine.
- [ ] **Day 14 (Nov 18)**: **Mock 3** (DECS, Micro Exchange, Raft); a full-resume mock for each resume variant; five behavioural stories (hardest bug, a trade-off, a failure, a result that changed your mind, working under constraints; [[Measurement traps]] and [[Evaluation honesty patterns]] hold real ones); a one-page numbers cheat sheet.

## Code-reading track

Line counts are **real code only**: blank lines, comments and docstrings are removed, which is about half of each file.
Each day has two kinds of file:
- **Deep files** (about 450–800 lines, about 3 h): you trace them line by line and must be able to explain every line. At about 200 lines an hour, careful reading of 600 lines takes 3 hours.
- **Skim files** (about 1 h, at most): read only the module docstring, the function signatures and their docstrings. Write a one-line purpose for each function. Open a function body only if the trace exercise sends you there.

Paths are relative to each repo; the branch is named where it matters. If a function is unclear, read its test: tests show how a function is meant to be used.

**Day 1 (Nov 5), warm-up, about 1 h.** Deep: `vLLMmini/src/mini_vllm/model/llama.py`. *Trace:* one token through one
layer, writing down the tensor shapes (hidden 2048, 32 query and 8 KV heads, head_dim 64).

**Day 2 (Nov 6), mini-vLLM I** (`vLLMmini/src/mini_vllm`). Deep, 646 lines:
- `block_allocator.py`, `block_table.py`: the free list, refcounts, logical → physical blocks.
- `kv_cache.py`: the paged tensor layout.
- `scheduler.py` (355): queues, the token budget, recompute preemption.

Skim, 390 lines: `paged_attention.py` (the naive reference), `engine.py`, `model/runner.py`, `constants.py`.
*Trace:* 3 requests of different lengths through 5 scheduler steps. Write the block tables after each step, and say what happens when the pool runs out.

**Day 3 (Nov 7), mini-vLLM II.** Deep, 591 lines:
- `prefix_cache.py`: radix insert and match, refcounted LRU eviction.
- `paged_attention_triton.py`: the program grid and the block loop.
- `benchmarks/serving.py`: how TTFT, TPOT and throughput were measured, and the three confounds it removes.

Skim, 578 lines: `server.py`, `metrics.py`, `workloads.py`.
*Trace:* two chat turns that share a system prompt. Which blocks are reused, and when could one be evicted?

**Day 4 (Nov 8), speculative decoding** (`speculative_decoding/src/specdec`). Deep, 424 lines (a lighter day; spend the spare time on the proof):
- `sampling.py`: accept/reject and residual resampling, mapped line by line to the paper.
- `speculative.py`: the draft loop, the verify pass, KV rollback.
- `cache_utils.py`, `compile_utils.py`: the 0.88× → 2.25× fix.

Skim, 440 lines: `models.py`, `baseline.py`, `stats.py`, `controller.py` (the cut controller; know why it was cut).
*Trace:* γ=4 where tokens 1–2 are accepted and token 3 is rejected and resampled. What is in the KV cache afterwards?

**Day 5 (Nov 9), CUDA** (`sysml`), plus the evening mock. Deep, 600 lines:
- `prob2/q1–q3/student.cu`: naive → coalesced → tiled → register-blocked matmul. For each, work out which thread computes which output, and the loads per output.
- `prob3/q3`, `prob3/q4/student.cu`: streams and gradient checkpointing, the two resume claims.

Skim, 538 lines: `prob3/q1–q2`; `flashattention/flash_lib.cu` (know the D+1 padding fix); `kvcache/model.py`, `radix_tree.py` (PA1, never benchmarked).
*Trace:* the shared-memory tile indices for one block of the TILE 64, R 4 kernel.

**Day 6 (Nov 10), seminar.** No code. Read the seminar prep pack's concept map, then `vaults/personal/Seminar/` and the report.

**Day 7 (Nov 11), CosyVoice** (`/data/jayavardhan/cosyvoice_rnd`). Deep, 595 lines, your own code:
- `scripts/phase4a_pretokenize.py`
- `phase4b_train_lora.py`: the PEFT config at about line 96, the training loop, DDP.
- `phase4c_generate.py`
- `phase5_eval_wer.py`

Skim: your other scripts (1,205 lines: `step1_run_hindi_baseline`, `phase2_codec_benchmark`, `phase3_*`, `phase5_eval_wer_whisper`, `compile_results`). For upstream code you didn't write, read only the class and method names: `cosyvoice/cli/cosyvoice.py` → `cli/model.py` → `llm/llm.py` (565) → `flow/flow.py`.
*Trace:* one Hindi clip from text plus audio to speech tokens to loss. Then compute the trainable LoRA parameter count by hand.

**Day 8 (Nov 12), CS728 and LeNet-5.** Your code in `interview_materials/cs728/` and `interview_materials/lenet/`. Size the deep set to about 600 lines once the code is there. Worked example for retrieval heads: `LongContext_RAG/heads/` (skim).
*Trace:* how a head's attention over the tool list becomes a Recall@1 ranking.

**Day 9 (Nov 13), Photo-Seek I** (`photo_seek/src/photoseek`, branch upgrade-2026-10). Deep, 674 lines: `search/query.py`, `search/router.py`,
`search/engine.py`, `embed/clip.py`, `index/faiss_store.py`.

Skim, 1,185 lines: `config.py`, `catalog/db.py` (the schema only), `ingest/scanner.py`, `ingest/exif.py`, `embed/base.py`, `cli.py`, and `eval/retrieval.py`, `eval/router.py`, `eval/offline.py` (read just the metric code).
*Trace:* "mom at the beach last summer", from the router to the StructuredQuery to the filter to FAISS to the results.

**Day 10 (Nov 14), Photo-Seek II, plus emotion steering.** Deep, 636 lines: `faces/cluster.py`, `faces/active.py`, `faces/head.py`.

Skim, 761 lines: `faces/detect.py`, `faces/metrics.py`, `faces/pipeline.py`, `embed/arcface.py`, `eval/faces.py`, `ui/app.py`. Also skim `emotion_steering/Project.ipynb`, reading the vector-extraction, hook and guard cells closely.
*Trace:* one merge correction. What changes in the DB and the clusters, and why did the adapter head add nothing?

**Day 11 (Nov 15), BountyBrain** (`bountybrain/bountybrain`, branch upgrade-2026-10). Deep, 752 lines: `rag/index.py`, `rag/hybrid.py`, `rag/reranker.py`, `kg/build.py`.

Skim, 1,698 lines: `kg/models.py` (the node and edge types), `kg/queries.py`, `clients/llm.py`, `agent/path.py`, `api/app.py`, and `eval/classification.py`, `eval/faithfulness.py` (read only the part where the confound was).
*Trace:* "explain SSRF": the alias table → hybrid retrieval → rerank → the cited LLM prompt.

**Day 12 (Nov 16), DECS and Micro Exchange.** Deep, 823 lines; the DECS part is simple C++:
- DECS: `project/server.cpp`, `KVCache.h` (check whether it is really LRU), `DBPool.h`.
- Micro Exchange (branch upgrade-2026-10): `include/core/ring_buffer.hpp`, `object_pool.hpp`, `include/engine/order.hpp`, `order_book.hpp`, `matching_engine.hpp`, `src/order_book.cpp`, `include/persistence/wal_logger.hpp`.

Skim, about 1,900 lines: `core/tsc_clock.hpp`, `core/affinity.hpp`, `engine/exchange.hpp`, `engine/workload.hpp`, `persistence/journal.hpp`, `persistence/snapshot.hpp`, `net/*.hpp`, `src/gateway.cpp`, `src/main.cpp`; `baseline/naive_engine.hpp` (only: why is it slow?); DECS `load_gen/script.sh`, `get_popular.js`, `put_only.js`, `dbpin.sh`.
*Trace:* in DECS, a GET that misses the cache. In Micro Exchange, an order that partially fills against two resting orders and is then cancelled.

**Day 13 (Nov 17), Raft** (`raftkv`, branch upgrade-2026-10). Deep, about 750 lines:
- `include/raft_node.h`, `src/storage.cpp`.
- In `src/raft_node.cpp`, the core functions: `become_follower`, `become_leader`, `maybe_advance_commit` (lines 216–282), `start_election`, `ticker_loop` (333–445), `replicator_loop`, `apply_loop`, `submit` (450–710), `handle_request_vote`, `handle_append_entries` (711–882).

Skim: `start_pre_vote`, `handle_pre_vote`, `take_snapshot`, `handle_install_snapshot` (the upgrade features); `transport.cpp`, `state_machine.cpp`, `fault.cpp`, `main.cpp`; `scripts/chaos_test.py`, `linearizability.py`, `shardkv.py`.
*Trace:* the leader crashes after appending an entry to 1 of 4 followers. Walk through the new election, and say whether the entry survives.

**Day 14 (Nov 18).** No new code. Redraw every diagram from memory, then re-read only the deep files behind your weak topics.

## Prep-pack prompt

Start each session in the project's folder. CosyVoice is `/data/jayavardhan/cosyvoice_rnd`. For projects with no repo
here, run the prompt in their `interview_materials/<name>/` folder and drop the `.git/info/exclude` step.

```text
I have technical interviews in 2 weeks and must be able to defend this project in depth, as its sole author. Read the
docs (README, RESULTS/PROJECT, PROGRESS, SETUP, CLAUDE.md), the git log and the source code. Do NOT change code, run
GPU jobs, commit or push. Write INTERVIEW_PREP.md in this folder and add it to .git/info/exclude so it stays untracked.

Project: <PROJECT>. My resume bullets: "<paste bullets>".

INTERVIEW_PREP.md must contain:
1. A 60-second pitch and a 3-minute deep explanation, in plain spoken English.
2. An architecture diagram (ASCII), with the key files and functions as path:line. Trace one request or input end to end through the code.
3. Every significant design decision: the alternatives, what was chosen, why, and the cost of that choice. Mark each as
   "documented" (cite the doc or commit) or "reconstructed from the code". Never invent history.
4. The hardest problems and bugs, taken from PROGRESS logs, commits and docs, written as STAR stories (situation, task, action, result).
5. A numbers sheet: every number on my resume, how it was measured (hardware, workload, n, baseline), caveats, and whether
   the current repo reproduces it. Flag any number the repo does not support, and give the honest wording.
6. Prerequisite concepts, each explained in 3–5 lines, plus the 1–2 canonical papers or sources.
7. 50 likely interview questions with crisp answers: why X, not Y; why this model or module; scaling; failure modes;
   what you would do differently; what comes next. Include hostile follow-ups.
8. Questions where the honest answer is "it didn't work" or "I didn't do that", and how to say it well.
9. 5 self-test exercises: a small code change, where I predict the effect before reading the answer.
End your reply with the path to the file.
```

## Mock-interview prompt

```text
Act as a senior interviewer for an ML-systems / research-engineer role. Grill me on <PROJECT> using INTERVIEW_PREP.md
and the code in this folder. Ask one question at a time and wait for my answer. Follow up on anything vague with
"why not X?", "what breaks at 10× scale?" and "how did you measure that?". Mix in questions on prerequisites. After 15
questions, grade each answer (strong / OK / weak), list what I got wrong, and give the 5 topics I should revise.
```

## Weak topics (add as you go)

- 
