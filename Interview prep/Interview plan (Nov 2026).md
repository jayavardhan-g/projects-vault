---
type: plan
dates: 2026-11-05 → 2026-11-18
status: active
tags: [interview, plan]
---
# Interview plan: 14 days, 13 projects (Nov 2026)

**Goal.** Interviews start around 2026-11-19. By then you should be able to defend every project on **resumes 4 and 5**
(`/data/jayavardhan/projects/resumes/25M0807-4.pdf`, `-5.pdf`) in depth, as their sole author. That means the code, the
prerequisites, the "why this and not that" choices, the hardest problems, and the numbers. The thesis is not covered
here. The welfare platform (resume 3 only) is out of scope.

**This note is the master tracker.** Tick boxes as you go, and add weak topics at the bottom. Claude sessions follow
[[Claude instructions - interview prep]].

## Revised schedule (2026-11-08): two days were missed, so the rest is compressed

**Day numbering restarts on Nov 8: Day 1 = Nov 8, Day 11 = Nov 18.** The older day sections further down keep their old numbers. This table says which section to study on which date. For Nov 8–10,
use the detailed day notes instead: they have exact functions, line numbers, maths, reading and a rebuild exercise.

| new day | date | study | where |
|---|---|---|---|
| **1** | Sun Nov 8 | mini-vLLM I: paging, scheduler | [[Day 01 - mini-vLLM I]] |
| **2** | Mon Nov 9 | mini-vLLM II: prefix cache, Triton, benchmarks | [[Day 02 - mini-vLLM II]] |
| **3** | Tue Nov 10 | Speculative decoding | [[Day 03 - Speculative decoding]] |
| **4** | Wed Nov 11 | CUDA kernels, plus a 60-min Mock 1 in the evening | old section Day 5 |
| **5** | Thu Nov 12 | Seminar and speech basics (morning, 3 h: blocks 1–2 and 4), then CosyVoice (afternoon; drop its S blocks) | old sections Day 6 and Day 7 |
| **6** | Fri Nov 13 | CS728 and LeNet-5 | old section Day 8 |
| **7** | Sat Nov 14 | Photo-Seek I and II in one day (deep C blocks and P only; drop all S blocks) | old sections Day 9 and Day 10 |
| **8** | Sun Nov 15 | Emotion steering (2 h) and BountyBrain, plus a 60-min Mock 2 | old sections Day 10 (emotion) and Day 11 |
| **9** | Mon Nov 16 | DECS and Micro Exchange | old section Day 12 |
| **10** | Tue Nov 17 | Raft, plus a 60-min Mock 3 | old section Day 13 |
| **11** | Wed Nov 18 | Full-resume mocks (resumes 4 and 5), weak topics, cheat sheet | old section Day 14 |

Each project day now has a **rebuild block**: rewrite the project's core mechanism yourself in about 100 lines
(60–90 min). This replaces rebuilding the whole project, which there is no time for.

## How a day works

- **Each day covers only that day's project or projects.** Don't open other projects, and don't ask Claude about them.
- **Theory and code go together, block by block.** Each block says whether it is theory (T), deep code (C), skim (S)
  or practice (P). The order is the one that makes sense for that topic: usually learn a concept, then read the
  function that implements it, then go back to the theory for whatever the code raised.
- **Deep (C):** trace every line and be able to explain it. A careful pace is about 200 lines of real code an hour.
  The line counts below are real code only, with blank lines, comments and docstrings removed.
- **Skim (S):** read only the module docstring, the function signatures and their docstrings. Write one line per function.
  Open a function body only if the trace exercise needs it.
- **Practice (P):** the trace exercise with real values, then the prep pack's questions for that day, said out loud.
  Write down every answer you fumble.
- Each day is about 6.5–7 hours. **If you are behind,** cut the S blocks first, then shorten the trace. Never cut P.
- Start every day with 15 minutes recalling yesterday's numbers and diagram, from memory.
- **Prep packs** are in `Interview prep/Prep packs/`, one note per project. They are your question bank and map. Ready:
  [[Photo-Seek - prep]], [[CUDA kernels - prep]], [[Speculative decoding - prep]], [[Raft - prep]], [[Micro Exchange - prep]], [[DECS KV server - prep]], [[BountyBrain - prep]],
  [[Emotion steering - prep]], [[Seminar - prep]]. Each study day has its own note in `Interview prep/Days/`, with its lesson and your notes. Still to write, one at a time and just
  in time: mini-vLLM (Day 1 evening), CosyVoice (Day 6 evening), CS728 and LeNet-5 (Day 7 evening, once you've copied the materials).

## Day 1 actions (one-off)

- [x] Stop all upgrade work.
- [x] DECS "CPU saturation at 11K TPS": **the claim holds.** The 26% is vmstat's whole-machine average. The CSV's `cpu` column (probably the pinned server core) hits 100% at 750–1000 VUs while TPS plateaus at about 11.1k. Writes plateau at about 3.1k with that core at 75%, so the database limits writes (likely commit/fsync).
- [ ] Push and merge Photo-Seek's `upgrade-2026-10`; GitHub `main` still shows the scaffold. Decide whether to push microexchange's and raftkv's branches.
- [ ] Settle the wording, using the prep pack's numbers sheet for each:
  - Photo-Seek: the adapter head added nothing. The gain comes from applying corrections (CPLFW BCubed 95.9 → 97.6 after 1k reviews).
  - Micro Exchange: 10.8M ops/s and 50× were measured on a laptop; the server gives 3.44M and 18.1×.
  - Emotion steering: 90% is the best cell (18/20); the mean is 36%.
  - CosyVoice: why WER can be above 1.
- [ ] Copy the CS728 and LeNet-5 code and reports into `/data/jayavardhan/projects/interview_materials/{cs728,lenet}/` before Day 7.

---

## Day 1 (Nov 5): LLM inference basics (afternoon only, about 3.5 h)

Lesson and self-check: [[Prereq - LLM inference basics]].

- [ ] **T, 60 min.** Forward pass, KV cache size formula, prefill (compute-bound) vs decode (memory-bound), the roofline and the H200's ridge point. Use the chat lesson, plus Horace He's *Making Deep Learning Go Brrrr*.
- [ ] **C, 60 min.** `vLLMmini/src/mini_vllm/model/llama.py` (about 200 lines). Trace one token through one layer and write down the tensor shapes (hidden 2048, 32 query and 8 KV heads, head_dim 64).
- [ ] **T, 45 min.** Pope et al., *Efficiently Scaling Transformer Inference*, §2–3. Then TTFT, TPOT and throughput.
- [ ] **P, 30 min.** The 6 self-check questions from the chat.
- [ ] Evening: generate the mini-vLLM prep pack, scoped to the Day 2–3 files.

## Day 2 (Nov 6): mini-vLLM I (paging and scheduling), about 6.75 h

Repo `vLLMmini/src/mini_vllm`, branch `main`.
- [ ] **P, 20 min.** The prep pack's pitch and diagram.
- [ ] **T, 45 min.** PagedAttention paper (Kwon et al.) §3–4: blocks, block tables, internal vs external fragmentation.
- [ ] **C, 75 min.** `block_allocator.py` (92), `block_table.py` (80), `kv_cache.py` (119): the free list, refcounts, logical → physical blocks, the tensor layout.
- [ ] **T, 30 min.** Orca's iteration-level (continuous) batching; preemption by recompute vs swap.
- [ ] **C, 105 min.** `scheduler.py` (355): queues, the token budget, preemption.
- [ ] **S, 40 min.** `paged_attention.py` (the naive reference), `engine.py`, `model/runner.py`, `constants.py`.
- [ ] **P, 45 min.** Trace 3 requests of different lengths through 5 scheduler steps. Write the block tables after each step, and say what happens when the pool runs out.
- [ ] **P, 45 min.** Prep-pack questions on these parts. Know where 16.8% → 3.9% fragmentation (4.3×) and 3.82× come from.

## Day 3 (Nov 7): mini-vLLM II (prefix cache, kernel, benchmarks), about 7 h

- [ ] **T, 40 min.** Radix trees, SGLang RadixAttention §3, LRU eviction with refcounts.
- [ ] **C, 75 min.** `prefix_cache.py` (197).
- [ ] **T, 40 min.** The Triton model (programs, `tl.load` with masks), what a fused decode kernel saves, CUDA graphs.
- [ ] **C, 60 min.** `paged_attention_triton.py` (145): the grid and the block loop.
- [ ] **T, 25 min.** Benchmarking: open vs closed loop, paired ratios on a shared GPU, confounds.
- [ ] **C, 60 min.** `benchmarks/serving.py` (249).
- [ ] **S, 40 min.** `server.py` (FastAPI, SSE), `metrics.py`, `workloads.py`.
- [ ] **P, 30 min.** Two chat turns that share a system prompt: which blocks are reused, and when can one be evicted?
- [ ] **P, 45 min.** Questions: 7.41× only at 8k (0% below 2k, and why), why real vLLM is 6.2× faster, why Triton and not CUDA.

## Day 4 (Nov 8): speculative decoding, about 6.75 h

Repo `speculative_decoding/src/specdec`.
- [ ] **T, 75 min.** Leviathan et al. and Chen et al.: the accept/reject rule, the proof that it preserves the target distribution, α, and expected tokens per step (1−α^{γ+1})/(1−α).
- [ ] **C, 45 min.** `sampling.py` (62): map it line by line to the algorithm.
- [ ] **C, 75 min.** `speculative.py` (222): the draft loop, the verify pass, KV rollback.
- [ ] **T, 30 min.** Launch overhead, CUDA graphs, `torch.compile(reduce-overhead)`, StaticCache.
- [ ] **C, 45 min.** `cache_utils.py` (91), `compile_utils.py` (49): the 0.88× → 2.25× fix.
- [ ] **S, 35 min.** `models.py`, `baseline.py`, `stats.py`, `controller.py` (know why the controller was cut).
- [ ] **T, 30 min.** Medusa, EAGLE and tree drafting, and why you didn't use them.
- [ ] **P, 30 min.** γ=4 with tokens 1–2 accepted and token 3 rejected: what is in the KV cache afterwards?
- [ ] **P, 45 min.** Questions: the three-draft ablation, the domain dependence (code 2.25–2.60×, chat 1.42×), the bf16 noise floor.

## Day 5 (Nov 9): CUDA kernels, plus Mock 1, about 7 h

Repo `sysml`. Your resume numbers come from **PA2 (`matmul/`)** and **PA3 (`mlp/`)**; `prob2/` and `prob3/` are lab quizzes.
- [ ] **T, 60 min.** SMs, warps, the memory hierarchy, coalescing, shared-memory banks, occupancy, arithmetic intensity of matmul.
- [ ] **C, 120 min.** `matmul/matmul_bench.cu` (385). Read it kernel by kernel: naive → coalesced → shared-memory tiling → 2D register blocking. For each, work out which thread computes which output and the global loads per output. Then compute the 51.3× and 48.2% from RESULTS.md.
- [ ] **T, 25 min.** CUDA streams and concurrency; gradient checkpointing (memory vs recompute).
- [ ] **C, 75 min.** `mlp/mlp_bench.cu` (496): the stream variants (batch split vs column split, 1.73×) and the checkpointed backward (13.3%). Skim the rest of the file.
- [ ] **S, 30 min.** `flashattention/flash_lib.cu` (online softmax, the D+1 padding fix); `kvcache/` (PA1, never benchmarked).
- [ ] **P, 90 min, evening.** **Mock 1**: mini-vLLM, speculative decoding, CUDA.

## Day 6 (Nov 10): seminar and speech basics, about 6 h (no code)

Sources: `interview_materials/seminar/INTERVIEW_PREP.md`, `vaults/personal/Seminar/`, `mtp/qwen/report/Seminar Report.docx`.
- [ ] **T, 45 min.** Audio basics: sampling, frames and hops, log-mel features, the Whisper encoder.
- [ ] **T, 75 min.** VQ, RVQ, FSQ; codebook collapse; bitrate maths; acoustic vs semantic tokens (HuBERT); token rates and sequence length.
- [ ] **T, 60 min.** WavTokenizer, SpeechGPT; cascaded vs end-to-end (latency, error propagation, paralinguistics).
- [ ] **T, 60 min.** The CosyVoice 1/2 pipeline: supervised semantic tokens, the LM, flow matching, HiFi-GAN. This sets up Day 7.
- [ ] **P, 60 min.** Seminar prep-pack questions and its cheat sheet.
- [ ] Evening: generate the CosyVoice prep pack.

## Day 7 (Nov 11): CosyVoice, about 7 h

Repo `/data/jayavardhan/cosyvoice_rnd`. The upstream `cosyvoice/` code is not yours; skim it only.
- [ ] **P, 20 min.** The prep pack's pitch and diagram.
- [ ] **T, 45 min.** The LoRA paper: low rank, r, the alpha/r scaling, which modules to adapt; computing the trainable parameter count.
- [ ] **S, 30 min.** Upstream `cosyvoice/cli/cosyvoice.py` → `cli/model.py` → `llm/llm.py`: class and method names only.
- [ ] **C, 50 min.** `scripts/phase4a_pretokenize.py`.
- [ ] **C, 75 min.** `scripts/phase4b_train_lora.py`: the PEFT config at about line 96, the training loop, DDP.
- [ ] **T, 30 min.** DDP; BF16 vs FP16; catastrophic forgetting and how it was tested.
- [ ] **C, 50 min.** `phase4c_generate.py`, `phase5_eval_wer.py`. WER = (S+D+I)/N, and why it can exceed 1.
- [ ] **S, 40 min.** `step1_run_hindi_baseline.py`, `phase2_codec_benchmark.py`, `phase3_*` (the distillation attempt), `compile_results.py`.
- [ ] **P, 30 min.** One Hindi clip from text plus audio to speech tokens to loss. Compute 9M = 1.36% by hand.
- [ ] **P, 45 min.** Prep-pack questions.
- [ ] Evening: generate the CS728 and LeNet-5 prep packs.

## Day 8 (Nov 12): CS728 tool retrieval and LeNet-5, about 7 h

Your code is in `interview_materials/cs728/` and `interview_materials/lenet/`. Size the C blocks once it's there.
- [ ] **T, 60 min.** Retrieval heads (Wu et al.), lost in the middle (Liu et al.), Recall@k, Hit Count head selection.
- [ ] **C, 90 min.** Your CS728 code. Skim `LongContext_RAG/heads/` for 20 minutes as a worked example.
- [ ] **P, 30 min.** How a head's attention over the tool list becomes a Recall@1 ranking. Then the questions.
- [ ] **T, 60 min.** Magnitude pruning and the lottery ticket; global vs layer-wise pruning; MNIST-C; PGD attacks; INT8 quantisation.
- [ ] **C, 90 min.** Your LeNet-5 code.
- [ ] **P, 45 min.** Questions: the starved conv1 layer, the PGD 51.98 → 82.18% result, p=0.83.

## Day 9 (Nov 13): Photo-Seek I (retrieval and router), about 7 h

Repo `photo_seek/src/photoseek`, branch `upgrade-2026-10`.
- [ ] **P, 20 min.** The prep pack's pitch and diagram.
- [ ] **T, 50 min.** CLIP: contrastive training, the shared embedding space, model variants, cosine similarity.
- [ ] **C, 40 min.** `embed/clip.py`; skim `embed/base.py` (the Protocol pattern).
- [ ] **T, 40 min.** FAISS index types (Flat, IVF, HNSW, PQ): exact vs approximate, and when to use each.
- [ ] **C, 40 min.** `index/faiss_store.py`.
- [ ] **T, 25 min.** Structured LLM output (JSON schemas) and Ollama as a local server.
- [ ] **C, 90 min.** `search/query.py`, `search/router.py`, `search/engine.py`.
- [ ] **S, 40 min.** `config.py`, `catalog/db.py` (the schema), `ingest/scanner.py`, `ingest/exif.py`, `cli.py`; `eval/retrieval.py`, `router.py`, `offline.py` (the metric code only).
- [ ] **P, 30 min.** "Mom at the beach last summer" from the router to the StructuredQuery to the filter to FAISS to the results.
- [ ] **P, 40 min.** Questions: R@1 39.4, the 84% vs 58% router result, what "offline" was proven by.

## Day 10 (Nov 14): Photo-Seek II (faces) and emotion steering, about 7.25 h

- [ ] **T, 50 min.** Face detection (SCRFD); ArcFace's additive angular margin.
- [ ] **S, 25 min.** `faces/detect.py`, `embed/arcface.py`.
- [ ] **T, 30 min.** Agglomerative clustering vs HDBSCAN vs k-means; thresholds; BCubed, pairwise F1, NMI.
- [ ] **C, 60 min.** `faces/cluster.py`.
- [ ] **T, 25 min.** Active learning (uncertainty vs random sampling); metric learning (contrastive and triplet losses).
- [ ] **C, 75 min.** `faces/active.py`, `faces/head.py`.
- [ ] **S, 30 min.** `faces/metrics.py`, `faces/pipeline.py`, `eval/faces.py`, `ui/app.py`.
- [ ] **P, 30 min.** One merge correction end to end, and why the adapter head added nothing.
- [ ] **T, 40 min.** Emotion steering: the residual stream, difference-of-means and CAA vectors, layer choice.
- [ ] **C, 50 min.** `emotion_steering/Project.ipynb`: the vector extraction, the hook and the perplexity-guard cells.
- [ ] **P, 30 min.** Emotion questions: the 90% best cell vs 36% mean, unscored emotions, why the middle layers work.

## Day 11 (Nov 15): BountyBrain, plus Mock 2, about 6.5 h

Repo `bountybrain/bountybrain`, branch `upgrade-2026-10`.
- [ ] **P, 20 min.** The prep pack's pitch and diagram.
- [ ] **T, 40 min.** Knowledge-graph modelling (CWE, CAPEC); BM25 vs dense retrieval; bi-encoders vs cross-encoders; RRF.
- [ ] **C, 60 min.** `kg/build.py`; skim `kg/models.py`.
- [ ] **C, 60 min.** `rag/index.py`, `rag/hybrid.py`, `rag/reranker.py`.
- [ ] **T, 25 min.** R@k, MRR, nDCG; RAG faithfulness; LLM judges and Cohen's κ.
- [ ] **S, 40 min.** `kg/queries.py`, `clients/llm.py`, `agent/path.py`, `api/app.py`, `eval/classification.py` (the confound), `eval/faithfulness.py`.
- [ ] **P, 25 min.** "Explain SSRF" through aliases → hybrid retrieval → rerank → the cited prompt. Then the questions.
- [ ] **P, 90 min.** **Mock 2**: seminar, CosyVoice, CS728, LeNet-5, Photo-Seek, emotion steering, BountyBrain.

## Day 12 (Nov 16): DECS and Micro Exchange, about 7 h

- [ ] **T, 45 min.** Thread-per-request vs event-loop servers; LRU design; connection pools; Little's law, the utilisation law, closed-loop load.
- [ ] **C, 60 min.** DECS: `project/server.cpp`, `KVCache.h` (is it really LRU?), `DBPool.h` (299 together).
- [ ] **S, 20 min.** DECS: `load_gen/script.sh`, `get_popular.js`, `put_only.js`, `dbpin.sh`.
- [ ] **T, 45 min.** Order books and price-time priority; the SPSC ring buffer; acquire/release memory ordering; false sharing and cache lines.
- [ ] **C, 75 min.** Micro Exchange (branch `upgrade-2026-10`): `include/core/ring_buffer.hpp`, `object_pool.hpp`, `include/engine/order.hpp`, `order_book.hpp`, `matching_engine.hpp`, `src/order_book.cpp`.
- [ ] **T, 20 min.** WAL, fsync, mmap, crash recovery.
- [ ] **C, 30 min.** `include/persistence/wal_logger.hpp`.
- [ ] **S, 40 min.** `persistence/journal.hpp`, `snapshot.hpp`, `net/*`, `src/gateway.cpp`, `src/main.cpp`, `baseline/naive_engine.hpp` (only: why is it slow?).
- [ ] **P, 30 min.** In DECS, a GET that misses the cache. In Micro Exchange, an order that partly fills against two resting orders and is then cancelled.
- [ ] **P, 45 min.** Questions: the 11K vs 3.1K TPS story, and the laptop 50× vs server 18.1×.

## Day 13 (Nov 17): Raft, about 7 h

Repo `raftkv`, branch `upgrade-2026-10`. Main is the original; know which features are which.
- [ ] **T, 75 min.** The Raft paper §5: election, log replication, safety, Figure 8, the persistence rules.
- [ ] **C, 30 min.** `include/raft_node.h`.
- [ ] **C, 90 min.** `src/raft_node.cpp`: `become_follower`, `become_leader`, `maybe_advance_commit` (lines 216–282), `start_election`, `ticker_loop` (333–445), `handle_request_vote`, `handle_append_entries` (711–882).
- [ ] **C, 50 min.** `replicator_loop`, `apply_loop`, `submit` (450–710).
- [ ] **C, 25 min.** `src/storage.cpp`: fsync plus atomic rename.
- [ ] **T, 40 min.** Linearizability; CAP; Raft vs Paxos; pre-vote and snapshots (§6–7).
- [ ] **S, 40 min.** `start_pre_vote`, `take_snapshot`, `handle_install_snapshot`; `transport.cpp`, `state_machine.cpp`, `fault.cpp`, `main.cpp`; `scripts/chaos_test.py`, `linearizability.py`, `shardkv.py`.
- [ ] **P, 30 min.** The leader crashes after replicating an entry to 1 of 4 followers. Does the entry survive?
- [ ] **P, 45 min.** Prep-pack questions.

## Day 14 (Nov 18): mocks and revision, about 7 h

- [ ] **P, 90 min.** **Mock 3**: DECS, Micro Exchange, Raft.
- [ ] **P, 2 × 60 min.** Full-resume mocks, one for resume 4 and one for resume 5.
- [ ] **P, 60 min.** Five behavioural stories: hardest bug, a trade-off, a failure, a result that changed your mind, a constraint. [[Measurement traps]] and [[Evaluation honesty patterns]] hold real ones.
- [ ] **P, 90 min.** Redo your weak topics. Re-read only the deep files behind them.
- [ ] **P, 45 min.** A one-page numbers cheat sheet.

---

## Prep-pack prompt (one project per session, just in time)

Run it in the project's folder the evening before that project's day, so each session covers only one project.

```text
I have technical interviews soon and must defend this project in depth, as its sole author. Focus ONLY on the files
listed for this project in /data/jayavardhan/projects/vault/Interview prep/Interview plan (Nov 2026).md (its day
section). Read those files, the README/RESULTS/PROGRESS docs and the git log. Do NOT change code, run jobs, commit or push.
Write it to /data/jayavardhan/projects/vault/Interview prep/Prep packs/<PROJECT> - prep.md (not into the repo).
Project: <PROJECT>. My resume bullets: "<paste bullets from resume 4 or 5>".
Include: a 60-second pitch; an ASCII diagram with path:line references; the design decisions (alternatives, choice,
why, cost; marked "documented" or "reconstructed"); the hardest problems as STAR stories; a numbers sheet (how each
resume number was measured, and whether the repo supports it); the prerequisites (3–5 lines each, with sources);
40 likely questions with crisp answers and hostile follow-ups; and honest "it didn't work" answers.
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
