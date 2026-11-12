---
type: study-day
day: 15
date: 2026-11-22
tags: [interview, day, revision, mock]
---
# Day 15 (Sun Nov 22): revision and the full-resume mock

Part of [[Interview plan (Nov 2026)]]. **No new material today.** About 7 h. Interviews start around Nov 23.

**By tonight you can:**
- give every project's 60-second pitch without notes;
- redraw every project's diagram from memory;
- quote each resume number *with its setup and caveat*, using the honest wording;
- tell 5 behavioural stories, each with a real source;
- survive a 90-minute mock across both resumes.

## Schedule

| # | Time | Kind | What |
|---|---|---|---|
| 1 | 60 min | recall | Redraw every diagram from memory, then check (A) |
| 2 | 45 min | practice | Every 60-second pitch, out loud, timed |
| 3 | 45 min | write | The one-page numbers cheat sheet: fill in and check (B) |
| — | break | | |
| 4 | 60 min | practice | Five behavioural stories (C) |
| 5 | 90 min | practice | **Full-resume mock** with Claude (D) |
| — | break | | |
| 6 | 60 min | practice | Redo the mock's 5 weakest topics; re-read only the deep files behind them |
| 7 | 20 min | wrap-up | Update the plan's weak topics; write the log entry |

## A. Diagrams to redraw (block 1)

One sheet each, 5 minutes each, no notes. Then check against the source.

| Project | Draw | Check against |
|---|---|---|
| mini-vLLM | engine → scheduler → block table / allocator / radix cache → model → KV cache | [[Day 01 - mini-vLLM I#Code, explained]], [[Day 02 - mini-vLLM II#Code, explained]] |
| Speculative decoding | draft γ tokens → one verify pass → accept/reject → residual → KV rollback | [[Day 04 - Speculative decoding#Code, explained]] |
| CUDA | the matmul chain, and the FlashAttention block (warps × 4 rows, the KV loop, online softmax) | [[Day 05 - CUDA I#Study guide]], [[Day 06 - CUDA II#Code, explained]] |
| CosyVoice | text → Qwen2 LM (+LoRA) → 25 Hz speech tokens → flow matching → mel → vocoder | [[Day 08 - CosyVoice#Code, explained]] |
| Seminar | the design-space map: representation → modelling → rendering → evaluation | [[Day 07 - Seminar and speech basics#Code, explained]] |
| CS728 | probe → hit counts per head → top-10 heads → attention mass per tool → ranking | [[Day 09 - CS728 and LeNet-5#Code, explained]] |
| LeNet-5 | the IMP loop; per-layer weight counts | [[Day 09 - CS728 and LeNet-5]] D2 |
| Photo-Seek | ingest → CLIP/FAISS and SCRFD/ArcFace → clustering ↔ active learning; query → parser → SQL prefilter → FAISS | [[Day 10 - Photo-Seek I#Code, explained]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained]] |
| Emotion steering | calibrate (diff of means per layer) → hook at layer L → guard ladder | [[Day 11 - Photo-Seek II and emotion steering#Code, explained]] part 2 |
| BountyBrain | the graph build; query → BM25 + dense → RRF → rerank → cited prompt | [[Day 12 - BountyBrain#Code, explained]] |
| DECS | k6 → thread pool → LRU → DB pool → Postgres (pinned) | [[Day 13 - DECS and Micro Exchange#Study guide]] part 1 |
| Micro Exchange | producer → SPSC ring → matching (log first, then book) → follower → snapshot | [[Day 13 - DECS and Micro Exchange#Study guide]] part 2 |
| Raft | election; AppendEntries with prev index/term; the commit rule | [[Day 14 - Raft#Code, explained]] |

## B. Numbers cheat sheet (block 3)

Each line = the number, its setup, and the honest caveat. Fill in any blank from the day note or the prep pack, then fold the sheet into your pocket.

| Project | Number | Setup | Say with it |
|---|---|---|---|
| mini-vLLM | 4.3× less fragmentation | wasted KV slots 16.8% → 3.9%, `independent` workload | slightly worse on uniform prompts |
| mini-vLLM | 3.82× throughput | 2415 vs 631 tok/s vs the HF baseline; TTFT p50 2448 → 179 ms | mostly more requests in flight |
| mini-vLLM | up to 7.4× | 7.41× prefill at **8k context**, 90% cached | 1.00× at 512; hit rate ≠ time saved |
| Spec. decoding | 2.25× / 2.60× | Qwen2.5-0.5B → 14B; code workloads; batch 1 | chat 1.42×; started at 0.88× (launch-bound) |
| CUDA | 51.3× / 48.2% | FP32 SGEMM, N = 32768, H200; vs deliberately uncoalesced naive | 4.7× vs coalesced naive; no tensor cores |
| CUDA | 1.73× | width 32K only, B = 32 forward, column split | likely locality, not concurrency |
| CUDA | 13.3% | **total** training memory; activations −32% | backward +26% |
| CosyVoice | 9M / 1.36% | 8,798,208 of 647,972,675; r 16, α 32, 7 projections | single-GPU jobs in parallel, not DDP |
| CosyVoice | WER 1.99 → 0.11 | IndicConformer ASR; **CSV not on server: verify** | WER > 1 = insertions |
| CS728 | R@1 1.3 → 40.7%; R@5 25.5 → 66.9% | top-10 heads by Hit Count | know the baseline's definition |
| LeNet-5 | 3.75% weights; 98.52% clean; 69.83% MNIST-C | IMP, 5 seeds, val-selected | |
| LeNet-5 | 3.89 pp → 0.13 pp (p = 0.83) | layer-wise vs global; starved conv1 | p = 0.83 = no evidence of a difference |
| LeNet-5 | PGD 51.98 → 82.18%; 245.6 → 75.0 KB | INT8 | 3.27× by arithmetic; check gradient masking |
| Photo-Seek | BCubed 95.9 → 97.6 | CPLFW, 5,606 test faces, 1,000 simulated reviews, margin | ArcFace frozen; adapter −0.07 |
| Photo-Seek | 84% vs 58% | all four filters right on 100 queries, hybrid vs rules | ±7–9 pt CI |
| Photo-Seek | R@1 39.4 | COCO-5k, zero-shot CLIP ViT-B/32 | not your model |
| Emotion | up to 90% | Joy+, L9, strength 5: 18/20 (CI 70–97%) | mean joy 36%; anger 3.3% |
| BountyBrain | 112k nodes (35k core) | 46.6k exploits are outside the core | not "30k+ across exploits" |
| BountyBrain | R@5 0.965, MRR 0.861 | hybrid + rerank, 200 known-item queries | source chunk stays indexed |
| DECS | 11K TPS, CPU-bound | get-popular, server core 100% | whole machine 26% |
| DECS | 3.1K TPS writes | **database-bound**, disk 6% busy | not I/O saturation |
| Micro Exchange | 3.44M ops/s, 18.1× | shared Xeon, 5 paired runs, 2M commands | laptop 10.8M / 50× was one run |
| Raft | failover ≈ 1.0 s | 3 nodes, SIGKILL the leader, n = 10 | quantised by 0.2 s polling |
| Raft | ~250 writes/s | 3 nodes, one fsync per write, loopback | relative, not capacity |

## C. Five behavioural stories (block 4)

Pick one real story per prompt. Sources are listed so you never invent one. Tell each in STAR form in under 2 minutes.

| Prompt | Candidates (source) |
|---|---|
| **Hardest bug** | mini-vLLM admission mutation (README 879; [[Day 02 - mini-vLLM II]]); Raft follower commit index (Raft pack story 1); FlashAttention bank conflict (CUDA S1) |
| **A trade-off** | recompute vs swap preemption (mini-vLLM); a hybrid LLM + rules parser (Photo-Seek 4.5); a flat FAISS index |
| **A failure** | Micro Exchange numbers didn't reproduce (S1); batch-split streams never helped (CUDA S6); the BountyBrain planner lost to a baseline (4.2) |
| **A result that changed your mind** | Photo-Seek: the adapter head added nothing; BountyBrain: the LLM didn't beat keyword, because the shortlist was the ceiling (4.4); prefix-cache hit rate ≠ time saved |
| **Working under a constraint** | no root on a shared server: no ncu, the occupancy API (CUDA S7); 232 threads on a shared server (Photo-Seek 4.6) |

Also read [[Measurement traps]] and [[Evaluation honesty patterns]] in `Lessons/` for more real material.

## D. The full-resume mock (block 5, 90 min)

Start a fresh Claude session with:

```text
Read /data/jayavardhan/projects/vault/Interview prep/Claude instructions - interview prep.md and follow it.
Today is Day 15. Run one 90-minute full-resume mock across resumes 4 and 5, following rule 7 of the instructions:
one question at a time, follow up on anything vague with "why not X?", "what breaks at 10× scale?" and "how did you
measure that?". Mix projects, prerequisites and one behavioural question. After about 15 questions, grade each answer
(strong / OK / weak), list what I got wrong, name the 5 topics to revise, and add them to the plan's weak topics.
```

Rules for you: answer out loud, no notes. When a number has a caveat, say the caveat before you're asked.

## E. Final checks

- [ ] Every resume claim you can't defend is reworded (or you have the honest sentence ready): Photo-Seek "fine-tuning ArcFace" and "1,000+ photos"; Micro Exchange 10.8M / 50×; emotion "up to 90%"; CUDA "13.3% activation memory"; BountyBrain "30k+ across exploits"; DECS "I/O saturation"; CosyVoice "4 H200 GPUs".
- [ ] You have a one-line answer on authorship and AI assistance for BountyBrain, Micro Exchange and Raft.
- [ ] CosyVoice WER CSVs or the presentation found and the numbers checked.
- [ ] Photo-Seek, Micro Exchange and Raft upgrade branches pushed, or you're ready to say what GitHub shows.

## My notes and fumbles

- 
