# Project reports: index

These are one self-contained report per project on resumes 4 and 5 (the thesis is excluded). Each report was written
from the project's own code, results files and git history, not from the Day notes or prep packs.

**How to use them:** paste one report into any LLM and ask it to teach, quiz or mock-interview you on that project.
Each report starts with an instruction block for the LLM.

Every report has the same sections:
1. summary;
2. resume bullets checked against evidence;
3. background to learn first;
4. architecture and a code walk with `file:line`;
5. results and how they were measured;
6. hardest problems;
7. design decisions;
8. limitations;
9. likely interview questions;
10. numbers cheat sheet.

Claims are labelled **[CODE] / [MEASURED] / [DOC] / [RECON]**, so the LLM knows what is verified and what is inferred.

| # | Report | Resume | Type |
|---|---|---|---|
| 01 | [[01 mini-vLLM - report]] | 4, 5 | self project |
| 02 | [[02 Speculative decoding - report]] | 4, 5 | self project |
| 03 | [[03 CUDA kernels (CS794) - report]] | 4, 5 | course (CS794) |
| 04 | [[04 Seminar - Speech in LLMs - report]] | 4, 5 | M.Tech seminar (CS694) |
| 05 | [[05 CosyVoice Indic LoRA - report]] | 4, 5 | course (CS692) |
| 06 | [[06 CS728 Retrieval heads - report]] | 4 | course (CS728), team of 3 |
| 07 | [[07 LeNet-5 compression and robustness - report]] | 4 | course (CS725), team; see authorship |
| 08 | [[08 Photo-Seek - report]] | 4, 5 | self project |
| 09 | [[09 Emotion steering - report]] | 4 | course (CS772) |
| 10 | [[10 BountyBrain - report]] | 4 | self project; see authorship |
| 11 | [[11 DECS HTTP KV server - report]] | 5 | course (CS744) |
| 12 | [[12 Micro Exchange - report]] | 5 | self project; see authorship |
| 13 | [[13 Raft KV store - report]] | 5 | self project |

## Resume claims to fix or qualify (details in each report's §2)

| Project | Claim | What the evidence says |
|---|---|---|
| CUDA | "reducing peak activation memory 13.3%" | 13.3% is the whole training step's peak; activations alone drop 32%. Also found from the code: `mm_tiled_AT` computes the wrong gradient, and one recompute is redundant. |
| CUDA | 1.73× with column-split streams | Only at N = 32768, where the baseline itself jumps 7× for 4× more work. The cause was not profiled. |
| Speculative decoding | the "bursty acceptance" explanation | `acceptance_rate` is accepted ÷ drafted. Solving for α gives 0.912, which fits the independence formula exactly. |
| CosyVoice | "17 MB" | The adapter file is 35.3 MB (fp32). |
| CosyVoice | "across 4 H200 GPUs" | Each adapter trained on 1 GPU; runs and generation were spread across 4. |
| CosyVoice | WER 1.99 → 0.11 | The result CSVs are not on the server, so this can't be verified here. |
| CS728 | "32 heads and 3 layers" | Llama-3.2-1B has 16 × 32 = 512 heads; the top 10 sit in 7 layers. |
| CS728 | "mitigated lost-in-the-middle" | Not measured for the selected heads. The 1.3% baseline is chance level. K was tuned on the test set. |
| LeNet-5 | authorship | The extension report lists Harshith Matta as sole author, and the code is on an unpushed branch. |
| LeNet-5 | PGD 51.98 → 82.18 | At ε = 0.05 for the 13.8%-weights model. INT8 was QAT and 1.07× slower. |
| Photo-Seek | "fine-tuning ArcFace embeddings" | ArcFace is frozen, and the adapter head adds +0.00 / −0.07. The gain comes from sampling and constrained refinement (−41% error vs −7% random). |
| Emotion steering | "up to 90%" | One best cell (joy, layer 9). Best cells across emotions range 15–90%; the pooled mean is 14.9%. Trust and anticipation can't be scored. |
| BountyBrain | "30k+ nodes" | The graph is 112k nodes and 97k edges (35k-node core). The first scaffold commit is by another author ("Tarunsai"). |
| DECS | "I/O saturation at 3.1K TPS" | Disk %util was only 2–3%. The limit is likely commit fsync / lock latency. Also: a ~43 ms latency floor (Nagle; httplib TCP_NODELAY is off) and a connection-leak bug. |
| Micro Exchange | 10.8M ops/s, 50× | Laptop only. The shared server gives 3.4M ops/s and 14–18×. The first engine commit is by another author ("Tarunsai"). |
| mini-vLLM | "up to 7.4×" | At 8k context with 90% cached; 1.00× on the real multi-turn workload. |
