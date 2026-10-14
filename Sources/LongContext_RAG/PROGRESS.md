# PROGRESS

> **This section is the only authoritative one.** Everything below the first
> `---` is a dated session log kept for reasoning and history; those entries
> were true when written and some are explicitly superseded (Session 1's "not
> yet started" list, for instance, is done). Do not act on them without
> checking here first.

## START HERE (next session)

Milestones 1 and 2 are **done and committed** (`25c747e`, `499b949`, `3184028`).
Session 4 added the head-signal diagnostic (uncommitted, see below).
Nothing is mid-flight; no background jobs are running.

**Milestones 3-4 are done. The mechanism works, and the headline is:**

> A dense-shortlist -> retrieval-head-rescore cascade improves multi-hop QA F1
> by **+0.054** over dense retrieval at a matched 256-token budget on 32k
> contexts (0.661 vs 0.607, 95% CI [+0.028, +0.085], p < 0.001, n=300), while
> using **99.2% less context than naive long-context and answering it +0.18 F1
> better** (0.661 vs 0.480), at 7.0x lower latency.

The head signal fails as a *replacement* for embeddings and succeeds as a
*complement*. Both halves were pre-registered, neither is a reframing:

- *Standalone*: missed the kill criterion on both counts (R@5 0.672 vs dense's
  0.770; precision@256 0.284 vs 0.321). Dead as a replacement.
- *Fused with dense*: significant F1 gain, verified against placebo controls and
  a same-shortlist ablation. See Session 4 below for the full chain.

**Nothing is committed.** The working tree carries all of Session 4. Show the
diff and get approval before committing (see CLAUDE.md).

### What is left

1. **Cross-task generalization (Milestone 5)** — the Definition-of-Done item that
   is still open. MuSiQue, TriviaQA and NQ-open are already in the datasets
   cache; 2WikiMultihopQA is metadata-only and needs a download.
2. **Ablations**: head count k is swept (1/5/10/25, k=5 best); head-selection
   method (attention-magnitude vs copy-precision) is not.
3. **Short-paper write-up**, which the spec calls for given the paper potential.
4. Optional: sweep the fusion weight per context length; only 32k was tuned.

Milestone checkpoints go to `CHECKPOINTS.md` (gitignored) instead of being asked
as questions — see the working agreement in CLAUDE.md. It now carries Milestones
1, 2 and 3-4 in interview-answer form, including the design rationale and the
likely "why X not Y" questions with answers. **Read it before changing anything
those decisions rest on.**

`README.md` (committed, project-facing) carries the same results written for a
reader who has never seen the repo — method, results, controls, limitations.
Keep the two in sync: `PROGRESS.md` is the working log, `README.md` is the
artifact.

### Cold start — how to run anything here

Working directory is `/data/jayavardhan/projects/LongContext_RAG`. There is no
activated env; call the interpreter by absolute path, and always run offline:

```bash
cd /data/jayavardhan/projects/LongContext_RAG
ENV=/mnt/warm_storage/jayavardhan/tools/conda-envs/lcrag/bin/python
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 $ENV -u -m data.test_hotpot
```

Every model and dataset this project needs is already local, so `HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1` should be set on **every** invocation — without it, a
transient network stall can wedge a run for minutes.

Long runs: `nohup $ENV -u -m <module> ... > logs/<name>.log 2>&1 &`, record the
PID, and read the log file. **Never pipe a long run through `tail`** (it buffers
until exit) and **never `pkill`** (see shared-box rules below).

Reproduce Milestone 2 end to end (~54 min on one H200, plus ~2 min for latency):

```bash
$ENV -u -m baselines.run_baselines --n 300 --context-lengths 4000 16000 32000 \
    --budgets 256 512 1024 2048 --pool-size 8000 --out results/baselines
$ENV -u -m baselines.run_latency --n 25 --pool-size 8000 --out results/baselines
$ENV -m eval.plot --summary results/baselines/Meta-Llama-3.1-8B-Instruct_baselines_summary.json
```

Reproduce Milestones 3-4 (~35 min total). **Order matters** — everything after
the first command reuses its `--dump-scores` npz, which is where the expensive
head pass is paid:

```bash
$ENV -u -m compress.run_diagnostic --n 300 --context-lengths 32000 \
    --pool-size 8000 --budgets 256 512 --k-values 1 5 10 25 \
    --dump-scores --out results/diagnostic          # ~15 min, writes the npz
$ENV -u -m compress.analyze_fusion                  # placebo controls, CPU only
$ENV -u -m compress.run_fusion_eval --budgets 256 512 1024   # ~6 min
$ENV -u -m compress.analyze_eval                    # containment + bootstrap, CPU
$ENV -u -m compress.run_cascade                     # ~7 min
$ENV -u -m compress.run_cascade_latency --n 25      # ~5 min
```

The npz (`results/diagnostic/*_scores.npz`, 3.2 MB) is **gitignored**, so a fresh
clone must re-run `run_diagnostic --dump-scores` before anything else works.

Smoke-test any change in ~90 s with
`--n 8 --context-lengths 4000 --budgets 256 1024 --pool-size 3000 --out results/smoke`
(delete `results/smoke/` afterwards; it is throwaway). For `compress`, the fast
check is `$ENV -u -m compress.test_spans` (9 checks, no GPU, ~40 s).

### Module map — do not rebuild any of this

```
paths.py                resolve_model() -> local mirror or HF id
data/hotpot.py          Chunk, LongQAInstance, DistractorPool, InstanceBuilder,
                        render_context()
data/budget.py          fill_budget() -- the ONE greedy packer all methods share
data/test_hotpot.py     18 checks (gold labelling, depth, budget binding)
heads/capture.py        AttentionCapture -- per-decode-step attention at q_len=1
heads/ablate.py         HeadMasker -- per-head zeroing at o_proj
heads/scorer.py         RetrievalScorer -- copy-precision
baselines/encoders.py   DenseEncoder (bge-m3, CLS + L2, plain transformers)
baselines/retrievers.py Retriever base + dense/bm25/truncate/oracle/random, rrf()
baselines/rerankers.py  CrossEncoderReranker (bge-reranker-v2-m3)
baselines/generate.py   Generator (length-aware batching), build_prompt()
baselines/run_baselines.py  two-phase driver: score all, then generate all
baselines/run_latency.py    unbatched latency, shuffled job order
compress/spans.py       SpanPromptBuilder -- exact per-sentence token spans
compress/head_scorer.py HeadAttentionScorer (12 variants from ONE forward pass),
                        HeadRetriever adapter, load_head_list()
compress/test_spans.py  9 span-exactness checks (no GPU)
compress/run_diagnostic.py   scoring-only driver: head variants, fusion, Spearman,
                             naive-outcome split, --dump-scores
compress/analyze_fusion.py   placebo controls: length / random / bm25
compress/run_fusion_eval.py  held-out weight selection + generation at budget
compress/analyze_eval.py     answer containment, paired_bootstrap()
compress/run_cascade.py      dense shortlist -> head rescore + denseonly control
compress/run_cascade_latency.py  shuffled-order cost, per-query vs pre-indexed
eval/metrics.py         exact_match, f1, gold_recall, full_support, recall_at_k
eval/plot.py            accuracy-vs-compression figures
```

### The API a new method plugs into

Still the entry point for Milestone 5 work. A new method is **just a scorer**.
Implement `Retriever` and it inherits the matched-budget comparison, all metrics,
and the plots for free:

```python
class Retriever:
    name = "..."
    def score(self, question: str, chunks: list[Chunk]) -> list[float]: ...
```

`Chunk` has `.text .title .para_idx .sent_idx .is_gold .n_tokens` and
`.embed_text` (the title-prefixed form used for scoring). Scores are
higher-is-better, one per chunk, order matching `chunks`.

Then register it in the `retrievers` list in `baselines/run_baselines.py`
(phase 1, "Phase 1: score chunks"). Everything downstream is generic. Selection
happens via
`fill_budget(chunks, scores, budget, builder.title_tokens(inst)) -> list[Chunk]`,
and `render_context(selected)` turns it back into titled paragraphs in document
order. Scoring is budget-independent and is computed once per instance, then
reused across all budgets — keep it that way.

### Reference numbers (300 questions)

Milestone 2 baselines, F1:

| context | naive | BM25@256 | dense@256 | oracle@256 |
|---|---|---|---|---|
| 4,000 | 0.576 | 0.555 | 0.610 | 0.753 |
| 16,000 | 0.500 | 0.501 | 0.603 | 0.736 |
| 32,000 | 0.480 | 0.507 | 0.605 | 0.747 |

Milestones 3-4, all at 32k / budget 256:

| method | full support | F1 | signal cost |
|---|---|---|---|
| oracle | 0.997 | 0.747 | — |
| fusion, full context | 0.757 | 0.670 | 1497 ms |
| head_sum_k5 alone | 0.437 | 0.662 | 1497 ms |
| **cascade S=1024** | 0.700 | **0.661** | **105 ms** |
| cross-encoder | 0.697 | 0.641 | — |
| dense | 0.610 | 0.607 | 397 ms (encode) |
| naive long-context | — | 0.480 | — |

Full tables in `results/baselines/*.md`, `results/diagnostic/*.md`,
`results/cascade/*.md`.

### What is established, and what it constrains next

1. **The bar is dense retrieval, not naive long-context**, and it has now been
   cleared by +0.054 F1 (p < 0.001) — but only by the *fused* signal, never by
   the head signal used as a replacement.
2. **Retrieval metrics do not reliably convert to F1.** The cross-encoder wins
   clearly on R@5 (0.836 vs 0.770) and is *not* significantly better on F1
   (p = 0.09). Do not report a retrieval win as if it were an accuracy win.
3. **Compress hard, not softly.** Every method peaks at 256-1024 tokens and
   declines at 2048, the oracle included. A bigger budget is not safer.
4. **Fusion helps only when the second signal is decorrelated.** dense+bm25,
   dense+length, dense+random and dense+cross-encoder all add ~nothing or hurt;
   dense+head adds a lot. Test decorrelation (Spearman on gold chunks) before
   assuming any new signal will fuse well.
5. **Always pair the statistics.** Random halves of this dataset differ by
   ~0.09 F1 on an identical method — larger than any gap between methods. Use
   `compress.analyze_eval.paired_bootstrap`; an unpaired comparison at n=150
   cannot resolve anything here.
6. **Any latency claim must include the cost of producing the signal**, and must
   shuffle job order — block-grouped timing produced a physically impossible
   result twice in this project now.

### Design constraints to carry into the mechanism

- Key it to the **top-k head set with k >= 5**, never a single head. `L15H30`
  alone is causally dispensable (Session 2).
- **Select over sentences**, title-prefixed, and spend the budget through
  `data.budget.fill_budget` — that is what makes the comparison matched. Do not
  write a second packer.
- **Do not rebuild machinery that exists.** `heads/capture.py`
  (`AttentionCapture`, per-decode-step attention at `q_len=1`) and
  `heads/ablate.py` (`HeadMasker`, per-head zeroing at `o_proj`) are the
  primitives the mechanism needs.
- The open design question for Milestone 3, worth raising before coding: the
  head signal needs *a forward pass over the long context to exist*, which is
  the cost compression is supposed to avoid. Options are a cheap prefill-only
  pass, a smaller draft model, or a two-stage cascade (dense shortlist -> head
  rescoring). Whichever is chosen, **the latency table must be recomputed to
  include the cost of computing the signal**, or the efficiency claim is not
  honest.

### Known rough edges (small, not blocking)

- `heads/run_head_id.py::pick_device` calls `torch.cuda.mem_get_info` on every
  GPU, which spins up a CUDA context on all 8 (~522 MiB each) before the first
  log line. `baselines/run_baselines.py::pick_device` does the same job via
  `nvidia-smi` without that cost — the heads copy could adopt it.
- `results/head_id/*_instances.jsonl` is 2.3 MB and committed. It is what makes
  the answered-only recompute reproducible without a re-run. Drop it from git if
  the repo should stay lean.
- `results/baselines/*_baselines.jsonl` is 12 MB (22,500 per-condition records)
  and was **deliberately left out of `3184028`** — a large blob is hard to
  remove from history later, while adding it is one command. The summary JSON,
  the markdown tables and the figures are committed, so every number in the
  README is reproducible without it. The file is still on disk; commit it if you
  want per-record re-analysis without a 54-minute re-run.

### Environment

- conda env `/mnt/warm_storage/jayavardhan/tools/conda-envs/lcrag` (python 3.11,
  torch 2.13+cu130, transformers 5.15). `rank_bm25` added this session.
  `sentence-transformers` and `faiss` are **not** installed and are **not
  needed** — see Session 3 for why.
- **Models are mirrored to local disk** at `/data/jayavardhan/lcrag_models/`
  (bge-m3, Llama-3.1-8B). Loading from the NFS HF cache is pathological — see
  Session 3. `paths.resolve_model` handles the lookup; nothing else needs to
  know.
- HotpotQA, MuSiQue, TriviaQA and NQ-open are already in the datasets cache and
  load offline. **2WikiMultihopQA is only metadata-cached** and would need a
  download — relevant to the Milestone 5 cross-task study.
- Everything runs with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.

Sanity checks (all offline, seconds to ~1 min):

```bash
python -m data.test_hotpot     # 18 checks, benchmark construction
python -m heads.test_capture   # 14 checks, attention capture
python -m heads.test_masking   #  6 checks, head masking (loads a model)
```

**Shared-box rules:** this server has ~20 other users, and a colleague
(harshith) runs jobs under the *same* `jayavardhan` UID. Kill background jobs
only by a PID recorded at launch — never `pkill -f <pattern>`. Do not touch
`/mnt/warm_storage/jayavardhan/harshith/**`.

---

## Session 4 — 2026-08-12 — head-signal diagnostic (the premise test)

### What was run

`compress/run_diagnostic.py`: scoring only, no generation. 300 questions at 32k,
the same instances as Milestone 2 (same seed, same shuffle, same per-instance
seed), so every number lines up with the M2 tables. 888 s on one H200.

Scoring-only was deliberate. Whether the head signal is worth a mechanism is
decided entirely by where it ranks the gold; generating answers too would have
multiplied the cost to confirm what the ranking already settles.

### The result — the premise failed

| method | R@1 | R@5 | gold recall@256 | full support@256 | precision@256 |
|---|---|---|---|---|---|
| oracle | 0.433 | 0.999 | 0.999 | 0.997 | 0.397 |
| **reranker** | 0.390 | **0.836** | 0.869 | 0.697 | 0.340 |
| **dense** | 0.349 | **0.770** | 0.818 | 0.610 | 0.321 |
| **head_sum_k5** | 0.319 | **0.672** | 0.712 | 0.437 | 0.284 |
| bm25 | 0.306 | 0.616 | 0.661 | 0.373 | 0.255 |
| head_sum_k1 | 0.044 | 0.128 | 0.136 | 0.013 | 0.062 |

`head_sum_k5` is the best of twelve variants (3 aggregations x k in 1/5/10/25).
Kill criterion missed on both halves. The signal is *real* — random scores
R@5 0.004, so k=5 is ~170x chance and beats BM25 — but it is not competitive
with a bi-encoder, let alone a cross-encoder.

### Why, mechanistically (the honest explanation)

**Retrieval heads are a *copying* signal, and multi-hop evidence selection is
not a copying task.** Attention at the answer token points at where the answer
is about to be copied from, not at the bridge fact needed to get there. The
full-support-to-recall ratio shows exactly this:

| method | full_support@256 / gold_recall@256 |
|---|---|
| reranker | 0.802 |
| dense | 0.746 |
| head_sum_k5 | **0.613** |

The head signal is disproportionately bad at getting *both* gold sentences,
which is the one thing multi-hop needs.

### The finding worth keeping

Every external retriever degrades on the questions the model answers wrong with
full context. The head signal does not:

| method | R@5 (naive correct) | R@5 (naive wrong) | delta |
|---|---|---|---|
| dense | 0.842 | 0.697 | **-0.145** |
| reranker | 0.902 | 0.771 | -0.131 |
| bm25 | 0.698 | 0.534 | -0.164 |
| head_sum_k5 | 0.658 | 0.685 | **+0.027** |

So the signal is **not circular** — it does not depend on the model having
already used the evidence. Deflationary reading to rule out before claiming
anything: a uniformly mediocre signal has less room to drop, so flatness may
reflect insensitivity rather than insight. Testing that needs the head-vs-dense
rank correlation, which the current run does not save (only per-method metrics
are written, not raw scores). Add score dumping if this is pursued.

### Baseline concern that was justified

The cross-encoder (bge-reranker-v2-m3, same family as the bi-encoder so the
comparison is about architecture, not training data) **beats dense** — R@5 0.836
vs 0.770, full support 0.697 vs 0.610. Had the head signal won against dense
alone, "did you compare against a cross-encoder?" would have taken the result
apart in an interview. It is now measured.

### The fusion run — the result that changed the conclusion

The deflationary reading above (flat = insensitive, not orthogonal) is **ruled
out**. Mean Spearman over instances:

| pair | rho |
|---|---|
| dense ~ reranker | +0.432 |
| reranker ~ head_sum_k5 | +0.354 |
| dense ~ head_sum_k5 | +0.213 |
| dense ~ head_sum_k5, **gold chunks only** | **+0.072** |

On the sentences that actually matter the two signals are essentially
uncorrelated. So fusing them should add information — and it does:

| ranking signal | R@5 | gold recall@256 | full support@256 | precision@256 |
|---|---|---|---|---|
| oracle | 0.999 | 0.999 | 0.997 | 0.397 |
| **dense + head (z, w=0.25)** | **0.844** | **0.882** | **0.747** | **0.348** |
| reranker | 0.836 | 0.869 | 0.697 | 0.340 |
| dense + head (z, w=0.5) | 0.822 | 0.868 | 0.733 | 0.342 |
| dense + reranker (RRF) | 0.832 | 0.870 | 0.703 | 0.340 |
| dense + head (RRF) | 0.807 | 0.845 | 0.677 | 0.334 |
| dense | 0.770 | 0.818 | 0.610 | 0.321 |

Fusion closes **35% of the dense-to-oracle full-support gap** (0.610 -> 0.747
against a 0.997 ceiling). Note the control in that table: fusing dense with the
*reranker* — a stronger but correlated signal — adds almost nothing over the
reranker alone (0.703 vs 0.697). Decorrelation is doing the work, not fusion.

### Placebo controls (`compress/analyze_fusion.py`)

The cheap explanations for the lift, tested at the same weight:

| ranking signal | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| dense alone | 0.349 | 0.770 | 0.869 | 0.937 |
| **dense + head** | 0.346 | **0.844** | **0.935** | **0.970** |
| dense + length | 0.332 | 0.736 | 0.856 | 0.930 |
| dense + random | 0.341 | 0.750 | 0.863 | 0.927 |
| dense + bm25 | 0.357 | 0.761 | 0.857 | 0.925 |

Gold sentences are 1.18x the mean sentence length, so a length prior was a real
candidate explanation -- `head_sum` accumulates attention over tokens and is
length-biased by construction. It does not survive: **every placebo makes dense
worse.** Only the model-internal signal helps. (dense+bm25 hurting is also
consistent with M2, where RRF fusion of dense and BM25 never beat dense.)

### Does the retrieval gain become QA accuracy? (`compress/run_fusion_eval.py`)

Not assumable — M2 showed recall gains *lowering* F1. So: generation at matched
budgets, through the same packer and generator as every M2 baseline. The fusion
weight was swept on a dev half (150 questions) by full support and frozen before
touching the test half, which fixes the earlier "0.25 chosen on reported data"
problem. Dev curve was flat (0.687-0.740 across w=0.10-0.50), **chosen w=0.20**.

Sanity check that the pipeline is comparable: dense@256 reproduces M2 exactly
(full support 0.610, F1 0.607 vs published 0.610 / 0.605).

All 300, budget 256:

| method | gold recall | full support | EM | F1 |
|---|---|---|---|---|
| fusion (w=0.20) | 0.886 | 0.757 | 0.527 | **0.670** |
| head_sum_k5 alone | 0.712 | 0.437 | 0.533 | 0.662 |
| reranker | 0.869 | 0.697 | 0.497 | 0.641 |
| dense | 0.818 | 0.610 | 0.470 | 0.607 |

### The significance work that changed what is claimable

Dev and test halves *disagreed* about the ranking of the top three methods, and
the same method differed by ~0.09 F1 between halves — the halves differ in
difficulty by more than the methods differ from each other. Since every method
answers identical questions, a paired bootstrap over questions is the right test
and is far tighter. 10k resamples, budget 256, n=300:

| comparison | mean F1 diff | 95% CI | p |
|---|---|---|---|
| fusion vs dense | +0.0632 | [+0.0296, +0.0990] | **0.0000** |
| head_sum_k5 vs dense | +0.0553 | [+0.0108, +0.1017] | **0.0158** |
| reranker vs dense | +0.0344 | [-0.0050, +0.0747] | 0.0906 |
| fusion vs reranker | +0.0289 | [-0.0094, +0.0685] | 0.1434 |
| fusion vs head_sum_k5 | +0.0079 | [-0.0285, +0.0463] | 0.6902 |

**Only "beats dense" survives.** Fusion is *not* distinguishable from the head
signal alone or from the cross-encoder at this sample size — do not claim it is.
Note also that the reranker's clear win on retrieval metrics (R@5 0.836 vs
0.770) did **not** convert into a significant F1 gain, which is M2's lesson
appearing a second time.

### Why the head signal beats dense on 0.44 full support (`analyze_eval.py`)

Because annotated-evidence recall is the wrong yardstick for it. Attention is
read at the step generating the first answer token, so the signal points at what
the model is about to *copy*. Share of selected contexts containing the answer
string (span answers only, yes/no excluded):

| method | answer present |
|---|---|
| fusion | 0.898 |
| head_sum_k5 | 0.863 |
| reranker | 0.842 |
| dense | 0.796 |

The head signal finds the *answer* better than dense while finding the annotated
*supporting facts* far worse. Both statements are true and the second one is not
a defect — it is what the signal is for.

### The cascade — making the signal affordable (`compress/run_cascade.py`)

The accuracy result costs more than it saves: reading the head signal over the
full 32k context takes **1497 ms**, so `fusion_fullctx` is *slower than naive
long-context*. Dead as an efficiency claim. The cascade replaces the full pass
with dense-shortlist -> head-rescore over the shortlist only.

| method | full support | F1 | head pass |
|---|---|---|---|
| fusion_fullctx | 0.757 | 0.670 | 1497 ms |
| **cascade S=1024** | 0.700 | **0.661** | **105 ms** |
| cascade S=4096 | 0.730 | 0.661 | 211 ms |
| cascade S=2048 | 0.710 | 0.652 | 126 ms |
| dense | 0.610 | 0.607 | — |

Paired bootstrap, n=300:

| comparison | mean F1 diff | 95% CI | p |
|---|---|---|---|
| cascade_S1024 vs dense | +0.0545 | [+0.0278, +0.0845] | **0.0000** |
| **cascade_S1024 vs cascade_S1024_denseonly** | **+0.0531** | [+0.0251, +0.0836] | **0.0002** |
| fusion_fullctx vs cascade_S1024 | +0.0087 | [-0.0172, +0.0344] | 0.5318 |

Two things worth being able to say out loud:

- **The `_denseonly` control is the cleanest result in the project.** Same
  shortlist, same budget, same generator — the *only* difference is whether the
  head signal rescored it. It reproduces dense exactly (F1 0.608 vs 0.607), so
  the entire gain is the head signal and none of it is shortlisting.
- **The cascade loses nothing measurable** versus the full-context pass
  (p=0.53) at **14x lower signal cost**.

### Latency, honestly (`compress/run_cascade_latency.py`)

The first cascade run timed each shortlist in its own block and reported S=2048
as *faster* than S=1024 — impossible for a prefill, and the same warm-up
contamination Session 3 hit. Re-measured with shuffled job order and 10 calls
discarded, it is monotonic: 105 / 126 / 211 ms for S = 1024 / 2048 / 4096.

**M2's latency table measures generation only**, so its "15x faster than naive"
silently assumes retrieval is free. Encoding this benchmark's per-question
haystack costs 397 ms, which is not free. Both regimes are now reported:

| method | per-query ms | vs naive | pre-indexed ms | vs naive |
|---|---|---|---|---|
| naive long context | 1381 | 1.00x | 1381 | 1.00x |
| dense@256 | 489 | 2.83x | 92 | 15.0x |
| **cascade@256 (S=1024)** | **594** | **2.33x** | **197** | **7.0x** |
| fusion_fullctx@256 | 1986 | 0.70x | 1589 | 0.87x |

`per_query` charges the encoder every question (what this benchmark literally
does — each question carries its own haystack); `pre_indexed` amortizes it (what
a deployed system with a fixed corpus does). The cascade costs ~2x dense and
buys +0.054 F1; the full-context version costs more than naive and is not worth
having.

### Methodological caveats (limitations, not excuses)

1. **One decode step.** Attention is read only at the first answer token. Wu et
   al. score over the whole generation. Averaging over several steps is untested
   — though for a short-answer prompt the whole answer sits in one place, so
   this is unlikely to recover the missing bridge sentence.
2. **Heads were identified on NIAH over Paul Graham essays**, not on multi-hop
   QA. Cross-domain head transfer is the risk `Project.md` names explicitly.
   The standalone result is evidence against transfer; the fusion result is
   evidence that what *does* transfer is real but partial.
3. **The fusion weight (0.25) was chosen from two candidates on the data being
   reported.** Both 0.25 and 0.5 beat dense, so "fusion helps" is not an
   artifact of the choice, but the exact +0.074 R@5 is optimistic. A proper
   weight sweep with held-out selection is owed before the number is quoted.
4. **Retrieval quality is not QA F1.** M2 showed recall gains can lower F1.
   Nothing here has been generated yet.

### Written this session

```
compress/spans.py            SpanPromptBuilder -- exact per-sentence token spans
compress/head_scorer.py      HeadAttentionScorer (12 variants from one pass),
                             HeadRetriever adapter for the M2 pipeline
compress/test_spans.py       9 checks: span/sentence identity, ordering,
                             non-overlap, prompt identity vs build_prompt
compress/run_diagnostic.py   scoring-only driver, naive-outcome split, fusion
                             variants, Spearman correlation, score dumping
compress/analyze_fusion.py   placebo controls (length / random / bm25)
compress/run_fusion_eval.py  held-out weight selection + generation at budget
compress/analyze_eval.py     answer containment + paired bootstrap
compress/run_cascade.py      dense shortlist -> head rescore, + denseonly control
compress/run_cascade_latency.py  shuffled-order cost measurement, both regimes
baselines/rerankers.py       CrossEncoderReranker (bge-reranker-v2-m3)
```

`bge-reranker-v2-m3` was already in the NFS HF cache and is now mirrored to
`/data/jayavardhan/lcrag_models/` like the others.

### Gotcha

The span test caught nothing in the builder but did catch a bad test: locating
the question block with `min(i for i ... if q in decode(ids[i:]))` returns 0,
since decoding from 0 also contains the question. Spans are verified by decoding
each one back to its own sentence and by asserting the assembled prompt is
byte-identical to `build_prompt(render_context(chunks))`.

---

## Session 3 — 2026-08-11 — Milestone 2: benchmark + baselines

### Decisions taken (with your go-ahead)

All four were raised as options with a recommendation; you took the recommended
option on each.

1. **Pooled-distractor HotpotQA**, not native distractor and not fullwiki.
   Native distractor contexts measure **1,312 Llama tokens on average** — there
   is no long context to compress and every method would tie. Fullwiki would
   have made retrieval failure indistinguishable from compression failure.
   Each question keeps its 2 gold + 8 hard distractor paragraphs and is padded
   from a train-split pool to 4k / 16k / 32k.
2. **Sentence as the retrieval unit**, title-prefixed for scoring. HotpotQA
   annotates supporting facts per sentence, so Recall is scored directly against
   the annotation with no span mapping.
3. **bge-m3 and BM25, both reported.** Multi-hop questions are entity-heavy,
   where BM25 stays competitive; reporting only a dense retriever invites "did
   you only beat a weak baseline?".
4. **Budget = tokens of retrieved context** handed to the generator, excluding
   question and scaffolding. Absolute token budgets, swept 256/512/1024/2048.

### The results

300 questions x 3 lengths x 4 budgets x 6 methods + naive = 22,500 generations,
54 min on one H200.

| context | naive F1 | dense@256 F1 | oracle@256 F1 |
|---|---|---|---|
| 4,000 | 0.576 | 0.610 | 0.753 |
| 16,000 | 0.500 | 0.603 | 0.736 |
| 32,000 | 0.480 | 0.605 | 0.747 |

Three things worth being able to say out loud in an interview:

- **Naive long-context decays with length (0.576 -> 0.480) while dense stays
  flat.** The gold is *always present* in the naive condition, so nothing was
  lost to retrieval — the decay is purely the model failing to use a long
  context. That is lost-in-the-middle, measured directly.
- **The oracle declines as its budget grows** (0.753 -> 0.630 at 4k) even though
  it always contains every gold sentence. Only the amount of surrounding non-gold
  text changes. This is the cleanest argument in the whole run that compression
  is worth doing.
- **Latency: 15.0x at 32k** (1,381 ms -> 92 ms) while being more accurate. The
  retrieval baseline dominates naive long-context on both axes at once.

Hybrid RRF never beat dense alone. Truncation collapses with length (0.179 ->
0.059 at a 256 budget), which confirms the padded haystack genuinely buries the
evidence rather than leaving it near the front.

### The one genuinely tricky technical bit

**Loading models from the NFS HF cache is pathological, and it is not the
network.** bge-m3 did not finish loading in 280 s; the same checkpoint on local
disk loads in **30 s**. Sequential reads off that mount run at 80 MB/s, so it
looks healthy — `dd` of the full 2.3 GB file takes 27 s.

A `faulthandler` stack dump found the loader threads parked in
`transformers/core_model_loading.py::_materialize_copy`. transformers
materializes checkpoint tensors through an **mmap**, and mmap page-faults over
NFS are brutal even where sequential reads are fine. Two wrong hypotheses got
ruled out first: thread-pool explosion (`GLOBAL_WORKERS` is `min(4, cpu_count)`,
so only 4) and a corrupt blob (the file reads back fine — the first stalled `dd`
was contending with the hung Python process still holding the mount busy).

Fix: mirror models to `/data/jayavardhan/lcrag_models/` and resolve through
`paths.resolve_model`, which falls back to the HF id when no mirror exists.

### Written this session

```
paths.py               local-mirror model resolution
data/hotpot.py         pooled-distractor construction, Chunk / LongQAInstance
data/budget.py         fill_budget -- the single greedy packer all methods share
data/test_hotpot.py    18 checks: gold labelling, depth control, budget binding
baselines/encoders.py  bge-m3 via plain transformers (CLS + L2, no ST dependency)
baselines/retrievers.py dense / BM25 / hybrid RRF / truncation / oracle / random
baselines/generate.py  batched greedy generation, length-aware batching
baselines/run_baselines.py  two-phase driver (score, then generate)
baselines/run_latency.py    unbatched latency vs naive long-context
eval/metrics.py        HotpotQA official EM/F1, gold recall, full support, R@k
eval/plot.py           accuracy-vs-compression figures
```

### Decisions that were mine to make, flagged in case you disagree

- **No FAISS.** Each instance has its own haystack of a few hundred to a few
  thousand sentences, so one exact matmul is faster than building a per-instance
  ANN index. An index only pays off against one large shared corpus (fullwiki).
  The spec mentions FAISS/BM25 as "standard retrieval tooling"; this is an
  implementation choice, not a change of baseline.
- **No `sentence-transformers`.** bge-m3's dense head is exactly CLS pooling +
  L2 normalisation, confirmed against the checkpoint's own
  `1_Pooling/config.json`. The wrapper would add a dependency without adding
  behaviour.
- **`full_support` reported alongside `gold_recall`.** For multi-hop, mean recall
  0.5 could mean "half the evidence for everyone" or "all the evidence for half";
  only the second is answerable. `full_support` disambiguates.
- **Added an oracle and a random condition** beyond the spec's two baselines.
  The oracle is what turned "dense scores 0.605" into "dense leaves 0.14 F1 on
  the table", which is the finding that motivates Milestone 3.

### Gotchas hit, so they are not re-hit

- Model loading from `/mnt/warm_storage` — see above. This is the big one.
- A progress print keyed on `len(out) % 500 < len(idx)` **silently stops firing**
  once long contexts shrink the batch below the step. Looked like a 7-minute
  hang; the job was at 97% GPU. Fixed to compare against a running marker.
- Batching by fixed count OOMs at 32k. `Generator` now bounds a batch by both
  count and total padded size.
- Latency grouped by condition is contaminated by warm-up: an identical
  330-token prompt read 196 ms in the first block and 77 ms later. Jobs are now
  shuffled before timing and 10 calls discarded.

---

## Session 2 — 2026-08-11 — verifying the head list

### Correction to a Session 1 claim

Choosing Llama over Qwen was justified partly on "Wu et al. published retrieval-head
lists for the Llama family, so the reimplementation can be checked against a
reference." **That was wrong for this checkpoint.** Their `head_score/` directory
covers only llama-2-7b-80k, llama-2-13b-64k, Mistral-7B-v0.2, Mixtral-8x7B,
Qwen1.5-14B, Qwen1.5-14B-Chat, Yi-6B-200K. There is **no Llama-3 or Llama-3.1
list**, so the promised cross-check was never available as described. (Neither
was it available for Qwen2.5 — the published Qwen is 1.5-14B — so the decision
was not necessarily wrong, but the stated reason for it was.)

### What the cross-model comparison does and does not show

`results/head_id/published_comparison.md`, vs `llama-2-7b-80k`:

| | heads >= 0.1 | share | max score |
|---|---|---|---|
| ours (Llama-3.1-8B) | 21/1024 | 2.1% | 0.725 |
| llama-2-7b-80k | 33/1024 | 3.2% | 0.943 |

Top-20 overlap **1/20**, Spearman **+0.015** over all 1024 heads.

The *shape* agrees with the paper's central claim (a low-single-digit percentage
of heads carry the copying). The *positions* do not — which is what the paper
predicts across different checkpoints, so it is not evidence of a bug. This
supports the implementation; it cannot verify the head list.

### Causal verification (this is the real check)

`heads/run_ablation.py` — zero heads at the input to `o_proj` (removing one
head's contribution without renormalizing the others' softmax) and measure
needle recall. 80 instances per condition, random heads of matched count as the
control, three seeds.

| heads masked | identified | random (mean of 3 seeds) | gap |
|---|---|---|---|
| 1 | 0.975 | 0.975 | **+0.000** |
| 5 | 0.688 | 0.975 | **+0.287** |
| 10 | 0.625 | 0.962 | **+0.337** |

Baseline (nothing masked) 0.975, consistent with the sweep's 0.988.

**Masking 5 identified heads costs 29 points of recall; masking 5 random heads
costs nothing.** The head list is causally special — which is precisely the
property the compression premise depends on.

### The result that changes the design

**Masking `L15H30` alone — copy-precision 0.865, twice the runner-up — changes
recall by exactly zero.** The single highest-scoring head is individually
dispensable; the capability is redundant across the top few.

Consequence for Milestones 3-4: **do not key the compression mechanism to one
head.** A mechanism built on `L15H30` would rest on a head the model does not
actually need. Use the top-k set, k >= 5.

Honest caveat: a null effect at k=1 is weaker evidence than the clear effect at
k=5. `L15H30` may still do the work in the unablated model, with others
compensating once it is removed. Separating "not needed" from "redundantly
covered" needs a different experiment (e.g. masking the rest of the top-k while
*keeping* L15H30, or patching its output).

Guard against the obvious failure mode: `heads/test_masking.py` checks the mask
mechanism itself, because a null ablation result and a dead hook look identical.
All 6 checks pass — and decisively for the point above, masking `L15H30` **does**
move the logits (max|delta| = 0.172), so the hook fires and the zero recall drop
is real redundancy rather than a no-op:

```
[PASS] masking L15H30 changes logits (max|d|=0.172)
[PASS] masking 5 heads changes more than 1 (0.250 > 0.172)
[PASS] masking all of layer 15 is largest (1.281)
[PASS] masking every head destroys the model (21.375)
[PASS] clean prediction is sensible (got ' Tokyo')
[PASS] all-heads-masked prediction differs (got ' the')
```

---

## Session 1 — 2026-08-10 — Milestone 1: retrieval-head identification

### Decisions taken (with your go-ahead)

1. **Base model: Llama-3.1-8B-Instruct.** Matches the spec's "Llama-3-8B class"
   and, unlike Qwen, Wu et al. published retrieval-head lists for the Llama
   family, so the reimplementation can be checked against a reference.
   - The official `meta-llama` repo is gated and this box's HF token is **not
     approved** for it (403 on file download; the metadata endpoint returns 200
     even without approval, which is what made the first check look fine).
   - Using `NousResearch/Meta-Llama-3.1-8B-Instruct` — an ungated mirror of the
     same weights. Request access to the official repo if you want to switch.
2. **Head-ID method: Wu et al. copy-paste retrieval score** as primary.
   Attention-mass-on-gold-evidence is deferred to the head-selection ablation
   (Milestone 5) so the choice becomes a measured result rather than an assertion.

### Environment

- conda env: `/mnt/warm_storage/jayavardhan/tools/conda-envs/lcrag` (python 3.11,
  torch 2.13.0+cu130, transformers 5.15.0, datasets 5.0.1)
- haystack corpus: 49 Paul Graham essays (645 KB) at
  `/mnt/warm_storage/jayavardhan/tools/lcrag_data/pg_essays`
- Only `/data/jayavardhan` and `/mnt/warm_storage/jayavardhan` are usable, and
  the `harshith` subfolder there is strictly off-limits.
- GPU 0 is shared with another job; `heads.run_head_id` auto-picks the freest GPU.

### The one genuinely tricky technical bit

transformers 5.x **removed `output_attentions`** — attention weights only exist
on the eager path. Running eager throughout is infeasible (a `[heads, T, T]`
matrix per layer, ~4 GB/layer at T=8k, quadratic in T).

The pipeline exploits the fact that the retrieval score is only ever read at
*decode* steps, where q_len=1 and the matrix is `[heads, 1, T]` (~1 MB/layer):

- prefill under SDPA with attention capture off;
- flip `config._attn_implementation` to `"eager"` for the single-token decode
  steps (it is re-read on every forward, and its setter does no validation);
- capture weights via a forward hook on each `self_attn` module;
- feed the **last prompt token inside the decode loop**, not the prefill, so the
  attention producing the first generated token is also captured at q_len=1.

`AttentionCapture.stack()` raises if any layer captured nothing, so a silent
fallback to SDPA fails loudly instead of producing all-zero scores.

### Written this session

```
heads/needles.py      8 needles across 4 kinds (natural fact, random digits,
                      rare proper noun, invented term)
heads/haystack.py     PromptBuilder — segment-wise tokenization for exact needle
                      token spans; NIAH instance construction at a given depth
heads/capture.py      AttentionCapture — hooks + sdpa/eager flip
heads/scorer.py       RetrievalScorer — the copy-paste score
heads/run_head_id.py  sweep driver + aggregation
README.md, .gitignore
```

### Status / what is next

- [x] Code written and syntax-clean
- [x] Mechanism validated — `python -m heads.test_capture`, 14/14 checks, offline
      on a tiny random Llama (no download needed). Covers: hooks fire per layer
      and only inside `recording()`; eager flip applied and restored; captured
      rows are real distributions over the right key count at q_len=1; needle
      spans token-exact; and the copy rule's four cases — a true hit plus three
      traps (right position/wrong token, right token/wrong position, the
      position-0 attention sink).
- [x] Llama-3.1-8B weights in place, all four shards sha256-verified
- [x] Full sweep: 240 instances in **154 s**, needle recall **0.988**
- [x] Documented head list -> `results/head_id/`
- [x] Metric artifact found and fixed; sweep re-run under copy-precision
- [ ] Sanity check against Wu et al.'s published Llama head list (not yet done —
      **do not claim agreement until it is**)

### Results — first head list

**21 of 1024 heads score above 0.1**, and the distribution is extremely sparse:
`L15H30 = 0.725` is nearly **2x** the runner-up (`L8H1 = 0.362`). That sparsity is
qualitatively what the retrieval-head literature reports, which is a good sign
for the whole premise of the project — if the signal were spread over hundreds of
heads there would be nothing to compress with.

Robustness check that did pass cleanly: recomputing over only the 237/240
instances the model actually answered changes the ranking not at all —
**top-20 overlap is 20/20**. So the head list is not an artifact of failures.

### Final head list (copy-precision)

**25 of 1024 heads above 0.1.** `L15H30 = 0.865` remains dominant, 2x the
runner-up `L8H1 = 0.430`. Answered-only recompute: 0.868 vs 0.865, top-20
overlap **20/20**.

`L15H30` copy-precision is now flat across all eight needles (0.800-0.918),
including the rare proper noun that previously read 0.283. The generalization
gap was the metric, not the model.

### Metric artifact — found and fixed

The per-needle-kind table appears to show retrieval heads collapsing on
`rare_proper_noun` needles (`L15H30`: 0.87 on natural facts vs **0.28** on the
rare name). Reported as-is that is a "heads don't generalize" headline — and it
would have been **wrong**.

That needle's recall is 0.97: the model retrieves it perfectly, emitting
`Dr. Ilsabet Quorrigan` verbatim. The score is low because Wu et al.'s
`|copied positions| / |needle length|` divides by the *whole needle* while the
model only copies the *answer span*. The `rare_name` needle is 26 tokens long but
its answer is ~10, capping the achievable score at 0.371.

Dividing each needle's observed max score by that ceiling gives a ratio of
**0.76-0.89 across all eight needles** — i.e. once the ceiling is accounted for,
every needle behaves the same and the "generalization gap" disappears:

| needle | klen | gen tokens | ceiling | observed max | ratio |
|---|---|---|---|---|---|
| glass_bridge | 25 | 25.0 | 1.000 | 0.885 | 0.89 |
| animal_fact | 23 | 23.0 | 1.000 | 0.880 | 0.88 |
| sf_sandwich | 23 | 22.1 | 0.961 | 0.849 | 0.88 |
| magic_osaka | 11 | 11.0 | 1.000 | 0.818 | 0.82 |
| magic_helsinki | 11 | 11.0 | 1.000 | 0.800 | 0.80 |
| invented_term | 21 | 19.9 | 0.948 | 0.778 | 0.82 |
| recipe_code | 17 | 10.1 | 0.594 | 0.504 | 0.85 |
| rare_name | 26 | 9.6 | 0.371 | 0.283 | 0.76 |

**Resolution (your call): copy-precision is the headline metric**, with Wu et
al.'s `|g_h|/|k|` reported in the adjacent column for comparability.

Copy-precision = of the decode steps where the emitted token *is* in the needle,
the fraction where head h's attention argmax pointed at it. Micro-averaged over
the sweep, so an instance with a single copy step cannot swing the average. It is
invariant to how much of the needle the answer covers, which is precisely the
failure above.

Results are also broken out **per individual needle**, not only per kind — a kind
containing one needle is confounded with that needle's identity, which is how the
artifact disguised itself as a generalization gap in the first place.

Interview framing: the honest version of this story is not "I reimplemented the
paper's metric" but "I reimplemented it, found its normalizer was confounded with
needle design, and measured the confound before believing the result."

The sweep + analysis are chained behind the download by
`$CLAUDE_JOB_DIR/tmp/run_sweep.sh` (logs to `logs/sweep.log`), so they start
without supervision. To run by hand once weights are present:

```bash
python -m heads.run_head_id --context-lengths 2000 4000 8000 --n-depths 10
python -m heads.analyze \
  --scores results/head_id/Meta-Llama-3.1-8B-Instruct_head_scores.json \
  --instances results/head_id/Meta-Llama-3.1-8B-Instruct_instances.jsonl
```

**Gotchas hit, so they are not re-hit:**
- Do not pipe long runs through `tail` — it buffers everything until exit; a
  healthy run looked hung for 13 minutes.
- `hf download` wedged twice (with *and* without `hf_transfer`), dropping to
  ~2 KB/s while a clean connection did 4 MB/s. Killing it and using
  `curl -sSL -C -` against `resolve/main/<file>` fixed it. Each wedged attempt
  also left a *fresh* `.incomplete` blob, so several partials accumulated per
  shard — resume the **largest** one, not the alphabetically first.
- CUDA context init on this box takes ~2 min before the first log line; a silent
  process is not necessarily a stuck one.
- The `Qwen2.5-7B` entry in the shared HF cache is incomplete — loading it
  triggers a 15 GB download.

### Not yet started

Baselines (Milestone 2), compression mechanism (3–4), main experiment (5).
No commits made yet — awaiting your review of the diff.
