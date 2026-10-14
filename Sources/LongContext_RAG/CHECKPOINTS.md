# Checkpoints

Revision material: what was built, why it was built that way, and the questions
an interviewer is most likely to push on. Written as the explanation you would
give out loud.

---

## Milestone 1 — Retrieval-head identification

### What it is, in one breath

Some attention heads in an LLM do the actual work of copying information out of
the context into the answer. Wu et al. call these retrieval heads and show they
are few, sparse, and causally responsible for long-context factuality. Milestone
1 reimplements their identification method on Llama-3.1-8B-Instruct and then
verifies the resulting head list *causally*, by ablation, rather than trusting
the score.

### How the score works

Plant a needle sentence at a known depth in a long distractor haystack, ask a
question only the needle answers, and watch each decode step. A head scores a
hit when its attention **argmax** lands on a needle token that is exactly the
token currently being generated — the head caught in the act of copying.

The headline metric is **copy-precision**: of the decode steps where the emitted
token was in the needle, what fraction did this head's argmax point at it.

### The finding that matters

**25 of 1024 heads score above 0.1, and `L15H30` scores 0.865 — roughly twice
the runner-up.** The signal is sparse and concentrated, which is the precondition
for the whole project: if the copying were spread across hundreds of heads there
would be nothing to compress with.

### Then the result that broke the obvious story

Masking `L15H30` alone — the top head by a factor of two — changes needle recall
by **exactly zero**. Masking five identified heads costs 29 points of recall
while masking five random heads costs nothing.

So the head list is causally special as a *set*, but the single best head is
individually dispensable: the capability is redundant across the top few. This
is why the compression mechanism keys on the top-k set with k ≥ 5 and never on
one head.

### Likely interview questions

**"How do you know your masking hook actually fired? A null result and a dead
hook look identical."**
That is exactly why `heads/test_masking.py` exists. It confirms masking `L15H30`
*does* move the logits (max |Δ| = 0.172), that masking 5 heads moves them more
than 1, that masking a whole layer moves them more still, and that masking every
head destroys the model. The zero recall drop is real redundancy, not a no-op.

**"You reimplemented the paper's metric — did you reproduce their head list?"**
No, and I can't: Wu et al. publish head scores for Llama-2, Mistral, Mixtral,
Qwen1.5 and Yi, but **not for any Llama-3.x model**. I checked, and an earlier
note in my own repo claiming otherwise was wrong. Against llama-2-7b-80k the
*shape* matches — 2.1% vs 3.2% of heads above threshold — while the positions do
not (top-20 overlap 1/20). That is what the paper itself predicts across
checkpoints, so it supports the implementation but cannot verify the list. The
verification is the ablation instead, which is the same evidence the paper uses.

**"Why copy-precision instead of the published normalizer?"**
This is the best story in Milestone 1. Wu et al. divide by the whole needle
length, `|g_h|/|k|`. But the model only copies the *answer span*, not the whole
needle. On a 26-token needle whose answer is ~10 tokens, a perfect head cannot
score above 0.371. My first sweep therefore showed retrieval heads apparently
collapsing on rare proper nouns — 0.87 on natural facts vs 0.28 on a rare name —
which reads as a headline "retrieval heads don't generalize" finding. It was
wrong. That needle's recall was 0.97: the model retrieved it perfectly. Dividing
each needle's score by its achievable ceiling gave a ratio of 0.76–0.89 across
all eight needles — uniform. The generalization gap was the metric, not the
model. Copy-precision is invariant to how much of the needle the answer covers,
which is precisely the failure. I report the published score alongside it.

The honest framing: not "I reimplemented the paper's metric" but "I
reimplemented it, found its normalizer was confounded with needle design, and
measured the confound before believing the result."

**"Why is the score only computed at decode time?"**
`output_attentions` was removed in transformers 5.x, and attention weights only
exist on the eager path. Running eager throughout is infeasible — an eager
prefill allocates a `[heads, T, T]` matrix per layer, about 4 GB per layer at
T=8k, growing quadratically. But the score is only ever *read* at decode steps,
where query length is 1 and the matrix is `[heads, 1, T]`, about 1 MB per layer.
So the pipeline prefills under SDPA and flips `config._attn_implementation` to
eager only for the single-token steps. The last prompt token is fed inside the
decode loop rather than the prefill, so even the attention producing the first
generated token is captured at q_len=1.

---

## Milestone 2 — Benchmark and baselines

### The question this milestone answers

Before building anything clever, what does a *strong* conventional RAG system
achieve at a matched context budget, and how much room is there above it?

### The thing to understand first: why the dataset had to be rebuilt

HotpotQA's distractor setting gives 10 paragraphs — I measured **1,312 Llama
tokens on average**. Llama-3.1 has a 128k window. At 1.3k tokens nothing is lost
in the middle, there is nothing to compress, and every method ties. Evaluating
long-context compression there would be measuring nothing.

So each question keeps its 2 gold and 8 hard TF-IDF distractor paragraphs and is
padded with paragraphs pooled from other questions up to 4k / 16k / 32k tokens.
That buys a context-length knob, exact sentence-level gold labels, and
controllable gold depth.

### The three results, and why each matters

**1. Retrieval beats stuffing the window, and the gap widens with length.**

| context | naive long-context F1 | dense @256 F1 |
|---|---|---|
| 4,000 | 0.576 | 0.610 |
| 16,000 | 0.500 | 0.603 |
| 32,000 | 0.480 | 0.605 |

At 32k, a 256-token budget — 1/128th of the context — is **+0.125 F1** and
**15× faster**.

**2. This is lost-in-the-middle measured, not asserted.** Naive decays 0.576 →
0.480 while dense stays flat at ~0.605. The crucial detail: in the naive
condition the gold sentences are *always present*. Nothing was lost to
retrieval. The decay is purely the model failing to use a long context.

**3. The result that justifies the whole project — the oracle gets worse as it
gets more budget.** The oracle condition always contains every gold sentence,
and its F1 still falls: **0.753 → 0.729 → 0.680 → 0.630** as the budget goes 256
→ 512 → 1024 → 2048 at 4k. The evidence is constant. The only thing changing is
how much non-gold text sits next to it.

If accuracy fell only when evidence was missing, the answer would be "retrieve
better" and compression would be a pure efficiency trick — a speed/accuracy
trade you accept a cost for. But accuracy falls *with the evidence present*, so
irrelevant context is itself harmful. That makes compression an **accuracy**
intervention, not just an efficiency one, and it is why the accuracy-vs-budget
curve has a peak instead of rising monotonically. It is also why the project's
premise — spend the context window on less, but better-chosen, text — is worth
pursuing at all.

### What this sets up for Milestone 3

Dense @256 reaches 0.605 F1 at 32k; the oracle reaches 0.747 at the *same
budget*. That **~0.14 F1 gap is a retrieval-signal limit, not a generation
limit** — the generator demonstrably can answer from 256 tokens, it just isn't
being handed the right 256 tokens. That gap is what the retrieval-head method
has to attack, and it means **the bar is dense retrieval, not naive
long-context.** Beating naive is not a result any more.

### Likely interview questions

**"Isn't a padded haystack synthetic? Aren't you making the task artificially
easy for retrieval?"**
It is semi-synthetic and I say so. Two guards: pooled paragraphs are excluded by
title whenever the title appears anywhere in the source question's own 10, so
gold content can't reappear at an unlabelled position; and the pool is drawn
from the **train** split so padding never pulls another evaluation question's
gold in. The residual risk — a pooled paragraph under a different title
happening to mention the answer — is inherent to any distractor pool, fullwiki
included. The alternative, fullwiki, would have been *worse* for this question:
gold is no longer guaranteed present, so retrieval failure gets confounded with
compression failure, and the comparison I actually care about gets muddier.

**"Did you only beat a weak baseline?"**
That is why BM25 is reported alongside bge-m3. Multi-hop questions are
entity-heavy, which is exactly where lexical retrieval stays competitive — BM25
lands at 0.555 vs dense 0.610 at 4k/256, close enough to be a real baseline
rather than a strawman. bge-m3 is a genuinely strong retriever, not MiniLM. And
the oracle bounds the whole thing from above so nobody has to take my word for
where the ceiling is.

**"Why sentences and not paragraphs or fixed windows?"**
HotpotQA annotates supporting facts *per sentence*, so a sentence unit lets
Recall be scored directly against the annotation with no span mapping and no
label noise. Paragraphs would have been too coarse — only ~10 units per native
example, which is not enough resolution for a compression-ratio curve. Fixed
windows are what production RAG does over arbitrary documents, but they straddle
sentence boundaries so gold labels have to be mapped by overlap, which adds
noise to the exact metric I most need to trust. Sentences are scored
title-prefixed because a bare HotpotQA sentence often drops its subject ("He was
born in…"), which no retriever could recover.

**"How do you know the comparison is actually matched?"**
Every method — dense, BM25, hybrid, truncation, oracle — differs *only* in the
scores it assigns to chunks. They all spend the budget through one shared greedy
packer (`data/budget.py`). So a difference in results is a difference in the
retrieval signal, not in how greedily one implementation packed its context.
Budget is defined as tokens of retrieved context handed to the generator,
excluding question and scaffolding, and it's measured on the rendered string:
a 1024 budget yields 973 tokens and never exceeds.

**"Why no FAISS? The spec mentions it."**
Each instance has its own haystack of a few hundred to a few thousand sentences.
One exact matmul is both exact and faster than building a per-instance ANN
index — an index only earns its cost against a single large shared corpus, which
is the fullwiki setting I deliberately didn't choose. It's an implementation
choice, not a change of baseline, and swapping it in later changes nothing about
the comparison.

**"Hybrid retrieval usually helps. Why didn't it?"**
It didn't here — RRF of dense and BM25 landed between the two and never beat
dense alone. The reading is that fusing a weaker lexical signal into a stronger
dense one costs more than it adds when the dense retriever is already good. The
practical consequence: when I later ablate combining the head signal with
embedding similarity, I should not assume fusion helps by default. I have direct
evidence from this dataset that it can hurt.

**"Your speedup number — is it fair?"**
It's measured unbatched, one request at a time, with 10 warm-up calls discarded,
because batching mixes wall-clock across conditions and would make per-condition
latency meaningless. I also found and fixed a real measurement bug: jobs were
grouped by context length, so whichever block ran first absorbed residual
warm-up — an identical 330-token prompt read 196 ms in the leading block and
77 ms elsewhere. Jobs are now shuffled before timing, which dropped the headline
32k speedup from an inflated 20.5× to an honest 15.0×.

**"What's the biggest threat to your headline result?"**
That the retrieval-head signal needs a forward pass over the long context in
order to exist — which is the exact cost compression is supposed to avoid. If I
report accuracy gains without charging for computing the signal, the efficiency
claim is dishonest. That's the open design question for Milestone 3, and the
latency table has to be recomputed to include it.

### The debugging story worth telling

Loading models from the NFS HF cache was pathological — bge-m3 didn't finish
loading in 280 s, while the same checkpoint on local disk loads in 30 s. What
makes it a good story is that the obvious diagnosis was wrong twice. It looked
like slow I/O, but `dd` reads the full 2.3 GB file in 27 s at 80 MB/s. It looked
like thread-pool explosion on a 216-core box, but transformers caps the loader
at `min(4, cpu_count)` workers. A `faulthandler` stack dump found the threads
parked in `core_model_loading._materialize_copy`: transformers materializes
checkpoint tensors through an **mmap**, and mmap page-faults over NFS are brutal
precisely where sequential reads are fine. Fix was to mirror models to local disk
behind `paths.resolve_model`.

The transferable lesson: sequential throughput told me nothing about random
access, and I only got to the answer by dumping the stack instead of reasoning
about it.

---

## Milestones 3–4 — The retrieval-head-guided compression mechanism

### What it is, in one breath

Standard RAG picks context with an embedding model that has never seen the
generator. This milestone adds a second signal that comes from *inside* the
generator — where its retrieval heads attend when it is about to answer — and
uses it to rescore a dense shortlist before spending a tight token budget. It
improves multi-hop QA F1 by +0.054 over dense retrieval at a matched 256-token
budget, and the gain is significant, controlled, and cheap.

### The result, in the form I would quote it

> A dense-shortlist → retrieval-head-rescore cascade improves HotpotQA F1 by
> +0.054 over bge-m3 dense retrieval at a matched 256-token budget on 32k
> contexts (0.661 vs 0.607, 95% CI [+0.028, +0.085], p < 0.001, n=300), while
> using 99.2% less context than naive long-context and answering +0.18 F1
> better, at 7× lower latency.

### The honest arc, which is the actual story

I set out to *replace* embedding similarity with the head signal. **That failed,
and I killed it on a criterion I set before running.** The standalone head signal
scores R@5 0.672 against dense's 0.770 and precision@256 0.284 against 0.321 —
worse on both, so the replacement framing is dead.

What saved the project was asking *why* it failed rather than tuning it. The head
signal is a **copying** signal: it points at where the answer is about to be
copied from. Multi-hop evidence selection is not a copying task — you also need
the bridge fact, which the model is not copying at that moment. The tell is the
ratio of full-support to gold-recall: 0.61 for the head signal against 0.75 for
dense and 0.80 for the cross-encoder. It is disproportionately bad at getting
*both* gold sentences, which is exactly the thing multi-hop needs.

So it is not a better retriever. It is a **different** one — and that is the
useful property, because a different signal can be fused.

### Why fusion was worth trying, and why it is not just "ensembling helps"

The evidence for fusion came before the fusion: mean Spearman between the head
signal and dense is +0.213 over all sentences, and **+0.072 restricted to the
gold sentences**. On the sentences that matter the two are essentially
uncorrelated. Two signals that disagree that much have information the other
lacks.

That is also why I do not accept "any second signal would help." I ran the
placebos at the same weight and every one of them made dense *worse*: length
0.736, random 0.750, BM25 0.761, against dense's own 0.770 — and dense+head
0.844. Length mattered as a control specifically because `head_sum` accumulates
attention over tokens and is length-biased by construction, and gold sentences
happen to be 1.18× the mean length, so a length prior was a live explanation. It
did not survive.

The other control is in the fusion table itself: fusing dense with the
*cross-encoder* — a stronger signal, but a correlated one (ρ = +0.43) — adds
almost nothing over the cross-encoder alone (full support 0.703 vs 0.697).
Decorrelation is doing the work, not fusion.

### The cascade, and why the mechanism needed one at all

There is a circularity at the heart of this idea and I want to state it before an
interviewer does: **the head signal needs a forward pass over the long context to
exist, which is the cost compression is supposed to avoid.** Measured, the
full-context pass costs 1497 ms, which makes the fused method *slower than naive
long-context* (1986 ms vs 1381 ms). As an efficiency result that is a
non-starter.

The cascade fixes it: dense scores everything cheaply, keep the top 1024 tokens,
and run the head pass over the shortlist only. Prefill dominates, so the pass
falls to 105 ms — 14× cheaper — and **loses nothing measurable in accuracy**
(cascade 0.661 vs full-context 0.670, p = 0.53).

The cost of the cascade is a ceiling: the head signal can only rerank what dense
already retrieved, so the shortlist's recall bounds the whole method. That is why
the shortlist starts at 1024 tokens rather than lower — M2 established that
dense reaches 0.957 full support by 2048 tokens, so a shortlist in that range
gives up very little.

### The single control I would point to if I could only show one

`cascade_S1024` versus `cascade_S1024_denseonly`. Same shortlist, same budget,
same packer, same generator — **the only difference is whether the head signal
rescored it.** The dense-only arm reproduces plain dense exactly (F1 0.608 vs
0.607), and the rescored arm is +0.053 above it (p = 0.0002). There is no room
left for the gain to have come from anything but the head signal.

### Likely interview questions

**"Your method needs a forward pass through the model to decide what to feed the
model. Isn't that circular?"**
It is, in the full-context form — measured at 1497 ms, which is slower than just
stuffing the window, so I do not claim it as an efficiency result. The cascade is
the answer: dense narrows 32k to 1024 tokens, the head pass runs on that, and it
costs 105 ms. I also tested whether the signal is circular in a second, worse
sense — whether it only finds evidence on questions the model already answers
correctly. It does not: every external retriever drops on the questions the model
gets wrong (dense −0.145 R@5), and the head signal is flat (+0.027).

**"Did you compare against a cross-encoder reranker, or only a bi-encoder?"**
Cross-encoder, bge-reranker-v2-m3 — same family as the bi-encoder, so the
comparison is about architecture rather than training data. It beats dense on
retrieval metrics (R@5 0.836 vs 0.770). But it does **not** beat dense
significantly on F1 (+0.034, p = 0.09), and my fusion is not significantly better
than it either (+0.029, p = 0.14). I will not claim I beat the cross-encoder.

**"Your head signal has worse evidence recall but better F1. Isn't your metric
broken?"**
No — the yardstick is wrong for that method. Attention is read at the step
generating the first answer token, so the signal points at what is about to be
copied. It finds the *answer string* better than dense (0.863 vs 0.796) while
finding the annotated *supporting facts* worse (0.437 vs 0.610 full support).
HotpotQA's supporting-fact annotation is not the only sufficient evidence, so a
method optimising answer-location scores badly on evidence-recall and well on F1.

**"Why is the fusion weight 0.20? Did you tune it on your test set?"**
Not any more. The first version picked 0.25 from two candidates on the reported
data, which is worth nothing. The reported number comes from sweeping seven
weights on a dev half by full support, freezing the winner, and reporting the
other half. The dev curve is flat (0.687–0.740 across 0.10–0.50), so the result
does not balance on the weight.

**"Is +0.054 F1 real, or noise on 300 questions?"**
Real: paired bootstrap, 10k resamples, CI [+0.028, +0.085], p < 0.001. Pairing
matters a lot here — my dev and test halves differed by ~0.09 F1 *on the same
method*, which is larger than any gap between methods, so an unpaired comparison
at n=150 would have been hopeless. Every method answers identical questions, so
pairing removes question difficulty from the comparison entirely.

**"What doesn't work / what would you do next?"**
Three things. The head list was identified on needle-in-a-haystack over essays,
not multi-hop QA, and the standalone failure is direct evidence that transfer is
partial — cross-task generalization is the open Definition-of-Done item. The
signal is read at a single decode step, so it structurally cannot find a bridge
fact the model is not currently copying. And I have swept the number of heads
(k=5 best) but not the head-*selection* method, so "copy-precision is the right
way to pick heads" is still an assumption rather than a measured result.

### The debugging story worth telling

The first cascade latency run reported S=2048 as *faster* than S=1024. That is
physically impossible for a prefill, and it is the same warm-up contamination
this project hit in Session 3 — the first block pays for cold kernels. Fixed the
same way: shuffle the job order so warm-up spreads across conditions, discard the
first ten calls. It came out monotonic (105 / 126 / 211 ms).

The related honesty catch: M2's latency table measures **generation only**, so
its headline "15× faster than naive long-context" quietly assumes retrieval is
free. Encoding this benchmark's per-question haystack costs 397 ms. Whether that
counts depends on the setting — this benchmark gives every question its own
haystack, but a deployed system indexes the corpus once — so both regimes are
now reported rather than the flattering one.
