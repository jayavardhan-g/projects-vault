---
type: interview-guide
day: 9
date: 2026-11-16
tags: [interview, cs728, retrieval-heads, lenet, pruning, robustness, quantisation, mock]
---
# Day 9, interview edition: CS728 tool retrieval and the LeNet-5 compression study (from zero, then defend it)

Part of [[Interview plan (Nov 2026)]]. Built from [[Day 09 - CS728 and LeNet-5]] (the older study note, kept) plus the sources below. Everything you need is on this page: every paper is summarised here, every piece of code is explained line by line, and every animation has a step-through player. You should not need another website.

**Sources on the server (all read for this note):**

| what | where | state |
|---|---|---|
| LeNet-5 code | `/data/jayavardhan/projects/interview_materials/lenet/` = GitHub `harshith2302/cs725_Project`, branch `main`, commit `3b2584b` | evaluation suite only; the pruning, distillation and quantisation code is on an **unpushed** branch `feat/compression-robustness` |
| LeNet-5 report | same folder, `fml_report.pdf` (12 pages, Aug 2026) | every LeNet number on your resume is in it |
| LeNet-5 Part 1 report | same folder, `legacy/docs/report.pdf` | the 5-person CS725 reproduction |
| CS728 code and report | **not on the server** | the numbers on your resume can't be checked yet |
| Retrieval-head code | `/data/jayavardhan/projects/LongContext_RAG/` (your repo, `main`, `bfb2ec9`) | the same machinery as CS728: the copy rule, head ranking, attention mass per span |

> [!important] What this note is for
> In an interview you will be asked to explain each project from scratch, then pushed: *why heads and not an embedding model, what was the baseline, why global pruning, is the robustness gain real, why 3.27× and not 4×.* This note builds every idea from zero, on tiny numbers, then shows the real code, then gives the questions with answers. **Start with §0**: it lists the resume lines that need new wording, and one authorship question you must settle before any interview.

**How to use it (about 6 hours):**

| pass | what | time |
|---|---|---|
| 1 | §0 (what to fix), then §7 and §17 (say it out loud) | 30 min |
| 2 | Part A, §1–§6: CS728, one idea at a time: idea → animation → code → question bank | 1.5 h |
| 3 | Part B, §8–§16: LeNet-5, same pattern | 2.5 h |
| 4 | §18 mock interview: cover the answer, say yours out loud, then open it | 1.5 h |

**Every claim carries a label.** Use the same discipline when you speak.

| label | means | example |
|---|---|---|
| **[IMPL]** | in code on the server (path:line given) | `pgd()` clamps the change to ±ε at `attacks.py:75` |
| **[REPORT]** | a number or statement in `fml_report.pdf` | 98.52 ± 0.14% clean at 3.75% of weights |
| **[RAN]** | checked today by running or recomputing it | the p = 0.83 recomputed from the report's own means and standard deviations |
| **[THEORY]** | what the maths or a hand count predicts | a linear score can drop by at most ε‖w‖₁ |
| **[RECON]** | reconstructed from the code or the docs; not documented | the per-head hit counter of CS728 |
| **[MISSING]** | not on the server; get it before the interview | the definition of the 1.3% baseline |
| **[FUTURE]** | not done; a possible next step | a per-position Recall@1 curve |

**Animations.** Each animated figure has a step-through player: click **▶ Step through it** under it (← / → step, space plays). New for this note: `d09i-*` (from `Days/figures/_gen/d09figs.py`) and four Manim videos `d09m-*` (from `Days/figures/_gen/manim_d09.py`); the older `d09-*` figures are reused.

---

## 0. Read this first: what to fix before you defend anything

### 0.1 Authorship of the LeNet-5 extension

> [!warning] Settle this before any interview
> - **Part 1** (reproducing LeNet-5) was a **5-person CS725 team project**: Ajay Rangoji, Harshith Matta, **G Jayavardhan**, Vineeth Babu Ch and Pujith Sai Kumar K (`legacy/docs/report.pdf`, `legacy/docs/project_plan.pdf`). Its code is `github.com/pujith22/LeNet-5-on-MNIST`.
> - **Part 2** (compression vs robustness, which is every number on your resume) is in `harshith2302/cs725_Project`. All 7 commits are by `harshith2302`, and `fml_report.pdf` lists **Harshith Matta (25M0834)** as its only author.
> - Your resume presents Part 2 as yours. An interviewer who opens the GitHub link will see someone else's name.
>
> **Decide what your role was and say exactly that.** Honest options, depending on what is true: "a team project in CS725; I worked on X", or "I extended our team's reproduction; the repo is under my teammate's account because …". If Part 2 is not your work, drop or reword these bullets. This note can't decide it for you; it only teaches the content so you can explain whatever part you own.

### 0.2 Your resume lines, checked against the evidence

| resume says | what the evidence says | say instead |
|---|---|---|
| CS728: Recall@1 **1.3% → 40.7%**, Recall@5 **25.5% → 66.9%** | **[MISSING]** no CS728 code or report on the server. The baseline that gives 1.3% is not defined anywhere you can show. | Fine to say, **if** you can name the baseline in one sentence (see §6.2). |
| "aggregating retrieval-specialized heads across **32 heads and 3 layers**" | **[MISSING]** reads oddly next to "top 10 heads". 32 heads in each of 3 layers? 32 heads in total? | Fix from your report. Never say a number you can't explain. |
| "Mitigated the **lost-in-the-middle** effect" | **[MISSING]** needs accuracy split by the correct tool's position. | If you only have overall recall: "aimed to reduce position bias", not "mitigated". |
| LeNet: **3.75%** of weights, **98.52%** clean, **69.83%** MNIST-C, validation split, **5 seeds** | ✓ **[REPORT]** Table 2, row "Standard, 3.75% weights" (layer-wise pruning, standard training). | As written. |
| "**7× larger** corruption loss … **3.89 pp** … traced to a starved conv1" | ✓ 7× = 3.89 (MNIST-C drop) ÷ 0.56 (clean drop) **[REPORT]** Table 5. conv1 keeps 6 of 150 weights **[REPORT]** Table 4, **[RAN]**. | Say what the 7× compares: "the corruption drop was 7× the clean drop". |
| "global pruning at matched sparsity closed the gap to **0.13 pp (p = 0.83)**, a **3.76 pp** gain" | ✓ **[REPORT]** Table 5; both p-values **[RAN]** (§13). "Matched" = 3.52% global vs 3.75% layer-wise (they are 15 rounds of two schedules, §11.3). | Add "Welch's t-test, 5 seeds" if asked. |
| "Improved PGD robustness from **51.98% to 82.18%** through iterative magnitude pruning" | Numbers ✓ **[REPORT]** Table 3, **but** at **ε = 0.05** (not the 0.3 used everywhere else), for the **13.8%**-weights model (not the 3.75% one). Both models fall to ~0% by ε = 0.2. The report says it "is not a defence". | "At a small budget (ε = 0.05), the pruned model resisted PGD far better (52% → 82%); at the standard ε = 0.3 both are at 0%." |
| "reducing model size **3.28×** (245.6 KB to 75.0 KB) via **INT8** quantization, at a cost of just **0.17 pp**" | 245.6 / 75.0 = **3.27** (the report rounds to 3.28). It is **QAT** (quantisation-aware training), not post-training. Latency got **1.07× worse** (p = 0.0005). The INT8 accuracy drop is listed as **not significant** (§5 of the report). Table 2 shows IMP + KD + INT8 at 98.65% vs IMP + KD at 98.64%, so know which two rows the 0.17 compares. | "3.3× smaller on disk via INT8 QAT, with no significant accuracy change, but slightly slower on CPU at batch 1." |

### 0.3 Two corrections to the older Day 9 note

- **The lottery-ticket paper (2019) resets survivors to their initial values (step 0).** "Rewinding" to an early training step came later (Frankle et al. 2020). This project ran **both**, plus plain fine-tuning: the two rewinds were indistinguishable; fine-tuning without reset was 6.73 points worse at 1.30% of weights, but not significant at 5 seeds (p = 0.062 raw, 0.174 after Holm) **[REPORT]** §3.2.1.
- **The p = 0.83 comes from Welch's t-test** (two independent groups of 5 seeds), not a paired test **[REPORT]** §5, **[RAN]**.

---

# Part A. CS728: zero-shot tool retrieval with retrieval heads

## 1. The project on one page

### 1.1 The problem, in plain words

An LLM **agent** can call tools: a weather API, a flight search, a calculator. The prompt lists each tool's name and description. For each user request, something has to pick the right tool.

The usual answer is a **retriever**: an embedding model, trained on query–tool pairs, that turns each description and each query into a vector and picks the closest. That needs training data and a second model.

Your project asks: **does the LLM already know where to look?** Inside the model, a few attention heads specialise in finding and copying information from the prompt. If we can find those heads, their attention over the tool list *is* a retriever, with no training at all ("zero-shot").

> [!quote] Say this
> "In CS728 I used the LLM's own attention as a tool retriever. A few heads specialise in pointing at the part of the prompt the model is about to use. I found them with a simple count on probe queries, kept the top 10, and ranked tools by how much attention those heads put on each tool's description. With no training, Recall@1 went from 1.3% to 40.7%."

### 1.2 The ten questions about the whole project

| question | answer | label |
|---|---|---|
| What problem? | Pick the right tool for a query from a list of tool descriptions in the prompt. | |
| Why not just ask the LLM? | It can be done, but the model may write a tool name that doesn't exist, and "lost in the middle" makes it miss tools in the middle of a long list. | [THEORY] |
| What is a retrieval head? | A head whose strongest attention lands on exactly the context token the model is about to copy (Wu et al. 2024). Fewer than 5% of heads. | §3 |
| How did you find them? | **Hit Count**: run probe queries whose correct tool is known; for each head, count how often its argmax lands on the correct tool; keep the top 10. | §4, [RECON] |
| How do they become a retriever? | Sum the chosen heads' attention over each tool's tokens; rank tools by that score. | §5 |
| Metric? | Recall@1 and Recall@5: share of queries whose correct tool is first / in the top 5. | §6 |
| Baseline? | 1.3% Recall@1. **[MISSING]**: you must name it (§6.2). | |
| Result? | Recall@1 1.3% → 40.7%, Recall@5 25.5% → 66.9%. | resume; [MISSING] |
| Trade-offs? | No training, no second model; but it needs the model's attention weights (slow "eager" path), and 40.7% is below a trained retriever's usual top-1. | §5.4, §6 |
| What would you do next? | Use heads to **re-rank** a shortlist from an embedding retriever (your LongContext_RAG "cascade" does this for passages); measure recall per tool position. | [FUTURE] |

### 1.3 Facts to fill in before the interview

The CS728 code is not on the server. Before any interview, find these in your report or code and write them in [[#My notes and fumbles]]:

- [ ] which LLM (and so how many layers × heads);
- [ ] which dataset or tool set, how many tools per prompt, how many queries;
- [ ] **what the 1.3% baseline is** (§6.2 lists the likely candidates);
- [ ] the probe set for Hit Count: how many queries, and whether they are separate from the test queries;
- [ ] what "32 heads and 3 layers" means;
- [ ] how lost in the middle was measured, if it was.

Until then, the code in this Part is **your LongContext_RAG repo**, which does the same three steps on passages instead of tools: the copy rule (`heads/scorer.py`), ranking heads (`heads/run_head_id.py`) and attention mass per span (`compress/head_scorer.py`). Wherever CS728 needs a step that repo doesn't have, the code is marked **[RECON]**.

---

## 2. From zero: tokens, attention and heads

### 2.1 Tokens and positions

The model never sees letters. A **tokenizer** cuts the text into pieces (**tokens**) and gives each piece an integer id. Each token has a **position**: 0, 1, 2, … in the prompt. In our toy prompt:

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| token | `<s>` | weather | api | flight | search | stock | price | calc | math | fly | Paris |
| part | start | tool A | tool A | tool B | tool B | tool C | tool C | tool D | tool D | query | query |

`<s>` is a special start-of-text token. Each tool's description is a **span**: tool B is positions 3–4. Real descriptions are dozens of tokens; here they are two.

### 2.2 Attention: one row of weights

Each layer of a Transformer lets every token **look at** earlier tokens and mix in what it finds. For the token being generated, one attention head does three things:

1. **Score** every earlier position: score = (this token's *query* vector · that position's *key* vector) / √d. (Query and key are two learned projections of each token's vector; d is their length. Day 1 §5 builds these from scratch.)
2. **Softmax** the scores: e^score for each, divided by the total, so the weights are positive and sum to 1.
3. **Mix**: the output is the weighted average of the positions' *value* vectors.

The row of weights from step 2 says where the head is looking. Its largest entry, the **argmax**, is the one position the head looks at most.

![[d09i-attn-row.svg]] [[d09i-attn-row.html|▶ Step through it]]

On the toy: the score 3.0 at position 3 becomes e³ = 20.09; the total is 48.62; so the weight is 20.09 / 48.62 = **0.413**, the largest. This head is looking at "flight", inside tool B.

### 2.3 A head, a layer, and how many there are

- A **head** is one independent copy of steps 1–3, with its own projections, so it learns its own way of scoring. Each produces its own row.
- A **layer** runs several heads side by side, then a small MLP.
- Llama-3.1-8B (the model in LongContext_RAG) has **32 layers × 32 heads = 1,024 heads**, so 1,024 rows per generated token.

We name a head by its layer and number: **L14H3** is head 3 of layer 14.

### 2.4 The attention sink

Many heads put a lot of weight on position 0 (`<s>`) whatever the content, as a place to "park" attention they don't need. It is called the **attention sink**. In the toy row `<s>` gets 0.186. Any retrieval score must ignore it, or a head that always parks on `<s>` would look busy. §5's code removes it explicitly.

### 2.5 Prefill and decode

- **Prefill:** the model reads the whole prompt in one pass (all positions at once).
- **Decode:** it then generates one token at a time; each new token attends over everything before it.

The attention row that matters for retrieval is the one at a **decode** step, the moment the model starts its answer. §5.4 explains why the code reads attention only there.

### 2.6 Check yourself

1. Scores [1.0, 0.0, 2.0]: softmax weights? Argmax? *(e¹ = 2.72, e⁰ = 1, e² = 7.39; total 11.11 → 0.245, 0.090, 0.665; argmax 2.)*
2. Why can't you read "which tool" from a single head's output vector? *(The output is a blend of value vectors; the **weights** say where it looked.)*
3. How many attention rows does Llama-3.1-8B make for one generated token? *(1,024.)*

---

## 3. Retrieval heads (Wu et al. 2024), and your code for the copy rule

### 3.1 What the paper says (you don't need to open it)

> [!abstract] Wu et al., *Retrieval Head Mechanistically Explains Long-Context Factuality* (2024), §1–3
> - **Question:** when a long-context model copies a fact from far back in its prompt, which part of the model finds it?
> - **Test:** needle-in-a-haystack. Hide a sentence (the **needle**, e.g. "The secret code is 4721") in a long text and ask for it.
> - **Retrieval score:** each time the model emits a needle token, check every head: does its argmax point at that same token in the needle? A head's score = the share of needle tokens it was "caught copying".
> - **Findings:** under 5% of heads score high, so retrieval is **sparse**. These heads exist in every model family tested, are already there in base models, and survive long-context training.
> - **Causal test:** masking the retrieval heads makes needle tests fail and the model invent facts; masking the same number of random heads barely matters.
> - **For CS728:** Hit Count is the same idea, scored on whether the head's argmax lands on the **correct tool** instead of on a copied token.

### 3.2 The copy rule on tiny numbers

A head **hits** at a decode step if **both** hold:
1. its argmax position is **inside the needle**, and
2. the token at that position **is the token being emitted**.

Both halves matter. Inside the needle but a different token: the head is attending, not copying. The right token but argmax elsewhere: the model got the token some other way.

![[d09-1-copy-hit.svg]] [[d09-1-copy-hit.html|▶ Step through it]]

### 3.3 Reading the code: every Python and PyTorch construct in this Part

| construct | meaning | tiny example |
|---|---|---|
| tensor shape `[layers, heads, keys]` | a 3-D array; `attn[l, h, j]` = weight head h of layer l puts on position j | `[2, 2, 10]` = 2 layers, 2 heads, 10 positions |
| `t.argmax(dim=-1)` | index of the largest value along the last axis; removes that axis | `[[0.1, 0.7, 0.2]] → [1]` |
| `a >= b`, `a & b` | element-wise comparison gives True/False; `&` is element-wise AND | `[T, F] & [T, T] → [T, F]` |
| `x[idx]` with a tensor `idx` | **gather**: look up x at each position listed in idx; result has idx's shape | `x = [11, 47, 21]`, `idx = [[2, 1]] → [[21, 47]]` |
| `t.clamp(min, max)` | clip every element into a range | `[-4, 2, 12].clamp(0, 9) → [0, 2, 9]` |
| `t.to(torch.int32)` | convert type; True/False → 1/0 | `[T, F] → [1, 0]` |
| `zeros.scatter_(dim, index, src)` | **write** `src` at the positions given by `index` along `dim` (the trailing `_` means in place) | write True at slot 2 of a 4-slot row |
| `a \|= b` | in-place OR: a position stays True once it has been True | `[F, T] \|= [T, F] → [T, T]` |
| `t.sum(-1)` | add along the last axis | `[[1, 0, 1]] → [2]` |
| `@torch.no_grad()` | decorator: don't record operations for gradients (saves memory; we only read) | |
| `with ctx:` | run the block inside a context manager that sets something up and always undoes it | `with capture.recording():` turns hooks on, then off |
| `set(xs)`, `x in s` | a set; membership test in O(1) | `47 in {47, 21} → True` |

### 3.4 The code: `copy_hits` (`LongContext_RAG/heads/scorer.py:121–143`) **[IMPL]**

```python
121  def copy_hits(attn, prompt_ids, generated_token, needle_start, needle_end):   # ◀ signature shortened; returns (argmax, hit mask)
     ⋮  # 128–137 left out: the docstring (says exactly the two-part rule of §3.2)
138      j = attn.argmax(dim=-1)                                   # ◀ [layers, heads, keys] → [layers, heads]: where each head looks most
139      in_span = (j >= needle_start) & (j < needle_end)          # ◀ rule part 1: is that position inside the needle?
     ⋮  # 140–141 comment: positions past the prompt are generated tokens, never in the needle
142      tok_at_j = prompt_ids[j.clamp(max=prompt_ids.numel() - 1)]   # ◀ gather: the token id sitting at each head's argmax
143      return j, in_span & (tok_at_j == generated_token)         # ◀ rule part 2, AND part 1: a hit per head
```

| name | type / shape | holds | toy value (first step) |
|---|---|---|---|
| `attn` | float `[layers, heads, keys]` | every head's attention row at this decode step | `[2, 2, 10]` |
| `prompt_ids` | int `[T]` | the prompt's token ids | `[11 12 13 14 15 16 47 21 17 18]` |
| `generated_token` | int | the token the model just emitted | 47 |
| `needle_start`, `needle_end` | int | needle = positions `start … end − 1` | 4, 8 |
| `j` | int `[layers, heads]` | argmax position per head | `[[0, 6], [7, 6]]` |
| `in_span` | bool `[layers, heads]` | argmax inside the needle? | `[[F, T], [T, T]]` |
| `tok_at_j` | int `[layers, heads]` | token at each argmax | `[[11, 47], [21, 47]]` |
| returned hit mask | bool `[layers, heads]` | caught copying? | `[[F, T], [F, T]]` |

**Why the `clamp` on line 142?** During decoding the keys include tokens the model has just generated, so `j` can be ≥ `len(prompt_ids)`. Indexing there would crash. Clamping keeps the lookup in range; such positions are outside the needle anyway, so `in_span` is already False and the result is unchanged.

**What breaks if you change it:**

| change | result |
|---|---|
| drop `in_span &` (keep only the token match) | a head looking at an **earlier copy** of "47" outside the needle counts as a hit |
| drop the token match (keep only `in_span`) | a head that stares at the needle while the model writes something else counts as a hit |
| drop the `clamp` | an IndexError the first time any head looks at a generated token |
| use total attention on the needle instead of argmax | diffuse heads that spread a little everywhere score well; argmax tests the sharp claim "this head points at the copied token" |

### 3.5 The caller: one decode step at a time (`scorer.py:56–106`) **[IMPL]**

```python
 56  @torch.no_grad()                                                       # ◀ reading only, no gradients
 57  def score_instance(self, inst: NIAHInstance) -> InstanceResult:        # ◀ one haystack-with-needle probe
 58      ids = torch.tensor([inst.input_ids], device=self.device)           # ◀ the prompt, shape [1, T]
 59      prompt_ids = ids[0]                                                # ◀ the same ids, shape [T]
 61      ns, ne, klen = inst.needle_start, inst.needle_end, inst.needle_len # ◀ needle span and its length
     ⋮  # 62–72 left out: layer and head counts, an empty KV cache, remember the attention mode
 74      self.model(input_ids=ids[:, :-1], past_key_values=cache, use_cache=True)   # ◀ prefill all but the last token (fast path)
     ⋮  # 75–76: restore the attention mode
 77      covered = torch.zeros((n_layers, n_heads, klen), dtype=torch.bool, ...)    # ◀ which needle positions each head has copied
 78      hit_steps = torch.zeros((n_layers, n_heads), dtype=torch.int32, ...)       # ◀ hit counter per head
 80      needle_tokens = set(prompt_ids[ns:ne].tolist())                    # ◀ the needle's token ids
 83      cur = ids[:, -1:]                                                  # ◀ last prompt token: the first one fed in the loop
 85      for _ in range(self.max_new_tokens):                               # ◀ generate up to 32 tokens
 86          with self.capture.recording():                                 # ◀ eager attention + hooks for THIS step only
 87              out = self.model(input_ids=cur, past_key_values=cache, use_cache=True)
 88          attn = self.capture.stack()                                    # ◀ [layers, heads, keys]
 89          nxt = int(out.logits[0, -1].argmax())                          # ◀ greedy: the emitted token
 90          if nxt in self.eos_ids:                                        # ◀ end of answer: stop
 91              break
     ⋮  # 92: remember the token for the decoded text
 93          if nxt in needle_tokens:                                       # ◀ a copy OPPORTUNITY (denominator of copy-precision)
 94              copyable_steps += 1
 96          j, hit = copy_hits(attn, prompt_ids, nxt, ns, ne)              # ◀ §3.4
 97          hit_steps += hit.to(torch.int32)                               # ◀ count hits per head
 98          rel = (j - ns).clamp(0, klen - 1)                              # ◀ argmax as a needle-relative index 0 … klen−1
 99          covered |= torch.zeros_like(covered).scatter_(                 # ◀ mark the copied needle position as True …
100              2, rel.unsqueeze(-1), hit.unsqueeze(-1)                    #   … only where hit is True
101          )
103          cur = torch.tensor([[nxt]], device=self.device)                # ◀ feed the emitted token next
     ⋮  # 105: decode the answer text
106      scores = (covered.sum(-1).float() / klen).cpu().tolist()           # ◀ score = copied needle positions ÷ needle length
```

| name | holds | toy value at the end |
|---|---|---|
| `cache` | the KV cache: keys and values of every token so far, so each decode step reads them instead of recomputing (Day 1 §2) | |
| `cur` | the single token fed at this step, shape `[1, 1]` | |
| `copyable_steps` | decode steps whose emitted token is in the needle | 2 |
| `hit_steps` | per head, steps where it hit | `[0, 2, 1, 1]` |
| `covered` | per head, which needle positions it ever copied | L0H1 {6, 7}, L1H0 {7}, L1H1 {6} |
| `scores` | per head, covered ÷ `klen` | `[0, 0.5, 0.25, 0.25]` |

![[d09i-copy-dbg.svg]] [[d09i-copy-dbg.html|▶ Step through it]]

*Animated, 14 steps. The code is on the left with the running line highlighted; the variables are on the right, and the ones that just changed are red.*

**`.unsqueeze(-1)` on line 100:** `scatter_` along axis 2 needs an index with the same number of axes as `covered` (3). `rel` is `[layers, heads]`; `.unsqueeze(-1)` makes it `[layers, heads, 1]`: "one slot to write per head".

### 3.6 From one probe to a ranking (`heads/run_head_id.py:121–142`) **[IMPL]**

```python
121  def _aggregate(results, n_layers, n_heads, threshold):
122      total = [[0.0] * n_heads for _ in range(n_layers)]                # ◀ sum of per-probe scores, per head
123      hits = [[0] * n_heads for _ in range(n_layers)]                   # ◀ sum of hit steps, per head
124      for r in results:                                                 # ◀ every probe
     ⋮  # 125–128: add r.scores and r.hit_steps into total and hits
130      mean = [[total[l][h] / n for h in range(n_heads)] for l in range(n_layers)]   # ◀ Wu et al.'s score, averaged
138      copyable = sum(r.copyable_steps for r in results)                 # ◀ all copy opportunities, pooled
139      precision = [                                                     # ◀ copy-precision: hits ÷ opportunities
140          [(hits[l][h] / copyable if copyable else 0.0) for h in range(n_heads)]
141          for l in range(n_layers)
142      ]
```

Two scores, and why:
- **Coverage** (`mean`, Wu et al.'s |g_h| / |k|) is capped by how much of the needle the answer copies: if the answer repeats 2 of 4 needle tokens, no head can pass 0.5.
- **Copy-precision** divides by the steps where copying was possible, so short answers aren't penalised. It is **micro-averaged** (steps pooled over all probes before dividing), so one probe with a single copy step can't swing a head's score.

The list comprehension `[[… for h in …] for l in …]` builds a list of lists: one row per layer, one entry per head.

### 3.7 Question bank: attention and the copy rule

**L1. What is an attention head, in one sentence?**
> [!success]- Answer
> One independent attention pattern inside a layer: for the current token it scores every earlier position, softmaxes the scores into weights that sum to 1, and mixes the earlier positions' value vectors with those weights. Each head has its own learned projections, so each learns its own way of choosing where to look.

**L1. What is a retrieval head?**
> [!success]- Answer
> A head that, while the model writes something it read in the prompt, puts its **largest** attention weight on exactly that token in the prompt. Wu et al. found fewer than 5% of heads behave like this, in every model family they tested, and masking them breaks long-context fact lookup while masking random heads doesn't.

**L2. State the copy rule, and say why both halves are needed.**
> [!success]- Answer
> A head hits at a decode step if its argmax position is inside the needle **and** the token there equals the token being emitted. Without the first half, a head looking at an earlier copy of the same word elsewhere would count. Without the second, a head staring at the needle while the model writes something unrelated would count. Together they mean "this head was pointing at the thing being copied, at the moment it was copied".

**L2. Why argmax and not the total attention mass on the needle?**
> [!success]- Answer
> Argmax tests a sharp, falsifiable claim: this head points at the copied token. Total mass rewards diffuse heads that spread a little attention everywhere, including over the needle. For **ranking tools** (§5) I do use mass, because there I need a graded score for every tool, not a yes/no.

**L2. Why does `copy_hits` clamp `j`?**
> [!success]- Answer
> At decode time the keys include tokens generated so far, beyond the end of the prompt, so `j` can exceed the last prompt index and the gather `prompt_ids[j]` would go out of range. Those positions can't be in the needle, so `in_span` is already False there; clamping only keeps the lookup legal and never changes a result.

**L3. "Your coverage score can't exceed 0.5 on some probes. Is that a bug?"**
> [!success]- Answer
> No, it's a property of the score: coverage = needle positions copied ÷ needle length. If the answer repeats only 2 of the needle's 4 tokens, even a perfect head copies 2 of 4. That's why the code also computes copy-precision, hits ÷ steps where copying was possible, micro-averaged over all probes; it ranks heads without that ceiling.

**L3. "Couldn't a head score high just because it attends to the most recent tokens?"**
> [!success]- Answer
> Not under this rule: a hit needs the argmax **inside the needle**, which sits at varying depths in the haystack, and the token there must match the one emitted. A recency head points at the last few tokens, which are the question, not the needle. Varying the needle's depth across probes is exactly what separates content-driven heads from positional ones.

---

## 4. Hit Count: choosing the heads

### 4.1 The idea

The copy rule needs a needle and a copied token. Tool selection has neither: the model doesn't copy the tool's description. So CS728 uses a **looser hit**:

> A head **hits** on a probe query if its argmax lands **anywhere inside the correct tool's span**.

Then:
1. run a set of **probe queries** whose correct tool is known;
2. for each head, count its hits (**Hit Count**);
3. keep the **top k** heads (k = 10 in your project).

No gradients, no training: a few hundred forward passes.

![[d09i-hitcount.svg]] [[d09i-hitcount.html|▶ Step through it]]

On the toy: L1H0 hits on all 5 probes, L1H2 on 4, L0H2 on 2. L0H0 never hits, because it always parks on `<s>`. With k = 2 we keep L1H0 and L1H2.

### 4.2 The code (reconstructed: your CS728 code is not on the server) **[RECON]**

```python
hits = torch.zeros(n_layers, n_heads)                   # ◀ one counter per head
for query, gold in probes:                              # ◀ probe set: (query, index of its correct tool)
    attn, spans = attention_at_answer(query, tools)     # ◀ [layers, heads, T] at the first answer token, as in §5.3
    j = attn.argmax(dim=-1)                             # ◀ where each head looks most: [layers, heads]
    s, e = spans[gold]                                  # ◀ the correct tool's token range
    hits += ((j >= s) & (j < e)).float()                # ◀ +1 for every head whose argmax is inside it
top = hits.flatten().topk(10).indices                   # ◀ the 10 largest counts, as flat indices 0 … L·H − 1
layers, heads = top // n_heads, top % n_heads           # ◀ flat index → (layer, head)
```

| name | holds | toy value |
|---|---|---|
| `hits` | per-head hit count | L0H0 0, L0H1 1, L0H2 2, L1H0 5, L1H1 1, L1H2 4 |
| `probes` | (query, correct tool) pairs | 5 pairs |
| `spans[gold]` | correct tool's token range | B → (3, 5) |
| `hits.flatten()` | the `[2, 3]` grid as a 6-long list | `[0, 1, 2, 5, 1, 4]` |
| `.topk(k).indices` | positions of the k largest values | k = 2 → `[3, 5]` |
| `top // n_heads`, `top % n_heads` | integer division and remainder | 3 → (1, 0) = L1H0; 5 → (1, 2) = L1H2 |

> [!note] What to check against your own code
> Was the hit "argmax inside the span" (as here) or "most attention mass on the right tool"? Were `<s>` and the query excluded first? Were the probe queries separate from the test queries? If they overlapped, the selection saw the test set, and an interviewer will ask.

### 4.3 Why k = 10, and what changes with k

- **Too few** (k = 1–2): one head's quirks dominate; if it fails on a query type, nothing compensates. (LongContext_RAG found that its single best head alone was "causally dispensable", `compress/head_scorer.py:21–22`.)
- **Too many** (k = 50+): you start adding heads that don't retrieve, and their spread-out attention blurs the ranking toward noise and position bias.
- **The right way to pick k** is on held-out queries: sweep k ∈ {1, 5, 10, 25} and choose by validation recall. LongContext_RAG computes every k in one pass for that reason (`head_scorer.py:59, 126–129`). Whether CS728 swept k: **[MISSING]**.

### 4.4 Question bank: Hit Count

**L1. What is Hit Count?**
> [!success]- Answer
> A head-selection rule: run probe queries with known correct tools, count for each head how many times its strongest attention lands inside the correct tool's description, and keep the heads with the highest counts (top 10 in my project). It needs only forward passes, no training.

**L2. Why is no training needed?**
> [!success]- Answer
> The model already learned, during pre-training, heads that look up information in context. Hit Count only **finds** them; it doesn't change any weight. The retriever is then a fixed read-out: sum the chosen heads' attention per tool.

**L2. How would you choose k?**
> [!success]- Answer
> On held-out queries, never on the test set: compute recall for several k (1, 5, 10, 25), and pick the smallest k near the best. Too few heads makes the ranking brittle; too many adds non-retrieval heads whose diffuse or positional attention blurs it.

**L3. "Your probe set and test set: could Hit Count have overfitted?"**
> [!success]- Answer
> If probes and test queries overlapped, yes: the selected heads would be tuned to those exact queries. The selection has few degrees of freedom (choosing 10 of ~1,000 heads), so the risk is modest, but the honest protocol is disjoint probe and test queries, ideally disjoint tools too. [Say what your project did; if they overlapped, say so and how you'd fix it.]

**L3. "Two heads always attend to the same token. Does keeping both help?"**
> [!success]- Answer
> Not much: they add the same information twice and push the score toward their shared preference. A greedy selection that skips heads highly correlated with ones already chosen would add diversity. I didn't do that [unless you did]; it's a reasonable next step.

---
## 5. Heads as a retriever: a score for every tool

### 5.1 The idea

Hit Count gave us k heads that land on the right tool. For a **new** query we don't know the right tool, so we can't check hits. Instead we give every tool a score:

1. Run the prompt (tools + new query) to the first answer token; read the chosen heads' attention rows.
2. **Drop** every position that isn't a tool token (the sink `<s>`, the query, the instructions).
3. **Renormalise** each head's row over the tool tokens, so it reads "share of this head's tool-directed attention".
4. **Add up** each tool's share: one number per (head, tool).
5. **Average** over the k heads, so each head counts equally.
6. **Rank** tools by that number. The top one is the prediction.

![[d09m-heads.mp4]]
*Manim video (37 s): attention rows → Hit Count over 5 probes → the chosen heads' mass per tool → Recall@k. Pause and scrub with the video controls.*

### 5.2 Reading the code: the new PyTorch pieces

| construct | meaning | tiny example |
|---|---|---|
| `attn[l_idx, h_idx]` with two index tensors | **fancy indexing**: pairs up the i-th layer index with the i-th head index | `l_idx = [1, 1]`, `h_idx = [0, 2]` → rows of L1H0 and L1H2: shape `[2, T]` |
| `torch.full((T,), -1)` | a tensor of length T filled with −1 | `[-1, -1, …]` |
| `seg[s:e] = i` | write `i` into positions s … e−1 | `seg[3:5] = 1` → positions 3, 4 belong to tool 1 |
| `x[bool_mask]`, `p[:, bool_mask]` | keep only the positions where the mask is True (`:` = every row) | drop the sink and the query |
| `p.sum(dim=1, keepdim=True)` | row sums, kept as a column `[K, 1]` so the division broadcasts over each row | |
| `.clamp_min(1e-12)` | never divide by exactly 0 | |
| `zeros(K, n).index_add_(1, idx, src)` | for each column j of `src`, add it into column `idx[j]` of the result: a "bucket sum" | 8 token shares → 4 tool totals |
| `mat[:k].mean(dim=0)` | average the first k rows (heads) | 2 heads → 1 score per tool |

### 5.3 The code: `HeadAttentionScorer.score_all` (`LongContext_RAG/compress/head_scorer.py:100–130`) **[IMPL]**

The same reduction CS728 needs, written for passages (sentences) instead of tools: a "chunk" here = a tool there.

```python
100  @torch.no_grad()
101  def score_all(self, question: str, chunks: list[Chunk]) -> dict[str, list[float]]:
102      attn, spans, _ = self.attention_for(question, chunks)      # ◀ [layers, heads, T] at the first answer token + each chunk's (start, end)
103      selected = attn[self._l_idx, self._h_idx].float()           # ◀ only the chosen heads: [K, T]
104      del attn                                                    # ◀ free the full [layers, heads, T] tensor
106      n_chunks = len(chunks)
107      seg = torch.full((selected.shape[1],), -1, dtype=torch.long, device=self.device)   # ◀ owner of each position; −1 = nobody
108      lengths = torch.ones(n_chunks, device=self.device)          # ◀ tokens per chunk (for the "mean" variant)
109      for i, (s, e) in enumerate(spans):
110          if e > s:
111              seg[s:e] = i                                        # ◀ positions s … e−1 belong to chunk i
112              lengths[i] = e - s
114      valid = seg >= 0                                            # ◀ True only on chunk tokens
115      seg_v = seg[valid]                                          # ◀ their owners, in order
116      p = selected[:, valid]                                      # ◀ attention on chunk tokens only: sink and question dropped
117      p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)         # ◀ renormalise each head's row to sum to 1
119      K = p.shape[0]
120      sums = torch.zeros(K, n_chunks, device=self.device).index_add_(1, seg_v, p)   # ◀ per head, per chunk: total share
     ⋮  # 121–123 left out: `maxes`, the same bucketing with max instead of sum
124      per_head = {"sum": sums, "mean": sums / lengths, "max": maxes}  # ◀ three ways to score a span
126      out: dict[str, list[float]] = {}
127      for agg, mat in per_head.items():
128          for k in self.k_values:                                 # ◀ k ∈ (1, 5, 10, 25), all from ONE forward pass
129              out[f"head_{agg}_k{k}"] = mat[:k].mean(dim=0).cpu().tolist()   # ◀ average the top-k heads
130      return out
```

| name | type / shape | holds | toy value |
|---|---|---|---|
| `self._l_idx`, `self._h_idx` | int `[K_max]` | layer and head of each chosen head, best first (set in `__init__`, `:72–73`) | `[1, 1]`, `[0, 2]` |
| `spans` | list of (start, end) | each chunk's (tool's) token range | A (1,3), B (3,5), C (5,7), D (7,9) |
| `selected` | float `[K, T]` | the chosen heads' rows | L1H0: .40 .03 .02 .25 .15 .02 .03 .02 .02 .03 .03 |
| `seg` | int `[T]` | chunk owning each position | −1 0 0 1 1 2 2 3 3 −1 −1 |
| `p` | float `[K, n_chunk_tokens]` | renormalised share per chunk token | L1H0: .056 .037 .463 .278 .037 .056 .037 .037 |
| `sums` | float `[K, n_chunks]` | per head, per chunk | L1H0: .093 .741 .093 .074 |
| `out["head_sum_k2"]` | list | per-chunk score | A .196, B .554, C .113, D .137 → B first |

![[d09i-score-dbg.svg]] [[d09i-score-dbg.html|▶ Step through it]]

**The three design choices, each measured rather than assumed** (`head_scorer.py:9–22` docstring):
- **Normalise over the spans only.** Raw attention is dominated by the sink and the question tokens, which can never be selected. In the toy, L1H0 puts 0.40 on `<s>`; renormalising over tools turns B's 0.40 of raw mass into 0.74 of its tool-directed mass.
- **Sum, mean or max over a span.** Sum rewards long descriptions (more tokens to collect mass on); mean divides that out but rewards a short span with one spike; max takes the single best token. Tool descriptions differ in length, so this choice matters for CS728. Which one CS728 used: **[MISSING]**.
- **Average heads equally.** A plain sum lets one diffuse head contribute most of the mass.

**What breaks if you change it:**

| change | result |
|---|---|
| skip line 116 (keep the sink and question) | every tool's share shrinks by the same sink-dominated denominator per head, but heads with a big sink weight count less, and the question tokens can't be "retrieved"; rankings get noisier |
| sum heads instead of averaging | a head with flat attention over 30 tools adds the same small amount everywhere, while a sharp head's vote is diluted |
| `mean` aggregation with very different description lengths | a 3-token tool with one spike beats a 40-token tool with steady attention |

### 5.4 Why attention is read only at decode steps (`heads/capture.py`) **[IMPL]**

Fast attention kernels (SDPA, FlashAttention) never store the weight matrix; they compute the output directly (Day 6). The weights only exist on the slow "eager" path, and an eager **prefill** over T tokens builds a `[heads, T, T]` matrix per layer:

$$32 \text{ heads} \times 8{,}192^2 \times 2 \text{ bytes} \approx 4.3 \text{ GB per layer}$$

A **decode** step has one query, so its matrix is `[heads, 1, T]`: 32 × 8,192 × 2 bytes ≈ 0.5 MB. So the code prefills with SDPA and switches to eager for the single-token steps only:

```python
 71  @contextmanager                                       # ◀ makes `with capture.recording():` possible
 72  def recording(self):
     ⋮  # 73: docstring
 74      cfg = self.model.config
 75      previous = cfg._attn_implementation                # ◀ remember the current mode (usually "sdpa")
 76      cfg._attn_implementation = "eager"                 # ◀ the slow path that returns weights
 77      self._enabled = True                               # ◀ hooks start saving
 78      self.buffer = [None] * self.n_layers
 79      try:
 80          yield self                                     # ◀ the body of the `with` block runs here
 81      finally:                                           # ◀ runs even if the body crashes
 82          self._enabled = False
 83          cfg._attn_implementation = previous            # ◀ always restore the fast path
     ⋮
 95      out = torch.cat([b[:, :, -1, :] for b in self.buffer], dim=0)   # ◀ stack(): each layer's [1, heads, 1, T] → [layers, heads, T]
```

The weights reach `self.buffer` through a **forward hook** (`:60–69`): a function PyTorch calls after each attention module runs, here copying out `output[1]` (the weights) with `.detach()` so no gradient graph is kept.

| construct | meaning |
|---|---|
| `@contextmanager` + `yield` | turns a function into a `with` block: code before `yield` runs on entry, the `finally` part on exit |
| `try: … finally:` | the `finally` part always runs, so a crash can't leave the model stuck in slow eager mode |
| `register_forward_hook(fn)` | PyTorch calls `fn(module, inputs, output)` every time that module runs |
| `b[:, :, -1, :]` | take the last query position (the only one, at a decode step) |

### 5.5 Question bank: heads as a retriever

**L1. How do you turn attention into a tool ranking?**
> [!success]- Answer
> At the first answer token, take the selected heads' attention rows, keep only positions inside tool descriptions, renormalise each head's row over those positions, sum each tool's share, average over heads, and rank tools by that score.

**L2. Why renormalise over the tool spans?**
> [!success]- Answer
> Raw attention is dominated by tokens that can't be answers: the attention sink at position 0 and the question itself. If a head puts 40% on the sink, its tool shares are all scaled down, and heads would count unequally for a reason that has nothing to do with the tools. Renormalising gives "share of this head's tool-directed attention", comparable across heads.

**L2. Sum, mean or max over a tool's tokens: which and why?**
> [!success]- Answer
> Sum rewards long descriptions; mean corrects for length but lets a short description with one spike win; max uses only the single best token. Because the right one isn't obvious, the LongContext_RAG scorer computes all three in one pass and picks on held-out data. [For CS728, say which you used and why.]

**L2. Why read attention only at the first answer token?**
> [!success]- Answer
> Two reasons. Semantically, that's when the model decides what to say, so its retrieval heads point at what it's about to use. Practically, attention weights are only visible on the slow eager path, and an eager prefill would build a [heads, T, T] matrix per layer (~4 GB per layer at 8k tokens), while a single decode step's matrix is [heads, 1, T], about half a megabyte. So: prefill with SDPA, then one eager step.

**L3. "What is the cost of this retriever at inference?"**
> [!success]- Answer
> One prefill of the whole tool list plus one eager decode step, which you'd pay anyway to answer. Compared with an embedding retriever, it needs no second model and no training, but the full prompt must fit in context and be prefilled each time (no precomputed tool index), and the eager step is slower than SDPA. With many tools, a cheap embedding shortlist first, then head re-ranking, is the better design.

---

## 6. Recall@k, the baseline, and "lost in the middle"

### 6.1 Recall@k from zero

For each test query, rank all tools; find the **rank** of the correct one (1 = first).

$$\text{Recall@}k = \frac{\#\{\text{queries whose correct tool has rank} \le k\}}{\#\{\text{queries}\}}$$

Six queries whose correct tools rank 1, 3, 1, 7, 2, 1:
- Recall@1 = 3 / 6 = **50%** (ranks of 1);
- Recall@5 = 5 / 6 = **83%** (all except the 7).

Recall@1 is plain top-1 accuracy. Recall@5 matters when a later step (the LLM itself, or a re-ranker) chooses among a shortlist of 5. With N tools, a random ranking gives Recall@1 = 1/N and Recall@5 = 5/N.

### 6.2 What was the 1.3% baseline? **[MISSING]**

A baseline of 1.3% Recall@1 is **very** low; that is the first thing an interviewer will notice. Possible readings:

| candidate baseline | what Recall@1 1.3% would mean | follow-up you'll get |
|---|---|---|
| all heads averaged ("plain attention") | averaging all heads drowns the few retrieval heads in sink and positional attention | "So the gain is from removing noise, not from finding something special?" |
| random ranking | 1/N = 1.3% means about 77 tools per prompt (and 5/N = 6.5% at Recall@5, so it doesn't match the 25.5%) | the 25.5% Recall@5 rules this out |
| the LLM asked to name the tool | the model rarely writes an exact tool name, so exact match is ~1% | "Did you allow fuzzy matching?" |
| a single random or worst head | | "Why compare with a strawman?" |

Recall@5 = 25.5% with Recall@1 = 1.3% means the baseline often ranks the right tool 2nd–5th but almost never 1st, typical of a score dominated by position (e.g. the first few tools always on top). **Check your report and write the one-sentence definition in [[#My notes and fumbles]].**

### 6.3 Lost in the middle (Liu et al. 2023), summarised

> [!abstract] Liu et al., *Lost in the Middle* (2023), §1–2
> - **Setup:** multi-document question answering. The model gets 10, 20 or 30 documents, one of which holds the answer, and the answer's position is moved from first to last.
> - **Result:** accuracy is high when the answer is first, fairly high when last, and lowest in the middle: a **U shape**.
> - **Striking:** with the answer in the middle, GPT-3.5-Turbo can do worse than with no documents at all.
> - **Also:** a longer context window doesn't make a model better at using the middle.
> - **Key-value retrieval** (a JSON list of random IDs) shows the same pattern, so it's about position, not about the question being hard.

![[d09i-lost-middle.svg]] [[d09i-lost-middle.html|▶ Step through it]]

**Why it happens [THEORY]:** a lot of attention is **positional**: the sink and early tokens get extra weight (primacy), and the last tokens sit next to the question (recency). A tool in the middle has neither.

**Why selected heads should help [RECON]:** heads chosen because they land on the **right** tool across probes whose correct tool sits at many positions are, by construction, the heads whose argmax follows content, not position. Summing their attention should flatten the U.

**How to show it [FUTURE / MISSING]:** report Recall@1 separately for each position bucket of the correct tool (first 10%, middle, last 10%), for the baseline and for the heads. A flatter curve for the heads is the evidence for the resume's "mitigated".

### 6.4 Question bank: metrics and position

**L1. Define Recall@1 and Recall@5.**
> [!success]- Answer
> The fraction of test queries whose correct tool is ranked first, or among the first five. Recall@1 is top-1 accuracy; Recall@5 matters when a later step picks from a shortlist of five.

**L2. "Your baseline is 1.3%. Isn't that a strawman?"**
> [!success]- Answer
> [Say what the baseline was.] If it was all-heads attention: it's the natural "no selection" control, and the point of the result is exactly that selection matters: averaging every head drowns the few retrieval heads in sink and positional attention. I'd also report a fair external baseline, such as an off-the-shelf embedding retriever, so the 40.7% has a real reference point.

**L2. What is "lost in the middle"?**
> [!success]- Answer
> Long-context models use information at the start and end of the prompt best and in the middle worst: accuracy against the answer's position is U-shaped (Liu et al. 2023). In tool selection, tools listed in the middle of a long list are the ones most likely to be missed.

**L3. "How do you know your heads reduce position bias, rather than just being more accurate overall?"**
> [!success]- Answer
> Only by splitting recall by the correct tool's position. If heads were uniformly better, both curves would keep their U shape with the heads' curve higher. Mitigation means the heads' curve is **flatter**: the middle gains more than the ends. [Say whether you measured this; if not, that's the experiment, and I'd phrase the claim as "aimed to reduce".]

**L3. "40.7% Recall@1. Would you ship this?"**
> [!success]- Answer
> Not as the only retriever. Its strengths are zero training data and no extra model. In production I'd use a cheap embedding retriever for a top-20 shortlist and the head score to re-rank it, or use the head score as one feature. My later LongContext_RAG project builds exactly that cascade for passages.

---

## 7. CS728: say it out loud

### 7.1 The 30-second version

> "In CS728 I built a tool retriever with no training, from the LLM's own attention. A few attention heads specialise in looking up information in the prompt. I found them with a simple count: on probe queries with a known correct tool, count how often each head's strongest attention lands inside that tool's description, and keep the top 10. For a new query, I rank tools by how much attention those heads put on each description. Recall@1 went from 1.3% to 40.7% and Recall@5 from 25.5% to 66.9%."

### 7.2 The 2-minute version

> "The setting is an agent with a list of tools in its prompt. The standard fix is a trained embedding retriever; I wanted to see whether the LLM already contains one.
>
> Wu et al. showed that under 5% of attention heads are retrieval heads: when the model copies something from its context, those heads point exactly at it. I adapted their idea to tools. On probe queries with known answers, a head hits if its argmax lands inside the correct tool's span. Counting hits ranks every head, and I kept the top 10.
>
> To score tools for a new query, I read those heads' attention at the first answer token, drop the attention sink and the query tokens, renormalise over tool tokens, sum per tool, average over heads, and rank.
>
> The baseline [name it] got 1.3% Recall@1; the selected heads got 40.7%, and 66.9% Recall@5. The gain is mostly from **selection**: most heads attend by position, so averaging everything mostly measures position.
>
> Limits: it needs the attention weights, so one slow eager decode step; the whole tool list must be prefilled per query; and 40.7% top-1 is below a trained retriever. The practical design is to use it as a re-ranker over an embedding shortlist."

### 7.3 Whiteboard

```
 prompt:  <s> | tool A | tool B | tool C | … | tool N | query   → first answer token
                         ▲
 1,024 heads → each has a row of weights over these positions
 Hit Count (probes with known tool):  hit = argmax ∈ span(correct tool);  keep top 10 heads
 Retrieve:  for each chosen head: drop sink+query → renormalise → sum per tool → mean over heads → rank
 Metric:    Recall@k = share of queries whose tool ranks ≤ k        1.3% → 40.7% (R@1)
```

### 7.4 Claims to avoid

| don't say | why | say instead |
|---|---|---|
| "I trained a retriever" | nothing was trained | "a training-free retriever read off attention" |
| "retrieval heads understand the query" | they're defined by where their argmax lands | "heads whose attention lands on the content being used" |
| "mitigated lost in the middle" (without per-position numbers) | unsupported without a position split | "aimed to reduce position bias; the per-position split is the check" |
| "1.3% → 40.7%, a 31× improvement" | ratios against a near-zero baseline mislead | quote both absolute numbers and name the baseline |
| "32 heads and 3 layers" (unless you can explain it) | contradicts "top 10 heads" on its face | the exact selection, in one sentence |

---

# Part B. LeNet-5: is compression free under corruption and attack?

## 8. The project on one page

### 8.1 The problem, in plain words

Pruning research usually reports one number: clean test accuracy. The lottery-ticket result says LeNet-5 can lose about 96% of its weights with no drop in clean MNIST accuracy, and people read that as "the removed weights were useless".

But a deployed model also sees blurred, shifted and noisy digits, and sometimes deliberate attacks. **Is compression that is free on clean data also free under distribution shift and attack?** The project measures that, carefully: 5 seeds, validation-only selection, every number from a JSON record.

> [!quote] Say this
> "We pruned LeNet-5 to 3.75% of its weights with almost no loss on clean MNIST, but it lost seven times as much accuracy on corrupted digits. The cause wasn't sparsity: the standard layer-wise recipe starves the first layer, which has only 150 weights. Global pruning at the same sparsity removes the corruption penalty entirely."

### 8.2 Setup **[REPORT] §3.1, [IMPL]**

| | |
|---|---|
| Model | LeNet-5, tanh, average pooling, dense C3 connections; 61,706 parameters (`src/models.py:51–79`) |
| Data | MNIST split 55,000 train / 5,000 validation / 10,000 test, fixed split seed 1234 (`src/data.py:23–25, 65–69`); MNIST-C (15 corruptions × 10,000 images) |
| Training | SGD, lr 0.01, momentum 0.9, batch 64, 20 epochs for every condition (`src/train.py:37–47`) |
| Selection | the epoch with the best **validation** accuracy; test touched once (`train.py:126–133`) |
| Seeds | 5 per configuration; mean ± sample standard deviation (`utils.py:115–118`) |
| Compression | IMP 20% per round × 20 rounds; Hinton distillation; INT8 QAT (eager mode, `torch.ao.quantization`) |
| Attacks | PGD-L∞ in raw pixel space; reported number = worst case of PGD-100 with 5 restarts |
| Compute | 260 recorded runs, 44 GPU-hours on A40s via SLURM; latency on CPU, batch 1, 1 thread, median of 1,000 |
| Statistics | Welch's t-test, two-sided, n = 5; Holm–Bonferroni within 13 families; 59 comparisons, 31 survive |

### 8.3 The headline table **[REPORT] Table 2**

| configuration | clean % | MNIST-C % | robust % (PGD, ε = 0.3) | size KB | latency ms |
|---|---|---|---|---|---|
| standard, dense | 99.08 ± 0.09 | 73.72 ± 0.92 | 0.00 | 245.6 | 0.366 |
| standard, 3.75% weights (layer-wise) | **98.52 ± 0.14** | **69.83 ± 2.41** | 0.00 | 245.6 | 0.365 |
| standard, 1.30% weights | 95.65 ± 0.79 | 64.03 ± 1.42 | 0.00 | 245.6 | 0.367 |
| adversarial training, dense | 98.78 ± 0.09 | 78.48 ± 0.79 | 90.24 ± 0.32 | 245.6 | 0.357 |
| adversarial training, 3.75% | 96.60 ± 0.79 | 69.99 ± 1.56 | 83.55 ± 1.75 | 245.6 | 0.356 |
| IMP + KD + INT8 | 98.65 ± 0.17 | 71.10 ± 1.47 | — | **75.0** | 0.396 |
| global-pruned, 3.52% | **98.90 ± 0.07** | **73.59 ± 0.98** | 0.00 | 245.6 | — |
| global-pruned, 1.15% | 98.20 ± 0.16 | 70.42 ± 1.75 | 0.00 | 245.6 | — |

Read the size column: **pruning never shrinks the file** (245.6 KB at every sparsity), because the zeros are still stored (§11.7). Only INT8 does.

### 8.4 What is implemented where

| piece | where | on the server? |
|---|---|---|
| model, data split, training, seeding, JSON records | `src/models.py`, `data.py`, `train.py`, `utils.py` | ✓ `main` |
| PGD, FGSM, budget audit | `src/attacks.py` | ✓ |
| clean / MNIST-C / latency / size accounting | `src/evaluate.py` | ✓ |
| Phase 0 audit (leakage, 5 seeds) | `AUDIT.md`, `scripts/`, `results/raw/*.json` | ✓ |
| IMP (`prune.py`), distillation (`distill.py`), QAT (`quantize.py`), the 260 run records, figures | branch `feat/compression-robustness` | ✗ **never pushed** (only `main` exists on GitHub) |

So the pruning code below is the report's own **Listing 1** (pseudo-code) plus a loop reconstructed from `PROJECT_PLAN.md` Phase 2, both marked.

### 8.5 The ten questions about the whole project

| question | answer | label |
|---|---|---|
| What problem? | Does compression that costs nothing on clean MNIST cost something under corruption (MNIST-C) and attack (PGD)? | |
| Why LeNet-5? | Small enough to run 260 runs × 5 seeds and check every claim; the lottery-ticket result is known to hold on it. | |
| Main result? | Layer-wise IMP at 3.75%: −0.56 clean but −3.89 MNIST-C (7×). Global at 3.52%: −0.13 MNIST-C, p = 0.83. | [REPORT] |
| Why does layer-wise hurt? | Same fraction per layer: conv1 (150 weights) drops to 6, and half its filters are empty. | §11–12 |
| Why does global help? | One threshold for all; conv1's weights are large, so it keeps 26.7% at round 20. | §11.6 |
| Robustness to attack? | At ε = 0.3 every standard model is at 0%. At ε = 0.05 the 13.8% model resists far better (82% vs 52%). Not a defence. | §14 |
| Size? | INT8 QAT: 245.6 → 75.0 KB (3.27×); slightly slower at batch 1. | §16 |
| How do you know it's real? | 5 seeds, Welch tests with Holm correction, validation-only selection, gradient-masking checks, every number from a JSON record. | §13 |
| Limitations? | Global empties a later conv layer to 0.3%; robustness grids still use layer-wise tickets; one architecture; n = 5. | [REPORT] §7 |
| Next? | Sensitivity-scaled per-layer rates; global-pruned ε-sweeps; a larger network. | [FUTURE] |

---
## 9. From zero: images, convolution, and LeNet-5

### 9.1 An image is a grid of numbers

An MNIST digit is a 28 × 28 grid of pixels. Each pixel is a brightness from 0 (black) to 255 (white); the code divides by 255, so the model sees numbers in **[0, 1]**. The label is the digit, 0–9. A classifier outputs 10 numbers (**logits**), one per digit; the largest is the prediction.

### 9.2 A convolution

A fully connected layer would give every output its own weight for every pixel: 784 weights per output. A **convolution** instead slides one small grid of weights (a **kernel**, or **filter**) over the image. At each stop it multiplies the overlapping pairs and adds them: one output number. The outputs form a new grid, a **feature map**.

![[d09i-conv.svg]] [[d09i-conv.html|▶ Step through it]]

Two consequences that matter for this project:
- **Weight sharing:** the same few weights are used at every position, so a convolution layer is tiny. LeNet's conv1 has 6 filters of 5 × 5 = **150 weights** for the whole image.
- **Size arithmetic:** a k × k kernel on an n × n image, no padding, stride 1, gives (n − k + 1) × (n − k + 1). So 32 × 32 with 5 × 5 → **28 × 28**.

A filter that has learned "vertical stroke here" fires on every vertical stroke in the image. The first layer's filters are the network's basic stroke and edge detectors; everything later is built from them. That's why losing conv1 matters (§12).

### 9.3 tanh and average pooling

- **tanh** squashes every number into (−1, 1). Without such a non-linear step between layers, stacked layers would collapse into one linear map.
- **Average pooling 2 × 2** replaces each 2 × 2 block by its mean: it halves height and width, keeping a coarse "is there a stroke around here".

### 9.4 LeNet-5 in your repo, layer by layer

![[d09i-lenet-shapes.svg]] [[d09i-lenet-shapes.html|▶ Step through it]]

![[d09m-lenet.mp4]]
*Manim video (33 s): the convolution toy, then the layer pipeline and weights per layer.*

| layer | output shape | weights | biases | share of weights |
|---|---|---|---|---|
| input (raw) | 1 × 28 × 28 | | | |
| normalize (pad 2, standardise) | 1 × 32 × 32 | | | |
| conv1 (6 filters, 5 × 5) + tanh | 6 × 28 × 28 | **150** | 6 | **0.24%** |
| pool2 (avg 2 × 2) | 6 × 14 × 14 | | | |
| conv3 (16 filters, 6 × 5 × 5) + tanh | 16 × 10 × 10 | 2,400 | 16 | 3.9% |
| pool4 | 16 × 5 × 5 | | | |
| conv5 (120 filters, 16 × 5 × 5) + tanh | 120 × 1 × 1 | **48,000** | 120 | **78.1%** |
| fc6 (120 → 84) + tanh | 84 | 10,080 | 84 | 16.4% |
| fc7 (84 → 10) | 10 logits | 840 | 10 | 1.4% |
| **total** | | **61,470** | **236** | = 61,706 parameters |

All shapes and counts checked by running the model **[RAN]**. 61,706 × 4 bytes = 241.0 KiB; the saved `state_dict` file is **245.6 KiB** **[RAN]**, the report's "245.6 KB" (the report's KB are KiB, 1,024 bytes).

The names follow LeCun's paper (C1, S2, C3, S4, C5, F6, output), which is why the layers are numbered 1, 3, 5, 6, 7. "Dense C3" means each conv3 filter sees all 6 conv1 maps (the 1998 paper used a sparse connection table).

### 9.5 Reading the code: the PyTorch model pieces

| construct | meaning |
|---|---|
| `class LeNet5(nn.Module)` | a model is a class that inherits from `nn.Module`; PyTorch then tracks its layers and weights |
| `__init__(self, …)` + `super().__init__()` | the constructor: create the layers once, store them as attributes (`self.conv1 = …`) |
| `forward(self, x)` | what happens to an input; calling `model(x)` runs it |
| `nn.Conv2d(in, out, kernel_size=5)` | a convolution with `out` filters, each of shape `in × 5 × 5`, plus `out` biases |
| `nn.AvgPool2d(kernel_size=2, stride=2)` | average each 2 × 2 block; no weights |
| `nn.Linear(in, out)` | fully connected: an `out × in` weight matrix plus `out` biases |
| `torch.flatten(x, 1)` | keep axis 0 (the batch), merge the rest into one axis: `[B, 120, 1, 1] → [B, 120]` |
| `register_buffer(name, t)` | store a tensor in the model that is saved with it but is **not** a trainable weight |
| `F.pad(x, (2, 2, 2, 2), value=0.0)` | add 2 rows/columns of zeros on each side: 28 × 28 → 32 × 32 |
| `.view(1, 1, 1, 1)` | reshape a single number to 4 axes so it broadcasts against `[B, C, H, W]` |
| `@property` | a method read like an attribute: `model.prunable_modules` without parentheses |

### 9.6 The code: `InputNormalize` and `LeNet5` (`src/models.py`) **[IMPL]**

```python
 22  class InputNormalize(nn.Module):                         # ◀ layer 0 of the network: pad + standardise
     ⋮  # 23–30: docstring
 31      def __init__(self, mean=MNIST_MEAN, std=MNIST_STD, pad=2):   # ◀ MNIST_MEAN = 0.1307, MNIST_STD = 0.3081 (lines 18–19)
 32          super().__init__()
 33          self.pad = pad
 34          self.register_buffer("mean", torch.tensor(mean).view(1, 1, 1, 1))   # ◀ saved with the model, never trained
 35          self.register_buffer("std", torch.tensor(std).view(1, 1, 1, 1))
 37      def forward(self, x):                                # ◀ x: raw pixels in [0, 1], shape [B, 1, 28, 28]
 38          if self.pad:
 39              x = F.pad(x, (self.pad,) * 4, mode="constant", value=0.0)   # ◀ 28×28 → 32×32, as in LeCun's paper
 40          return (x - self.mean) / self.std                # ◀ centre and scale INSIDE the model

 51  class LeNet5(nn.Module):
 58      def __init__(self, activation="tanh", width=1, num_classes=10):
 59          super().__init__()
 60          c1, c3, c5, f6 = 6 * width, 16 * width, 120 * width, 84 * width   # ◀ width = 2 gives the distillation teacher
 62          self.normalize = InputNormalize()
 63          self.conv1 = nn.Conv2d(1, c1, kernel_size=5)        # ◀ 6 × 1 × 5 × 5 = 150 weights
 64          self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
 65          self.conv3 = nn.Conv2d(c1, c3, kernel_size=5)       # ◀ 16 × 6 × 5 × 5 = 2,400
 66          self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
 67          self.conv5 = nn.Conv2d(c3, c5, kernel_size=5)       # ◀ 120 × 16 × 5 × 5 = 48,000 (a 5×5 kernel on a 5×5 map = fully connected)
 68          self.fc6 = nn.Linear(c5, f6)                        # ◀ 84 × 120 = 10,080
 69          self.fc7 = nn.Linear(f6, num_classes)               # ◀ 10 × 84 = 840
 70          self.act = _activation(activation)                  # ◀ nn.Tanh() (lines 43–48)
 72      def forward(self, x):
 73          x = self.normalize(x)                               # ◀ [B, 1, 32, 32]
 74          x = self.pool2(self.act(self.conv1(x)))             # ◀ [B, 6, 28, 28] → [B, 6, 14, 14]
 75          x = self.pool4(self.act(self.conv3(x)))             # ◀ [B, 16, 10, 10] → [B, 16, 5, 5]
 76          x = self.act(self.conv5(x))                         # ◀ [B, 120, 1, 1]
 77          x = torch.flatten(x, 1)                             # ◀ [B, 120]
 78          x = self.act(self.fc6(x))                           # ◀ [B, 84]
 79          return self.fc7(x)                                  # ◀ [B, 10] logits
     ⋮
 82      def prunable_modules(self):                             # ◀ (a @property) the five layers whose .weight gets pruned
```

| name | holds |
|---|---|
| `x` | a batch of images; B = batch size (64 in training, 1,000 in evaluation) |
| `self.mean`, `self.std` | 0.1307 and 0.3081: MNIST's pixel mean and standard deviation |
| `width` | multiplies every channel count; `LeNet5(width=2)` is the wider teacher for distillation |
| `prunable_modules` | `[("conv1", …), ("conv3", …), ("conv5", …), ("fc6", …), ("fc7", …)]`: biases are never pruned |

### 9.7 Why normalisation lives inside the model

This is one of the project's three rules, and interviewers like it.

An attack (§14) changes the input by at most ε. **Which input?** If standardisation were done in the data loader, the attack would perturb the standardised tensor, where 1 unit = 0.3081 pixel units. An ε = 0.3 attack there is really ε = 0.3 × 0.3081 ≈ **0.092** in pixel space: a much weaker attack, so every robustness number would look far too good.

With `InputNormalize` as layer 0, the model eats raw pixels, the attack perturbs raw pixels, and ε = 0.3 means what the literature means. `assert_pixel_space_model` (`attacks.py:21–28`) refuses to attack a model without this layer.

The project checked this concretely (`METHODOLOGY.md`): the course plan expected ~17.6% PGD accuracy for a standard LeNet at ε = 0.3; the correct pixel-space attack gives **0.0%** (matching Madry et al.), while the normalisation bug gives 3.0%.

### 9.8 Question bank: LeNet-5 and convolutions

**L1. Walk me through LeNet-5's shapes.**
> [!success]- Answer
> Input 1 × 28 × 28, padded to 32 × 32 and standardised inside the model. conv1, six 5 × 5 filters: 6 × 28 × 28 (32 − 5 + 1). Average pool: 6 × 14 × 14. conv3, sixteen filters over all 6 maps: 16 × 10 × 10. Pool: 16 × 5 × 5. conv5, 120 filters of 16 × 5 × 5: 120 × 1 × 1, effectively fully connected. Then fc6 to 84 and fc7 to 10 logits. tanh after each conv and fc6. 61,706 parameters.

**L1. Why does a convolution have so few weights?**
> [!success]- Answer
> Weight sharing: one small kernel is reused at every position of the image, so the weight count depends on the kernel size and channel counts, not on the image size. conv1 has 6 × 5 × 5 = 150 weights for the whole 32 × 32 image; a fully connected layer producing the same 6 × 28 × 28 outputs would need millions.

**L2. Where are most of LeNet-5's weights, and why does that matter for pruning?**
> [!success]- Answer
> conv5 holds 48,000 of 61,470 weights (78%), fc6 10,080, conv3 2,400, fc7 840, and conv1 only 150 (0.24%). Any rule that removes the same fraction from every layer treats a 150-weight layer like a 48,000-weight one, so the tiny first layer is cut to a handful of weights long before the big layers feel anything.

**L2. Why is normalisation a layer inside the model?**
> [!success]- Answer
> So that attacks operate on raw pixels. If standardisation sat in the data loader, an ε = 0.3 perturbation of the standardised tensor would be only ε ≈ 0.092 in pixel space, and every robustness number would be inflated. The repo enforces this: the attack code refuses a model without an InputNormalize layer.

**L3. "Why average pooling and tanh in 2026?"**
> [!success]- Answer
> Faithfulness: Part 1 reproduced LeCun et al. (1998), and Part 2 deliberately kept that model so the compression results sit on a known reproduction. Max pooling and ReLU train faster and are what you'd use today (Part 1 tested an Adam + ReLU variant), but changing the architecture would mix two questions.

---

## 10. Training honestly: validation selection and seeds

### 10.1 The three data splits

- **Training set (55,000):** the model learns from it.
- **Validation set (5,000):** used to **choose** things: which epoch to keep, which hyperparameters.
- **Test set (10,000):** used **once**, at the end, to report the number.

If you choose the epoch by **test** accuracy, you've used the test set to pick the best of 20 noisy numbers, so the reported number is biased upward. It's no longer an estimate of accuracy on new data.

### 10.2 The code **[IMPL]**

`src/data.py:65–69`: a fixed split, the same for every training seed:
```python
 65  def train_val_indices(n_total: int = 60_000):
 66      """Fixed permutation split. Same for every training seed, by design."""
 67      g = torch.Generator().manual_seed(SPLIT_SEED)        # ◀ SPLIT_SEED = 1234 (line 23): its own random stream
 68      perm = torch.randperm(n_total, generator=g).tolist() # ◀ a shuffled list of 0 … 59,999
 69      return perm[:N_TRAIN], perm[N_TRAIN : N_TRAIN + N_VAL]   # ◀ first 55,000 = train, next 5,000 = validation
```

`src/train.py:125–135`: selection on validation only:
```python
125          # Selection uses validation only.
126          if val_acc > best_val_acc:                        # ◀ inside the epoch loop: a new best on VALIDATION
127              best_val_acc, best_val_epoch = val_acc, epoch
128              best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}   # ◀ snapshot the weights
     ⋮  # 130: wall-clock time
133      test_at_best_val = record.epochs[best_val_epoch - 1]["test_acc"]   # ◀ the honest number: test at the val-chosen epoch
135      leaky_epoch = max(record.epochs, key=lambda e: e["test_acc"])      # ◀ the leaky number, kept only to measure the leak
```

| name | holds |
|---|---|
| `record.epochs` | one dict per epoch: train loss, val loss/acc, test loss/acc |
| `best_val_epoch` | the epoch with the highest validation accuracy (1-based, hence `- 1`) |
| `state_dict()` | a dict of every weight tensor by name; `.clone()` copies it so later training can't change the snapshot |
| `max(…, key=lambda e: …)` | the element with the largest key; `lambda e: e["test_acc"]` is a one-line function |

Every run also writes a JSON record (config hash, seed, git SHA, machine, metrics, per-epoch curve) atomically (`utils.py:94–101`: write to `.tmp`, then rename). Tables are rebuilt from those files, never typed by hand.

### 10.3 What the audit found **[REPORT] Table 1, README**

| budget | protocol | standard | augmented | difference | p (Welch) |
|---|---|---|---|---|---|
| 10 epochs | best epoch on validation | 98.88 ± 0.05 | 99.08 ± 0.09 | +0.20 | 0.003 |
| 20 epochs | best epoch on validation | 99.08 ± 0.09 | 99.16 ± 0.04 | +0.08 | **0.115** |
| 20 epochs | best epoch on test (leaky) | 99.13 ± 0.07 | 99.24 ± 0.03 | +0.11 | 0.023 |

- Selecting on test inflated accuracy by about **0.05 pp**, the same size as the effect being studied.
- With honest selection at 20 epochs, augmentation's gain is **not** significant (p = 0.115); with leaky selection it looks significant (p = 0.023). **Leakage manufactured significance.**
- Seed-to-seed standard deviation is about **0.09 pp**, so single-seed differences below that are noise. That's why every number here has 5 seeds.

### 10.4 Question bank: protocol

**L1. Why a validation split?**
> [!success]- Answer
> To make choices (epoch, hyperparameters, temperature) without touching the test set. If the test set chooses, the reported test number is the maximum of several noisy estimates and is biased upward.

**L2. How big was the leak, and why does 0.05 pp matter?**
> [!success]- Answer
> About 0.05 pp on average. Small in absolute terms, but the effects in this project are of the same size: augmentation's gain at 20 epochs was 0.08 pp. With test-set selection that gain looked significant (p = 0.023); with validation selection it wasn't (p = 0.115).

**L2. Why is the split seed separate from the training seed?**
> [!success]- Answer
> So all five training seeds use exactly the same validation images. Seeds then vary only the initialisation and data order, and differences between configurations aren't confounded with which 5,000 images happened to be held out.

**L3. "You select by validation accuracy over 20 epochs. Isn't that also optimistic?"**
> [!success]- Answer
> The **validation** number is optimistic (it's a maximum of 20), which is why it's never reported. The test accuracy at that chosen epoch is an unbiased estimate, because the test set played no part in the choice.

---

## 11. Pruning from zero: masks, IMP and the lottery ticket

### 11.1 What pruning is

A trained network has many weights near zero that barely affect the output. **Magnitude pruning** assumes small |w| means unimportant: set the smallest weights to zero and keep them there. "3.75% of weights remain" means **96.25% sparsity**.

**Masks.** Each weight tensor W gets a 0/1 **mask** M of the same shape; the network uses W ⊙ M (element-wise product). After every optimiser step the mask is re-applied, so a pruned weight can't grow back.

| | w₁ | w₂ | w₃ | w₄ |
|---|---|---|---|---|
| W | 0.9 | −0.8 | 0.7 | 0.1 |
| M | 1 | 1 | 1 | 0 |
| W ⊙ M | 0.9 | −0.8 | 0.7 | **0** |

### 11.2 One-shot vs iterative, and what happens between rounds

Removing 96% at once usually breaks a network. **Iterative magnitude pruning (IMP)** removes a fraction per round, and lets the network recover between rounds:

```
train → prune 20% of surviving weights → reset or keep the survivors → retrain with the mask fixed → repeat
```

What "reset or keep" means is the difference between three methods, and **this project ran all three** **[REPORT] §3.2.1**:

| variant | after pruning, the surviving weights … | from |
|---|---|---|
| **rewind to init** (lottery ticket) | go back to their values at initialisation (step 0), then retrain | Frankle & Carbin 2019 |
| **late rewinding** | go back to their values at an early training step k | Frankle et al. 2020 |
| **fine-tuning** | keep their trained values; continue training | Han et al. 2015 |

Result: the two rewinds were indistinguishable at every level; fine-tuning was 6.73 points worse at 1.30% of weights, but with a per-seed standard deviation of 5.89 that is **not significant** at n = 5 (p = 0.062 raw, 0.174 Holm). "The lottery-ticket ordering is directionally reproduced but statistically unestablished."

> [!abstract] The two papers, summarised
> - **Frankle & Carbin, *The Lottery Ticket Hypothesis* (2019).** A dense random network contains a small subnetwork (a "winning ticket") that, trained alone **from the same initial values**, matches the full network in a similar number of steps. Found by IMP with reset to initialisation. Tickets are often under 10–20% of the weights on MNIST and CIFAR-10 networks (LeNet-300-100 among them). The same mask with **fresh random** weights learns slower and ends worse, so the initial values matter. Their convolutional experiments prune the output layer at half the rate.
> - **Han et al., *Learning both Weights and Connections* (2015).** Train, remove weights below a threshold, retrain the rest (keeping their values), repeat. Iterating removes more than one big cut at the same accuracy. Convolutional layers are more sensitive to pruning than fully connected ones. AlexNet shrinks 9×, VGG-16 13×, LeNets about 12×.

### 11.3 The schedule, and where 3.75% comes from **[RAN]**

`PROJECT_PLAN.md` Phase 2.1: prune **20% per round, the output layer (fc7) at half rate (10%)**, for 20 rounds. Per round, each layer keeps 80% of its survivors (fc7 90%):

| round | conv1 | conv3 | conv5 | fc6 | fc7 | total left |
|---|---|---|---|---|---|---|
| 0 | 150 | 2,400 | 48,000 | 10,080 | 840 | 100% |
| 9 | 21 | 322 | 6,442 | 1,353 | 325 | **13.77%** |
| 15 | **6** | 85 | 1,688 | 354 | 173 | **3.75%** |
| 20 | **2** | 27 | 553 | 116 | 102 | **1.30%** |

These reproduce the report's 13.8%, 3.75% and 1.30% exactly, and conv1's 21, 6 and 2 survivors (report Table 4). **Global** pruning at 20% per round gives 0.8¹⁵ = **3.52%** and 0.8²⁰ = **1.15%**: the report's global rows. So "matched sparsity" means *the same round*, 3.75% vs 3.52%.

![[d09i-imp-rounds.svg]] [[d09i-imp-rounds.html|▶ Step through it]]

### 11.4 The pruning step: the report's Listing 1 **[REPORT] §3.1**

The real `prune.py` is on the unpushed branch. The report shows its core as pseudo-code:

```python
 1  def prune_step(model, scope, rate=0.20):
 2      # rank weights by magnitude, then zero the smallest `rate` fraction
 3      if scope == "layerwise":
 4          for layer in prunable_layers(model):                      # ◀ conv1, conv3, conv5, fc6, fc7
 5              threshold = magnitude_quantile(layer.weight, rate)    # ◀ a threshold PER LAYER: rate of this layer's |w| lie below it
 6              layer.mask &= (layer.weight.abs() >= threshold)       # ◀ keep the big ones; &= so pruned weights stay pruned
 7      elif scope == "global":
 8          all_weights = concat_magnitudes(model)                    # ◀ every |w| of every prunable layer, in one vector
 9          threshold = magnitude_quantile(all_weights, rate)         # ◀ ONE threshold for the whole network
10          for layer in prunable_layers(model):
11              layer.mask &= (layer.weight.abs() >= threshold)
12      apply_masks(model)                                            # ◀ weight ← weight × mask
13      return model
```

| name | holds |
|---|---|
| `scope` | `"layerwise"` or `"global"`: the only difference between the two schedules |
| `rate` | fraction of **surviving** weights to remove this round (0.20; fc7 0.10 in the layer-wise runs) |
| `magnitude_quantile(v, rate)` | the value below which `rate` of the magnitudes lie. Not shown in the report; it must look only at surviving (non-zero) weights, or from round 2 on the quantile would land among the zeros and nothing new would be pruned **[RECON]** |
| `layer.mask` | a 0/1 (bool) tensor the shape of `layer.weight` |
| `&=` | in-place AND: a 0 stays 0 forever |

![[d09i-prune-dbg.svg]] [[d09i-prune-dbg.html|▶ Step through it]]

**The loop around it** (reconstructed from `PROJECT_PLAN.md` Phase 2.1; not your code) **[RECON]**:

```python
theta0 = copy.deepcopy(model.state_dict())          # ◀ the initial weights, kept for rewinding
train(model, epochs=20)                             # ◀ round 0: train dense
for r in range(1, 21):
    prune_step(model, scope, rate=0.20)             # ◀ shrink the masks
    rewind(model, theta0)                           # ◀ survivors ← their values at init (late rewinding: at epoch k instead)
    train(model, epochs=20, masks=True)             # ◀ retrain; re-apply masks after every optimizer.step()
    save_checkpoint(model, r)                       # ◀ one ticket per sparsity level
```

### 11.5 Layer-wise vs global on 16 weights

The same two layers as the animation above: layer 1 is small with **large** weights (0.9, −0.8, 0.7, 0.1); layer 2 has 12 small ones (0.3, −0.05, 0.2, 0.02, −0.25, 0.15, 0.01, −0.4, 0.12, 0.06, −0.03, 0.35). This time **keep only a quarter** (4 of 16), as in the figure below.
- **Layer-wise:** each layer keeps 25%. Layer 1 keeps **1** (0.9); its −0.8 and 0.7 are dropped although they beat every weight in layer 2. Layer 2 keeps 3 (−0.4, 0.35, 0.3).
- **Global:** keep the 4 largest |w| of all 16: 0.9, −0.8, 0.7, −0.4. Layer 1 keeps **3**, layer 2 keeps 1.

(The animation above did the same with half kept: layer-wise 2 + 6, global 3 + 5.)

![[d09-2-pruning.svg]] [[d09-2-pruning.html|▶ Step through it]]

### 11.6 Why global pruning spares conv1 **[RAN]**, and its own pathology

PyTorch initialises a layer's weights uniformly in ±1/√(fan-in), where fan-in = inputs per output. conv1's fan-in is 1 × 5 × 5 = 25; conv5's is 16 × 5 × 5 = 400. So conv1 **starts** with weights about 4× larger:

| layer | fan-in | init bound 1/√fan-in | mean \|w\| at init (measured) |
|---|---|---|---|
| conv1 | 25 | 0.200 | **0.100** |
| conv3 | 150 | 0.082 | 0.041 |
| conv5 | 400 | 0.050 | **0.025** |
| fc6 | 120 | 0.091 | 0.045 |
| fc7 | 84 | 0.109 | 0.055 |

A single global threshold therefore cuts conv5 hardest and conv1 least: at round 20 global keeps **26.7%** of conv1 where layer-wise keeps 1.3% **[REPORT] §4.2**.

**Global's own pathology [REPORT] §7:** it empties a later convolutional layer to **0.3%** while keeping conv1 at 26.7%. So it doesn't avoid destroying a layer; it destroys a **different** one. The report doesn't name the layer; conv5, with the smallest weights, is the likely one **[RECON]**. The principled fix, which neither schedule implements, is a per-layer rate scaled by each layer's **sensitivity** (how much loss rises when that layer alone is pruned) **[FUTURE]**.

![[d09m-prune.mp4]]
*Manim video (34 s): 20 rounds of layer-wise pruning, layer by layer, then the clean vs MNIST-C cost of each schedule (report Table 5).*

### 11.7 Pruning doesn't shrink the file (`src/evaluate.py:105–124`) **[IMPL]**

```python
105  def model_size(model) -> dict:
     ⋮  # 106–113: docstring ("a pruned model saved with torch.save is byte-for-byte the same size")
114      counts = count_params(model)                          # ◀ total, non-zero, sparsity (utils.py:65–77)
115      fp32_kb = _file_size_kb(lambda p: torch.save(model.state_dict(), p))   # ◀ the REAL file on disk
     ⋮  # 116–118: comment on the CSR estimate
119      csr_bytes = 0
120      for _, m in getattr(model, "prunable_modules", []):
121          w = m.weight.detach()
122          nnz = int((w != 0).sum())                         # ◀ surviving weights
123          rows = w.shape[0]
124          csr_bytes += nnz * (4 + 4) + (rows + 1) * 4      # ◀ CSR: value + column index per non-zero, + a row-pointer array
```

A dense tensor stores every entry, zeros included, so torch.save writes the same 245.6 KiB at any sparsity. A **sparse format** (CSR: compressed sparse row) stores only the non-zeros with their positions, 8 bytes each. At 3.75% (2,306 weights) the weights would take about 18 KiB. The code reports both and labels the CSR size "theoretical": quoting it as "the model size" would be a misrepresentation.

**Unstructured sparsity doesn't make it faster either**: dense hardware still multiplies the zeros. Speed needs sparse kernels or **structured** pruning (removing whole filters or channels). The report's latencies confirm it: 0.366 ms dense, 0.365 ms at 3.75%.

### 11.8 Question bank: pruning

**L1. What is iterative magnitude pruning?**
> [!success]- Answer
> Repeat: train, remove the smallest-magnitude fraction of the surviving weights (20% per round here), then reset or keep the survivors and retrain with the pruned weights held at zero by a mask. Twenty rounds of 20% leave about 1% of the weights.

**L1. Layer-wise vs global pruning?**
> [!success]- Answer
> Layer-wise removes the same fraction from every layer, each with its own threshold. Global ranks all weights in the network together and applies one threshold, so layers with larger weights keep more of them.

**L2. Where does 3.75% come from?**
> [!success]- Answer
> 15 rounds of the layer-wise schedule: 20% per round for the four hidden layers (0.8¹⁵ = 3.5%) and 10% for the output layer, which is pruned at half rate as in the lottery-ticket paper (0.9¹⁵ = 21% of its 840). Together that's 2,306 of 61,470 weights = 3.75%. Global pruning at the same round gives 0.8¹⁵ = 3.52%, which is the "matched sparsity" comparison.

**L2. Rewind to init, late rewinding, fine-tuning: what's the difference, and what did you find?**
> [!success]- Answer
> After each prune the survivors either go back to their initial values (lottery ticket), to their values at an early epoch (late rewinding), or keep their trained values (fine-tuning). The two rewinds were indistinguishable; fine-tuning was 6.73 points worse at 1.30% of weights, but with a seed standard deviation of 5.89 that's p = 0.062 raw and 0.174 after Holm correction: directionally the lottery-ticket result, not established at 5 seeds.

**L2. Why does global pruning keep conv1?**
> [!success]- Answer
> conv1's weights are larger. PyTorch initialises weights in ±1/√fan-in; conv1 has fan-in 25 and conv5 400, so conv1's weights start about 4× larger (mean |w| 0.100 vs 0.025), and training keeps them relatively large. A single network-wide threshold therefore removes conv5's weights first; at round 20 conv1 still has 26.7%.

**L3. "So global pruning is the answer?"**
> [!success]- Answer
> No. It moves the damage: at the most extreme level it empties a later conv layer to 0.3% while keeping conv1 at 26.7%. The two schedules bracket the question. The principled version scales each layer's rate by its measured sensitivity, which neither run implements. What the data does show is that at 3.5% global is strictly better on both clean and MNIST-C accuracy.

**L3. "96% sparse: is it 25× smaller and faster?"**
> [!success]- Answer
> Neither, as measured. The saved file is the same 245.6 KB because dense tensors store the zeros; a CSR format would be ~18 KB for the weights, but that's theoretical here. Latency is unchanged (0.366 vs 0.365 ms) because dense kernels multiply zeros too. Speed needs structured pruning or sparse kernels. The real size reduction came from INT8.

---

## 12. MNIST-C and the hidden cost

### 12.1 MNIST-C, summarised

> [!abstract] Mu & Gilmer, *MNIST-C* (2019)
> - MNIST test digits under **15 corruptions**, 10,000 images each: shot noise, impulse noise, glass blur, motion blur, shear, scale, rotate, brightness, translate, stripe, fog, spatter, dotted line, zigzag and Canny edges.
> - The corruptions are never seen in training: they test robustness to **distribution shift**, not to an adversary.
> - CNNs at ~99% clean accuracy lose far more on these, so clean accuracy hides fragility. In your table, the dense LeNet goes from 99.08% clean to 73.72% MNIST-C.

### 12.2 The code (`src/evaluate.py:30–44`) **[IMPL]**

```python
 29  @torch.no_grad()
 30  def mnist_c_accuracy(model, corruption_loaders: dict, device) -> dict:
     ⋮  # 31–35: docstring ("the mean alone hides which corruptions break which models")
 36      per = {name: clean_accuracy(model, ld, device) for name, ld in sorted(corruption_loaders.items())}   # ◀ accuracy on each of the 15
 37      accs = list(per.values())
 38      return {
 39          "per_corruption": per,
 40          "mean_acc": sum(accs) / len(accs),                # ◀ "MNIST-C accuracy" = the plain mean over 15 corruptions
 41          "worst_corruption": min(per, key=per.get),        # ◀ name of the hardest one
 42          "worst_acc": min(accs),
 43          "n_corruptions": len(per),
 44      }
```

`{name: … for name, ld in …}` is a dict comprehension: one entry per corruption. The loaders come from `get_mnist_c_loaders` (`data.py:108–146`), which converts the stored `(N, 28, 28, 1)` uint8 arrays to the same `(N, 1, 28, 28)` floats in [0, 1] as clean MNIST.

### 12.3 The 7× **[REPORT] Table 5**

| weights left | schedule | Δ clean (pp) | Δ MNIST-C (pp) | ratio |
|---|---|---|---|---|
| 3.75% | layer-wise | −0.56 | **−3.89** (p = 0.019) | **7.0×** |
| 3.52% | global | −0.18 | −0.13 (p = 0.83) | 0.7× |
| 1.30% | layer-wise | −3.43 | −9.70 | 2.8× |
| 1.15% | global | −0.88 | −3.30 | 3.8× |

"7×" = the corruption drop divided by the clean drop: 3.89 / 0.56 = 6.9. **Validating on clean accuracy alone would call this compression nearly free.** Head to head at matched sparsity, global beats layer-wise by **3.76** points on MNIST-C (p = 0.021) and 0.38 on clean (p = 0.002): better on both.

A residual cost survives near 1%: global still loses 3.30 points of MNIST-C there. So sparsity **does** eventually cost robustness, but far later and far less than the layer-wise numbers suggested.

### 12.4 The starved conv1 **[REPORT] Table 4, §3.6**

| weights left | conv1 weights alive | conv1 filters entirely empty |
|---|---|---|
| 100% | 150 / 150 | 0 / 6 |
| 13.8% | 21 / 150 | 0 / 6 |
| 3.75% | **6 / 150** | **3 / 6** |
| 1.30% | 2 / 150 | 4 / 6 |

At 3.75%, the survivors are isolated single weights: no 5 × 5 structure is left, so they can't be oriented edge detectors. The network still classifies clean digits (later layers compensate), but anything that needs robust low-level features, like blurred, noisy or shifted strokes, fails. **That is the confound:** under layer-wise pruning "compressed" and "first layer destroyed" happen together. Global pruning separates them, because it keeps conv1 alive at the same total sparsity, and the corruption penalty disappears.

### 12.5 The effect depends on how the model was trained **[REPORT] Table 6**

Switching layer-wise → global at ~13.5% weights changes MNIST-C by:

| training | effect of switching to global |
|---|---|
| standard | **+2.15** (p = 0.010) |
| augmented (rotations, shifts) | +0.34 (p = 0.74) |
| adversarial | **+3.20** (p = 0.003) |

Augmentation already teaches tolerance to the spatial changes a damaged conv1 can't handle, so it has little to gain. The adversarial arm also became much **more stable** under global pruning (robust accuracy 88.51 ± 1.17 vs 84.72 ± 5.86).

### 12.6 Question bank: corruption robustness

**L1. What is MNIST-C?**
> [!success]- Answer
> The MNIST test set under 15 corruptions (noise, blur, shear, scale, rotation, brightness, translation, fog, spatter, stripes, dotted lines, zigzags, Canny edges), 10,000 images each. It tests robustness to distribution shift the model never trained on. I report the mean over the 15, and the per-corruption numbers.

**L2. What does "7× larger corruption loss" mean exactly?**
> [!success]- Answer
> At 3.75% of weights with layer-wise IMP, clean accuracy fell 0.56 points from dense but MNIST-C accuracy fell 3.89 points: the corruption drop was about 7 times the clean drop. A pruning study that validated only on clean accuracy would have called that compression nearly free.

**L2. How did you find that conv1 was the cause?**
> [!success]- Answer
> Two steps. First, counting: at 3.75% under layer-wise pruning conv1 has 6 of its 150 weights and 3 of its 6 filters are entirely empty, so the "edge detector" layer is gone. That's a hypothesis, because sparsity and conv1 damage happen together. Second, a control: global pruning at the same round (3.52%), same seeds, same rewind and budget, keeps conv1 at a healthy fraction. The MNIST-C penalty dropped from −3.89 (p = 0.019) to −0.13 (p = 0.83). Changing only the schedule removed the effect, so the schedule, not the sparsity, caused most of it.

**L3. "Is it the conv1 layer specifically, or just that layer-wise is worse in general?"**
> [!success]- Answer
> The control shows the schedule matters; the conv1 count is the mechanism I'd point to, but strictly it's not isolated. The clean test would be layer-wise pruning with conv1 exempt (or pruned at a lower rate) and everything else unchanged; if the penalty disappears, conv1 is the cause. That's the follow-up I'd run. Supporting evidence: the effect is small for augmented training, which teaches exactly the spatial tolerance a damaged first layer loses.

**L3. "Global pruning still loses 3.3 points at 1.15%. So sparsity does hurt?"**
> [!success]- Answer
> Yes, eventually, and I say so: at about 1% of weights global still loses 3.30 MNIST-C points against 0.88 clean. The claim is narrower: at 3.5%, which is where people stop and declare compression free, most of the hidden cost came from the layer-wise recipe, and a better selection rule removed it at no price.

---

## 13. Statistics: mean ± std, Welch's t-test and Holm

### 13.1 Why statistics at all

Training is random: initial weights, data order. Train the same configuration with 5 seeds and you get 5 different accuracies. Report **mean ± standard deviation** (sample std, divide by n − 1). Two configurations differ "really" only if the gap is large compared with that seed noise.

### 13.2 Welch's t-test, step by step

You have two groups of 5 numbers (5 seeds each), with means m₁, m₂ and standard deviations s₁, s₂.

1. **Difference:** d = m₁ − m₂.
2. **Noise of each mean:** a mean of n numbers wobbles with variance s²/n.
3. **Standard error of the difference:** SE = √(s₁²/n + s₂²/n).
4. **t = d / SE:** how many noise units the difference is.
5. **Degrees of freedom** (Welch–Satterthwaite): df = (s₁²/n + s₂²/n)² / [(s₁²/n)²/(n−1) + (s₂²/n)²/(n−1)].
6. **p-value:** the probability of a |t| at least this large **if there were no real difference**, from the t-distribution with df degrees of freedom.

"Welch" means the two groups may have **different** spreads (layer-wise has s = 2.41, dense 0.92). The classic Student's t-test assumes equal spreads.

![[d09i-welch.svg]] [[d09i-welch.html|▶ Step through it]]

**Recomputed from the report's own table [RAN]:**

| comparison | d | SE | t | df | p (report) |
|---|---|---|---|---|---|
| global 3.52% vs dense, MNIST-C | −0.13 | 0.601 | −0.22 | 7.97 | **0.834** (0.83) |
| layer-wise 3.75% vs dense, MNIST-C | −3.89 | 1.154 | −3.37 | 5.14 | **0.019** (0.019) |
| global vs layer-wise, MNIST-C | +3.76 | 1.163 | 3.23 | 5.29 | **0.021** (0.021) |
| global vs layer-wise, clean | +0.38 | 0.070 | 5.43 | 5.88 | **0.0017** (0.002) |
| PGD ε = 0.05: 13.8% vs dense | +30.2 | 4.60 | 6.57 | 5.32 | **0.001** (0.0010) |

### 13.3 What a p-value does and doesn't say

- **p = 0.83:** a gap this small is entirely expected if global-pruned and dense models are equally robust. It is **not** proof that they're equal. With 5 seeds and a spread of ~1 point, a true gap of 0.5 points could easily hide.
- **p = 0.019:** a gap this large would be rare (under 2%) if there were no difference.
- **"Not significant" means "not resolvable at n = 5"**, not "no effect". The report says exactly that.

### 13.4 Many tests: Holm–Bonferroni

Run 59 tests at α = 0.05 and you expect about 3 false alarms even if nothing is real. **Holm's method** controls that, within a family of m related tests:
1. sort the p-values: p(1) ≤ p(2) ≤ … ≤ p(m);
2. compare p(1) with α/m, p(2) with α/(m−1), …, p(i) with α/(m − i + 1);
3. stop at the first one that fails; it and all larger ones are not significant.

Toy, 3 tests, α = 0.05: p = 0.010, 0.030, 0.040. 0.010 ≤ 0.05/3 = 0.0167 ✓; 0.030 ≤ 0.05/2 = 0.025 ✗ → stop. Only the first survives.

The report: 59 comparisons in 13 families, Holm within each, **31 survive**. Every p-value was computed twice, with `scipy.stats.ttest_ind` and with a hand-written Welch, agreeing to 10⁻⁶. The failures are listed too: lottery ticket vs fine-tuning, the distillation temperature grid, the INT8 accuracy drop, and adversarial vs standard MNIST-C at pruned levels.

### 13.5 Question bank: statistics

**L1. What does p = 0.83 mean?**
> [!success]- Answer
> If global-pruned and dense models truly had the same MNIST-C accuracy, a difference at least as big as the one we saw (0.13 points) would turn up 83% of the time. So there's no evidence of a difference. It does not prove they're equal: with 5 seeds, small differences can't be resolved.

**L2. Why Welch's t-test and not a paired test?**
> [!success]- Answer
> The two groups are independent runs: a seed in one configuration isn't the "same subject" as that seed in another in any controlled sense (pruning changes the whole training trajectory), and the spreads differ a lot (2.41 vs 0.92). Welch's test handles unequal variances. A paired test would be appropriate if each seed's two runs shared everything except the treatment and were strongly correlated.

**L2. Why Holm correction?**
> [!success]- Answer
> Fifty-nine tests at α = 0.05 would give about three false positives by chance alone. Holm keeps the chance of any false positive within each family at 5%, and it's uniformly more powerful than plain Bonferroni: the smallest p-value is compared with α/m, the next with α/(m−1), and so on, stopping at the first failure.

**L3. "n = 5 is tiny. Why should I believe any of it?"**
> [!success]- Answer
> For the big effects, n = 5 is plenty: the layer-wise MNIST-C drop is 3.4 standard errors, and the PGD gap at ε = 0.05 has non-overlapping per-seed distributions. For small effects it isn't, and the report says so: 28 of 59 comparisons don't survive correction, and those are reported as "not resolvable at n = 5" rather than as "no effect". The seed noise floor is about 0.09 points, so effects above roughly 0.1–0.2 points are resolvable.

**L3. "How would you compute the p-value without scipy?"**
> [!success]- Answer
> d = m₁ − m₂; SE = √(s₁²/n₁ + s₂²/n₂); t = d / SE; df by Welch–Satterthwaite; p = 2 × (1 − F(|t|)), where F is the cumulative t-distribution with df degrees of freedom. For p = 0.83: d = −0.13, SE = √(0.92²/5 + 0.98²/5) = 0.601, t = −0.22, df ≈ 8, and a |t| of 0.22 on 8 df has a two-sided p of about 0.83.

---
## 14. PGD from zero, your attack code, and what the robustness result really says

### 14.1 Adversarial examples and the ε-ball

Take a digit the model classifies correctly. Change each pixel by a tiny amount, chosen to push the model toward a wrong answer. If the prediction flips, that input is an **adversarial example**. **Robust accuracy** = the share of test images whose prediction survives the attack.

"Tiny" is made precise with the **L∞ ball**: no pixel may move more than ε.

$$\lVert x - x_0 \rVert_\infty \le \varepsilon \iff x_0 - \varepsilon \le x \le x_0 + \varepsilon \text{ for every pixel}$$

Pixels must also stay in [0, 1]. On MNIST the standard budget is **ε = 0.3** (Madry et al.).

### 14.2 FGSM, then PGD

- **FGSM** (one step): x = x₀ + ε · sign(∇ₓ loss). `sign` turns each gradient entry into −1, 0 or +1, so every pixel moves the full ε in the direction that raises the loss. Why the sign: under an L∞ budget each pixel may move independently by up to ε, so the biggest first-order increase in loss comes from pushing **every** pixel the full amount.
- **PGD** (projected gradient descent): many small steps of size α, each followed by a **projection** back into the box and into [0, 1]:

$$x_{t+1} = \text{clip}_{[x_0 - \varepsilon,\, x_0 + \varepsilon] \cap [0, 1]}\Big(x_t + \alpha \cdot \text{sign}\big(\nabla_x \mathcal{L}(x_t, y)\big)\Big)$$

Many small steps can follow a curved loss surface that one FGSM step overshoots. A **random start** inside the box and several **restarts** make it stronger.

> [!abstract] Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* (2018), §2
> - Robustness as a min-max problem: the attacker picks the worst perturbation inside the ε-ball; training picks weights that minimise that worst-case loss.
> - PGD with a random start is their attack, and they argue it's the strongest attack that uses only first-order (gradient) information.
> - **Adversarial training**: train on PGD examples instead of clean ones.
> - MNIST, L∞, ε = 0.3: about 89% accuracy against their strongest attack; a naturally trained model is at ~0%.

### 14.3 Reading the code: the PyTorch pieces for attacks

| construct | meaning |
|---|---|
| `x.clone().detach()` | a copy with its own memory, cut off from any gradient history |
| `.requires_grad_(True)` | ask autograd to track this tensor, here the **input**, so we can get d loss / d input |
| `torch.autograd.grad(loss, x)` | returns (d loss / d x,) as a tuple without touching any weight's `.grad`; `(grad,) = …` unpacks the 1-tuple |
| `grad.sign()` | −1, 0 or +1 per element |
| `torch.empty_like(x).uniform_(-eps, eps)` | random numbers uniform in [−ε, ε], same shape as x |
| `.clamp(-eps, eps)` | clip each element into [−ε, ε] |
| `model.eval()` / `model.train(flag)` | switch layers like dropout and batch norm to inference behaviour, then restore the previous mode |
| `@torch.enable_grad()` | turns gradients back on even if the caller disabled them; attacks need them |

### 14.4 The code: `_grad_wrt_input` and `pgd` (`src/attacks.py`) **[IMPL]**

```python
 31  def _grad_wrt_input(model, x, y, loss_fn):
 32      x = x.clone().detach().requires_grad_(True)         # ◀ a fresh leaf tensor we can differentiate with respect to
 33      loss = loss_fn(model(x), y)                          # ◀ cross-entropy of the current adversarial batch
 34      (grad,) = torch.autograd.grad(loss, x)               # ◀ d loss / d INPUT, same shape as x: [B, 1, 28, 28]
 35      return grad

 49  def pgd(model, x, y, eps=EPS_DEFAULT, alpha=ALPHA_DEFAULT, steps=STEPS_DEFAULT, random_start=True, loss_fn=None):
     ⋮  # 59–61: docstring; loss_fn defaults to nn.CrossEntropyLoss(); remember the training flag
 62      model.eval()                                         # ◀ attack the inference-mode model
 64      x_orig = x.clone().detach()                          # ◀ the clean images: the centre of every box
 65      if random_start:
 66          adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)   # ◀ a random point inside the box
 67          adv = adv.clamp(0.0, 1.0)                        # ◀ … that is still a valid image
 68      else:
 69          adv = x_orig.clone()
 71      for _ in range(steps):                               # ◀ 40 by default (EPS 0.3, ALPHA 0.01, lines 16–18)
 72          grad = _grad_wrt_input(model, adv, y, loss_fn)
 73          adv = adv.detach() + alpha * grad.sign()         # ◀ one signed step uphill on the loss
 74          # Project back into the eps-ball, then into the valid pixel range.
 75          adv = x_orig + (adv - x_orig).clamp(-eps, eps)   # ◀ clip the CHANGE to ±eps per pixel
 76          adv = adv.clamp(0.0, 1.0)                        # ◀ stay inside [0, 1]
     ⋮  # 78: restore the model's training flag
 79      return adv.detach()
```

| name | holds | toy value (2-pixel model) |
|---|---|---|
| `x`, `y` | a batch of clean images and their labels | x = (0.30, 0.20), y = 0 |
| `eps`, `alpha`, `steps` | budget, step size, iterations | 0.15, 0.04, 4 |
| `x_orig` | untouched clean copy | (0.30, 0.20) |
| `adv` | the adversarial batch being built | (0.26, 0.24) → … → (0.15, 0.35) |
| `grad` | d loss / d adv | (−0.803, +0.401) at the start |

![[d09i-pgd-dbg.svg]] [[d09i-pgd-dbg.html|▶ Step through it]]

![[d09-3-pgd.svg]] [[d09-3-pgd.html|▶ Step through it]]

**What breaks if you change it:**

| change | result |
|---|---|
| project with `x_orig` replaced by the previous `adv` (line 75) | the box drifts with every step; the final perturbation can exceed ε, so "robust accuracy" is measured under a bigger attack than stated |
| drop line 76 | pixels outside [0, 1]: not valid images |
| drop `.detach()` on line 73 | each step's graph chains onto the last; memory grows with every step |
| use `grad` instead of `grad.sign()` | a different (L2-style) attack; under an L∞ budget it's weaker per step |
| normalisation in the data loader | the attack runs at an effective ε ≈ 0.09 (§9.7) |

**The budget audit** (`attacks.py:114–126`): after every batch the code **measures** the largest change and the pixel range, and raises an error if `max_linf > eps + 1e-5` or any pixel left [0, 1]. A projection bug then shows up as a crash, not as inflated robustness.

### 14.5 What the robustness result really says **[REPORT] Table 3, §3.5**

![[d09m-pgd.mp4]]
*Manim video (33 s): PGD steps and the projection on the 2-pixel toy, then your ε-sweep.*

Worst-case accuracy under PGD (PGD-100, 5 restarts):

| model | ε = 0.05 | ε = 0.1 | ε = 0.15 | ε = 0.3 |
|---|---|---|---|---|
| standard, dense | 51.98 ± 9.51 | 2.33 ± 0.90 | 0.01 | 0.00 |
| standard, 13.8% weights (layer-wise) | **82.18 ± 3.92** | 14.74 ± 5.02 | 0.13 | 0.00 |
| adversarially trained, dense | 98.09 | 97.29 | 96.19 | **90.55** |

- The resume's 51.98% → 82.18% is the **ε = 0.05** column, for the **13.8%** model (p = 0.001 **[RAN]**). At ε = 0.1 the gap is 12 points (p = 0.0046).
- **Both** standard models are near 0% by ε = 0.15–0.2 and exactly 0% at the standard ε = 0.3. The report: "The effect is real but it is not a defence." Real robustness comes from adversarial training (90.55% at ε = 0.3).
- **Limitation [REPORT] §7:** the ε-sweep used layer-wise tickets only, so the small-ε gain "might be a first-layer effect rather than a sparsity effect". The report reads it as "the other face of the first-layer damage".

### 14.6 The hostile question: is it gradient masking?

**Gradient masking**: the model isn't more robust, but its gradients have become useless (flat, noisy, saturated), so a gradient attack fails to find adversarial examples that still exist. A sudden robustness gain from a change that wasn't meant to add robustness is the classic warning sign, so the project tested it before believing it **[REPORT] §3.5, §6.1**:

| check | what masking would look like | what was found |
|---|---|---|
| **transfer attack**: craft examples on the dense model, test on the pruned one | transfer works **better** than white-box PGD on the pruned model | transfer **68.15%** accuracy vs white-box **14.10%** at ε = 0.1: white-box is far stronger → no masking |
| accuracy vs ε | stays high as ε grows | decays monotonically to 0 |
| PGD vs FGSM | FGSM (one step) does as well or better | PGD dominates FGSM |
| more steps | PGD-100 weaker than PGD-40 | PGD-100 no weaker |

### 14.7 Two measurement corrections that changed answers **[REPORT] §6**

**PGD-40 with one restart overstated robustness.** Adversarially trained models scored 93.40 ± 0.24% under the attack they were trained with (PGD-40, 1 restart), but **90.63 ± 0.53%** under PGD-100 with 5 restarts (p = 0.0001), lower on all five seeds. More steps stopped helping by PGD-200; more **restarts** kept finding new adversarial examples. Restarts were the binding constraint. Every reported robust number uses the stronger attack.

**Adversarial training of sparse networks needs an ε ramp.** Training at the full ε = 0.3 from epoch 1 collapsed every pruned ticket to a constant prediction: loss stuck at ln 10 = 2.303 (a uniform guess over 10 classes) and all 5 seeds at 11.35%, the frequency of the most common digit. Ramping ε up over the first epochs fixed it:

| recipe | 21.3% weights | 3.75% weights |
|---|---|---|
| SGD 1e-2, ε ramped | 96.9 → 95.0 | 11.2 → **93.2** |
| Adam 1e-4, ε ramped | 84.5 → 74.3 | 55.2 → 54.0 |
| SGD 1e-2, ε = 0.3 from epoch 1 | 11.2 (stuck) | 11.2 (stuck) |

Why: when every example is already adversarial to a network that has learned nothing, there's no useful gradient to start from. Reporting the collapse would have produced a false headline: "pruned networks can't be adversarially trained."

### 14.8 Question bank: adversarial robustness

**L1. What is PGD?**
> [!success]- Answer
> Projected gradient descent on the input: start at (or randomly near) the clean image, repeatedly step by α times the sign of the loss gradient with respect to the input, and after each step project back into the allowed box (each pixel within ε of the original) and into [0, 1]. Robust accuracy is the accuracy on the result.

**L1. Why the sign of the gradient?**
> [!success]- Answer
> Under an L∞ budget each pixel may move independently by up to ε. To first order the loss changes by gradient · perturbation, which is maximised by moving every pixel the full amount in the direction of its gradient's sign.

**L2. Your attack settings?**
> [!success]- Answer
> L∞, ε = 0.3 for the headline column (the Madry MNIST setting), in raw pixel space because normalisation is inside the model. Step size 0.01, random start. The reported numbers are the worst case over PGD-100 with 5 restarts, after PGD-40 with one restart turned out to overstate robustness by about 3 points. The ε-sweep goes from 0.05 to 0.4.

**L2. "Pruning raised robust accuracy from 52% to 82%." Explain exactly.**
> [!success]- Answer
> At ε = 0.05, a small budget, the model pruned to 13.8% kept 82.18% accuracy under PGD against 51.98% for the dense model, p = 0.001, with non-overlapping seed distributions. At ε = 0.1 the gap is 12 points, and by ε = 0.2 both are at zero; at ε = 0.3 both are 0%. So it's a real small-ε effect but not a defence. For actual robustness, adversarial training gives 90.55% at ε = 0.3.

**L3. "That's suspicious. Gradient masking?"**
> [!success]- Answer
> It was the first thing we tested. If the pruned model's gradients were broken, examples crafted on the dense model would transfer better than white-box PGD on the pruned model. They don't: at ε = 0.1 transfer leaves 68.15% accuracy while white-box PGD leaves 14.10%, so white-box is far stronger. Accuracy also decays monotonically with ε to 0, PGD beats FGSM, and PGD-100 is no weaker than PGD-40. Those are the standard sanity checks.

**L3. "Why might pruning genuinely help at small ε?"**
> [!success]- Answer
> In the linear picture, a small input change moves a score by up to ε · ‖w‖₁, and pruning removes most of the weights, so fewer paths add up small perturbations. But here pruning also destroyed most of conv1, and the report flags that the small-ε gain may be a first-layer effect, not a sparsity effect: the ε-sweep only used layer-wise tickets. Running the sweep on globally pruned tickets would separate the two.

**L3. "Can you attack the INT8 model?"**
> [!success]- Answer
> Not directly: a converted INT8 model uses integer kernels with no gradients. The plan was to attack the fake-quantised float graph (the QAT model in eval mode) and label it as a surrogate. The headline table leaves the INT8 robust column empty rather than attack the FP32 model and call it INT8.

---

## 15. Knowledge distillation (short)

### 15.1 From zero

A trained **teacher** model outputs probabilities over the 10 digits. Even when it's sure a digit is a 7, it gives the 1 a bit more probability than the 4. Those "dark knowledge" ratios tell a **student** which mistakes are reasonable. **Distillation** trains the student to match the teacher's **softened** probabilities, as well as the true labels.

**Temperature T** softens: divide the logits by T before the softmax. Logits [4, 2, 0]: T = 1 → [0.87, 0.12, 0.02]; T = 4 → [0.51, 0.31, 0.19].

The loss (Hinton et al. 2015; `PROJECT_PLAN.md` Phase 2.2):

$$\mathcal{L} = \alpha \, T^2 \, \text{KL}\big(\text{softmax}(z_t / T) \,\|\, \text{softmax}(z_s / T)\big) + (1 - \alpha)\, \text{CE}(z_s, y)$$

The T² keeps the soft term's gradients at the same scale as T changes.

### 15.2 In this project **[REPORT] §3.3**

- Teacher: LeNet-5 with **double width** (`LeNet5(width=2)`); student: the pruned 3.75% ticket.
- T and α chosen on **validation** from a 3 × 3 grid (T ∈ {1, 4, 10}, α ∈ {0.3, 0.7, 0.9}).
- Result: clean 98.52 → **98.64**, MNIST-C 69.83 → **71.49**, at identical size and latency.
- Distillation vs standard training at the same budget: +0.17 clean (p = 0.011), above the ~0.09 seed noise.
- The T/α grid "found nothing": the whole grid spans 0.16 points, below what one seed can resolve. The gain is from distilling at all, not from a particular T.

**Q (L2). Why did distillation help the pruned model?**
> [!success]- Answer
> The pruned student has little capacity, so it benefits from a richer target than one-hot labels: the teacher's soft probabilities carry similarity information between digits. It recovered part of the pruning cost, +0.12 clean and +1.66 MNIST-C, at no change in size or latency. The temperature choice didn't matter within the grid: the effect is from distilling at all.

---

## 16. INT8 from zero, QAT, and where the 75.0 KB comes from

### 16.1 Why fewer bits

A float32 weight takes 4 bytes; an int8 takes 1. For weights spread roughly symmetrically around zero, 256 evenly spaced levels are usually enough.

### 16.2 Symmetric quantisation on 4 numbers

Weights w = (0.40, −1.30, 0.07, 0.95). Scale s = max|w| / 127 = 1.30 / 127 = 0.010236. Store q = round(w / s); read back q · s.

| w | w / s | q (1 byte) | q · s | error |
|---|---|---|---|---|
| 0.40 | 39.08 | 39 | 0.3992 | 0.0008 |
| −1.30 | −127.00 | −127 | −1.3000 | 0.0000 |
| 0.07 | 6.84 | 7 | 0.0717 | −0.0017 |
| 0.95 | 92.81 | 93 | 0.9520 | −0.0020 |

![[d09-4-int8.svg]] [[d09-4-int8.html|▶ Step through it]]

- **Error ≤ s/2** per weight. One outlier inflates s, and so every other weight's error.
- **Per-channel** quantisation gives each output channel (each filter) its own scale, so one large filter doesn't coarsen the others. PyTorch's default for conv and linear weights.
- **Affine** quantisation adds a **zero point** z for one-sided ranges (activations after ReLU): q = round(x/s) + z, x ≈ (q − z)s.

### 16.3 PTQ vs QAT, and which this is

- **Post-training quantisation (PTQ):** quantise a trained model, using a little calibration data for activation ranges.
- **Quantisation-aware training (QAT):** insert **fake-quantise** nodes during training: the forward pass rounds as INT8 would, the backward pass lets gradients through as if no rounding happened (the straight-through estimator). The weights learn to sit well on the grid. Then **convert** to real INT8 kernels.

This project used **eager-mode QAT** with `torch.ao.quantization`, real conversion, and **measured** the file and CPU latency **[REPORT] §3.1**. (Not PTQ, as the older note assumed.)

### 16.4 Where the 75.0 KB comes from **[RAN] with a stand-in**

The branch's `quantize.py` isn't on the server, so a stand-in was built: the same LeNet-5, PyTorch's default fbgemm QAT config (per-channel symmetric int8 weights), converted. Its `state_dict` file is **75.03 KiB**, the report's number exactly.

![[d09i-int8-bytes.svg]] [[d09i-int8-bytes.html|▶ Step through it]]

| part | FP32 file (KiB) | INT8 file (KiB) |
|---|---|---|
| weights | 240.1 (61,470 × 4 B) | **60.0** (61,470 × 1 B) |
| per-channel scales + zero points | — | **3.7** (236 channels × 16 B) |
| biases, buffers, activation scales | 0.9 | 1.0 (biases stay float32) |
| file container (one record per tensor) | 4.5 | **10.3** (many more small tensors) |
| **total** | **245.6** | **75.0** |

245.6 / 75.0 = **3.27×** (3.28× in the report is a rounding). The weights alone shrink exactly 4×; the scales and the per-tensor file overhead don't.

### 16.5 It got slower **[REPORT] §3.4**

Batch-1 median CPU latency rose from 0.359 to 0.383 ms, a significant **1.07× slowdown** (p = 0.0005). On a model this small, quantising the input and dequantising the output at each boundary costs more than the cheaper integer arithmetic saves. INT8 bought **size**, not speed. (Latency method, `evaluate.py:48–92`: 1 thread, 100 warm-up runs, median of 1,000 timed runs, because a single scheduler hiccup skews a mean.)

### 16.6 Question bank: quantisation

**L1. How does INT8 quantisation work?**
> [!success]- Answer
> Store each weight as an 8-bit integer q = round(w/s) with a float scale s per channel, s = max|w|/127, and compute with q·s. Rounding error is at most s/2 per weight. Weights become 4× smaller.

**L2. PTQ or QAT, and why?**
> [!success]- Answer
> QAT: fake-quantise during training so the weights adapt to the rounding, then convert to real INT8 kernels. For LeNet at 8 bits PTQ would likely be enough; QAT was used because the pipeline already retrains (pruning, distillation), and it gives the safest accuracy. The accuracy change was not significant.

**L2. Why 3.27× and not 4×?**
> [!success]- Answer
> Only the weights shrink 4× (240 → 60 KiB). The INT8 file also stores a scale and zero point per output channel (3.7 KiB), keeps biases in float32 (~1 KiB), and the file format writes a record per tensor, and a quantised model has many more small tensors (~10 KiB). 60 + 3.7 + 1 + 10.3 = 75.0. I checked that breakdown by converting a stand-in LeNet-5, which came out at 75.03 KiB.

**L3. "Why is the INT8 model slower?"**
> [!success]- Answer
> At batch 1 on a 61k-parameter model the arithmetic is tiny, so fixed costs dominate: quantising the input, dequantising the output, and per-layer requantisation. Integer kernels save arithmetic that barely exists here. On a larger model or larger batch, INT8 usually speeds things up. We measured 1.07× slower, p = 0.0005, and reported it rather than claiming a speedup from the bit width.

**L3. "Pruned to 3.75% and INT8: why isn't the file 3.75% × 75 KB?"**
> [!success]- Answer
> Because neither format stores sparsity: dense tensors store the zeros, in float32 or int8. Combining the two would need a sparse format, e.g. CSR (value + index per non-zero), which the code computes only as a theoretical size. I report the real on-disk size.

---

## 17. LeNet-5: say it out loud

### 17.1 The 30-second version

> "This was a CS725 project on LeNet-5. Iterative magnitude pruning removes 96% of its weights with almost no clean-accuracy loss, but we asked whether that holds under corrupted inputs and attacks. With the standard layer-wise recipe it doesn't: at 3.75% of weights, clean accuracy dropped 0.56 points but corrupted-digit accuracy dropped 3.89, seven times more. We traced it to the first conv layer, which has only 150 weights and was cut to 6. Global pruning at the same sparsity removed the penalty: 0.13 points, p = 0.83. Everything was 5 seeds with validation-only selection."

### 17.2 The 2-minute version

> "The setup is LeNet-5 on MNIST, with MNIST-C for distribution shift and PGD for attacks. Two rules mattered: model selection only on a validation split (the original reproduction had selected on test, which inflated accuracy by about 0.05 points and even manufactured a significant result), and normalisation inside the model so the attack's ε is in pixel space.
>
> We pruned with iterative magnitude pruning, 20% per round with rewinding, and measured clean, MNIST-C and PGD accuracy at every level, with 5 seeds.
>
> Clean accuracy stayed flat to about 14% of the weights, reproducing the lottery-ticket result. But MNIST-C fell faster at every level. At 3.75%: −0.56 clean, −3.89 corrupted. Counting weights showed why: layer-wise pruning removes the same fraction from every layer, so conv1 kept 6 of 150 weights and half its filters were empty.
>
> To separate 'sparse' from 'first layer destroyed', we re-ran with global pruning, same seeds and budget. The corruption penalty disappeared (−0.13, p = 0.83), and global beat layer-wise by 3.76 points. So the recipe, not the sparsity, caused most of the hidden cost.
>
> Two smaller findings: INT8 QAT cut the file 3.3× but was slightly slower on CPU, and pruned models resisted small-ε PGD better, which we checked wasn't gradient masking, but it isn't a defence."

### 17.3 The 5-minute walkthrough (expand each with the section)

1. **Question and setup (45 s):** compression is validated on clean accuracy; is that enough? LeNet-5, MNIST-C, PGD; 5 seeds; Welch + Holm (§8).
2. **Protocol (45 s):** validation-only selection and the 0.05 pp leak; normalisation inside the model (§9.7, §10).
3. **IMP and the schedule (60 s):** masks, 20%/round, rewind; 3.75% derivation (§11).
4. **The finding (60 s):** 7× table; conv1 6/150; global control, p = 0.83 vs 0.019 (§12, §13).
5. **Attacks (45 s):** ε = 0.05 gain, not a defence, gradient-masking checks, PGD-40 vs PGD-100 × 5 (§14).
6. **Compression extras (30 s):** distillation +0.12 clean; INT8 3.27×, slower (§15, §16).
7. **Limits (15 s):** global empties a later layer; ε-sweep only layer-wise; one small network; n = 5.

### 17.4 Whiteboard

```
 LeNet-5: 1×32×32 →conv1(150 w)→ 6×28×28 →pool→ conv3(2,400) → pool → conv5(48,000) → fc6(10,080) → fc7(840)
 IMP:  train → prune 20% (fc7 10%) → rewind → retrain   × 15 rounds = 3.75%
 layer-wise: same fraction per layer → conv1 6/150      global: one threshold → conv1 keeps ~27%
            Δclean   ΔMNIST-C
 layer-wise  −0.56   −3.89 (p=.019)   ← 7×
 global      −0.18   −0.13 (p=.83)
 PGD ε=0.05: 52% → 82% (13.8% weights); ε=0.3: 0% for both  |  INT8 QAT: 245.6 → 75.0 KiB, 1.07× slower
```

### 17.5 Likely follow-ups

| follow-up | where |
|---|---|
| "Walk me through LeNet's shapes." | §9.4 |
| "Why normalisation inside the model?" | §9.7 |
| "How did you select the model?" | §10 |
| "What is IMP; rewind or fine-tune?" | §11.2 |
| "Where does 3.75% come from?" | §11.3 |
| "Layer-wise vs global?" | §11.5–11.6 |
| "What does 7× mean?" | §12.3 |
| "How do you know it's conv1?" | §12.4, §12.6 |
| "What test, what does p = 0.83 mean?" | §13 |
| "Is the robustness gain real?" | §14.5–14.6 |
| "Why 3.27× not 4×; why slower?" | §16.4–16.5 |
| "Did pruning make it smaller or faster?" | §11.7 |
| "What was your part?" | §0.1 |

### 17.6 Claims to avoid

| don't say | why | say instead |
|---|---|---|
| "pruning improved adversarial robustness" (alone) | only at ε = 0.05–0.1, for the 13.8% model; 0% at ε = 0.3 | "at small ε the pruned model resisted PGD much better; it's not a defence" |
| "3.28× smaller" without "on disk, INT8" | pruning alone doesn't shrink the file | "INT8 QAT: 245.6 → 75.0 KiB on disk, 3.27×" |
| "INT8 made it faster" | it was 1.07× slower | "smaller, slightly slower at batch 1" |
| "global pruning fixes it" | it empties a later conv layer to 0.3% | "global is strictly better at 3.5%; both schedules have a pathology" |
| "p = 0.83 proves no difference" | absence of evidence at n = 5 | "no detectable difference over 5 seeds" |
| "96% sparse, so 25× smaller" | file and latency unchanged | "unstructured sparsity needs a sparse format or kernels" |
| "I implemented …" (anything you didn't) | the repo and report carry another name | your actual role (§0.1) |
| "post-training quantisation" | it was QAT | "quantisation-aware training" |

---

# Part C. Mock interview

## 18. Eighty questions with model answers

**How to drill:** read the question, answer out loud in under a minute, *then* open the answer. Mark each strong / OK / weak in [[#My notes and fumbles]]. Answers marked [say yours] depend on CS728 facts that aren't on the server.

### 18.1 Fundamentals (20)

**F1. What does softmax do, and why is it used for attention weights?**
> [!success]- Answer
> It maps any vector of scores to positive numbers that sum to 1: e^xᵢ / Σ e^xⱼ. For attention that gives a proper weighted average over positions, and the exponential sharpens differences, so a clearly better match gets most of the weight.

**F2. What are Q, K and V in one sentence each?**
> [!success]- Answer
> The query is what the current token is looking for; each key is what an earlier token offers to be matched on; each value is what that token contributes if it's attended to. Score = q·k/√d, weights = softmax of the scores, output = weighted sum of the values.

**F3. Why divide by √d?**
> [!success]- Answer
> Dot products of d-dimensional vectors grow in spread like √d. Without scaling, large d gives huge scores, softmax saturates to nearly one-hot, and gradients vanish. Dividing by √d keeps the scores in a reasonable range.

**F4. Why many heads?**
> [!success]- Answer
> Each head can learn a different notion of relevance (previous token, syntax, the attention sink, content lookup) and they run in parallel at the same total cost as one wide head. Retrieval heads are one such specialisation.

**F5. What is the attention sink?**
> [!success]- Answer
> Many heads put a large share of attention on the first token regardless of content, a place to park attention they don't need. It carries no retrieval signal, so any attention-based score must exclude it, which is why my scorer renormalises over tool tokens only.

**F6. Prefill vs decode?**
> [!success]- Answer
> Prefill processes the whole prompt in one parallel pass and fills the KV cache; decode generates one token at a time, each attending over the cached keys and values. The attention matrix at a decode step is [heads, 1, T], which is why my head scores are read there.

**F7. How does an embedding retriever work, and how is the head retriever different?**
> [!success]- Answer
> An embedding retriever encodes queries and documents into vectors with a trained encoder and ranks by cosine similarity; the document vectors can be precomputed. The head retriever needs no training and no second model, but scores tools inside the LLM's forward pass, so nothing is precomputed and the whole list must be in the prompt.

**F8. Recall@k vs precision@k?**
> [!success]- Answer
> With one correct item per query, Recall@k is whether that item is in the top k, averaged over queries. Precision@k is the fraction of the top k that are relevant, so with one relevant item it's at most 1/k. Recall@k is the natural metric for single-answer retrieval.

**F9. What is a convolution, and what is weight sharing?**
> [!success]- Answer
> A small kernel of weights slides over the image; at each position, multiply overlapping pairs and sum to get one output. Weight sharing means the same kernel is used everywhere, so the layer has few weights and detects the same pattern anywhere in the image.

**F10. Output size of a 5 × 5 convolution on 32 × 32, no padding, stride 1?**
> [!success]- Answer
> 32 − 5 + 1 = 28, so 28 × 28 per filter.

**F11. What does cross-entropy measure?**
> [!success]- Answer
> The negative log of the probability the model gives the correct class after softmax. It's 0 for a confident correct answer and large for a confident wrong one. A uniform guess over 10 classes gives ln 10 ≈ 2.303: the value the collapsed adversarial runs got stuck at.

**F12. SGD with momentum?**
> [!success]- Answer
> Each step moves the weights by −lr × a running average of recent gradients (momentum 0.9 means about the last 10 steps), which smooths noise and speeds progress along consistent directions. Here: lr 0.01, momentum 0.9, batch 64.

**F13. Why train, validation and test splits?**
> [!success]- Answer
> Train fits the weights; validation chooses everything else (epoch, hyperparameters); test estimates performance on new data and must be used once, or it becomes part of the selection and its number becomes optimistic.

**F14. What is magnitude pruning?**
> [!success]- Answer
> Set the weights with the smallest absolute values to zero and keep them at zero with a mask, on the assumption that small weights matter least. Iterative versions prune a little at a time and retrain in between.

**F15. State the lottery-ticket hypothesis.**
> [!success]- Answer
> A randomly initialised dense network contains a sparse subnetwork which, trained in isolation from the **same initial values**, reaches the full network's accuracy in a similar number of steps. It's found by iterative magnitude pruning with reset to initialisation; the same mask with new random weights does worse.

**F16. Structured vs unstructured pruning?**
> [!success]- Answer
> Unstructured removes individual weights anywhere, giving irregular sparsity that dense hardware can't exploit. Structured removes whole filters, channels or heads, giving a smaller dense model that's actually faster. This project used unstructured pruning, so neither file size nor latency changed.

**F17. Adversarial example vs corrupted input?**
> [!success]- Answer
> A corruption (blur, noise, fog) is a natural, model-independent shift: MNIST-C tests average-case robustness. An adversarial example is chosen by an optimiser against this specific model to be as harmful as possible within a budget: worst-case robustness. A model can be good at one and terrible at the other.

**F18. FGSM vs PGD?**
> [!success]- Answer
> FGSM is one step of size ε along the gradient's sign. PGD takes many smaller steps, projecting back into the ε-box after each, usually from a random start and with restarts. PGD is much stronger; FGSM is a useful sanity check (PGD must beat it).

**F19. What is knowledge distillation?**
> [!success]- Answer
> Train a small student to match a larger teacher's softened output probabilities (logits divided by a temperature T), plus the usual label loss. The soft probabilities carry information about which classes are similar, which helps a low-capacity student.

**F20. What is a p-value?**
> [!success]- Answer
> The probability of a result at least as extreme as the observed one if the null hypothesis (no difference) were true. Small p is evidence against the null; a large p is not evidence for it, only an absence of evidence against it, especially with few samples.

### 18.2 CS728 in depth (15)

**C1. Explain the project end to end.**
> [!success]- Answer
> Tool selection for an LLM agent without training a retriever. I located retrieval heads by Hit Count on probe queries (a head hits when its argmax lands inside the correct tool's description), kept the top 10, and at test time ranked tools by those heads' attention mass on each tool's tokens, read at the first answer token. Recall@1 rose from 1.3% to 40.7%, Recall@5 from 25.5% to 66.9%.

**C2. Which model and dataset?**
> [!success]- Answer
> [Say yours.] Include the number of layers and heads (so the interviewer can picture "10 of N"), how many tools per prompt and how many test queries.

**C3. What exactly was the baseline?**
> [!success]- Answer
> [Say yours, in one sentence.] Then: the gap is mostly selection. Most heads attend by position or to the sink, so an unselected aggregate mostly measures position; picking the few content-driven heads is what moves Recall@1.

**C4. Why does Recall@5 (25.5%) far exceed Recall@1 (1.3%) for the baseline?**
> [!success]- Answer
> The baseline often puts the right tool near the top but almost never first, the signature of a positional bias: a few positions (the first or last tools) systematically take rank 1. That's also why removing position-driven heads improves Recall@1 most.

**C5. Why the first answer token?**
> [!success]- Answer
> It's when the model decides what to say, so its retrieval heads point at what it will use; and it's a single-query decode step, where the attention matrix is [heads, 1, T] and can be read on the eager path cheaply. An eager prefill would need [heads, T, T] per layer.

**C6. How did you make sure the attention weights were available?**
> [!success]- Answer
> SDPA and FlashAttention never materialise the weights. In my LongContext_RAG code a context manager flips the model's attention implementation to eager for the one decode step and registers forward hooks on every attention module, which copy the weights out with detach; a `finally` block always restores the fast path. [Say what CS728 used, e.g. `output_attentions=True` on an older transformers version.]

**C7. Why top 10?**
> [!success]- Answer
> [Say how you chose it.] The principle: choose k on held-out queries; too few heads makes it brittle, too many dilutes it with non-retrieval heads. Retrieval heads are under 5% of heads, so k in the tens is the natural range.

**C8. What does "32 heads and 3 layers" mean?**
> [!success]- Answer
> [Say yours precisely.] If it means the aggregation considered heads from 3 layers, 32 heads each, say that and how it relates to the top 10.

**C9. How would you measure lost in the middle?**
> [!success]- Answer
> Bucket test queries by the correct tool's position in the list (start, middle, end), or shuffle the list and place the correct tool at each position in turn, and report Recall@1 per position for the baseline and the head retriever. Mitigation = a flatter curve, not just a higher one.

**C10. Sum, mean or max over a tool's tokens?**
> [!success]- Answer
> [Say yours.] Sum favours long descriptions; mean corrects length but rewards single spikes in short ones; max uses only the best token. Choose on validation; with descriptions of very different lengths, mean or max is usually safer.

**C11. What happens if two tools have near-identical descriptions?**
> [!success]- Answer
> Attention mass splits between them and the ranking between the two becomes a coin flip; Recall@5 stays fine but Recall@1 suffers. Embedding retrievers have the same problem. Tool descriptions that state distinguishing details, or a second-stage re-ranker, help.

**C12. Does this scale to 1,000 tools?**
> [!success]- Answer
> Not as is: all descriptions must fit in context and be prefilled per query, and attention spreads thinner over more candidates. I'd shortlist with an embedding retriever (top 20–50) and re-rank the shortlist with the heads; my later LongContext_RAG project does that cascade for passages.

**C13. Are retrieval heads the same across tasks?**
> [!success]- Answer
> Wu et al. found retrieval heads are largely universal within a model (present in base models, stable across long-context training and tasks). But Hit Count selects for my task specifically; I'd expect strong overlap with needle-in-a-haystack retrieval heads, and that's a cheap check: compare my top 10 with the heads ranked by the copy rule.

**C14. Could you causally verify the heads matter?**
> [!success]- Answer
> Yes: mask the selected heads (zero their output) and measure the model's own tool-calling accuracy, against masking the same number of random heads, as Wu et al. did for needle retrieval. My LongContext_RAG repo has ablation code for exactly that (`heads/ablate.py`).

**C15. What would you change with more time?**
> [!success]- Answer
> Disjoint probe and test tools; choose k and the aggregation on validation; a per-position recall curve for the lost-in-the-middle claim; an embedding-retriever baseline; and the cascade: embedding shortlist + head re-rank.

### 18.3 LeNet-5 implementation and results (20)

**L1. What was the research question?**
> [!success]- Answer
> Whether compression that's free on clean MNIST is also free under distribution shift (MNIST-C) and adversarial attack (PGD), for LeNet-5, measured with 5 seeds and validation-only selection.

**L2. Describe the model.**
> [!success]- Answer
> LeNet-5 with tanh, average pooling and dense C3 connections, normalisation and 2-pixel padding as layer 0 so it takes raw 28 × 28 pixels; 61,706 parameters, 78% of the weights in conv5.

**L3. What were the training settings?**
> [!success]- Answer
> SGD, lr 0.01, momentum 0.9, batch 64, 20 epochs for every condition; best epoch chosen on a fixed 5,000-image validation split; 5 seeds; every run writes a JSON record with config hash, seed, git SHA and machine spec.

**L4. What was wrong with the original reproduction?**
> [!success]- Answer
> It chose its reported epoch by test accuracy and set no seed. Re-run properly, the leak was about +0.05 pp, the same size as the augmentation effect it was used to show; at 20 epochs augmentation was significant only under the leaky protocol (p = 0.023 vs 0.115).

**L5. Describe the pruning schedule.**
> [!success]- Answer
> Iterative magnitude pruning: 20% of surviving weights per round, the output layer at half rate, 20 rounds, rewinding survivors between rounds and retraining 20 epochs with masks fixed. Checkpoints at every level; the headline is round 15, 3.75% of weights.

**L6. What is the headline result?**
> [!success]- Answer
> At 3.75% of weights with layer-wise IMP: clean 98.52% (−0.56 from dense) but MNIST-C 69.83% (−3.89), a 7× larger corruption cost (p = 0.019). Global pruning at 3.52%: clean 98.90%, MNIST-C 73.59%, −0.13 from dense, p = 0.83.

**L7. Why did conv1 starve?**
> [!success]- Answer
> It has only 150 weights, and layer-wise pruning removes the same 20% from it each round. After 15 rounds 6 weights remain and 3 of its 6 filters are entirely empty, so the first-layer feature detectors no longer exist.

**L8. Why does global pruning spare conv1?**
> [!success]- Answer
> One threshold over all weights, and conv1's weights are larger: PyTorch initialises with bound 1/√fan-in, and conv1's fan-in is 25 against conv5's 400, so its weights start about 4× bigger. Global pruning removes conv5's weights first; conv1 keeps 26.7% at round 20.

**L9. What does "matched sparsity" mean here?**
> [!success]- Answer
> The same round of each schedule: 15 rounds is 3.75% layer-wise (because the output layer is pruned at half rate) and 3.52% global. Global is actually slightly sparser, and still better.

**L10. What did MNIST-C per-corruption numbers show?**
> [!success]- Answer
> The code records all 15 and the worst one, because the mean hides which corruptions break which models. [If you know which corruptions drove the drop, say them; spatial ones like shear, scale and translate are where a damaged first layer and the augmentation result point.]

**L11. Did distillation help?**
> [!success]- Answer
> Yes, modestly: the pruned student went from 98.52 to 98.64 clean and 69.83 to 71.49 MNIST-C, at the same size and latency. Distillation vs standard training was +0.17 clean, p = 0.011. The temperature/α grid made no detectable difference.

**L12. What did INT8 give?**
> [!success]- Answer
> 245.6 → 75.0 KiB on disk (3.27×) via QAT, no significant accuracy change, and a 1.07× slower batch-1 CPU latency (p = 0.0005).

**L13. What did adversarial training give, and what went wrong at first?**
> [!success]- Answer
> Dense adversarially trained LeNet reached 90.55% at ε = 0.3 (PGD-100 × 5). At first, training sparse tickets at full ε from epoch 1 collapsed to constant predictions (loss ln 10, 11.35% accuracy on every seed); ramping ε over the first epochs fixed it, giving 93.2% for the 3.75% ticket in one recipe.

**L14. Why was PGD-40 with one restart a problem?**
> [!success]- Answer
> It overstated robustness of adversarially trained models: 93.40% vs 90.63% under PGD-100 with 5 restarts, lower on all five seeds. A step-and-restart sweep showed steps saturate by 200 but restarts keep finding examples, so restarts were the binding constraint.

**L15. How many runs and how were tables produced?**
> [!success]- Answer
> 260 recorded runs (plus 27 failed ones kept), each a JSON record; every table and figure is regenerated from those files by a script, never hand-edited, so a number can always be traced to a run.

**L16. What does the significance table contain?**
> [!success]- Answer
> 59 Welch tests in 13 families, Holm-corrected within each family; 31 survive. The ones that don't are listed too: lottery ticket vs fine-tuning, the KD grid, the INT8 accuracy drop, adversarial vs standard MNIST-C at pruned levels.

**L17. How was latency measured?**
> [!success]- Answer
> CPU, batch 1, one thread, 100 warm-up iterations, median of 1,000 timed runs (median because one scheduler hiccup skews the mean), with the machine spec recorded alongside. Also batch-128 throughput.

**L18. What are the main limitations?**
> [!success]- Answer
> Global pruning empties a later conv layer to 0.3%, so neither schedule is clean; the pipeline and ε-sweep experiments used layer-wise tickets only; tickets come from standard training even when evaluated under other conditions; one small architecture on one dataset; n = 5 limits small effects.

**L19. Which result are you least sure of?**
> [!success]- Answer
> The small-ε robustness gain from pruning. It's statistically solid and not gradient masking, but it was measured only on layer-wise tickets, so it may come from the damaged first layer rather than from sparsity. A global-ticket ε-sweep would tell.

**L20. One sentence: what's the takeaway?**
> [!success]- Answer
> The standard layer-wise pruning recipe carries a hidden corruption-robustness cost that clean validation can't see, and it comes from how weights are selected, not how many are removed: global pruning at the same sparsity removes it.

### 18.4 Hostile follow-ups (15)

**H1. "LeNet on MNIST is a toy. Why should I care?"**
> [!success]- Answer
> Because it's small enough to do properly: 5 seeds, every variant, a control experiment, 59 corrected tests. The finding, that uniform per-layer pruning starves small layers and that clean accuracy hides it, is a mechanism, and the same arithmetic applies to any network with a small first layer. I wouldn't claim the exact numbers transfer; I'd claim the check does.

**H2. "Your 7× is a ratio of two small numbers."**
> [!success]- Answer
> Fair, so I quote both: −0.56 clean and −3.89 MNIST-C. The MNIST-C drop alone is 3.4 standard errors from zero (p = 0.019). The ratio is there to make one point: validating on clean accuracy would call this compression free.

**H3. "p = 0.83 with 5 seeds proves nothing."**
> [!success]- Answer
> Agreed that it doesn't prove equality. What it shows is that the 3.89-point penalty, which was clearly detectable with the same 5 seeds (p = 0.019), is gone. And the head-to-head comparison is positive evidence: global beats layer-wise by 3.76 points, p = 0.021.

**H4. "Isn't the global result just because it's at 3.52% instead of 3.75%?"**
> [!success]- Answer
> That goes the wrong way for the objection: 3.52% is sparser, so if anything global is handicapped, and it still wins on both clean and MNIST-C.

**H5. "Global pruning just destroys a different layer. So what did you learn?"**
> [!success]- Answer
> That which layer you destroy matters a lot: losing conv1 costs corruption robustness, while at 3.5% global's damage doesn't. Both schedules have a pathology at the extreme, which is why I'd move to sensitivity-scaled per-layer rates; but at the sparsity people actually stop at, global is strictly better.

**H6. "Couldn't more retraining epochs fix the layer-wise model?"**
> [!success]- Answer
> Not without breaking the comparison: every condition had the same 20-epoch budget, a rule the project adopted after finding that augmentation's effect depended on budget. And conv1 with 3 of 6 filters empty has no capacity to recover those features however long it trains.

**H7. "Your PGD gain is gradient masking."**
> [!success]- Answer
> We tested that: transfer from the dense model leaves 68.15% accuracy at ε = 0.1, while white-box PGD on the pruned model leaves 14.10%. If gradients were masked, transfer would be the stronger attack. Accuracy also decays monotonically with ε, PGD beats FGSM, and more steps don't weaken the attack.

**H8. "Then pruning is a robustness method?"**
> [!success]- Answer
> No. The gain exists only at ε = 0.05–0.1; by ε = 0.2 both models are at zero, and at the standard ε = 0.3 both score 0%. Adversarial training gives 90.55% there. I describe it as a side effect, probably tied to first-layer damage, not a defence.

**H9. "3.28× is wrong: 245.6 / 75.0 is 3.27."**
> [!success]- Answer
> Right, 3.27; the 3.28 is a rounding in the report. The more important point is that it's measured on disk, not derived from bit width, and it's less than 4× because of per-channel scales and per-tensor file overhead.

**H10. "INT8 slower? Then why quantise?"**
> [!success]- Answer
> For size: 3.3× less storage and memory traffic, which matters on a device. On this tiny model at batch 1 the fixed quantise/dequantise overhead outweighs the cheaper arithmetic. I'd expect a speedup on a bigger model or batch, but I'd measure it rather than assume it.

**H11. "Did you write this code?"** (LeNet)
> [!success]- Answer
> [Answer truthfully, per §0.1.] Part 1 was our five-person team's reproduction; say what your part was in Part 2. Don't claim the repository or report as solely yours if they aren't.

**H12. "1.3% → 40.7%: you picked a weak baseline to look good."** (CS728)
> [!success]- Answer
> [Name the baseline.] The comparison isolates one design choice, selecting heads versus not, and that's what it's meant to show. The fair external comparison is a trained embedding retriever, which I'd expect to beat 40.7% top-1; my claim is a zero-training retriever, best used as a re-ranker.

**H13. "Attention isn't explanation. Why trust attention weights?"**
> [!success]- Answer
> I don't use them as an explanation; I use them as a feature and check it by its outcome, retrieval accuracy on held-out queries. Wu et al. also verified causally that masking retrieval heads breaks retrieval while masking random heads doesn't, so these specific heads' weights do track what the model uses.

**H14. "Your Hit Count heads were chosen on the same queries you report?"**
> [!success]- Answer
> [Say what you did.] If they overlapped: the risk is limited (choosing 10 heads is a small model), but the honest fix is disjoint probe and test sets, and I'd rerun that way.

**H15. "What would make you abandon the LeNet conclusion?"**
> [!success]- Answer
> If layer-wise pruning with conv1 exempted still showed the corruption penalty, the conv1 mechanism would be wrong (though the schedule result would stand). And if global pruning on a larger network showed the same 7× gap, the result would be specific to LeNet's shape.

### 18.5 Debugging and methodology (10)

**D1. A clean-trained model shows 80% PGD accuracy at ε = 0.3. What do you check?**
> [!success]- Answer
> That's a bug until proven otherwise; the expected value is ~0%. First, normalisation: is the attack in pixel space? With normalisation in the loader, ε = 0.3 is really ~0.09. Then the projection (is the change clipped around the clean input?), random start, the loss sign, `model.eval()`, and gradient masking (transfer, FGSM vs PGD, accuracy vs ε). The repo's budget audit measures max |change| and the pixel range every batch.

**D2. All five seeds of a run give exactly 11.35%. What happened?**
> [!success]- Answer
> The model predicts one constant class: 11.35% is the share of the most common digit in the test set, and the loss sits at ln 10. It's collapsed training, here adversarial training of sparse tickets at full ε from epoch 1. Fix: ramp ε up over the first epochs.

**D3. Pruning makes accuracy collapse after the first round. Likely cause?**
> [!success]- Answer
> The mask is wrong: applied to the wrong tensors, inverted (keeping the small weights), not re-applied after optimizer steps, or the quantile computed over all weights including already-pruned zeros. IMP's first rounds should leave clean accuracy flat on LeNet.

**D4. A smoke run's numbers appear in your results table. How do you prevent that?**
> [!success]- Answer
> The repo has two guards: smoke runs write to `results/smoke/` instead of `results/raw/`, and the table builder rejects any record whose config has `smoke` set. One guard can be bypassed by accident; two independent ones make the silent corruption much less likely.

**D5. Two numbers differ by 0.1 pp. Is it real?**
> [!success]- Answer
> Compare with the seed noise: the standard deviation across seeds is about 0.09 pp, so with 5 seeds the standard error of a difference is about 0.06. A 0.1 gap is under 2 standard errors: run Welch's test, and treat it as unresolved unless it survives correction.

**D6. Your latency numbers vary run to run. What do you change?**
> [!success]- Answer
> Pin threads (one), warm up, report the median of many iterations rather than the mean, record the machine and library versions, and compare conditions on the same machine in the same session. The repo does all of these.

**D7. Why is the train/val split seed fixed separately?**
> [!success]- Answer
> So all seeds share the same validation images; otherwise differences between configurations would mix in which images happened to be held out.

**D8. How do you know your pruning implementation is right?**
> [!success]- Answer
> It must reproduce the lottery-ticket behaviour: clean accuracy flat down to roughly 10–15% of weights. Here it was flat to 13.8%. Also check the counts per layer against the schedule (they match 13.8%, 3.75%, 1.30% exactly) and that pruned weights stay exactly zero after training.

**D9. Your CS728 retriever ranks the first tool first for almost every query. Diagnosis?**
> [!success]- Answer
> Positional heads or the sink are dominating: check that the sink and the query tokens are excluded and rows renormalised, that the selected heads aren't positional (their argmax should move with the correct tool's position across probes), and shuffle tool order to confirm.

**D10. The eager capture runs out of memory on long prompts. Why, and the fix?**
> [!success]- Answer
> Probably the prefill also ran eagerly, building a [heads, T, T] matrix per layer (~4 GB per layer at 8k), or the captured weights kept their autograd graph. Fix: prefill with SDPA and switch to eager only for the single decode step, and `detach()` the captured weights, as `capture.py` does.

---

## Sources (optional: everything needed is summarised above)

- Wu et al. (2024), *Retrieval Head Mechanistically Explains Long-Context Factuality*: §3.1.
- Liu et al. (2023), *Lost in the Middle: How Language Models Use Long Contexts*: §6.3.
- LeCun et al. (1998), *Gradient-Based Learning Applied to Document Recognition*: §9.
- Frankle & Carbin (2019), *The Lottery Ticket Hypothesis*; Frankle et al. (2020), *Linear Mode Connectivity and the Lottery Ticket Hypothesis*; Han et al. (2015), *Learning both Weights and Connections*: §11.2.
- Mu & Gilmer (2019), *MNIST-C*: §12.1.
- Madry et al. (2018), *Towards Deep Learning Models Resistant to Adversarial Attacks*; Goodfellow et al. (2015), *Explaining and Harnessing Adversarial Examples*: §14.
- Hinton et al. (2015), *Distilling the Knowledge in a Neural Network*: §15.

## My notes and fumbles

- CS728 facts (§1.3): model = ; dataset = ; tools per prompt = ; baseline = ; probe set = ; "32 heads and 3 layers" = ; lost-in-the-middle measured? 
- LeNet: my role in Part 2 = 
-
