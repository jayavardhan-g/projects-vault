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
| CS728 code and report | `/data/jayavardhan/projects/interview_materials/cs728/` (from your upload `PA3_CS728.zip`, 2026-11-17): `utils.py`, `run2.py`, `code3.py`, `run3.py`, `data/`, `Programming Assignment 3.pdf` | complete for Parts 2–3; the Part 1 code (BM25, MiniLM, UAE) is not in the zip |
| Retrieval-head code (background only) | `/data/jayavardhan/projects/LongContext_RAG/` (your later repo, `main`, `bfb2ec9`) | Wu et al.'s strict copy rule, used in §3 as background |

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
| **[REPORT]** | a number or statement in a report: `fml_report.pdf` (LeNet) or `Programming Assignment 3.pdf` (CS728) | 98.52 ± 0.14% clean at 3.75% of weights |
| **[RAN]** | checked by running or recomputing it | the p = 0.83 recomputed from the report's own numbers; the CS728 token spans checked with the real tokenizer |
| **[THEORY]** | what the maths or a hand count predicts | a linear score can drop by at most ε‖w‖₁ |
| **[RECON]** | reconstructed from the code or the docs; not documented | the LeNet IMP outer loop (§11) |
| **[MISSING]** | not on the server; get it before the interview | the LeNet compression branch, never pushed |
| **[FUTURE]** | not done; a possible next step | the CS728 position curve for the selected heads |

**Animations.** Each animated figure has a step-through player: click **▶ Step through it** under it (← / → step, space plays). New for this note: `d09i-*` (from `Days/figures/_gen/d09figs.py`) and four Manim videos `d09m-*` (from `Days/figures/_gen/manim_d09.py`); the older `d09-*` figures are reused. `d09r-*` are images taken from your CS728 report.

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
| CS728: Recall@1 **1.3% → 40.7%**, Recall@5 **25.5% → 66.9%** | ✓ **[REPORT]** p. 9 (0.0132 → 0.4068, 0.2554 → 0.6694). But 1.3% is **all 512 heads averaged**, which is chance level (1 in 100 tools). On its own the line reads as an inflated baseline. Your report also has BM25 18.5%, MiniLM 38.8% and **UAE-Large-V1 61.6%**, which beats the heads. | "Selecting 10 of Llama-3.2-1B's 512 heads raised Recall@1 from 1.3% (all heads) to 40.7%, beating BM25 (18.5%) and MiniLM (38.8%) without fine-tuning." (§6.2) |
| "aggregating retrieval-specialized heads across **32 heads and 3 layers**" | ✗ matches nothing. The model has 16 layers × 32 heads = 512; the best K was **10** heads, spread over layers 2–10 (§4.5). | "the best 10 of 512 attention heads, chosen with 200 labelled queries" |
| "Mitigated the **lost-in-the-middle** effect" | ✗ The report **measured** the U (attention to the correct tool vs its position, all heads, p. 4) but never measured it for the selected heads. | "measured a U-shaped position bias in attention-based ranking" (§6.4) |
| (not on the resume, but you'll be asked) | K = 10 / 20 / 30 was compared on the **test** set; 40.7% is the best of three (§4.4). The project was a 3-person team (G Jayavardhan, Pavan Kumar, Ajay Rangoji). | Say both before you're asked; say "we". |
| LeNet: **3.75%** of weights, **98.52%** clean, **69.83%** MNIST-C, validation split, **5 seeds** | ✓ **[REPORT]** Table 2, row "Standard, 3.75% weights" (layer-wise pruning, standard training). | As written. |
| "**7× larger** corruption loss … **3.89 pp** … traced to a starved conv1" | ✓ 7× = 3.89 (MNIST-C drop) ÷ 0.56 (clean drop) **[REPORT]** Table 5. conv1 keeps 6 of 150 weights **[REPORT]** Table 4, **[RAN]**. | Say what the 7× compares: "the corruption drop was 7× the clean drop". |
| "global pruning at matched sparsity closed the gap to **0.13 pp (p = 0.83)**, a **3.76 pp** gain" | ✓ **[REPORT]** Table 5; both p-values **[RAN]** (§13). "Matched" = 3.52% global vs 3.75% layer-wise (they are 15 rounds of two schedules, §11.3). | Add "Welch's t-test, 5 seeds" if asked. |
| "Improved PGD robustness from **51.98% to 82.18%** through iterative magnitude pruning" | Numbers ✓ **[REPORT]** Table 3, **but** at **ε = 0.05** (not the 0.3 used everywhere else), for the **13.8%**-weights model (not the 3.75% one). Both models fall to ~0% by ε = 0.2. The report says it "is not a defence". | "At a small budget (ε = 0.05), the pruned model resisted PGD far better (52% → 82%); at the standard ε = 0.3 both are at 0%." |
| "reducing model size **3.28×** (245.6 KB to 75.0 KB) via **INT8** quantization, at a cost of just **0.17 pp**" | 245.6 / 75.0 = **3.27** (the report rounds to 3.28). It is **QAT** (quantisation-aware training), not post-training. Latency got **1.07× worse** (p = 0.0005). The INT8 accuracy drop is listed as **not significant** (§5 of the report). Table 2 shows IMP + KD + INT8 at 98.65% vs IMP + KD at 98.64%, so know which two rows the 0.17 compares. | "3.3× smaller on disk via INT8 QAT, with no significant accuracy change, but slightly slower on CPU at batch 1." |

### 0.3 Two corrections to the older Day 9 note

- **The lottery-ticket paper (2019) resets survivors to their initial values (step 0).** "Rewinding" to an early training step came later (Frankle et al. 2020). This project ran **both**, plus plain fine-tuning: the two rewinds were indistinguishable; fine-tuning without reset was 6.73 points worse at 1.30% of weights, but not significant at 5 seeds (p = 0.062 raw, 0.174 after Holm) **[REPORT]** §3.2.1.
- **The p = 0.83 comes from Welch's t-test** (two independent groups of 5 seeds), not a paired test **[REPORT]** §5, **[RAN]**.

---

# Part A. CS728: picking a tool with the LLM's own attention

## 1. The project on one page

### 1.1 The problem, in plain words

An LLM **agent** can call tools: a PDF converter, a flight search, a stock-price API. The prompt lists each tool's name and description. For each user request, something has to pick the right tool.

The usual answer is a **retriever**: an embedding model, trained on query–passage pairs, that turns each description and each query into a vector and picks the closest one. That needs a second model trained for the job.

CS728 Programming Assignment 3 asks: **does the LLM already know where to look?** Put all 100 tool descriptions and the query in one prompt, run the model once, and read its **attention**: which tool's tokens do the query's tokens look at? Most attention heads look at other things (position, grammar). If we can find the few heads that look at the right tool, their attention *is* a retriever, with no weight changed.

The assignment had three parts, and your report covers all three:

| part | what | your result |
|---|---|---|
| 1 | standard retrievers: BM25, MiniLM, UAE-Large-V1 | 18.5%, 38.8%, 61.6% Recall@1 |
| 2 | attention of **all** heads as a retriever; does position matter ("lost in the middle")? | 1.3% Recall@1; a U-shaped position curve |
| 3 | attention of **selected** heads only ("retrieval heads"), chosen on training queries | 40.7% Recall@1 with the best 10 heads |

> [!quote] Say this
> "In CS728 we used an LLM's attention as a tool retriever. With Llama-3.2-1B reading 100 tool descriptions and a query, averaging all 512 attention heads ranks the right tool first only 1.3% of the time, about chance, because attention mostly follows position. We scored each head on 200 labelled training queries, counting how often it gave the correct tool the most attention, and kept the best 10. Those 10 heads alone reach 40.7% Recall@1 and 66.9% Recall@5, better than BM25 and MiniLM, with no weights trained. A proper embedding model, UAE-Large-V1, still wins with 61.6%."

### 1.2 Setup, checked against the code **[IMPL] [RAN]**

Code and report: `/data/jayavardhan/projects/interview_materials/cs728/` (copied from your upload `PA3_CS728.zip`, 2026-11-17). Report: `Programming Assignment 3.pdf`, 9 pages, authors **G Jayavardhan (25M0807), Pavan Kumar (25M0803), Ajay Rangoji (25M0802)**.

| item | value | where |
|---|---|---|
| model | `meta-llama/Llama-3.2-1B-Instruct`, fp16, eager attention, `output_attentions=True` | `run2.py:96, 107`, `utils.py:11–22` |
| size | **16 layers × 32 query heads = 512 heads**; 8 key/value heads (grouped-query attention); head dim 64 | config **[RAN]** |
| data | **MetaTool**: the 100 most frequent tools; 5,000 test and 1,500 training queries, disjoint | `utils.py:86–112`; overlap 0 **[RAN]** |
| head selection | first **200** training queries (`--train_samples 200`) | `run3.py:88, 103–110` |
| prompt | ≈ **2,980 tokens** (2,970–3,003): instruction, 100 tools, the query | **[RAN]** on 30 test prompts |
| one tool | `tool_id: <name>\ntool description: <text>`: 13 to 69 tokens, mean 28 | `utils.py:37`; **[RAN]** |
| tool order | **reshuffled for every query** (seed 64) | `run2.py:139–140`, `run3.py:132–133`, `code3.py:34–35` |
| metric | Recall@1 and Recall@5 over the 5,000 test queries | `run3.py:173–187` |
| queries per tool | very uneven: 16 to 376 test queries per tool | **[RAN]** |

Two checks I ran with the real tokenizer (CPU only, no model):
- **The tool spans are exact.** `PromptUtils` computes each tool's token positions by tokenizing pieces separately (§2.7). Decoding those positions from the full prompt gave back every tool's text exactly: **0 mismatches in 3,000** (30 prompts × 100 tools).
- **The query span is right.** Token 2929 is `Query`, token 34192 is `Correct`; the span between them is the query text plus its newline. The comment in `run3.py:78` calls 34192 "tool_id"; that comment is wrong, the code is right.

### 1.3 What is in each file

| file | what | who wrote it |
|---|---|---|
| `utils.py` | load the model; build the prompt and the token span of every tool; load the data | course skeleton (prompt format, spans) |
| `run2.py` | Part 2: score tools with **all** heads; Recall@k; the position plot | skeleton + the team's TODO bodies |
| `code3.py` | Part 3: `select_retrieval_heads`, the Hit Count (and two commented-out alternatives) | the team (a `TODO 3` in the skeleton) |
| `run3.py` | Part 3: score tools with the **selected** heads; Recall@k | skeleton + the team's `TODO 2/3` bodies |
| `data/` | `tools.json` (100 tools), `train_queries.json`, `test_queries.json` | course |
| report PDF | all three parts, all numbers | the team of three |

"Who wrote it" is read off the `TODO` and `Do NOT change` markers in the files. The Part 1 code (BM25, MiniLM, UAE) is **not** in the zip; its numbers are **[REPORT]** only.

> [!warning] Your share of the work
> This was a three-person assignment and the files carry no record of who wrote what. In an interview say "we" for the project and "I" only for the parts you did. Know all of it anyway: you will be asked about every line.

### 1.4 The ten questions about the whole project

| question | answer | label |
|---|---|---|
| What problem? | Pick the right tool for a query from 100 tool descriptions in the prompt. | |
| Which model and data? | Llama-3.2-1B-Instruct (16 × 32 = 512 heads); MetaTool, 100 tools, 5,000 test queries. | [IMPL] |
| How do you score a tool? | Mean attention from the query's tokens to the tool's tokens, averaged over heads. | §5, [IMPL] |
| Baseline? | The same score with **all 512 heads** averaged: Recall@1 1.3%, Recall@5 25.5%. Near chance for Recall@1 (1%). | §6.2, [REPORT] |
| How did you choose heads? | **Hit Count**: on 200 labelled training queries, count how often each head gives the correct tool its highest score; keep the top K. | §4, [IMPL] |
| Result? | Top 10 heads: Recall@1 **40.7%**, Recall@5 **66.9%**. | [REPORT] |
| Against real retrievers? | Beats BM25 (18.5%) and MiniLM (38.8%); loses to UAE-Large-V1 (61.6%). | [REPORT] |
| What about position? | With all heads, attention to the correct tool is ~6× higher in the first 5 places than in the middle: a U shape. | §6.4, [REPORT] |
| Weak spots? | K was compared on the test set; the selected heads' position curve was not measured; one prompt of 3,000 tokens per query with eager attention. | §4.4, §6.4 |
| What next? | Choose K on held-out training queries; measure the heads' position curve; use the heads to re-rank an embedding shortlist. | [FUTURE] |

---

## 2. From zero: tokens, attention and heads

### 2.1 Tokens and positions

The model never sees letters. A **tokenizer** cuts the text into pieces (**tokens**) and gives each piece an integer id. Each token has a **position**: 0, 1, 2, … in the prompt. In our toy prompt:

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| token | `<s>` | weather | api | flight | search | stock | price | calc | math | fly | Paris |
| part | start | tool A | tool A | tool B | tool B | tool C | tool C | tool D | tool D | query | query |

`<s>` is a special start-of-text token. Each tool's description is a **span**: tool B is positions 3–4, written (3, 5) in Python's "start, one past the end" style. In CS728 a tool is 13 to 69 tokens and the whole prompt about 2,980.

### 2.2 Attention: one row of weights

Each layer of a Transformer lets every token **look at** earlier tokens and mix in what it finds. For one token, one attention head does three things:

1. **Score** every earlier position: score = (this token's *query* vector · that position's *key* vector) / √d. (Query and key are two learned projections of each token's vector; d is their length, 64 in Llama-3.2-1B. Day 1 §5 builds these from scratch.)
2. **Softmax** the scores: e^score for each, divided by the total, so the weights are positive and sum to 1.
3. **Mix**: the output is the weighted average of the positions' *value* vectors.

The row of weights from step 2 says where the head is looking.

![[d09i-attn-row.svg]] [[d09i-attn-row.html|▶ Step through it]]

On the toy: the score 3.0 at position 3 becomes e³ = 20.09; the total is 48.62; so the weight is 20.09 / 48.62 = **0.413**, the largest. This head, reading from the query token "Paris", is looking mostly at "flight", inside tool B.

### 2.3 The attention matrix, and why only the query rows matter

Do step 1–2 for **every** token and stack the rows: an N × N matrix per head, where row i is "where token i looks". It is **causal**: token i only sees positions 0…i, so the upper triangle is 0.

In the CS728 prompt the query comes **after** all 100 tools. So the rows of the query's tokens are the only rows that can see every tool, and they are exactly the question "which tool does the query look at?". CS728 reads those rows and nothing else.

### 2.4 A head, a layer, and how many there are

- A **head** is one independent copy of steps 1–3, with its own query projection, so it learns its own way of scoring. Each makes its own matrix.
- A **layer** runs several heads side by side, then a small MLP.
- Llama-3.2-1B has **16 layers × 32 heads = 512 heads**.
- **Grouped-query attention:** it has only 8 key/value heads; each is shared by 4 query heads. The 4 still have different query projections, so they still make 4 different attention patterns. For CS728 that means 512 distinct rows per token.

We name a head by its layer and number, both counted from 0: **L8H19** is head 19 of layer 8 (your best head).

### 2.5 The attention sink

Many heads put a lot of weight on position 0 (`<s>`) whatever the content, as a place to "park" attention they don't need. It is called the **attention sink**. In the toy row `<s>` gets 0.186. CS728's score only adds up columns inside tool spans, so the sink's column is never counted; but a head that parks 80% on the sink has only 20% left for tools, so its tool scores are small and it counts less in the average (§5.3).

### 2.6 One forward pass, and why it must be "eager"

CS728 runs the model **once** over the whole prompt (this is the **prefill**; no answer is generated) with `output_attentions=True`. Fast attention kernels (SDPA, FlashAttention; Day 6) never store the weight matrix, so the code loads the model with the slow **eager** implementation, which does. §5.4 works out what that costs.

### 2.7 The real prompt and the spans: `PromptUtils` (`utils.py`) **[IMPL]**

```python
 28      self.prompt_seperator = " \n\n"                                    # ◀ goes before every tool
 29      user_header = '<|start_header_id|>user<|end_header_id|>'           # ◀ Llama-3 chat format
 30      asst_header = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
 31      self.item_instruction = f" Here are all the available tools:"
     ⋮
 37      self.doc_text = lambda idx, doc_name, doc_info: f"tool_id: {doc_name}\ntool description: {doc_info}"
 38      self.add_text1 = f"Now, please output ONLY the correct tool_id for the query below."
     ⋮
 49  def create_prompt(self, query):
 50      query_prompt = f"Query: {query}"+ "\nCorrect tool_id:"
 51      prompt = self.prompt_prefix + \                                    # ◀ header + instruction
 52              self.all_docs_info_string + \                              # ◀ the 100 tools, in this query's shuffled order
 53              self.prompt_seperator + \
 54              self.add_text1 + \
 55              self.prompt_seperator + \
 56              query_prompt + \                                           # ◀ "Query: … \nCorrect tool_id:"
 57              self.prompt_suffix                                         # ◀ the assistant header: the model would answer next
     ⋮
 67      doc_st_index = self.prompt_prefix_length + 1 # inlcudes " \n\n"    # ◀ first tool starts after the prefix and one separator token
 68      for ix, key in enumerate(shuffled_keys):
     ⋮
 72          doc_text = self.doc_text(...).strip()
 73          doc_text_len = len(self.tokenizer(doc_text, add_special_tokens=False).input_ids)   # ◀ this tool's length in tokens
 75          doc_spans.append((doc_st_index, doc_st_index + doc_text_len))  # ◀ its (start, end) in the full prompt
 76          doc_st_index =  doc_st_index + 1 + doc_text_len                # ◀ skip it and the next " \n\n" (1 token)
```

The query span is found by token ids (`run3.py:63–82`, same as `run2.py:79–91`):

```python
 71      i = len(input_ids_list) - 1
 72      while i >= 0:                                                      # ◀ walk BACKWARDS to the last "Query" token
 73          if input_ids_list[i] == 2929:                                  # ◀ 2929 = "Query" [RAN]
 74              position_of_query = i
     ⋮
 77      actual_query_st_pos = position_of_query + 2                        # ◀ skip "Query" and ":"
 79      for i in range(actual_query_st_pos, len(input_ids_list)):
 80          if input_ids_list[i] == 34192:                                 # ◀ 34192 = "Correct" [RAN] (the comment says "tool_id")
 81              break
 82      return (actual_query_st_pos, i)                                    # ◀ the query's tokens (plus its "\n")
```

| name | holds | example |
|---|---|---|
| `doc_spans` | one (start, end) per tool, in prompt order | first test query: `[(11, 38), (39, 67), (68, 118), …, (2899, 2936)]` **[RAN]** |
| `dict_doc_name_id` | tool name → its index in **this** prompt's order (`utils.py:26`) | its correct tool `PDF&URLTool` → 46, span (1352, 1421) **[RAN]** |
| `query_span` | (start, end) of the query tokens | (2954, 2964) for "How can I make this document publically available?" **[RAN]** |

**Why the spans can go wrong, and why here they don't.** Tokenizing a piece on its own can give different tokens from tokenizing it inside a longer text (merges across the boundary). The code assumes each separator `" \n\n"` is exactly one token and that nothing merges across it. I checked: the decoded spans match all 3,000 tools of 30 prompts exactly **[RAN]**. Hard-coded token ids (2929, 34192) only work for the Llama-3 tokenizer; with another model they would silently find nothing. A safer version would compute the spans from the tokenizer's character offsets.

### 2.8 Check yourself

1. Scores [1.0, 0.0, 2.0]: softmax weights? Largest? *(e¹ = 2.72, e⁰ = 1, e² = 7.39; total 11.11 → 0.245, 0.090, 0.665; position 2.)*
2. Why read the **query's** rows and not a tool's rows? *(Causal attention: a tool's tokens can't see the query, which comes later. Only the query rows can "look back" at all 100 tools.)*
3. How many attention matrices does one CS728 forward pass produce? *(512: 16 layers × 32 heads, each about 2,980 × 2,980.)*
4. Why does the sink not need special handling in CS728? *(Scores sum only columns inside tool spans; position 0 is in none.)*

---

## 3. Retrieval heads (Wu et al. 2024): the idea behind Part 3

### 3.1 What the paper says (you don't need to open it)

> [!abstract] Wu et al., *Retrieval Head Mechanistically Explains Long-Context Factuality* (2024), §1–3
> - **Question:** when a long-context model copies a fact from far back in its prompt, which part of the model finds it?
> - **Test:** needle-in-a-haystack. Hide a sentence (the **needle**, e.g. "The secret code is 4721") in a long text and ask for it.
> - **Retrieval score:** each time the model emits a needle token, check every head: does its strongest attention point at that same token in the needle? A head's score = the share of needle tokens it was "caught copying".
> - **Findings:** under 5% of heads score high, so retrieval is **sparse**. These heads exist in every model family tested, are already there in base models, and survive long-context training.
> - **Causal test:** masking the retrieval heads makes needle tests fail and the model invent facts; masking the same number of random heads barely matters.
> - **For CS728:** the assignment borrows the idea that **a few heads do the looking-up**, but scores heads on a simpler, tool-level test (§4): does the head give the correct tool more attention than any other tool?

### 3.2 The strict copy rule, on tiny numbers (background)

Wu et al.'s rule is stricter than CS728's. A head **hits** at a generation step if **both** hold:
1. its strongest-attention position is **inside the needle**, and
2. the token at that position **is the token being emitted**.

![[d09-1-copy-hit.svg]] [[d09-1-copy-hit.html|▶ Step through it]]

CS728 can't use this rule: the model never copies a tool description, it only has to pick a tool. So the assignment replaced "copied the right token" by "gave the right tool the most attention".

> [!note] Optional: the strict rule in code
> Your later project LongContext_RAG implements the strict rule (`/data/jayavardhan/projects/LongContext_RAG/heads/scorer.py:121–143`, `copy_hits`). If an interviewer asks how you would reproduce Wu et al. exactly, this is the walk-through:
>
> ![[d09i-copy-dbg.svg]] [[d09i-copy-dbg.html|▶ Step through it]]

### 3.3 Question bank: retrieval heads

**L1. What is an attention head, in one sentence?**
> [!success]- Answer
> One independent attention pattern inside a layer: for each token it scores every earlier position, softmaxes the scores into weights that sum to 1, and mixes the earlier positions' value vectors with those weights. Each head has its own learned projections, so each learns its own way of choosing where to look.

**L1. What is a retrieval head?**
> [!success]- Answer
> A head that looks up information in the context: in Wu et al.'s definition, while the model writes something it read in the prompt, the head puts its largest attention weight on exactly that token. Fewer than 5% of heads behave like this, and masking them breaks long-context lookup while masking random heads doesn't. In CS728 we used a looser, task-level version: heads that give the correct tool the most attention.

**L2. Why did CS728 not use Wu et al.'s copy rule?**
> [!success]- Answer
> The copy rule needs the model to copy tokens from the context while generating. In tool selection nothing is copied from the description; the question is only which tool the query attends to. So we scored heads on whether the correct tool got the highest mean attention from the query, using a single forward pass and no generation.

**L3. "Couldn't a head score high just because it attends to the first or last tool?"**
> [!success]- Answer
> Only on queries whose correct tool happens to sit there, which is rare: the tool order is reshuffled for every query, so the correct tool lands in any of the 100 places. A purely positional head hits about 1% of the time. The heads we kept hit 40–93 times out of 200 training queries, far above that, so they must follow content.

---

## 4. Hit Count: choosing the heads (`code3.py`) **[IMPL]**

### 4.1 The idea

For each of 200 training queries (correct tool known):
1. build the prompt (tools in a fresh random order) and run the model once;
2. for **every** head, score every tool: the mean of the attention block "query tokens × that tool's tokens";
3. the head **hits** if the correct tool has its highest score.

Count hits per head over the 200 queries; keep the K heads with the most hits. No gradients, no weights changed: 200 forward passes.

The assignment suggested this rule ("as mentioned in the assignment itself", report p. 5); you implemented it and two alternatives (§4.4).

![[d09i-hitcount.svg]] [[d09i-hitcount.html|▶ Step through it]]

On the toy: L1H0 picks the correct tool on all 5 queries, L1H2 on 4. L0H0 always picks the first tool and L0H1 always the last; they hit only when the answer happens to sit there. With K = 2 we keep L1H0 and L1H2.

### 4.2 Reading the code: the Python and PyTorch pieces

| construct | meaning | tiny example |
|---|---|---|
| `model(**inputs).attentions` | with `output_attentions=True`: a tuple, one tensor per layer, each `[batch, heads, N, N]` | 16 tensors of `[1, 32, 2980, 2980]` |
| `attentions[l][0]` | layer l, batch item 0 → `[heads, N, N]` | |
| `m[a:b, :]` | rows a … b−1, all columns | the query's rows |
| `m[:, c:d].mean()` | mean of the columns c … d−1 (over every remaining row too): one number | the query × tool block |
| `torch.argsort(x, descending=True)` | indices that sort x from largest to smallest | `[.2, .7, .1] → [1, 0, 2]` |
| `ranked[0].item()` | the first index, as a Python int | the head's top tool |
| `t.flatten()` | `[16, 32]` → 512 numbers, row by row | head (l, h) → index 32·l + h |
| `idx // n`, `idx % n` | integer division and remainder | 275 → (8, 19) = L8H19 |
| `with torch.no_grad():` | don't record operations for gradients (we only read) | |
| `del …; gc.collect(); torch.cuda.empty_cache()` | free the 9 GB of attention before the next query | |

### 4.3 The code: `select_retrieval_heads` (`code3.py:7–121`)

```python
  7  def select_retrieval_heads(train_queries, model, tokenizer, tools, device, max_heads=20):
     ⋮  # 8–21: the assignment's TODO and docstring
 23      num_layers = model.config.num_hidden_layers                     # ◀ 16
 24      num_heads = model.config.num_attention_heads                    # ◀ 32
 27      head_scores = torch.zeros(num_layers, num_heads, device=device) # ◀ one hit counter per head: [16, 32]
 29      for qix in tqdm(range(len(train_queries))):                     # ◀ 200 training queries
     ⋮  # 30–32: the query text and its gold tool name
 34          tool_ids = list(tools.keys())
 35          random.shuffle(tool_ids)                                    # ◀ a fresh tool order per query, as at test time
     ⋮  # 37–46: PromptUtils → the spans and name → index map
 48          gold_tool_id = map_docname_id[gold_tool_name]               # ◀ where the correct tool sits in THIS order
 50          prompt = putils.create_prompt(query=question)
 51          inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
 54          with torch.no_grad():
 55              attentions = model(**inputs).attentions                 # ◀ ONE forward pass: 16 × [1, 32, N, N]
     ⋮  # 57–70: the query span, same logic as §2.7
 72          for layer_idx in range(num_layers):
 73              attn = attentions[layer_idx][0]                         # ◀ [32, N, N]
 74              for head_idx in range(num_heads):
 75                  head_attn = attn[head_idx]                          # ◀ [N, N]: this head's matrix
 76                  query_attn = head_attn[query_start:query_end, :]    # ◀ only the query's rows: [q, N]
 78                  doc_scores = torch.zeros(len(item_spans), device=device)
 79                  for doc_i, (ds, de) in enumerate(item_spans):
 80                      doc_scores[doc_i] = query_attn[:, ds:de].mean() # ◀ tool score = mean of the query × tool block
 83                  ranked = torch.argsort(doc_scores, descending=True) # ◀ this head's ranking of the 100 tools
 84                  if ranked[0].item() == gold_tool_id:                # ◀ HIT: the correct tool is this head's first choice
 85                      head_scores[layer_idx, head_idx] += 1
     ⋮  # 88–100: the two alternatives, commented out (§4.4)
102          del attentions, inputs, input_ids                           # ◀ free GPU memory before the next query
     ⋮
107      flat_scores = head_scores.flatten()                             # ◀ [16, 32] → 512
108      top20= torch.argsort(flat_scores, descending=True)[:max_heads]  # ◀ the K best heads (the name says 20; K is a parameter)
111      for idx in top20:
112          layer = idx.item() // num_heads                             # ◀ flat index → layer …
113          head = idx.item() % num_heads                               # ◀ … and head
114          selected_heads.append((layer, head))
121      return selected_heads
```

| name | type / shape | holds | your run |
|---|---|---|---|
| `train_queries` | list of dicts | `text`, `gold_tool_name`, `qid` | the first 200 of 1,500 |
| `attentions` | tuple of 16 × `[1, 32, N, N]` fp16 | every head's full matrix | N ≈ 2,980 |
| `query_attn` | `[q, N]` | the query's rows for one head | q ≈ 10–40 |
| `doc_scores` | `[100]` | mean attention per tool, for one head | |
| `head_scores` | `[16, 32]` | hits per head | best: L8H19 = **93**, L6H11 = 83, L2H20 = 73 **[REPORT]** |
| `selected_heads` | list of (layer, head) | the K best | §4.5 |

**What breaks if you change it:**

| change | result |
|---|---|
| don't reshuffle the tools per query (drop line 35) | the correct tool for a given tool name sits in the same place every time; a head can "hit" by memorising positions, and the selection would not transfer to test, where the order is shuffled |
| `.sum()` instead of `.mean()` on line 80 | long descriptions (69 tokens) collect more attention than short ones (13): length, not relevance, starts to win |
| score with the last prompt token's row instead of the query rows | the row of the assistant header; it sees everything too, but mixes in the instruction; it's a reasonable alternative, not what you did |
| keep the 9 GB `attentions` alive across queries (drop line 102) | out of GPU memory after a few queries |

### 4.4 Three ways to score a head, and what K to keep **[REPORT]**

Your report (pp. 5–8) tried three head-scoring rules, all with K = 20, then varied K for the first:

| head score (summed over 200 training queries) | Recall@1 | Recall@5 |
|---|---|---|
| **1. Hit Count**: +1 when the correct tool is the head's first choice | **32.78%** | 61.78% |
| 2. Reciprocal rank: + 1 / rank of the correct tool (1, ½, ⅓, …) | 32.72% | **62.30%** |
| 3. Raw attention: + mean attention on the correct tool | 14.50% | 57.28% |

| K (Hit Count) | Recall@1 | Recall@5 |
|---|---|---|
| **10** | **40.68%** | **66.94%** |
| 20 | 32.78% | 61.78% |
| 30 | 26.64% | 57.78% |

**Reading the tables:**
- **Hit Count vs reciprocal rank.** Hit Count only rewards first place, which is what Recall@1 measures; reciprocal rank also rewards 2nd–5th place, so it picks heads that are good at the shortlist (Recall@5 +0.5). The two choose almost the same heads (the report's lists share 19 of 20), so the difference is small.
- **Why raw attention is much worse.** It rewards heads that put a lot of attention on the correct tool **in absolute terms**, even if they put as much on other tools. A head that attends strongly to everything (or to whatever sits first) scores well. Hit Count and reciprocal rank are **relative**: the correct tool has to beat the other 99.
- **Fewer heads is better.** Recall falls as K grows: past the first 10, the added heads hit less often (the 30th has 30 hits in 200, vs 93 for the best), and averaging them in dilutes the good ones.

> [!warning] K was chosen on the test set
> The K = 10 / 20 / 30 numbers are all on the 5,000 **test** queries, and 40.7% is the best of the three. Choosing a setting by its test score makes that score slightly optimistic. The clean protocol: split the 1,500 training queries, select heads on one part, choose K on the other, then run the test set once. With only three values of K the bias is probably small, but say it before the interviewer does.

**One more thing the numbers suggest [THEORY].** The best head alone was right on 93 of 200 training queries (46.5%). That is on the same queries used to pick it, so it is inflated, but it is above the 10-head result on test (40.7%). K = 1, 3 and 5 were never tried; they might do as well or better.

### 4.5 Where the selected heads are **[REPORT]**

![[d09r-cs728-heads.png]]
*From your report (p. 8): the 20 selected heads by layer (x) and head number (y); bigger circle = more hits. Layers are counted from 0 (0–15).*

The top 10: L8H19 (93 hits), L6H11 (83), L2H20 (73), L4H16 (71), L7H12 (71), L8H13 (68), L6H8 (65), L10H1 (65), L5H9 (64), L5H11 (61). They sit in layers 2–10, mostly 5–10, none in the last five layers (11–15 hold only 1 of the 20). The report's reading: the middle layers do the query–tool matching. That is consistent with other findings that lookup heads sit in the middle of the network, but it is a description, not a test.

### 4.6 Question bank: Hit Count

**L1. What is Hit Count?**
> [!success]- Answer
> A way to pick heads. On 200 training queries with known correct tools, each head scores all 100 tools by the mean attention from the query's tokens to each tool's tokens; the head gets +1 when the correct tool scores highest. We kept the heads with the most hits: the top 10 of 512 worked best.

**L2. Is this "training-free"?**
> [!success]- Answer
> No weights change, so no training in the usual sense. But it is **supervised selection**: it uses 200 labelled queries to choose 10 of 512 heads. I'd call it "no fine-tuning, a small labelled set to pick heads".

**L2. Why reshuffle the tool order for every training query?**
> [!success]- Answer
> So a head can't score by position. If tool X were always 37th, a head that always looks at place 37 would hit on every X query. With a fresh order each time, only heads whose attention follows the content of the query beat chance, and the test set is shuffled the same way.

**L2. Why did fewer heads do better?**
> [!success]- Answer
> The hit counts fall quickly: 93 for the best head, about 60 for the 10th, 30 for the 30th. Every head gets equal weight in the average, so adding weaker heads pulls the score toward their noisier, more positional attention. K = 10 was best of the three tried; smaller K wasn't tried.

**L3. "Your K was picked on the test set."**
> [!success]- Answer
> Yes. We compared K = 10, 20 and 30 on the 5,000 test queries and reported the best. With three options the optimism is small, but it's not a clean held-out number. I'd select heads on part of the 1,500 training queries and choose K on the rest, then run the test set once.

**L3. "Why did raw attention on the correct tool do so badly?"**
> [!success]- Answer
> It's absolute, not relative. A head that spreads lots of attention over every tool, or always over the first tools, gives the correct tool a high value too, without ever ranking it first. Hit Count and reciprocal rank only reward a head when the correct tool beats the other 99, which is what retrieval needs.

**L3. "Two of your heads always agree. Does keeping both help?"**
> [!success]- Answer
> Not much: they add the same information twice. We didn't check head correlations. A greedy selection that skips a head highly correlated with ones already chosen would add diversity; that's a cheap next step.

---

## 5. Heads as a retriever: a score for every tool

### 5.1 The idea

For a **test** query we don't know the correct tool, so we can't count hits. Instead, using only the chosen heads:

1. run the prompt (tools in a fresh random order) once; take each chosen head's attention matrix;
2. keep the **query's rows**;
3. for each tool, take the **mean** of the block "query rows × that tool's columns";
4. **average** over the chosen heads;
5. **rank** the tools by that number; the top one is the prediction.

![[d09m-heads.mp4]]
*Manim video (37 s): one attention row → Hit Count over 5 training queries → scoring the tools of a test query → Recall@k. Pause and scrub with the video controls.*

### 5.2 The code: `query_to_docs_attention_heads` (`run3.py:34–60`) and the ranking (`:165–176`) **[IMPL]**

```python
 34  def query_to_docs_attention_heads(attentions, query_span, doc_spans, selected_heads):
     ⋮  # 35–47: the assignment's TODO 2 and docstring
 49      doc_scores = torch.zeros(len(doc_spans), device=attentions[0].device)   # ◀ one score per tool: [100]
 50      query_start, query_end = query_span
 52      for layer_idx, head_idx in selected_heads:                  # ◀ the 10 chosen heads only
 53          head_attn = attentions[layer_idx][0, head_idx]          # ◀ [N, N]
 54          query_attn = head_attn[query_start:query_end, :]        # ◀ the query's rows: [q, N]
 56          for doc_i, (doc_start, doc_end) in enumerate(doc_spans):
 57              doc_scores[doc_i] += query_attn[:, doc_start:doc_end].mean()   # ◀ mean of the block, added per head
 59      doc_scores /= len(selected_heads)                           # ◀ → the mean over heads
 60      return doc_scores
     ⋮
165          ranked_docs = torch.argsort(doc_scores, descending=True) # ◀ tools, best first
166          gold_rank = 1000
167          for i, tool_id in enumerate(ranked_docs):
168              if tool_id == gold_tool_id:                         # ◀ where did the correct tool land?
169                  gold_rank = i                                   # ◀ 0 = first place
170                  break
173          if gold_rank<1:
174              correct_at_1 += 1                                   # ◀ Recall@1 counter
175          if gold_rank<5:
176              correct_at_5 += 1                                   # ◀ Recall@5 counter
```

![[d09i-score-dbg.svg]] [[d09i-score-dbg.html|▶ Step through it]]

*Animated, 10 steps: the code on the left with the running line highlighted, the variables on the right (the ones that just changed are red).*

| name | type / shape | holds | toy value |
|---|---|---|---|
| `attentions` | tuple of 16 × `[1, 32, N, N]` | all heads, all tokens | toy: N = 11 |
| `query_span` | (start, end) | the query's tokens | (9, 11) = "fly Paris" |
| `doc_spans` | list of (start, end) | each tool's tokens | A (1,3), B (3,5), C (5,7), D (7,9) |
| `selected_heads` | list of (layer, head) | from §4 | toy [(1,0), (1,2)]; yours the top 10 |
| `doc_scores` | `[n_tools]` | mean over heads of the block means | A .055, B .158, C .031, D .041 |
| `ranked_docs` | `[n_tools]` | tool indices, best first | [B, A, D, C] |
| `gold_rank` | int | position of the correct tool in the ranking | 0 → a Recall@1 and Recall@5 hit |

On the toy: L1H0's block for tool B is (0.25 + 0.15 + 0.30 + 0.12) / 4 = **0.205**; L1H2's is 0.110; the mean over the two heads is **0.158**, the highest, so B is ranked first.

### 5.3 The design choices, and what breaks if you change them

- **The query's rows only.** They are the only rows that see every tool (§2.3), and they ask exactly "what does the query look at?".
- **Mean over the block, not sum.** A sum would favour long descriptions (69 tokens collect more mass than 13); the mean divides that out. The price: a short description with one strongly attended token can win.
- **No renormalisation.** The block means are raw attention weights. Within one head, renormalising over tool tokens would only reweight the query rows (each row divided by its own total on tools), so it would change that head's ranking little. Across heads it matters more: a head that parks most of its attention on the sink contributes small numbers to every tool and so counts less in the average. A version that renormalises each head over tool tokens would give every head an equal vote; it wasn't tried.
- **Equal weight per head.** `doc_scores /= len(selected_heads)` is a plain mean. Weighting heads by their hit count is a natural variant.

| change | result |
|---|---|
| sum over heads instead of mean | same ranking (dividing every tool by the same 10 changes nothing); only the printed scale changes |
| `.max()` instead of `.mean()` over the block | rewards a single strongly attended token; less sensitive to description length, more to one-word matches |
| all 512 heads (Part 2's scorer) | Recall@1 drops from 40.7% to 1.3%: position-driven heads swamp the few good ones |

### 5.4 The cost: one prompt, 9 GB of attention, a Python loop

`load_model_tokenizer` (`utils.py:11–22`) asks for every weight matrix:

```python
 14      model = AutoModelForCausalLM.from_pretrained(model_name,
 15                                                  output_attentions = True,       # ◀ return every layer's weights
 16                                                  attn_implementation="eager",    # ◀ the only path that has them
 17                                                  dtype=dtype, ...)               # ◀ fp16 (run2.py:107, run3.py)
```

**Memory [RAN arithmetic].** Each layer returns `[1, 32, N, N]` in fp16. With N = 2,984 (the mean prompt):

$$32 \times 2{,}984^2 \times 2\text{ bytes} \approx 0.57\text{ GB per layer}, \qquad \times 16 \text{ layers} \approx 9.1\text{ GB per query}$$

That is why every loop ends with `del attentions … torch.cuda.empty_cache()`. It fits on a large GPU but would not scale: attention memory grows with N², so 1,000 tools (≈ 30k tokens) would need about 900 GB.

**Time.** Hit Count runs a Python loop over 512 heads × 100 tools = 51,200 small `.mean()` calls per query, each a separate GPU kernel. A vectorised version would compute all tool means at once (e.g. a cumulative sum over columns, then differences at the span edges).

**How it could be done more cheaply [THEORY].** Only the query's rows and only the 10 chosen heads are needed. Your later LongContext_RAG project shows the pattern: prefill with the fast kernel, then read attention on the slow path only where needed, with forward hooks that keep just the needed slice (`LongContext_RAG/heads/capture.py:60–95`). For CS728 that would cut 9 GB to a few MB.

### 5.5 Question bank: heads as a retriever

**L1. How do you turn attention into a tool ranking?**
> [!success]- Answer
> One forward pass over the prompt with attention weights returned. For each chosen head, take the rows of the query's tokens; for each tool, take the mean of the block of those rows and that tool's columns; average over the chosen heads; rank tools by the result.

**L2. Why the mean over the block and not the sum?**
> [!success]- Answer
> Tool descriptions range from 13 to 69 tokens. A sum would give long descriptions more attention mass just for being long. The mean removes length; its weakness is that a short description with one strongly attended word can win.

**L2. You don't remove the attention sink. Is that a bug?**
> [!success]- Answer
> No: the score only sums columns inside tool spans, and the sink at position 0 is in no span, so it never enters. What it does do is shrink a head's numbers if that head parks most of its attention there, so such heads count less in the average. Renormalising each head over tool tokens would equalise heads; we didn't try it.

**L2. Why do you need eager attention?**
> [!success]- Answer
> The fast kernels (SDPA, FlashAttention) compute the output without ever storing the N × N weight matrix. Only the eager implementation materialises it, and `output_attentions=True` returns it for every layer. That's about 0.57 GB per layer and 9 GB per query at 3,000 tokens in fp16.

**L3. "What is the cost of this retriever at inference?"**
> [!success]- Answer
> A full prefill of all 100 descriptions plus the query, about 3,000 tokens, per query, on the slow eager path, with 9 GB of attention returned. An embedding retriever encodes the 100 descriptions once and then needs one small encoder pass per query. So the head retriever is far more expensive per query; its advantage is that no retriever model is trained. In practice I'd use it to re-rank a short embedding shortlist, or read only the 10 heads' query rows with hooks.

**L3. "Would this scale to 1,000 tools?"**
> [!success]- Answer
> Not as built. The prompt would be about 30,000 tokens, attention memory grows with its square (around 900 GB for all layers in fp16), and attention spreads thinner over more candidates. I'd shortlist the top 20–50 with an embedding model and use the heads only on that shortlist.

---

## 6. Recall@k, the baseline, and "lost in the middle"

### 6.1 Recall@k from zero

For each test query, rank all tools; find the **rank** of the correct one (1 = first).

$$\text{Recall@}k = \frac{\#\{\text{queries whose correct tool has rank} \le k\}}{\#\{\text{queries}\}}$$

Six queries whose correct tools rank 1, 3, 1, 7, 2, 1:
- Recall@1 = 3 / 6 = **50%** (ranks of 1);
- Recall@5 = 5 / 6 = **83%** (all except the 7).

Recall@1 is plain top-1 accuracy. Recall@5 matters when a later step (the LLM itself, or a re-ranker) chooses from a shortlist of 5. With 100 tools, a random ranking gives Recall@1 = **1%** and Recall@5 = **5%**.

A caution for your data: test queries per tool range from 16 to 376 **[RAN]**, so Recall@k is dominated by the most frequent tools. Per-tool (macro-averaged) recall would weight every tool equally; it wasn't reported.

### 6.2 The 1.3% baseline, named **[IMPL] [REPORT]**

The baseline is Part 2's scorer, `query_to_docs_attention` (`run2.py:34–49`): the same block mean, but over **all** heads.

```python
 40      doc_scores = torch.zeros(len(doc_spans), device=attentions[0].device)
 43      for layer in range(num_layers):                              # ◀ all 16 layers
 44          avg_attn = attentions[layer][0].mean(dim=0)              # ◀ average all 32 heads → one [N, N] matrix
 45          query_attn = avg_attn[query_start:query_end, :]          # ◀ the query's rows
 46          for i, (doc_start, doc_end) in enumerate(doc_spans):
 47              doc_scores[i] += query_attn[:, doc_start:doc_end].mean()   # ◀ block mean, as in §5
 48      doc_scores /= num_layers                                     # ◀ → mean over all 512 heads
```

Result over 5,000 test queries: **Recall@1 0.0132, Recall@5 0.2554** (report p. 3, with the run's output):

![[d09r-cs728-run2-log.png]]

**What the two numbers say.**
- **Recall@1 1.3% ≈ chance (1%).** Averaged over all heads, the query's attention does not pick out the right tool for first place. §6.4 shows why: the first few places get about 6× the attention of the middle, so position, not content, mostly decides first place.
- **Recall@5 25.5% ≫ chance (5%).** There is some content signal in the average; it gets the right tool into the top 5 a quarter of the time but almost never beats the position bonus for first place **[THEORY]**.

**Is it a fair baseline?** For one claim, yes: *"selecting heads matters"*. Same model, same prompt, same score; only "all heads" changes to "10 heads". It is **not** a fair stand-in for "existing methods". For that, your report has three real baselines:

| method (report p. 9) | trained for retrieval? | Recall@1 | Recall@5 |
|---|---|---|---|
| attention, all 512 heads | no | 1.32% | 25.54% |
| BM25 (word overlap) | no | 18.54% | 34.74% |
| attention, top 30 heads | no (heads picked on 200 labelled queries) | 26.64% | 57.78% |
| attention, top 20 heads | " | 32.78% | 61.78% |
| MiniLM (msmarco-MiniLM-L-6-v3, 384-d) | yes | 38.82% | 61.06% |
| **attention, top 10 heads** | " | **40.68%** | **66.94%** |
| UAE-Large-V1 (1,024-d) | yes | **61.58%** | **86.90%** |

> [!important] How to state the result honestly
> "Selecting 10 of Llama-3.2-1B's 512 heads with 200 labelled queries raised Recall@1 from 1.3% (all heads averaged) to 40.7%. That beats BM25 (18.5%) and the small MiniLM retriever (38.8%) without training anything, but a strong embedding model, UAE-Large-V1, gets 61.6%."
> Don't say "1.3% → 40.7%" alone, and never as "31× better": a ratio against a near-chance baseline means nothing.

### 6.3 Lost in the middle (Liu et al. 2023), summarised

> [!abstract] Liu et al., *Lost in the Middle* (2023), §1–2
> - **Setup:** multi-document question answering. The model gets 10, 20 or 30 documents, one of which holds the answer, and the answer's position is moved from first to last.
> - **Result:** accuracy is high when the answer is first, fairly high when last, and lowest in the middle: a **U shape**.
> - **Striking:** with the answer in the middle, GPT-3.5-Turbo can do worse than with no documents at all.
> - **Also:** a longer context window doesn't make a model better at using the middle.
> - **Key-value retrieval** (a JSON list of random IDs) shows the same pattern, so it's about position, not about the question being hard.

![[d09i-lost-middle.svg]] [[d09i-lost-middle.html|▶ Step through it]]

**Why it happens [THEORY]:** a lot of attention is **positional**: early tokens get extra weight (primacy, and the sink), and the last tokens sit next to the question (recency). A tool in the middle has neither.

### 6.4 Your measurement: the U in attention **[IMPL] [REPORT]**

`analyze_gold_attention` (`run2.py:52–76`) takes, for every test query, the all-heads score of the **correct** tool and the place the shuffle put it (0–99, `run2.py:184–189`), groups places into 20 bins of 5, and plots the mean score per bin:

```python
 53      positions = [r["gold_position"] for r in results]   # ◀ where the correct tool sat (0–99)
 54      scores = [r["gold_score"] for r in results]         # ◀ its all-heads score
 57      num_bins = 20
 58      bin_size = 100 // num_bins                          # ◀ 5 places per bin
     ⋮
 64          bin_data = [s for p, s in zip(positions, scores) if low <= p < high]
     ⋮
 67              bin_scores.append(np.mean(bin_data))        # ◀ mean score of the correct tool in this bin
```

![[d09r-cs728-position.png]]
*Your report's plot (p. 4): mean all-heads attention from the query to the correct tool, against the correct tool's place in the list.*

![[d09i-position-curve.svg]] [[d09i-position-curve.html|▶ Step through it]]

Read off the plot: **0.00061** for places 0–4, about **0.0001** for places 5–90 (lowest ≈ 0.00009 around place 35), back up to **0.00029** for places 95–99. The first five places get about **6×** the attention of the middle; the last five about **3×**. That is the lost-in-the-middle U, measured on attention.

**What it does and doesn't show:**
- It shows the U for the **all-heads** score, which explains the 1.3% baseline.
- It measures **attention**, not accuracy. The accuracy version (Recall@1 per position bin) would be the direct analogue of Liu et al.
- It was **not** measured for the 10 selected heads. So "the selected heads reduce position bias" is a hypothesis, and your resume's "mitigated the lost-in-the-middle effect" is not supported as written. Say: *"we measured a U-shaped position bias in attention-based ranking; selecting heads raised accuracy from 1.3% to 40.7%."*
- The experiment that would support "mitigated" **[FUTURE]**: rerun `analyze_gold_attention` with `query_to_docs_attention_heads`, and also bin Recall@1 by position, for both scorers. A flatter curve for the heads = mitigation.

### 6.5 Question bank: metrics, baseline and position

**L1. Define Recall@1 and Recall@5.**
> [!success]- Answer
> The fraction of test queries whose correct tool is ranked first, or among the first five. With 100 tools, random ranking gives 1% and 5%.

**L2. "What exactly is the 1.3% baseline?"**
> [!success]- Answer
> The same attention score with all 512 heads averaged: for each tool, the mean attention from the query's tokens to the tool's tokens, averaged over every head and layer. It's the "no selection" control. Its Recall@1 is at chance level because attention averaged over all heads mostly follows position.

**L2. "Isn't 1.3% a strawman?"**
> [!success]- Answer
> For the claim "selecting heads matters" it's the right control: only that one choice differs. For the claim "this is a good retriever" it's not, so I compare with BM25 (18.5%), MiniLM (38.8%) and UAE-Large-V1 (61.6%). The selected heads beat the first two and lose clearly to UAE.

**L2. Why is the baseline's Recall@5 (25.5%) so much higher than its Recall@1 (1.3%)?**
> [!success]- Answer
> Averaged attention does carry some content signal: the right tool reaches the top five a quarter of the time, five times chance. But the first few places get about six times the attention of the middle, so first place almost always goes to whatever tool sits at the start, which is the correct one only 1% of the time.

**L2. What is "lost in the middle"?**
> [!success]- Answer
> Long-context models use information at the start and end of the prompt best and the middle worst: accuracy against the answer's position is U-shaped (Liu et al. 2023). In our run the all-heads attention to the correct tool showed the same U: about 0.0006 in the first five places, 0.0001 in the middle, 0.0003 in the last five.

**L3. "Did your selected heads fix lost in the middle?"**
> [!success]- Answer
> We didn't measure that. We measured the U for the all-heads score, and that the selected heads are far more accurate overall. To claim mitigation I'd plot the same curve, and Recall@1 per position, for the selected heads: a flatter curve would show it; a higher curve with the same U would mean they're better but still position-biased.

**L3. "40.7% Recall@1. Would you ship this?"**
> [!success]- Answer
> No, not alone: UAE-Large-V1 gets 61.6% at a fraction of the cost per query. The interesting result is that a general 1B chat model contains a few heads that match queries to tool descriptions about as well as a small trained retriever. Practically I'd use them as a re-ranking feature on an embedding shortlist.

---

## 7. CS728: say it out loud

### 7.1 The 30-second version

> "In a CS728 assignment we used an LLM's attention as a tool retriever. Llama-3.2-1B reads 100 tool descriptions and a query; we score each tool by the mean attention from the query's tokens to the tool's tokens. Averaging all 512 heads gives Recall@1 of 1.3%, basically chance, because attention mostly follows position: we measured a U-shaped position bias. We picked the heads that most often ranked the correct tool first on 200 labelled training queries; with the best 10, Recall@1 rose to 40.7% and Recall@5 to 66.9%. That beats BM25 and MiniLM with no retriever trained, though a strong embedding model, UAE-Large-V1, gets 61.6%."

### 7.2 The 2-minute version

> "An agent has to pick a tool from a list of descriptions. The usual tool is an embedding retriever; we wanted to see whether the LLM already contains one.
>
> The setup: MetaTool's 100 most common tools, 5,000 test queries, Llama-3.2-1B-Instruct. The prompt lists all 100 descriptions in a fresh random order, then the query. One forward pass with attention weights returned; for each tool, the score is the mean of the attention block from the query's tokens to that tool's tokens.
>
> With all 512 heads averaged, Recall@1 was 1.3%, about chance. Binning the correct tool's score by its position showed why: the first five places get about six times the attention of the middle. Lost in the middle, in attention.
>
> Following the retrieval-heads idea, that only a few heads do the looking-up, we scored every head on 200 training queries: a hit when the correct tool gets the head's highest score. Keeping the top 10 gave 40.7% Recall@1 and 66.9% Recall@5; 20 and 30 heads did worse, so a few heads carry the signal. We also tried reciprocal-rank scoring, about the same, and raw attention on the correct tool, much worse because it isn't relative to the other tools.
>
> Limits: K was compared on the test set; we didn't measure whether the selected heads reduce the position bias; and it costs a 3,000-token eager forward pass per query, with 9 GB of attention. UAE-Large-V1 still wins at 61.6%. The useful version is a re-ranker on an embedding shortlist."

### 7.3 Whiteboard

```
 prompt (≈2,980 tokens):  header | tool 1 | tool 2 | … | tool 100 | instruction | Query: … | assistant
                                     ▲ order reshuffled per query                     ▲ query rows see every tool
 score(tool) = mean( attention[query rows, tool columns] ),  averaged over the chosen heads
 baseline:  all 512 heads (16 layers × 32)                  → R@1 1.3%  (chance 1%),  R@5 25.5%
 Hit Count: 200 train queries; head hits if its top tool = correct; keep top K
            K = 10 → R@1 40.7%, R@5 66.9%   (K = 20: 32.8%, K = 30: 26.6%)
 compare:   BM25 18.5%   MiniLM 38.8%   UAE-Large-V1 61.6%
 position:  all-heads attention to the correct tool: 0.00061 (first 5) / 0.0001 (middle) / 0.00029 (last 5)
```

### 7.4 Claims to avoid

| don't say | why | say instead |
|---|---|---|
| "improved Recall@1 from 1.3% to 40.7%" (alone) | 1.3% is chance level; it reads as an inflated baseline | name the baseline (all heads averaged) and give BM25 / MiniLM / UAE alongside |
| "31× improvement" | a ratio against a near-chance number | absolute numbers |
| "32 heads and 3 layers" | matches nothing in the code or report | "the best 10 of 512 heads (16 layers × 32)" |
| "mitigated the lost-in-the-middle effect" | only the all-heads curve was measured | "measured a U-shaped position bias in attention-based ranking" |
| "training-free retriever" (unqualified) | 200 labelled queries pick the heads | "no fine-tuning; heads chosen with 200 labelled queries" |
| "beats embedding retrievers" | UAE-Large-V1 gets 61.6% | "beats BM25 and MiniLM; below UAE-Large-V1" |
| "Llama 3.1" | the code uses Llama-3.2-1B-Instruct (the report's "Llama3.1 1B" is a typo) | "Llama-3.2-1B-Instruct" |
| "I built …" for the whole thing | a three-person assignment | "we", and "I" for your own parts |

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

**How to drill:** read the question, answer out loud in under a minute, *then* open the answer. Mark each strong / OK / weak in [[#My notes and fumbles]]. Answers in [square brackets] are for you to fill with your own part of the work.

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
> Many heads put a large share of attention on the first token regardless of content, a place to park attention they don't need. It carries no retrieval signal. In CS728 it never enters a tool's score, because scores only sum columns inside tool spans; but a head that parks most of its weight there gives small numbers to every tool, so it counts less in the average over heads.

**F6. Prefill vs decode?**
> [!success]- Answer
> Prefill processes the whole prompt in one parallel pass and fills the KV cache; decode generates one token at a time, each attending over the cached keys and values. CS728 only needs the prefill: the query's rows of the attention matrix already show which tools the query looks at, so no answer is generated.

**F7. How does an embedding retriever work, and how is the head retriever different?**
> [!success]- Answer
> An embedding retriever (MiniLM, UAE-Large-V1 in CS728 Part 1) encodes queries and documents into vectors with a trained encoder and ranks by cosine similarity; the document vectors are computed once. The head retriever needs no fine-tuning and no second model, but scores tools inside the LLM's forward pass, so nothing is precomputed and all 100 descriptions must be in every prompt.

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
> A CS728 assignment on tool selection from attention. Llama-3.2-1B-Instruct reads 100 MetaTool tool descriptions in a random order and then the query; each tool's score is the mean attention from the query's tokens to the tool's tokens. With all 512 heads averaged, Recall@1 was 1.3%, about chance, and the attention to the correct tool was U-shaped in its position. We counted, on 200 labelled training queries, how often each head ranked the correct tool first, kept the top 10, and got 40.7% Recall@1 and 66.9% Recall@5 on 5,000 test queries: above BM25 (18.5%) and MiniLM (38.8%), below UAE-Large-V1 (61.6%).

**C2. Which model and dataset?**
> [!success]- Answer
> Llama-3.2-1B-Instruct in fp16 with eager attention: 16 layers × 32 query heads = 512 heads (8 shared key/value heads). MetaTool's 100 most frequent tools; 5,000 test queries and 1,500 training queries, disjoint; the first 200 training queries picked the heads. One prompt is about 2,980 tokens.

**C3. What exactly was the baseline?**
> [!success]- Answer
> The same score with no selection: for each layer, average the 32 heads' attention matrices, take the query rows, take each tool's block mean, then average over the 16 layers (`run2.py:34–49`). Recall@1 1.3%, Recall@5 25.5%. It shows that selection matters; for "is it a good retriever" the real comparisons are BM25, MiniLM and UAE.

**C4. Why does the baseline's Recall@5 (25.5%) far exceed its Recall@1 (1.3%)?**
> [!success]- Answer
> The averaged attention carries some content signal, enough to put the right tool in the top five a quarter of the time (chance is 5%). But the first few places get about six times the attention of the middle (our position plot), so first place almost always goes to whatever sits first: the correct tool only 1% of the time.

**C5. Which attention rows do you read, and why?**
> [!success]- Answer
> The rows of the query's own tokens. Attention is causal and the query comes after all 100 tools, so the query rows are the only ones that can see every tool, and they answer exactly "which tool does the query look at?". We locate them by token ids: after the last "Query" token, up to the "Correct" of "Correct tool_id:".

**C6. How did you get the attention weights, and what did it cost?**
> [!success]- Answer
> The model is loaded with `attn_implementation="eager"` and `output_attentions=True`, because fast kernels never store the weights. That returns every layer's [1, 32, N, N] matrix: at N ≈ 3,000 in fp16 about 0.57 GB per layer, 9 GB per query. We delete it and empty the CUDA cache after each query. Cheaper: keep only the query rows of the 10 chosen heads with forward hooks.

**C7. Why top 10?**
> [!success]- Answer
> We tried K = 10, 20 and 30 with Hit Count: 40.7%, 32.8% and 26.6% Recall@1. Hit counts fall off fast (93 for the best head, about 60 for the 10th, 30 for the 30th), and every head gets equal weight, so weaker heads dilute the good ones. Two caveats: K was compared on the test set, and smaller K (1–5) wasn't tried.

**C8. What does Hit Count actually count?**
> [!success]- Answer
> For each training query and each of the 512 heads: score all 100 tools by that head's block mean, sort, and add 1 if the correct tool is first (`code3.py:72–85`). Over 200 queries a head's count is how often it alone would have got Recall@1 right. The best, layer 8 head 19, got 93 of 200.

**C9. How did you measure lost in the middle, and what would you add?**
> [!success]- Answer
> For every test query we stored the correct tool's all-heads score and its position (0–99), binned positions in fives and plotted the mean score: about 0.0006 in the first bin, 0.0001 in the middle, 0.0003 in the last: a U. I'd add the same curve for the selected heads, and Recall@1 per position bin for both scorers; only a flatter curve for the heads would show they reduce the bias.

**C10. Sum, mean or max over a tool's tokens?**
> [!success]- Answer
> Mean over the block (query rows × tool columns). Descriptions range from 13 to 69 tokens, so a sum would favour long ones. The mean's weakness is that a short description with one strongly attended word can win; max over tool tokens would be the alternative to test.

**C11. Why did "raw attention on the correct tool" select bad heads?**
> [!success]- Answer
> It's absolute: a head that gives lots of attention to all tools, or to whatever comes first, also gives the correct tool a lot, without ever ranking it first. Hit Count and reciprocal rank are relative to the other 99 tools. Result: 14.5% Recall@1 vs 32.8% for Hit Count at K = 20.

**C12. Does this scale to 1,000 tools?**
> [!success]- Answer
> Not as built: about 30,000 tokens per prompt, attention memory growing with its square (roughly 900 GB of returned attention in fp16), and attention spread thinner. I'd shortlist the top 20–50 with an embedding model and score only those with the heads.

**C13. Where are the selected heads?**
> [!success]- Answer
> The top 10 are in layers 2–10, most in 5–10; of the top 20 only one is in layers 11–15. Middle layers doing query–document matching fits other interpretability findings, but we only described it; we didn't test it.

**C14. Could you verify causally that these heads matter?**
> [!success]- Answer
> Yes: zero the 10 heads' outputs and let the model answer with the tool name itself; compare its accuracy with zeroing 10 random heads, as Wu et al. did for needle retrieval. We didn't do it; my later LongContext_RAG repo has head-ablation code for this (`heads/ablate.py`).

**C15. What would you change with more time?**
> [!success]- Answer
> Choose K (including 1–5) and the head-scoring rule on a held-out part of the training queries; plot the position curve and per-position Recall@1 for the selected heads; report per-tool recall, since some tools have 376 test queries and some 16; read only the needed attention rows with hooks instead of returning 9 GB per query; and try the heads as a re-ranker on a UAE shortlist.

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
> The 1.3% is all 512 heads averaged, and yes, it's at chance level. It's the right control for one claim, that selecting heads matters, because nothing else changes. It isn't the comparison for "good retriever": against BM25 (18.5%) and MiniLM (38.8%) the heads win; against UAE-Large-V1 (61.6%) they lose clearly. That's the full picture, and it's how I'd write it.

**H13. "Attention isn't explanation. Why trust attention weights?"**
> [!success]- Answer
> I don't use them as an explanation; I use them as a feature and check it by its outcome, retrieval accuracy on held-out queries. Wu et al. also verified causally that masking retrieval heads breaks retrieval while masking random heads doesn't, so these specific heads' weights do track what the model uses.

**H14. "Your Hit Count heads were chosen on the same queries you report?"**
> [!success]- Answer
> No: heads were chosen on 200 training queries, disjoint from the 5,000 test queries (I checked: no shared ids or texts). But K was compared on the test set (10, 20, 30), and we reported the best. With three options the optimism is small; the clean fix is choosing K on held-out training queries.

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
> Position-driven heads are dominating, as in the all-heads baseline (1.3%). Check: are you averaging all heads instead of the selected ones; is the tool order really reshuffled per query (otherwise selection can reward position); do the spans line up with the tokens (decode them and compare); is the query span right (the hard-coded token ids only fit the Llama-3 tokenizer).

**D10. The CS728 scorer runs out of GPU memory on longer prompts. Why, and the fix?**
> [!success]- Answer
> `output_attentions=True` with eager attention returns all 16 layers' [1, 32, N, N] matrices: 9 GB at N ≈ 3,000 in fp16, growing with N². Also check that the previous query's tensors were deleted (`del attentions; torch.cuda.empty_cache()`). The real fix is to keep only what's used: forward hooks that save the query rows of the 10 selected heads, or prefill with a fast kernel and recompute attention only for those heads.

---

## Sources (optional: everything needed is summarised above)

- Wu et al. (2024), *Retrieval Head Mechanistically Explains Long-Context Factuality*: §3.1.
- Liu et al. (2023), *Lost in the Middle: How Language Models Use Long Contexts*: §6.3.
- CS728 Programming Assignment 3 report and code (your team): `interview_materials/cs728/`, §1–§7.
- Wang et al. (2023), MetaTool benchmark (the tool set and queries); Li & Li (2023), AnglE / UAE-Large-V1; Robertson & Zaragoza (2009), BM25: §6.2 (named only; nothing else needed).
- LeCun et al. (1998), *Gradient-Based Learning Applied to Document Recognition*: §9.
- Frankle & Carbin (2019), *The Lottery Ticket Hypothesis*; Frankle et al. (2020), *Linear Mode Connectivity and the Lottery Ticket Hypothesis*; Han et al. (2015), *Learning both Weights and Connections*: §11.2.
- Mu & Gilmer (2019), *MNIST-C*: §12.1.
- Madry et al. (2018), *Towards Deep Learning Models Resistant to Adversarial Attacks*; Goodfellow et al. (2015), *Explaining and Harnessing Adversarial Examples*: §14.
- Hinton et al. (2015), *Distilling the Knowledge in a Neural Network*: §15.

## My notes and fumbles

- CS728: my own part of the team's work = ; new resume wording (§0.2) = 
- LeNet: my role in Part 2 = 
-
