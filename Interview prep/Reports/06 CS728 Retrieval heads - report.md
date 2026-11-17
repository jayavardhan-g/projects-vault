# Project report: Attention-head selection for zero-shot tool retrieval (CS728, Organization of Web Information)

> **For the LLM reading this.** This report describes Programming Assignment 3 of **CS728: Organization of Web
> Information** (instructor **Prof. Soumen Chakrabarti**, Spring 2026). The team: **G Jayavardhan (25M0807), Pavan Kumar
> (25M0803), Ajay Rangoji (25M0802)**. It is listed on his resume as a course project, and he must be able to defend it.
>
> Sources:
> - the team's code `utils.py`, `run2.py`, `code3.py`, `run3.py` and data in `/data/jayavardhan/projects/interview_materials/cs728/`;
> - the team's write-up `Programming Assignment 3.pdf`.
>
> The Part 1 code (BM25, MiniLM, UAE) is **not** in the folder. Only its numbers are in the PDF.
>
> Labels:
> - **[CODE]**: what the code does, with `file:line`.
> - **[PDF]**: the team's reported result.
> - **[CHECKED]**: verified by running the tokenizer and data on CPU.
> - **[RECON]**: the report writer's analysis.

---

## 1. Summary

**Task: zero-shot tool retrieval.** Given a user query and descriptions of 100 tools (from the **MetaTool** benchmark),
rank the tools so the correct ("gold") tool comes first. Metrics: Recall@1 and Recall@5.

**Idea.** Put all 100 tool descriptions plus the query into one long prompt (about 2,970 tokens) for
**Llama-3.2-1B-Instruct**. Run one forward pass with eager attention, read the attention weights, and score each tool by
**how much the query tokens attend to that tool's tokens**. No fine-tuning and no generation: the LLM's attention *is*
the retriever.

The three parts of the assignment:
1. **Part 1, classic retrievers:** BM25 (lexical), msmarco-MiniLM-L-6-v3 (384-d bi-encoder), UAE-Large-V1 (1,024-d bi-encoder). R@1 = 18.5% / 38.8% / 61.6%.
2. **Part 2, attention from all heads, plus "lost in the middle":** averaging attention over all 16 layers × 32 heads gives R@1 = **1.32%** (chance is 1%) and R@5 = **25.5%**. A plot of attention to the gold tool against its position shows a U shape: high at the start, low in the middle, rising at the end.
3. **Part 3, select retrieval heads:** score each of the 512 heads on 200 training queries by **Hit Count** (+1 when that head's top-ranked tool is the gold tool). Keep the top K heads and rank test tools using only them. **K = 10 gives R@1 = 40.7%, R@5 = 66.9%.** That beats MiniLM and loses to UAE. Two other head-scoring rules were compared (reciprocal rank, raw attention).

---

## 2. Resume bullets vs evidence

| Bullet (resume 4) | Verdict | Evidence |
|---|---|---|
| "Improved zero-shot tool retrieval by identifying the top 10 retrieval-specialized attention heads via a Hit Count strategy, increasing Recall@1 from 1.3% to 40.7% and Recall@5 from 25.5% to 66.9%" | **Supported, with context to add.** | [PDF] All heads: 0.0132 / 0.2554. Top-10 Hit Count heads: 0.4068 / 0.6694. **Say what the 1.3% baseline is:** the mean over all 512 heads, which is roughly chance (1% for 100 tools). Also give the reference points: BM25 18.5, MiniLM 38.8, UAE 61.6 (R@1). K = 10 was chosen by comparing K = 10, 20, 30 **on the test set** (§7.2). |
| "Mitigated the 'lost-in-the-middle' effect by selectively aggregating retrieval-specialized heads…" | **Not supported as worded.** | The position curve was only measured for **all-heads** attention (Part 2). No position analysis was done for the selected heads, so "mitigated" is a claim without a measurement. Safer: "observed a lost-in-the-middle pattern in all-head attention; head selection raised overall recall". |
| "…across 32 heads and 3 layers" | **Doesn't match anything.** | Llama-3.2-1B has **16 layers × 32 heads = 512 heads**. The top 10 heads sit in **7 different layers** (2, 4, 5, 6, 7, 8, 10). Replace with "selected 10 of 512 heads (16 layers × 32 heads)". |
| (implicit) "without fine-tuning" | **True.** | Forward passes only. |
| Authorship | **Team of 3.** | The PDF lists three authors. Describe your own role honestly. |

---

## 3. Background (teach first)

### 3.1 Tool retrieval
LLM agents can call tools such as APIs or plugins. With many tools you can't put every description in the context, or
choose reliably among them, so a **retriever** first picks the likely tools. **MetaTool** (Huang et al., ICLR 2024) is
a benchmark of user queries labelled with the plugin (tool) that should handle them.

### 3.2 Retrieval methods
- **BM25:** a lexical score. For each query term, term frequency in the document is weighted by inverse document
  frequency, with saturation (k₁) and length normalisation (b). It fails on synonyms ("calculate" vs "compute").
- **Bi-encoder (dense retrieval):** encode the query and each document separately into vectors and rank by cosine
  similarity. Fast, because documents are encoded once. Examples: MiniLM (fine-tuned on MS MARCO, Bing queries) and
  UAE-Large-V1 (a strong universal embedding model).
- **Cross-encoder:** query and document go through the model *together*, which is more accurate but slower.
  **Attention-based retrieval in one long prompt is like a cross-encoder over all 100 documents at once**: the query
  tokens can attend to every tool's tokens.

### 3.3 Recall@k
The fraction of queries whose gold tool is in the top k. With 100 tools and random ranking, R@1 = 1% and R@5 = 5%.

### 3.4 Attention weights
For one head, the attention matrix A has shape [N, N] (N = prompt tokens). Row i is a probability distribution over
the earlier tokens that token i attends to; it is causal. `output_attentions=True` with
`attn_implementation="eager"` returns A for every layer and head. Fused kernels such as SDPA and FlashAttention never
materialise A.

### 3.5 Retrieval heads (Wu et al., 2024, "Retrieval Head Mechanistically Explains Long-Context Factuality")
A small fraction of attention heads (under 5%) do "copy-paste" retrieval from context. They are universal across
contexts, sparse, and causally important: masking them hurts needle-in-a-haystack retrieval. **This project uses the
same intuition**: only a few heads match the query to the right tool, and the rest add noise.

### 3.6 Lost in the middle (Liu et al., TACL 2024)
LLMs use information at the **beginning and end** of a long context better than information in the **middle**, giving
a U-shaped accuracy curve over position. Here, the gold tool's position in the shuffled list varies, so the effect can
be plotted.

---

## 4. Data and prompt (`utils.py`)

### 4.1 Data [CODE + CHECKED]
- Built by `get_queries_and_items_check` (`utils.py:86-112`) from MetaTool (`all_clean_data.csv`, `plugin_des.json`):
  - count queries per tool and keep the **100 most frequent tools**;
  - keep their queries and shuffle;
  - **test = the first 5,000**, train = the next 1,500.
- Files: `data/tools.json` (100 tools → description), `data/test_queries.json` (5,000), `data/train_queries.json` (1,500).
  Each query is `{text, gold_tool_name, qid}`.
- **[CHECKED]** Train and test share 0 qids, and all 100 tools appear as gold in the test set.
- Example tool: `stellarexplorer`, "Tool for exploring space through images including Mars Rover Photos…".
- Example query: "How can I make this document publically available?" → `PDF&URLTool`.

### 4.2 The prompt (`PromptUtils`, `utils.py:24-82`)
```
<|start_header_id|>user<|end_header_id|> Here are all the available tools: \n\n
tool_id: <name_1>\ntool description: <desc_1> \n\n
tool_id: <name_2>\ntool description: <desc_2> ... (100 tools, in a shuffled order)
 \n\nNow, please output ONLY the correct tool_id for the query below. \n\n
Query: <query>\nCorrect tool_id:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```
- **Tool spans:** `create_doc_pool_string` (`:61-82`) tokenizes each `tool_id: … tool description: …` block on its own to get its length. It then records `(start, end)` token positions for each tool, starting after the prefix and advancing by 1 + length for the separator.
  **[CHECKED]** Decoding the spans from the real tokenizer gives exactly the first and last tool blocks.
- **Query span** (`get_query_span`, `run2.py:79-91`, `run3.py:63-82`): search backward for token **2929 = "Query"**, skip 2 tokens (`Query` and `:`), then search forward for token **34192 = "Correct"**.
  **[CHECKED]** The span decodes to `" How can I make this document publically available?\n"`. It is exactly the query text, plus the newline.
- **[CHECKED]** Prompt length is about **2,972 tokens**.
- The tool order is **re-shuffled for every query** (`random.shuffle`, seeded with 64). The gold tool's position therefore varies, which makes the position analysis possible and prevents a fixed position bias from deciding the result.

### 4.3 The model (`load_model_tokenizer`, `utils.py:11-22`)
- `meta-llama/Llama-3.2-1B-Instruct`, **fp16**, `attn_implementation="eager"`, `output_attentions=True`. Eager is required to get attention matrices.
- 16 layers × 32 query heads (8 KV heads, GQA) = **512 attention heads**.
- Note: the team PDF says "Llama3.1 1B" in one place, but the code uses **Llama 3.2 1B Instruct**.
- Cost [RECON]: 16 layers × 32 heads × 2,972² fp16 attention values ≈ 9 GB of attention tensors per query. That is why each query is run alone and the code calls `del attentions; gc.collect(); torch.cuda.empty_cache()` after each one.

---

## 5. Part 2: all-head attention ranking and lost in the middle (`run2.py`)

### 5.1 Scoring (`query_to_docs_attention`, `run2.py:34-49`)
```
for each layer:
    avg_attn = attentions[layer][0].mean(dim=0)          # average over 32 heads → [N, N]
    query_attn = avg_attn[q_start:q_end, :]               # rows = query tokens
    for each tool i:  score[i] += query_attn[:, s_i:e_i].mean()   # mean over query rows × tool columns
score /= num_layers
```
Tool score = the **mean** attention weight in the block of (query-token rows × tool-token columns), averaged over heads and layers. Using the mean rather than the sum avoids favouring longer descriptions [RECON].

### 5.2 Results [PDF]
- **R@1 = 0.0132, R@5 = 0.2554** over 5,000 test queries.
- [RECON] R@1 is at chance (1%). R@5 is about 5× chance. Averaging all heads mixes a few useful heads with hundreds that attend to other things: syntax, position, attention sinks such as the first token, the instruction text.

### 5.3 Position plot (`analyze_gold_attention`, `run2.py:52-76`)
- Gold position (0–99) is binned into 20 bins of 5. The y-axis is the mean gold-tool score per bin.
- **Observed [PDF]:** attention is highest for tools at the **beginning**, drops in the **middle**, and rises toward the **end**, near the query. This is the lost-in-the-middle U shape.
- [RECON] Plausible mechanisms:
  - primacy, and attention sinks near the start;
  - recency: tokens close to the query get more attention, partly because of RoPE's distance decay.

---

## 6. Part 3: selecting retrieval heads (`code3.py`, `run3.py`)

### 6.1 Head scoring on training queries (`select_retrieval_heads`, `code3.py:7-121`)
- `run3.py` calls it with `train_queries[:200]`. **Only 200 of the 1,500 training queries** are used (`--train_samples 200`).
- For each training query: build a shuffled prompt, one forward pass, find the query span. Then for **every (layer, head)**:
  - `query_attn = attn[head][q_start:q_end, :]`;
  - tool score = `query_attn[:, s:e].mean()` for each tool;
  - `ranked = argsort(scores, descending)`.
- **Approach 1, Hit Count (shipped):** `if ranked[0] == gold: head_scores[layer, head] += 1` (`:82-85`).
- **Approach 2, reciprocal rank (commented out):** `+= 1/(rank+1)` (`:88-95`).
- **Approach 3, raw attention (commented out):** `+= mean attention(query → gold tool)` (`:97-100`).
- Flatten the 16 × 32 scores and take the top K by argsort: `layer = idx // 32`, `head = idx % 32` (`:106-118`).

### 6.2 Test-time ranking with the selected heads (`query_to_docs_attention_heads`, `run3.py:34-60`)
```
for (layer, head) in selected_heads:
    query_attn = attentions[layer][0, head][q_start:q_end, :]
    for each tool i: score[i] += query_attn[:, s_i:e_i].mean()
score /= len(selected_heads)
```
Each tool's score is averaged over the K heads. Tools are ranked, then R@1 and R@5 are computed over the 5,000 test queries (`run3.py:164-187`).

### 6.3 Top heads found by Hit Count [PDF]
Top 10 (layer, head: hits out of 200 training queries):
- L8 H19: 93
- L6 H11: 83
- L2 H20: 73
- L4 H16: 71
- L7 H12: 71
- L8 H13: 68
- L6 H8: 65
- L10 H1: 65
- L5 H9: 64
- L5 H11: 61

So the best single head picks the gold tool first on 46.5% of training queries. The top 30 heads are concentrated in **layers 5–10 (middle layers)**.

### 6.4 Results [PDF]
**Head-scoring rule, K = 20:**
| Approach | R@1 | R@5 |
|---|---|---|
| 1. Hit Count | 0.3278 | 0.6178 |
| 2. Reciprocal rank sum | 0.3272 | 0.6230 |
| 3. Mean attention to gold | 0.1450 | 0.5728 |

- The team's reading: Hit Count rewards only rank-1 hits, so it optimises R@1. Reciprocal rank also credits near-misses, so R@5 goes up slightly.
- Raw attention mass is worst. [RECON] A head can give the gold tool high *absolute* attention while giving others even more, for example heads that attend broadly to the whole tool list. What matters is the *relative* ranking, and the rank-based rules measure that.

**Number of heads, Hit Count:**
| K | R@1 | R@5 |
|---|---|---|
| **10** | **0.4068** | **0.6694** |
| 20 | 0.3278 | 0.6178 |
| 30 | 0.2664 | 0.5778 |
| all 512 (Part 2) | 0.0132 | 0.2554 |

Fewer, better heads win, because adding heads dilutes the signal with heads that do other jobs.

**Overall comparison:**
| Method | R@1 | R@5 |
|---|---|---|
| Attention, all heads | 0.0132 | 0.2554 |
| BM25 | 0.1854 | 0.3474 |
| Attention, top 30 | 0.2664 | 0.5778 |
| Attention, top 20 | 0.3278 | 0.6178 |
| MiniLM (msmarco) | 0.3882 | 0.6106 |
| **Attention, top 10** | **0.4068** | **0.6694** |
| UAE-Large-V1 | 0.6158 | 0.8690 |

The team's explanation:
- BM25 fails on vocabulary mismatch.
- MiniLM is trained for retrieval (on MS MARCO) but is small.
- A 1B LLM never trained for retrieval, using just 10 heads, matches or beats MiniLM.
- UAE wins because it is large and trained for embedding and retrieval.

---

## 7. Critical analysis (be ready for these) [RECON]

1. **The 1.3% baseline is weak.** It is chance-level, not a sensible retriever. The honest framing: "Averaging all
   attention heads gives chance-level R@1; selecting 10 heads turns the same forward pass into a retriever at 40.7%,
   between MiniLM (38.8%) and UAE (61.6%)." Don't present 1.3 → 40.7 as a 30× improvement over a real baseline.
2. **K was chosen on the test set.** K = 10, 20 and 30 were each evaluated on the 5,000 test queries, and the best was
   reported. A clean protocol picks K on held-out *training* queries (1,300 were unused).
3. **Only 200 training queries** for head scoring, with hit counts in the 30–93 range. The ranking of heads is noisy below the top few.
4. **One seed, one shuffle per query.** No variance reported.
5. **Lost in the middle was not re-measured for the selected heads.** That would be the natural next experiment: bin R@1 of the top-10 heads by gold position.
6. **Cost.** An approximately 3k-token forward pass per query with eager attention is far more expensive than a bi-encoder lookup. The attention approach is interesting because it needs **no training**, not because it is efficient.
7. **The prompt contains the instruction "output ONLY the correct tool_id",** so the query attends while in a "select a tool" frame. That may be part of why some heads become selectors.

---

## 8. Design decisions

| Decision | Why | Source |
|---|---|---|
| Mean attention over (query rows × tool columns) | normalises for tool and query length | [CODE] + [RECON] |
| Eager attention, fp16 | needed for attention matrices; memory | [CODE] |
| Shuffle tools per query | removes fixed position bias; enables the position plot | [CODE] (assignment "do not change") |
| Query span = after "Query:" up to "Correct" | only the user's words, not the instruction | [CODE], [CHECKED] |
| Hit Count head scoring | directly optimises rank-1 correctness | [PDF] |
| K = 10 | best R@1/R@5 among 10/20/30 | [PDF] (chosen on test) |
| Llama-3.2-1B-Instruct | assignment default; fits with full attention outputs | [CODE] |

---

## 9. Likely interview questions
1. *How do you turn attention into a retrieval score?* → §5.1 and §6.2.
2. *Why does averaging all heads fail?* → most heads do other jobs (syntax, position, sinks); averaging dilutes the few retrieval heads.
3. *What is Hit Count, and why did it beat raw attention?* → rank-based vs absolute mass, §6.4.
4. *Why does fewer heads give better recall?* → dilution, §6.4.
5. *Which layers are the heads in, and why might that be?* [RECON] → middle layers (5–10). Early layers handle local and lexical features, late layers focus on output prediction, and middle layers carry semantic matching.
6. *What is lost in the middle, and what did you see?* → §3.6 and §5.3. Be honest that it was measured for all heads only.
7. *How is this different from a bi-encoder or a cross-encoder?* → §3.2: like a cross-encoder over all candidates, untrained.
8. *What's wrong with your evaluation?* → §7: K tuned on test, chance-level baseline, 200 training queries, one seed.
9. *How would you scale to 10,000 tools?* [RECON] → bi-encoder first stage (UAE) to get the top 50, then attention heads or a cross-encoder to re-rank inside the context window.
10. *Why must you use eager attention?* → fused kernels don't materialise the attention matrix.
11. *How did you find the query span?* → token ids 2929 ("Query") and 34192 ("Correct").
12. *What was your contribution in a 3-person team?* → he must answer this himself, honestly.

---

## 10. Numbers cheat sheet
| Number | Meaning |
|---|---|
| 100 tools, 5,000 test / 1,500 train queries (200 used) | data |
| ~2,972 tokens | prompt length per query |
| 16 × 32 = 512 | attention heads in Llama-3.2-1B |
| 1% / 5% | chance R@1 / R@5 |
| 1.32% / 25.5% | all-head attention R@1 / R@5 |
| **40.7% / 66.9%** | top-10 Hit Count heads R@1 / R@5 |
| 32.8% / 61.8% | top-20 heads |
| 26.6% / 57.8% | top-30 heads |
| 18.5 / 38.8 / 61.6 | BM25 / MiniLM / UAE R@1 |
| 34.7 / 61.1 / 86.9 | BM25 / MiniLM / UAE R@5 |
| L8 H19, 93/200 | best single head (hit count) |
| layers 5–10 | where most top heads sit |

---

## Appendix A: Key code, annotated

Real line numbers from `/data/jayavardhan/projects/interview_materials/cs728/`. `⋮` = left out; `◀` = plain-English note.

### Reading the code: constructs used below
| Construct | Meaning |
|---|---|
| `AutoModelForCausalLM.from_pretrained(..., output_attentions=True, attn_implementation="eager")` | load the LLM so every forward returns all attention matrices (fused SDPA/Flash kernels never build them) |
| `model(**inputs).attentions` | tuple with one tensor per layer, each `[batch=1, heads=32, N, N]` |
| `A[i:j, :]`, `A[:, s:e]` | slicing rows (query tokens) and columns (tool tokens) |
| `.mean()` | average over all elements of the slice |
| `torch.argsort(x, descending=True)` | indices that sort `x` from largest to smallest, i.e. the ranking |
| `idx // 32`, `idx % 32` | turn a flat index over 16×32 heads back into (layer, head) |

### A.1 Building the prompt and the tool spans: `utils.py`
```python
24  class PromptUtils:
25      def __init__(self, tokenizer, doc_ids, dict_all_docs):
29          user_header = '<|start_header_id|>user<|end_header_id|>'                       # ◀ Llama-3 chat format
30          asst_header = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
31          self.item_instruction = f" Here are all the available tools:"
37          self.doc_text = lambda idx, doc_name, doc_info: f"tool_id: {doc_name}\ntool description: {doc_info}"
38          self.add_text1 = f"Now, please output ONLY the correct tool_id for the query below."
⋮
49      def create_prompt(self, query):
50          query_prompt = f"Query: {query}" + "\nCorrect tool_id:"
51          prompt = self.prompt_prefix + self.all_docs_info_string + self.prompt_seperator + \
54                   self.add_text1 + self.prompt_seperator + query_prompt + self.prompt_suffix
⋮
61      def create_doc_pool_string(self, shuffled_keys, all_docs):
67          doc_st_index = self.prompt_prefix_length + 1          # ◀ first tool starts after the prefix + " \n\n"
68          for ix, key in enumerate(shuffled_keys):              # ◀ 100 tools in THIS query's random order
72              doc_text = self.doc_text(...).strip()
73              doc_text_len = len(self.tokenizer(doc_text, add_special_tokens=False).input_ids)   # ◀ tokens in this tool block
75              doc_spans.append((doc_st_index, doc_st_index + doc_text_len))   # ◀ [start, end) token positions of tool ix
76              doc_st_index = doc_st_index + 1 + doc_text_len    # ◀ +1 for the separator token
```
Checked with the real tokenizer: the prompt is about 2,972 tokens, and the spans decode exactly to the first and last tool blocks.

### A.2 Finding the query tokens: `run3.py` (identical to `run2.py:79-91`)
```python
63  def get_query_span(input_ids, tokenizer):
68      input_ids_list = input_ids.tolist()
70      i = len(input_ids_list) - 1
71      while i >= 0:
72          if input_ids_list[i] == 2929:          # ◀ token 2929 = "Query" (searching backward finds the LAST one)
73              position_of_query = i
74              break
75          i -= 1
76      actual_query_st_pos = position_of_query + 2   # ◀ skip "Query" and ":"
78      for i in range(actual_query_st_pos, len(input_ids_list)):
79          if input_ids_list[i] == 34192:          # ◀ token 34192 = "Correct" (from "\nCorrect tool_id:")
80              break
82      return (actual_query_st_pos, i)            # ◀ [start, end): just the user's question (+ trailing "\n")
```

### A.3 Part 2: all-head scoring: `run2.py`
```python
34  def query_to_docs_attention(attentions, query_span, doc_spans):
40      doc_scores = torch.zeros(len(doc_spans), device=attentions[0].device)
43      for layer in range(num_layers):                         # ◀ 16 layers
44          avg_attn = attentions[layer][0].mean(dim=0)          # ◀ AVERAGE over all 32 heads → [N, N]
45          query_attn = avg_attn[query_start:query_end, :]      # ◀ rows = query tokens
46          for i, (doc_start, doc_end) in enumerate(doc_spans):
47              doc_scores[i] += query_attn[:, doc_start:doc_end].mean()   # ◀ mean attention query → tool i
48      doc_scores /= num_layers
⋮
180     ranked_docs = torch.argsort(doc_scores, descending=True)
181     gold_rank = (ranked_docs == gold_tool_id).nonzero(as_tuple=True)[0].item()   # ◀ 0 = ranked first
191     if gold_rank < 1: correct_at_1 += 1
193     if gold_rank < 5: correct_at_5 += 1
```
This gave R@1 0.0132 (chance is 0.01) and R@5 0.2554.

### A.4 Part 3: Hit-Count head selection on training queries: `code3.py`
```python
7   def select_retrieval_heads(train_queries, model, tokenizer, tools, device, max_heads=20):
27      head_scores = torch.zeros(num_layers, num_heads, device=device)   # ◀ [16, 32] score per head
29      for qix in tqdm(range(len(train_queries))):                       # ◀ 200 training queries (run3 default)
34          tool_ids = list(tools.keys())
35          random.shuffle(tool_ids)                                      # ◀ fresh tool order per query
50          prompt = putils.create_prompt(query=question)
55          attentions = model(**inputs).attentions
⋮       # 57-70: same query-span search as A.2
72          for layer_idx in range(num_layers):
73              attn = attentions[layer_idx][0]                           # ◀ [32, N, N]
74              for head_idx in range(num_heads):
76                  query_attn = attn[head_idx][query_start:query_end, :] # ◀ ONE head, query rows
78                  doc_scores = torch.zeros(len(item_spans), device=device)
79                  for doc_i, (ds, de) in enumerate(item_spans):
80                      doc_scores[doc_i] = query_attn[:, ds:de].mean()   # ◀ this head's score for each tool
83                  ranked = torch.argsort(doc_scores, descending=True)
84                  if ranked[0].item() == gold_tool_id:
85                      head_scores[layer_idx, head_idx] += 1             # ◀ HIT COUNT: +1 if this head ranks gold first
⋮       # 88-100: the commented-out alternatives (reciprocal rank 1/(rank+1); mean attention to the gold tool)
107     flat_scores = head_scores.flatten()                               # ◀ 512 scores
108     top20 = torch.argsort(flat_scores, descending=True)[:max_heads]   # ◀ best K heads (K = 10 / 20 / 30)
111     for idx in top20:
112         layer = idx.item() // num_heads                               # ◀ flat index → (layer, head)
113         head = idx.item() % num_heads
114         selected_heads.append((layer, head))
```

### A.5 Part 3: ranking test tools with only the selected heads: `run3.py`
```python
34  def query_to_docs_attention_heads(attentions, query_span, doc_spans, selected_heads):
49      doc_scores = torch.zeros(len(doc_spans), device=attentions[0].device)
52      for layer_idx, head_idx in selected_heads:                # ◀ e.g. (8,19), (6,11), (2,20), …
53          head_attn = attentions[layer_idx][0, head_idx]        # ◀ [N, N] for that one head
54          query_attn = head_attn[query_start:query_end, :]
56          for doc_i, (doc_start, doc_end) in enumerate(doc_spans):
57              doc_scores[doc_i] += query_attn[:, doc_start:doc_end].mean()
59      doc_scores /= len(selected_heads)                         # ◀ average over the K heads
⋮
104     train_queries=train_queries[:args.train_samples],        # ◀ 200 (argparse default, :88)
⋮
165     ranked_docs = torch.argsort(doc_scores, descending=True)
167     for i, tool_id in enumerate(ranked_docs):
168         if tool_id == gold_tool_id: gold_rank = i; break
173     if gold_rank<1: correct_at_1 += 1                         # ◀ K = 10 → 0.4068
175     if gold_rank<5: correct_at_5 += 1                         # ◀ K = 10 → 0.6694
```
| Name | What it is |
|---|---|
| `attentions` | tuple of 16 tensors `[1, 32, ~2972, ~2972]` (fp16) |
| `query_span` | `(start, end)` from A.2 |
| `doc_spans` / `item_spans` | 100 `(start, end)` tool spans from A.1 |
| `gold_tool_id` | index of the gold tool in this query's shuffled order |
