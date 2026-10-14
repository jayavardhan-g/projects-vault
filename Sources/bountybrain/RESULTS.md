# Results

All numbers produced on 2026-08-16 from the committed code and the datasets
`fetch_data.py` downloads. Raw JSON for every table is in `artifacts/`, which
is committed precisely so these can be checked rather than taken on trust.

Reproduce with:

```bash
python fetch_data.py
python build_graph.py --rebuild     # graph + metrics
python index_corpus.py              # chunk + embed
python serve_llm.py                 # only needed for the LLM arms
python eval.py --all
python report.py                    # re-render the README tables
```

Environment: 1× H100-class GPU for vLLM (Qwen2.5-14B-Instruct), 1× for
embeddings (`bge-small-en-v1.5`) and reranking (`bge-reranker-base`).

---

## 1. Corpus and knowledge graph

Built from eight bulk-downloaded, openly licensed datasets. No scraping.

| | count |
|---|---:|
| **Nodes** | **111,896** |
| **Edges** | **96,805** |
| exploit | 46,647 |
| CVE | 29,971 |
| signature (nuclei) | 13,569 |
| product | 10,901 |
| document (retrievable) | 9,224 |
| weakness (CWE) | 969 |
| technique (CAPEC) | 615 |

Documents by type: 6,397 disclosed writeup records · 1,243 prose articles ·
969 CWE pages · 615 CAPEC patterns. Chunked into **11,355** indexed passages.

### Structure

| metric | value |
|---|---|
| density | 1.55 × 10⁻⁵ |
| mean / median / max degree | 1.73 / 1.0 / 3,761 |
| connected components | 46,998 |
| largest component | 35,425 (31.7%) |
| isolated nodes | 24,905 |

The whole-graph figures look sparse, and that is honest rather than
disappointing: 77k of the 112k nodes are CVE and exploit records that no
nuclei template references, so they sit in two-node components. Excluding them
(`export_graph.py --core-only`) leaves the part that carries the reasoning —
**35,278 nodes, 63,340 edges, mean degree ≈ 3.6**.

### CWE taxonomy

10 pillar roots · 687 leaves · max depth 5 · mean leaf depth 2.58
(299 Variant, 539 Base, 114 Class, 10 Pillar, 7 Compound), from 1,441 typed
weakness→weakness edges.

### Coverage of the 969 weakness classes

| backed by | count | share |
|---|---:|---:|
| an observed CVE | 581 | 60.0% |
| an external document (writeup / article / CAPEC) | 365 | 37.7% |
| a CAPEC attack pattern | 336 | 34.7% |
| a nuclei detection signature | 166 | 17.1% |
| a disclosed bug bounty writeup | 89 | 9.2% |
| a Bugcrowd VRT payout priority | 57 | 5.9% |

"Has *any* document" is 100% by construction — every CWE has its own catalogue
page in the corpus — so it is excluded as meaningless.

### Entity resolution

**1,389 aliases → 969 classes**, from three layers: official CWE names,
official `Alternate_Terms` plus short names hidden in parentheticals, and 109
hand-curated community terms (IDOR, SSTI, WAF bypass, …).

Applied to pentester.land's 671 distinct curated labels this resolves **93.1%**
of vulnerability-labelled writeups (up from **76.5%** before the parenthetical
and curated layers). 22 labels are platform/engagement tags ("Android",
"Cloud", "Security code review") and are excluded from the denominator rather
than force-mapped.

### Economics (from disclosed bounties)

Best-paying weakness classes with ≥10 paid writeups:

| CWE | name | paid | mean | median | max |
|---|---|---:|---:|---:|---:|
| CWE-787 | Out-of-bounds Write | 30 | $42,326 | $20,000 | $215,000 |
| CWE-78 | OS Command Injection | 11 | $29,964 | $4,000 | $288,500 |
| CWE-94 | Code Injection | 134 | $21,184 | $5,550 | $288,500 |
| CWE-269 | Improper Privilege Management | 52 | $20,806 | $8,000 | $200,000 |
| CWE-89 | SQL Injection | 40 | $12,934 | $3,000 | $288,500 |
| CWE-918 | SSRF | 67 | $12,695 | $3,500 | $288,500 |

---

## 2. Retrieval — four-way ablation

200 model-generated queries over held-out passages, stratified across the four
document types. Queries are committed to `bountybrain/fixtures/retrieval_queries.json`.

| mode | R@1 | R@5 | R@10 | MRR | nDCG@10 | median ms |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (lexical only) | 0.655 | 0.850 | 0.890 | 0.735 | 0.764 | 16 |
| dense (embeddings only) | 0.665 | 0.880 | 0.925 | 0.758 | 0.790 | 38 |
| hybrid (RRF fusion) | 0.675 | 0.910 | 0.955 | 0.772 | 0.808 | 53 |
| **hybrid + cross-encoder rerank** | **0.800** | **0.965** | **0.975** | **0.861** | **0.873** | 149 |

By document type (hybrid + rerank):

| document type | n | R@5 | MRR |
|---|---:|---:|---:|
| writeup | 50 | 1.000 | 0.952 |
| technique (CAPEC) | 50 | 0.980 | 0.881 |
| article | 50 | 0.940 | 0.794 |
| weakness (CWE) | 50 | 0.940 | 0.819 |

**Reading it.** Fusion buys little on its own (R@1 0.675 vs 0.665 dense) — the
two retrievers largely agree on *which* documents matter. The cross-encoder is
what pays: +0.125 R@1 over fusion, because the win is in *ordering* the
candidates, not in finding them. It costs ~3× latency, so the cheaper arms
remain the right default for bulk work.

**Caveats.**
- Relevance is known-item: only the source document counts, so genuinely
  relevant results from other documents are scored as misses. Absolute recall
  is a **lower bound**; the between-arm comparison is unaffected.
- Queries are LLM-generated with an instruction not to copy distinctive tokens.
  That instruction is not verified mechanically — some lexical overlap
  certainly survives, which flatters BM25.
- Writeup documents are short metadata records, which is why they score
  highest: less text, less ambiguity.

**Bug found and fixed here.** The first run reported nDCG@10 = 1.113. The ideal
DCG assumed a single relevant item, but a gold document is chunked and can
contribute several relevant hits. Normalising against all relevant hits found
fixes it (0.873). The impossible value is what exposed the bug.

---

## 3. CWE classification — and a negative result

> **⚠️ These numbers are superseded and are being re-run.** The comparison
> below was **confounded**: `classify_keyword` reads the alias table straight
> from the CWE catalogue and so had all 109 curated community terms (IDOR,
> SSTI, …), while `classify_embedding` and `classify_llm` shortlist from
> `build_candidate_index()`, which embeds each class from its *stored*
> `aliases_json` — and the builder never persisted the curated layer. The
> baseline held vocabulary the other two arms were denied. The builder now
> persists it (`_merged_aliases`), so re-running with a forced candidate-index
> rebuild produces the first apples-to-apples comparison. Until then, treat
> the tables below as a record of the confounded run, not as a finding.
>
> The confound is expected to have hurt the LLM and embedding arms most on the
> **writeup-title** set, where practitioner slang dominates, and least on
> nuclei, whose formal template prose rarely uses it — which is the pattern the
> old numbers show. Whether the keyword baseline still wins after the fix is
> genuinely open.

Two independent gold sets, three strategies, identical inputs. 500 items each.

**Gold set: nuclei templates** (name + description + impact)

| strategy | accuracy | hierarchy ±2 | macro F1 | abstained | accuracy when answered |
|---|---:|---:|---:|---:|---:|
| **keyword** | **0.398** | 0.444 | 0.216 | 0.354 | 0.616 |
| llm | 0.368 | 0.440 | 0.097 | 0.314 | 0.536 |
| embedding | 0.184 | 0.300 | 0.044 | 0.000 | 0.184 |

**Gold set: disclosed writeups** (title only)

| strategy | accuracy | hierarchy ±2 | macro F1 | abstained | accuracy when answered |
|---|---:|---:|---:|---:|---:|
| **keyword** | **0.402** | 0.426 | 0.299 | 0.524 | 0.845 |
| llm | 0.280 | 0.342 | 0.114 | 0.392 | 0.461 |
| embedding | 0.188 | 0.314 | 0.065 | 0.000 | 0.188 |

**The retrieval-augmented LLM classifier does not beat a well-built alias
table.** That is the honest headline, and it is worth more than a flattering
number would be.

What the numbers say:

- On **nuclei** the two are close (0.398 vs 0.368) and near-identical on the
  hierarchy-aware score (0.444 vs 0.440). The LLM lands in the right
  neighbourhood about as often; it just picks a sibling more frequently.
- On **writeup titles** the LLM is clearly worse (0.280 vs 0.402). A title
  states an *outcome* ("Escalating From Reader To Contributor…") and the model
  must infer the root weakness, where it commits confidently to a plausible
  wrong class.
- The LLM's **macro F1 is much lower** on both sets (0.097 / 0.114 vs
  0.216 / 0.299), which says it concentrates predictions on common classes and
  does poorly on the tail — the opposite of what the shortlist was meant to
  buy.
- **Embedding-only is weak everywhere** (~0.18). Matching a terse vulnerability
  description against a formal CWE definition is a hard asymmetric task for a
  small bi-encoder, and it never abstains, so every hard case becomes a wrong
  answer.

**A fairness problem with the writeup gold set.** Its gold labels are produced
by applying the alias table to the curated `Bugs` field, while the keyword
baseline searches the *title* for entries from that same table. The inputs
differ (title vs. label field), so it is not straightforwardly circular — but
both share a vocabulary, and whenever a title happens to name the bug the
keyword baseline wins for free. **The nuclei set is the fair comparison**, and
there the gap is small. This caveat should be read as limiting the
writeup-set conclusion, not the nuclei one.

**What this changes.** The keyword classifier, not the LLM, is the right
default for bulk CWE assignment here: it is faster, free, and at least as
accurate. The LLM's value in this system is generating cited prose, which the
retrieval evaluation supports, not label assignment.

---

## 4. Recon planner — next-move ranking

400 held-out cases across 147 products. For each, 20% of a product's detection
templates are held out as gold **and excluded from the evidence pool**, so the
planner must generalise from the rest of the stack rather than being graded on
the fact it was handed.

| strategy | P@1 | P@3 | P@5 | P@10 | MRR |
|---|---:|---:|---:|---:|---:|
| **planner (value-weighted)** | **0.427** | 0.562 | 0.650 | 0.720 | 0.517 |
| planner (likelihood-only) | 0.422 | **0.570** | **0.662** | **0.738** | **0.522** |
| prior (global class frequency) | 0.325 | 0.532 | 0.598 | 0.720 | 0.445 |
| random | 0.095 | 0.282 | 0.440 | 0.720 | 0.245 |

**Beating the prior is the bar that matters** — it is what shows the planner
uses the recon fingerprint rather than the base rate of XSS being common. It
clears it by +31% relative on P@1 and +16% on MRR.

**A defect this evaluation caught.** The first version *lost* to the prior
(P@1 0.280 vs 0.325). Its applicability term was a log-scaled count of matching
templates, which saturates at 1.0 for any popular product — every ranking was
then decided by stack-independent terms. Replacing it with a smoothed
conditional estimate, P(weakness | product) shrunk toward the global prior by a
pseudo-count of 25, took P@1 from **0.280 → 0.427**.

**Value weighting costs ranking accuracy, as it should.** Likelihood-only is
slightly better at predicting the next finding (P@3 0.570 vs 0.562); the
value-weighted variant deliberately trades that to surface findings worth more.
Both are exposed in the UI; reporting only the flattering one would hide a real
trade-off.

**Residual leak, stated.** The global prior used for smoothing is computed over
all templates including held-out ones. That is ~5% of edges spread across
classes, and the `prior` baseline is computed the same way, so the comparison
stands — but the per-product evidence exclusion is the strict part, and this is
not.

---

## 5. Defects found and fixed

Recorded because they are the substance of the work, not incidental:

| defect | consequence | fix |
|---|---|---|
| nDCG ideal assumed one relevant chunk | nDCG@10 = 1.113, impossible | normalise against all relevant hits found |
| planner applicability saturated at 1.0 | lost to the trivial prior baseline | smoothed conditional P(weakness \| product) |
| curated aliases never persisted to the graph | "IDOR" resolved nowhere in the UI/API, **and** the classification comparison was confounded — only the keyword baseline could see the curated vocabulary | builder merges the curated layer into `aliases_json` |
| chain walk followed `CanPrecede` only | CAPEC states half its links as `CanFollow`, so the Attack Chains page was empty for every topic | traverse both directions |
| nuclei `cve-id` is a string, not a list | `"CVE-2021-22205"` became 14 findings, one per character | normalise to a list before iterating |
| every parser grouped by registrable domain | GitLab and Jira hosts merged into one blended plan | carry `host`; plan per host |
| host spellings differ across tools | `git.acme.example` and `https://git.acme.example` became two assets | normalise at every assignment site |
| 6 curated aliases pointed at CWE-840/CWE-16 | those are *categories*, not weaknesses — dangling edges | remapped; a test now asserts every alias resolves |
| ExploitDB ships ~460 duplicate rows | `edb_id` UNIQUE constraint violation | de-duplicate in the loader |
| Qdrant closed from `__del__` at teardown | ImportError traceback after every successful run | deterministic `atexit` close |

---

## 6. Honest limitations

- **Writeup bodies were never fetched.** Only pentester.land metadata is
  ingested. Writeup "documents" are short records, not article text.
- **Retrieval recall is a lower bound** (known-item relevance).
- **The LLM classifier result is currently unmeasured**, not merely negative —
  the run that produced it gave the keyword baseline a vocabulary advantage.
  See the warning in §3 and step 1 of PROGRESS.md.
- **Coverage is thin in places** — only 17.1% of weakness classes have a
  detection signature and 9.2% a disclosed writeup, which limits the planner to
  well-represented stacks.
- **Planner weights are hand-set, not learned.** The evaluation measures
  whether they work; it does not claim they are optimal.
- **Faithfulness of generated tutorials is not yet measured.** Citations are
  required by the prompt and rendered in the UI, but no suite verifies that
  every claim traces to a retrieved passage.
