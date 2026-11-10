---
type: index
tags: [interview, diagrams, mini-vllm]
---
# mini-vLLM diagrams (Days 1–2)

Interactive Archify pages, built from the `vLLMmini` repo at commit `3679394`. Open each `.html` in a browser.
Every box links to the exact source lines it describes. Each page has cards with the numbers to say in an interview.

**Suggested order:** start with the system map. Then read the Day 1 pages before the Day 1 code blocks, and the Day 2 pages before the Day 2 ones.

| Day | Diagram | Use it for |
|---|---|---|
| 1–2 | [System map](day1-2-system-map.html) | Which class owns what; CPU policy vs GPU numerics |
| 1 | [One decoder layer](day1-decoder-layer-dataflow.html) | B0: tensor shapes, where K,V are written and read |
| 1 | [Life of a KV block](day1-block-lifecycle.html) | B1: refcounts, free list, publish and evict (Q6, Q7) |
| 2 | [One schedule() iteration](day2-schedule-workflow.html) | B1: running first, evict before preempt, admission gate |
| 2 | [Life of a request](day2-request-lifecycle.html) | Q1, Q4: WAITING → RUNNING → FINISHED, preemption by recompute |
| 2 | [Inside one engine.step()](day2-engine-step-sequence.html) | Q1: the full call order in one iteration |
| 2 | [Radix cache snapshot](day2-radix-tree.html) | B2, Q6, Q9: block-wise matching, LRU leaf eviction (illustrative values) |

Sources for regeneration live in `.archify/` (one `candidate.json` per diagram).
