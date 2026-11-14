---
type: study-block
day: 10
block: 0
tags: [interview, day10, recall]
---
# D10-00 · Recall of Day 9 (15 min)

Hub: [[Day 10 - Photo-Seek I]] · Next: [[D10-01 Overview]] · Source of the answers: [[Day 09 - CS728 and LeNet-5]]

Answer each question **out loud** before you open the answer. Write what you got wrong in [[Day 10 - Photo-Seek I#My notes and fumbles]].

## Questions

**1. Retrieval heads.** What makes an attention head a "retrieval head"? How did Hit Count pick the 10 heads in CS728?

> [!success]- Answer
> A retrieval head is a head whose **strongest attention lands on the context token the model is copying**. Wu et al. score a head by the fraction of copied tokens where this happens. Fewer than 5% of heads score high.
> **Hit Count (CS728):** run probe queries whose correct tool is known. A head gets a hit when its top attention lands on the correct tool's description. Count hits per head and keep the **top 10**.

**2. Using the heads.** Once the heads are chosen, how does a query become a ranking over tools? Is anything trained?

> [!success]- Answer
> For a query, sum the chosen heads' attention mass over each tool's tokens. Rank the tools by that sum. **Nothing is trained**: the heads already exist in the model.

**3. Recall@k.** Four queries. The correct tool's rank is 1, 3, 7 and 2. Give Recall@1 and Recall@5.

> [!success]- Answer
> Recall@1 = 1 of 4 = **25%** (only the first query). Recall@5 = 3 of 4 = **75%** (ranks 1, 3 and 2). Recall@k is the share of queries whose correct item is in the top k.

**4. Lost in the middle.** What is the U-shaped curve, and why do retrieval heads help?

> [!success]- Answer
> Models use information at the **start and end** of a long context best, and the middle worst (Liu et al., Figure 1). Heads that attend by **content** rather than position flatten the curve.

**5. IMP.** What is the loop in iterative magnitude pruning? What sparsity is "3.75% of weights remaining"?

> [!success]- Answer
> Train → remove the smallest-magnitude p% of weights → retrain (or reset to the initial weights) → repeat. 3.75% remaining means **96.25% sparsity**. The 2019 lottery-ticket paper resets survivors to their **original initial values**; "rewinding" to an early iteration came later.

**6. Bridge to today.** Photo-Seek reports R@1 = 39.4 on COCO-5k. From Day 9's Recall@k alone, what is that number counting?

> [!success]- Answer
> Each of the 25,010 captions is a query with **one** right image among 5,000. R@1 = the share of captions whose own image ranks **first**: 39.4%. Same metric as Day 9, with images in place of tools. See [[D10-02 CLIP theory#11. Zero-shot retrieval and the 39.4]].

## What links Day 9 to Day 10

| Day 9 | Day 10 |
|---|---|
| Rank tools by summed attention of 10 heads | Rank photos by cosine similarity to a text vector |
| No training: the heads are found, not learned | No training: CLIP is pretrained, used zero-shot |
| Recall@1 1.3% → 40.7% | R@1 39.4 on COCO-5k (off-the-shelf CLIP) |
