---
type: study-block
day: 10
block: 2
tags: [interview, day10, clip, contrastive]
---
# D10-02 · CLIP theory (50 min)

Hub: [[Day 10 - Photo-Seek I]] · Previous: [[D10-01 Overview]] · Next: [[D10-03 CLIP code]]

This note follows the twelve questions in order. The worked 3-pair example started in the day note ([[Day 10 - Photo-Seek I#4. Deep dive: CLIP and cosine similarity from scratch|§4]]); here it is finished (both loss directions, temperature) and the missing ideas are added. Every number below was recomputed today.

> [!important] The one sentence
> CLIP is **two encoders trained together** so that a photo and its own caption become unit vectors pointing the same way. Photo-Seek **does not train it**; it only runs the trained encoders.

## 1. What is an embedding?

A list of numbers that stands for a thing. Photo-Seek's CLIP gives **512 numbers** per photo and per sentence. A single number means nothing; the **direction** of the whole list carries the meaning.

Shrink 512 dimensions to 2 and an embedding is an arrow on paper:

| thing | angle | vector (cos θ, sin θ) |
|---|---|---|
| dog photo | 10° | (0.985, 0.174) |
| cake photo | 80° | (0.174, 0.985) |
| hill photo | 45° | (0.707, 0.707) |
| "a dog" | 20° | (0.940, 0.342) |
| "a cake" | 70° | (0.342, 0.940) |
| "a hill" | 35° | (0.819, 0.574) |

## 2. Why must images and text be vectors?

A computer cannot compare a JPEG with a sentence. One is 150,528 pixel values (224 × 224 × 3), the other a string of characters. There is no meaningful "distance" between them.

But it **can** compare two lists of 512 numbers: one multiply-and-add per position. So both are turned into vectors, and "is this photo about this sentence?" becomes "how close are these two arrows?". That is cheap enough to do 5,000 times per query.

## 3. What is a *shared* embedding space?

Two models can each produce 512 numbers and still be useless together. If the image model puts dogs near (1, 0) and the text model puts the word "dog" near (0, 1), the numbers are not comparable.

**Shared** means: the same direction means the same thing for both encoders. The dog photo (10°) and "a dog" (20°) are neighbours. Nothing in the architecture forces this. **Only the training loss does** (section 6).

## 4. Why two encoders?

- **Different inputs need different front ends.** An image is cut into 32 × 32 pixel patches. A sentence is cut into word-piece tokens. One network cannot take both raw forms.
- **Separate encoders make search cheap.** Each photo is encoded **once**, at index time, with no query in sight. At query time only the short text is encoded. This design is called a *bi-encoder* (or two-tower).
- **The alternative** is one network that reads the image and the text together (a cross-encoder). It can be more accurate, but it must run once **per (query, photo) pair**: 5,000 forward passes per query instead of 1. [mine]

| | CLIP ViT-B-32 image tower [lib] | text tower [lib] |
|---|---|---|
| input | 224 × 224 RGB → 7 × 7 = 49 patches + 1 class token | up to 77 tokens |
| network | 12-layer Transformer, width 768 | 12-layer Transformer, width 512 |
| output | class token → linear projection → **512** | end-of-text token → linear projection → **512** |

Both end in a linear projection to the **same size**, 512. That is what lets the two outputs be compared.

## 5. How are image–text pairs used?

Training data is (image, caption) pairs scraped from the web. OpenAI's CLIP used about 400 million. Photo-Seek's checkpoint `laion2b_s34b_b79k` was trained by the open_clip authors on LAION-2B: the name says **34 billion samples seen, batch size 79k** [doc].

Nobody labels anything. The caption **is** the label. The only fact used is "this image came with this text".

## 6. What is contrastive learning?

Learning by comparison: **pull matching things together, push non-matching things apart.**

The model is never told "this is a dog". It is only asked: *of these N captions, which one came with this photo?* To answer, it must put the photo near its caption and far from the others.

## 7. The N × N similarity matrix

Take a batch of N = 3 pairs. Encode 3 photos and 3 captions, normalise all six vectors to length 1, and take every photo against every caption:

$$S_{ij} = \text{image}_i \cdot \text{text}_j \qquad S = I\,T^{\top} \quad (N \times N)$$

With the 2-D table above (each entry is $\cos$ of the angle gap):

| | "a dog" | "a cake" | "a hill" |
|---|---|---|---|
| dog photo | **0.985** | 0.500 | 0.906 |
| cake photo | 0.500 | **0.985** | 0.707 |
| hill photo | 0.906 | 0.906 | **0.985** |

Shapes at real size [lib]: `I` is `[N, 512]`, `T` is `[N, 512]`, `S` is `[N, N]`. With N = 79,000 the matrix has 6.2 billion entries.

## 8. Positive and negative pairs

- **Positives:** the diagonal. N of them (3 here). Photo *i* with its own caption.
- **Negatives:** everything off the diagonal. $N^2 - N$ of them (6 here). They cost nothing: they are just the other pairs in the batch ("in-batch negatives").

This is why CLIP wants **huge batches**. At N = 79,000 each photo is compared against 78,999 wrong captions, and some of those are *hard* (a different dog on a different beach). At N = 3 the task is too easy to teach much.

One flaw to know: a negative can be secretly correct. Two photos of dogs in one batch are each other's "wrong" answer. With web-scale data this noise is accepted. [mine]

## 9. Why a symmetric cross-entropy

**Row direction (photo → caption).** Read a row as scores, multiply by a scale $s$, softmax, and ask for the diagonal entry to win. With $s = 10$:

| row | scores × 10 | p(own caption) | loss = −ln p |
|---|---|---|---|
| dog photo | (9.85, 5.00, 9.06) | 0.683 | 0.381 |
| cake photo | (5.00, 9.85, 7.07) | 0.935 | 0.068 |
| hill photo | (9.06, 9.06, 9.85) | 0.523 | 0.648 |

Mean row loss = **0.366**.

**Column direction (caption → photo).** Read a column the same way:

| column | scores × 10 | p(own photo) | loss |
|---|---|---|---|
| "a dog" | (9.85, 5.00, 9.06) | 0.683 | 0.381 |
| "a cake" | (5.00, 9.85, 9.06) | 0.683 | 0.381 |
| "a hill" | (9.06, 7.07, 9.85) | 0.659 | 0.418 |

Mean column loss = **0.393**. Total loss = (0.366 + 0.393) / 2 = **0.38**.

**Why both?** Look at the cake. In its **row** the cake photo is confident (0.935): its nearest wrong caption is far. But in the "a cake" **column**, the hill photo scores 0.906, so the caption is unsure (0.683). One direction alone would miss that problem. Photo-Seek searches **text → image**, which is the column direction, so it matters that it was trained.

$$L = \tfrac12\Big(\underbrace{-\tfrac1N \sum_i \ln \frac{e^{sS_{ii}}}{\sum_j e^{sS_{ij}}}}_{\text{rows}} \; + \; \underbrace{-\tfrac1N \sum_j \ln \frac{e^{sS_{jj}}}{\sum_i e^{sS_{ij}}}}_{\text{columns}}\Big)$$

**What the loss "wants":** diagonal entries near 1, off-diagonal entries low. Random guessing on 3 pairs gives $\ln 3 = 1.10$; perfect alignment gives 0.

## 10. Temperature scaling

Cosines live in $[-1, 1]$. Fed to a softmax raw, they barely differ. The scale $s$ stretches them. Temperature is $\tau = 1/s$.

The hill photo's row, (0.906, 0.906, 0.985), at three scales:

| scale $s$ | temperature $\tau$ | probabilities | p(own caption) |
|---|---|---|---|
| 1 | 1 | (0.325, 0.325, 0.351) | 0.351: almost a coin toss among three |
| 10 | 0.1 | (0.239, 0.239, 0.523) | 0.523 |
| 100 | 0.01 | (0.000, 0.000, 0.999) | 0.999 |

Total loss on the whole batch: **0.949** at $s = 1$, **0.380** at $s = 10$, **0.278** at $s = 14.3$, about **0.000** at $s = 100$.

- **Small $s$:** a cosine gap of 0.08 is invisible. Gradients are tiny; the model cannot learn fine distinctions.
- **Large $s$:** small gaps become decisive. Too large too early makes training unstable.
- **CLIP learns it** [lib]: the parameter is `logit_scale`, stored as a log, initialised to $\ln(1/0.07)$ so $s \approx 14.3$, and clamped so $s \le 100$.

> [!note] Temperature here vs temperature in the LLM
> Same maths (divide logits before a softmax), different job. Here it shapes a **training loss**. In [[D10-06 Ollama and structured output]] it controls **randomness when sampling tokens**. And at search time CLIP's temperature is **not used at all**: ranking by $S$ and ranking by $sS$ give the same order.

## 11. Zero-shot retrieval and the 39.4

After training, searching needs no more training:

1. encode every photo once (index time);
2. encode the query text;
3. rank photos by cosine.

Query "a dog" at 15°: dog photo $\cos 5° = 0.996$, hill photo $\cos 30° = 0.866$, cake photo $\cos 65° = 0.423$.

**Zero-shot classification** is the same thing turned around: one image against several texts ("a photo of a dog", "a photo of a cake"); the best text is the label.

**"Zero-shot"** means CLIP saw none of *your* photos or *your* queries in training.

**The number** [doc: `eval/results/retrieval_coco_karpathy_ViT-B-32.json`]: COCO-5k, 5,000 images, all 25,010 captions as queries. **R@1 39.4, R@5 65.4, R@10 75.6.** Standard error ≈ ±0.3 points. It is off-the-shelf CLIP plus exact FAISS, measured on an H200. Through the real `photoseek search` path with the rule parser it is **35.9**, because the rules turn words like "baby" and "mother" into person filters that match nobody (410 captions were affected).

> [!warning] Training vs inference: keep them apart
> | | Training CLIP | What Photo-Seek does |
> |---|---|---|
> | who | OpenAI / the open_clip authors | you |
> | data | billions of web pairs | your photos; no captions |
> | computes | N × N matrix, loss, gradients, weight updates | one forward pass per image or text |
> | temperature | learned, used in the loss | unused |
> | code | not in your repo | `embed/clip.py` under `torch.inference_mode()` |
> If asked "how did you train CLIP?": "I did not. I use the pretrained `laion2b_s34b_b79k` weights zero-shot."

## 12. CLIP's limitations

| Limit | Example | What Photo-Seek does about it |
|---|---|---|
| **No identities** | "mom" means nothing to CLIP | Face pipeline + SQL `EXISTS` filter [code] |
| **No dates, no metadata** | "last summer" is not in the pixels | Rule/LLM parser + SQL on `taken_at` [code] |
| **Weak at counting and spatial relations** [lib] | "two dogs left of a cat" ≈ "a dog and a cat" | Nothing; a known gap |
| **Weak at negation** [lib] | "a beach without people" still likes people | Nothing |
| **Fine-grained classes** [lib] | dog breeds, car models | Nothing |
| **Text limit of 77 tokens** [lib] | long queries are cut | Queries are short; the parser removes filler |
| **Small absolute scores** | the top hits in `offline.json` score only 0.27–0.37 [doc] | Rank and return top k; never threshold on a score |
| **Inherits web bias and noise** [lib] | skewed toward common Western web content | Nothing; say so |
| **Not the best checkpoint** | ViT-L-14 or SigLIP would score higher | No ablation was run; the reason for B-32 is **not documented** [doc: decision 7] |

And the plainest limit: R@1 39.4 means that for **6 in 10** captions the right image was **not** first.

## Cosine similarity, in one place

Built from simple vectors in [[Day 10 - Photo-Seek I#4.2 Cosine similarity: "how close in angle?"|day note §4.2–4.3]]. The essentials:

$$\cos(a, b) = \frac{a \cdot b}{\lVert a\rVert\,\lVert b\rVert}$$

- $a = (3, 4)$, $b = (4, 3)$: dot 24, lengths 5 and 5, cosine **0.96**.
- $c = (6, 8)$ points the same way as $a$ but is twice as long. Raw dot $a \cdot c = 50$ looks "better" than 24 only because $c$ is long. Cosine = $50 / (5 \cdot 10) = 1.0$.
- **If both have length 1, the bottom is 1 and cosine = dot product.** Photo-Seek normalises every vector once, so FAISS's plain inner product is the cosine ([[D10-05 FAISS code#3. Why inner product on unit vectors is cosine]]).

## Check

1. In a batch of 79,000 pairs, how many positives and negatives does the matrix hold?
2. Why does CLIP use two separate encoders instead of one joint network?
3. The "a cake" column is (0.500, 0.985, 0.906) at $s = 10$. Which wrong photo takes the most probability, and why does the row direction not show this?
4. What happens to the hill row's confidence when $s$ goes from 10 to 100?
5. Is temperature used when Photo-Seek ranks photos?
6. Name three things CLIP cannot do that Photo-Seek handles outside CLIP.

> [!success]- Answers
> 1. 79,000 positives; $79{,}000^2 - 79{,}000 \approx 6.24$ billion negatives.
> 2. Different raw inputs, and photos can be encoded once ahead of time; a joint network would need one pass per (query, photo) pair.
> 3. The hill photo (0.906), with about 0.31 of the probability. In the cake photo's **row**, the competing captions are 0.500 and 0.707, so that direction looks fine.
> 4. 0.523 → 0.999: a 0.08 cosine gap becomes decisive.
> 5. No. Scaling all scores by the same positive number does not change their order.
> 6. Identify people, understand dates, filter by place label: all done by the parser and SQL.

## Five things to remember

1. Two towers, one 512-d space; the **loss** is what makes the space shared.
2. Batch of N → N × N cosine matrix → diagonal = positives, the rest = free negatives.
3. Symmetric cross-entropy: rows (photo → caption) and columns (caption → photo); 0.38 in the worked example.
4. Temperature stretches cosines for the softmax; learned in training, irrelevant for ranking.
5. **Pretrained, zero-shot.** R@1 39.4 on COCO-5k is CLIP's number, not a trained result of yours.
