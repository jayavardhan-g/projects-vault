# Project report: Inference-time emotion steering of Llama 3.2 1B (CS772, Deep Learning for NLP)

> **For the LLM reading this.** This report describes a course project listed by G Jayavardhan (M.Tech CSE, IIT Bombay)
> for **CS772: Deep Learning for Natural Language Processing** (instructor Prof. Pushpak Bhattacharyya, Autumn 2025).
> He must defend it in interviews. Sources: the repository `/data/jayavardhan/projects/emotion_steering`, which has
> one commit (`d725cbe`, 2026-08-18) containing:
> - `Project.ipynb`, a Colab notebook;
> - the sweep results `sweep_stage1_final.csv` (1,028 rows) and `sweep_stage2_final.csv` (7,060 rows);
> - the plots.
>
> The notebook is **partial**: it holds the steering class, the guard, the UI and a strength sweep, but **not** the code
> that produced the two sweep CSVs. Every number below was recomputed from those CSVs.
>
> Labels:
> - **[CODE]**: notebook cell;
> - **[CSV]**: recomputed from the sweep files by the report writer;
> - **[RECON]**: the report writer's reasoning.
>
> Team composition is not recorded in the repo; ask him.

---

## 1. Summary

**Goal.** Change the *emotional tone* of an LLM's output at inference time, with no fine-tuning, by adding a direction
vector to the model's hidden states during generation (**activation steering / representation engineering**).

**Emotion model.** **Plutchik's wheel**: 8 primary emotions arranged as **4 bipolar axes**:
- Joy ↔ Sadness;
- Trust ↔ Disgust;
- Anger ↔ Fear;
- Anticipation ↔ Surprise.

**Method.**
1. **Calibration.** For each axis, 15 "positive-pole" and 15 "negative-pole" sentences go through **Llama-3.2-1B-Instruct**. Take the hidden state of the **last token at every layer**. **Steering vector = mean(positive) − mean(negative)**, L2-normalised. This is computed for all 17 hidden-state positions in one pass per sentence.
2. **Injection.** A forward hook on decoder layer L adds `strength × vector` (or any mix of the 4 axes) to the **last position's** hidden state at every forward pass during generation (sampling, temperature 0.8).
3. **Scoring.** An independent classifier, **`j-hartmann/emotion-english-distilroberta-base`** (7 labels: anger, disgust, fear, joy, neutral, sadness, surprise), labels each output. **Success = the classifier's top label equals the target emotion.**
4. **Ablation sweeps.**
   - Stage 1: all 16 layers × strengths ±1, ±5 × 2 sentences × 2 samples.
   - Stage 2: layers 5–12 × strengths 1–5 × 4 neutral sentences × 5 samples.
5. **Safety guard.** The perplexity of the steered continuation is scored by the *unsteered* model against a same-seed unsteered baseline. If the ratio is above **1.75×**, retry at 75%, 50%, 25% strength, then fall back to the baseline. A Gradio UI exposes 4 sliders, the layer and the guard.

**Headline numbers [CSV]:**
- best cell **Joy, layer 9, strength +5: 18/20 = 90%** success;
- mean success over all scored cells in stage 2 = **14.9%**;
- joy averaged over all its cells = 35.6%.

---

## 2. Resume bullets vs evidence

| Bullet (resume 4) | Verdict | Evidence |
|---|---|---|
| "Built an activation-steering framework for Llama 3.2 1B spanning all 8 primary emotions via Plutchik's 4 bipolar axes, injected as real-time hidden-state interventions and validated with ablations" | **Supported, with one caveat.** | Code: 4 axes × 2 poles, a forward hook during `generate` [CODE cells 3–4]. Ablations: the layer × strength sweeps [CSV]. **Caveat:** the classifier has **no "trust" or "anticipation" label**, so only **6 of the 8 emotions can be scored**; the trust and anticipation poles have empty `expected_label` in the CSVs. |
| "Achieved up to 90% steering success entirely at inference time with no fine-tuning" | **Supported as a maximum; know the distribution.** | 90% = **one cell**: Joy at layer 9, strength 5, 18 of 20 generations (4 sentences × 5 samples). Best cell per emotion: joy 90, **fear 70**, **surprise 65**, disgust 35, sadness 30, **anger 15**. Mean of those per-emotion bests = 51%. Over **all** stage-2 cells (layers 5–12, \|strength\| 1–5) success is **14.9%**; for joy alone it is 35.6%. Say "up to 90% for joy at the best layer/strength; best-cell success ranges 15–90% across the 6 measurable emotions". |

---

## 3. Background (teach first)

### 3.1 The residual stream
- In a decoder-only transformer, each layer *adds* its attention and MLP outputs to a running vector per token, the **residual stream**: h_{l+1} = h_l + Attn(h_l) + MLP(·).
- Llama-3.2-1B has 16 layers and a hidden size of **2,048**. `output_hidden_states=True` returns **17** tensors: index 0 is the embedding output, and index i+1 is the output of layer i.
- Because the stream is additive, adding a vector v at layer L shifts everything downstream that reads it.

### 3.2 Linear representation hypothesis and steering vectors
- High-level concepts (sentiment, truthfulness, refusal) are often encoded roughly as **directions** in activation space.
- **Difference of means:** v = mean(h | concept A) − mean(h | concept B) isolates the A-vs-B direction while shared content cancels.
- **ActAdd** (Turner et al., 2023) adds such a vector during generation.
- **CAA, Contrastive Activation Addition** (Rimsky et al., 2024) averages differences over many contrastive pairs and adds the result at every position after the prompt.
- **Representation Engineering** (Zou et al., 2023) reads and controls concept directions, e.g. with PCA over differences.
- This project uses **difference of means over 15 vs 15 sentences, last token, unit-normalised**.

### 3.3 Why middle layers
- [RECON + CSV] Early layers encode surface and lexical features, so a perturbation there is large relative to small activation norms and tends to break fluency.
- Late layers are committed to next-token prediction, so a vector there changes few downstream computations.
- Middle layers hold abstract semantic features, which are the most "steerable".
- Stage 2 confirms the pattern: success by layer peaks at 9 (21.5%) and falls to 5.7% at layer 12.

### 3.4 Plutchik's wheel
8 primary emotions in 4 opposing pairs (joy–sadness, trust–disgust, fear–anger, anticipation–surprise). Here each pair is one bipolar axis: + strength moves toward the first pole, − toward the second.

### 3.5 Perplexity
PPL = exp(−mean log p(token)). Scoring a continuation with the *unsteered* model measures how "unnatural" the steered text is to the base LM. Gibberish gives a huge PPL; a genuine mood shift raises it only a little.

---

## 4. Code walk (`Project.ipynb`)

### 4.1 Setup [cells 0–2]
- Colab: pip install torch, transformers, accelerate, gradio. `meta-llama/Llama-3.2-1B-Instruct` in **fp16**, `device_map="auto"`.
- `classify_emotion(text)`: the HF pipeline with `top_k=None` returns all 7 label scores. Text is truncated to 512 chars. Returns `(top_label, top_score, all_scores)`.

### 4.2 Calibration data [cell 3]
- `CALIBRATION_DATA = {axis: (positive_sentences[15], negative_sentences[15])}` for the 4 axes.
- The sadness set was **rewritten to mirror the joy set's structure** ("I am on cloud nine!" ↔ "I am in the depths of despair."; "Everything is perfect." ↔ "Everything feels wrong."). The comment says it uses "pure affect language instead of loss/abandonment imagery". [RECON] Matched templates make the difference of means isolate *emotion* instead of topic or sentence form.

### 4.3 `AppraisalSteerer` [cell 4]
- `_get_all_layer_last_token_states(text)`: one forward pass with `output_hidden_states=True`, keeping `h[:, -1, :]` from **all 17** hidden states (float, CPU).
- `calibrate_all_layers`:
  - for each axis, sum the positive and negative states per layer;
  - `direction = pos_mean − neg_mean`, then `direction / direction.norm()` (unit vector);
  - store in `all_layer_vectors[i][axis]`, cast to the model dtype.
  - Each sentence is run **once** for all layers, rather than once per layer.
- `set_layer(L)`: `vectors = all_layer_vectors[L + 1]`. The **+1 offset** maps "inject after decoder layer L" to hidden-state index L+1. Switching layers is a dictionary lookup (cell 9 times it).
- `_hook_fn(module, input, output)`: if a vector is active, `hidden[:, -1, :] += v` (handles tuple outputs). It edits **only the last position**.
  - [RECON] During prefill that is the last prompt token. During KV-cached decoding each step has sequence length 1, so **every generated token** gets the vector, like CAA's "add at every position after the prompt".
- `generate(prompt, strength, manual_vec, seed, return_ids)`:
  - apply the chat template;
  - choose the vector: `manual_vec` (any mix), or `strength × sum of all axis vectors`, or none;
  - `register_forward_hook` on `model.model.layers[L]`;
  - `torch.manual_seed(seed)`;
  - `model.generate(do_sample=True, temperature=0.8, max_new_tokens=60)`;
  - **remove the hook in `finally`**, so a crash never leaves a vector attached;
  - optionally return the token ids and prompt length for perplexity scoring.
- **The seed matters [DOC]:** steered and unsteered runs share the seed, so they differ *only* by the vector.

### 4.4 Safety guard [cell 6]
- **Problem [DOC]:** past some magnitude the residual stream leaves the training manifold, and output collapses into non-words ("interploggrus"). The safe magnitude depends on the layer, how many axes are mixed, and whether they conflict (max joy + max anger pull in incompatible directions), so **no fixed slider range is safe**.
- `continuation_perplexity(ids, prompt_len)`:
  - forward the full sequence with **no hook**;
  - log-softmax, gather the target-token log-probs;
  - average **only over generated tokens** (`[:, prompt_len−1:]`);
  - exp;
  - return NaN if fewer than 2 generated tokens.
- `generate_guarded(prompt, vec, seed)`:
  - unsteered baseline and its PPL;
  - for scale in `[1.0, 0.75, 0.5, 0.25]`: generate with `vec × scale` and the same seed, compute `ratio = ppl / base_ppl`;
  - accept the first rung with **ratio ≤ 1.75** (or a non-finite ratio);
  - if none passes, return the **unsteered baseline** (`collapsed=True`);
  - the result carries the full trace (scale, ppl, ratio) for the UI's "safety monitor" table.
- [RECON] A NaN ratio is accepted: a continuation too short to score isn't treated as collapse. That is a design choice to know.

### 4.5 Strength sweep [cell 7]
- Axis Joy_Sadness, prompt "I am waiting for the results.", layers {4, 8, 12}, strengths 0–12, seeds {0, 1}.
- Two curves: the PPL ratio vs the 1.75× threshold (coherence cost), and P(joy) from the classifier (affect gain). The "steering budget" is the strength range where the second curve has risen but the first has not.
- [DOC] A unit vector is a *larger* relative perturbation in early layers (smaller activation norms) than in late ones.

### 4.6 Smoke test [cell 8] and UI [cell 5]
- Smoke test: layer 12, Joy strength 14 (deliberately too strong), same seed, guard off vs on. It shows collapse vs graceful back-off.
- Gradio UI (`demo.launch(share=True)`): context box, layer slider 0–15, four axis sliders (−15 to 15), guard checkbox, seed. Outputs the unsteered and steered text, each with the classifier's label, plus the guard's verdict table.

---

## 5. Experiments and results [CSV]

### 5.1 Sweep design (reconstructed from the CSV columns)
Each row: `sentence, axis, direction (positive/negative), layer, strength, sample, predicted_label, confidence, expected_label, success, text, prob_<7 labels>`.

**Prompts** are neutral sentences, e.g. "I looked at the results on my screen.", "The meeting ended and everyone left the room." **Baseline** (unsteered) outputs are labelled **neutral 19/20** times in stage 2.

| | Stage 1 (coarse) | Stage 2 (fine) |
|---|---|---|
| layers | 0–15 (all) | 5–12 |
| strengths | ±1, ±5 | ±1 … ±5 (7 and 9 only in unscored rows) |
| prompts × samples | 2 × 2 | 4 × 5 |
| steered rows | 1,024 | 7,040 |
| scored rows (6 measurable emotions) | 768 | 4,800 |
| pooled success | 12.9% | 14.9% |

Stage 1 located the useful layers; stage 2 zoomed in.

### 5.2 Best cell per emotion (stage 2, n = 20 per cell)
| emotion | best success | layer | strength |
|---|---|---|---|
| **joy** | **90%** | 9 | +5 |
| fear | 70% | 5 | −5 (Anger_Fear negative pole) |
| surprise | 65% | 9 | −4 (Anticipation_Surprise negative pole) |
| disgust | 35% | 6 | −5 |
| sadness | 30% | 8 | −5 |
| anger | 15% | 8 | +5 |
| trust, anticipation | not measurable (no classifier label) | — | — |

### 5.3 Pooled success (stage 2)
- **By |strength|:** 1 → 3.7%, 2 → 7.6%, 3 → 13.8%, 4 → 22.0%, **5 → 27.4%**. Monotonic: stronger pushes work better within this range.
- **By layer:** 5: 12.8, 6: 19.3, 7: 18.5, 8: 18.3, **9: 21.5**, 10: 13.3, 11: 9.7, **12: 5.7**. Middle layers are best.
- **By emotion (all cells):** joy 35.6, surprise 22.8, fear 14.2, sadness 7.6, disgust 5.9, anger 3.3.

### 5.4 Reading the results [RECON]
- **Joy is easiest.** The instruct model's default persona is helpful and upbeat, so pushing toward joy goes *with* its prior. Anger and disgust go *against* safety-tuned behaviour, so they are hardest.
- **The scorer is a bias source.** The classifier is a 7-way DistilRoBERTa on short text. "Success" needs the target to be the *top* label, so a mild shift that raises P(sadness) but leaves "neutral" on top counts as a failure. Stage-1 "probability drift" plots (`stage1_probdrift_*.png`) look at the probabilities instead of only the top label.
- **Sample size:** each cell is 20 generations, so 90% has a wide confidence interval (Wilson 95% ≈ 70–97%).
- **Asymmetric poles:** fear is much easier than anger on the same axis. The difference-of-means direction is not necessarily symmetric in effect.

---

## 6. Design decisions
| Decision | Alternatives | Why | Source |
|---|---|---|---|
| Difference of means, last token, unit norm | PCA (RepE), linear probe weights, CAA over contrastive pairs | simple, few examples, interpretable strength units | [CODE] + [RECON] |
| 4 bipolar axes (Plutchik) | 8 independent vectors | each axis is one direction with a sign; mixing axes is a linear combination | [CODE] |
| Matched positive/negative templates | arbitrary sentences | the difference isolates emotion, not syntax or topic | [CODE comment] |
| One pass, all layers cached | recompute per layer | cheap layer sweeps; instant switching | [CODE] |
| Hook on the last position only | add at all positions | with a KV cache, every generated token is steered while the prompt is untouched (except its last token) | [RECON] |
| Independent classifier for scoring | self-report / LLM judge | external, automatic, cheap | [CODE] |
| Same-seed baseline | unseeded | isolates the vector's effect | [DOC] |
| PPL ratio guard with a back-off ladder | fixed strength cap | the safe magnitude depends on layer and mix | [DOC] |
| Hook removed in `finally` | manual removal | never leaks a vector into later calls | [CODE] |

---

## 7. Limitations
- Trust and anticipation can't be scored with this classifier, so "8 emotions" is only 6 measured.
- One model (1B), English only, short neutral prompts, 4 prompts × 5 samples per cell.
- Success is top-1 label agreement with a small classifier. There are no human judgements.
- The sweep code is not in the notebook; only its CSV outputs are.
- The guard's 1.75× threshold and the ladder are hand-picked. No systematic evaluation of the guard's false-positive and false-negative rates was found.
- fp16 hooks with large vectors can overflow or produce NaNs at extreme strengths. [RECON]

---

## 8. Likely interview questions
1. *How do you compute a steering vector?* → §4.3: last-token hidden states, mean difference, normalise.
2. *Where do you inject, and why layer 9?* → stage-2 layer curve (§5.3), middle-layer argument (§3.3).
3. *What does strength mean physically?* → the multiple of a unit vector added to a 2,048-d hidden state. It must be compared with the typical hidden-state norm at that layer.
4. *Why does over-steering produce gibberish?* → activations leave the training distribution; downstream layers see out-of-distribution inputs.
5. *Explain your perplexity guard.* → §4.4. Why score with the unsteered model and only the continuation? Why the same seed?
6. *Your resume says 90%. What's the average?* → §2 and §5.3. Be upfront.
7. *Why is joy easy and anger hard?* → §5.4.
8. *How would you score trust and anticipation?* [RECON] → a classifier with Plutchik labels (e.g. trained on GoEmotions-mapped or Plutchik-annotated data), an LLM judge, or human rating.
9. *Steering vs prompting vs fine-tuning?* → steering: no training, continuous strength, composable, works where prompting fails, but less reliable and can degrade fluency. Fine-tuning (LoRA) is more reliable but needs data and changes weights.
10. *Is this a safety risk?* → the same technique can remove refusal directions. The guard only protects fluency, not safety.
11. *What is the +1 offset in `set_layer`?* → hidden_states[0] is the embedding output.
12. *How would you make the evaluation stronger?* → more prompts and samples, confidence intervals, probability-shift metrics, a human study, a fluency metric next to success, more models.

---

## 9. Numbers cheat sheet
| Number | Meaning |
|---|---|
| 16 layers, hidden 2,048, 17 hidden states | Llama-3.2-1B-Instruct |
| 15 + 15 sentences per axis | calibration set |
| 4 axes / 8 emotions / 6 measurable | Plutchik / classifier coverage |
| **90% (18/20)** | joy, layer 9, strength +5 (best cell) |
| 70 / 65 / 35 / 30 / 15% | best cells for fear / surprise / disgust / sadness / anger |
| 14.9% | pooled stage-2 success (all cells) |
| 27.4% | pooled success at \|strength\| = 5 |
| layer 9 (21.5%) vs layer 12 (5.7%) | middle vs late layers |
| 19/20 neutral | unsteered baseline labels |
| 1.75× / [1.0, 0.75, 0.5, 0.25] | guard threshold / back-off ladder |
| temperature 0.8, 60 new tokens | generation settings |
