> Prep pack for **Emotion steering**, written 2026-11-05 from `emotion_steering/INTERVIEW_PREP.md`. The vault copy is the one to study and edit.

# Interview Prep: Inference-Time Emotion Steering (CS772)

Written 2026-11-05 from the repo at `/data/jayavardhan/projects/emotion_steering` (branch `main`, one commit `d725cbe`, "Emotion Steering", 2026-08-18).
Nothing was run on a GPU. Every number below was recomputed from the CSVs with the Python standard library.

## Read this first

- **The repo has no README, PROGRESS or RESULTS file.** The only "docs" are the comments and docstrings inside `Project.ipynb`, plus the figure titles. The git history is one squashed commit with no message content. So "documented" below always means "a notebook comment says so".
- **How to read code references.** Cells 4 to 8 are stored as single JSON strings, so file line numbers are useless. I cite `cN:L`: cell index N (0-based), line L inside that cell's source. Example: `c4:69` is line 69 of the `AppraisalSteerer` cell.
- **The notebook cannot run top to bottom.** No cell ever builds `steerer = AppraisalSteerer(...)`. The Gradio cell says "Calibration now lives in cell 5" (`c5:2`), but no such cell exists. `c11` recalibrates an existing `steerer`. The only saved outputs are the pip install and a gated-repo 401 error from loading Llama (`c1`).
- **The sweep code that made the CSVs and the `stage1_*.png` figures is not in the repo.** The notebook's own sweep (`c7`) is a different, smaller sweep (prompt "I am waiting for the results.", layers 4/8/12, strengths 0 to 12). Its output, `steering_budget.png`, is not committed.
- **The CSVs contain no perplexity column.** So no coherence number from the guard is saved anywhere.
- `sweep_stage2_checkpoint.csv` is a strict subset of `sweep_stage2_final.csv` (all 5,870 rows identical). `stage1_heatmaps(1).png` is a byte-identical copy of `stage1_heatmaps.png`.
- **Correction to the brief.** Fear is not about 3%. In stage 2, **Anger is 3.3%** (26/800) and **Fear is 14.2%** (114/800). The `Anger_Fear` axis has Anger as its positive pole and Fear as its negative pole (`c3`, CSV `expected_label`). The projects-vault note `Projects/Emotion steering - Llama activation steering.md` has the same swap ("Anger 14%, Fear (Anger_Fear+) only 3%"). Fix it there too.

---

## 1. Pitch

### 60-second version

I built a small system that changes the emotional tone of a language model without training it.
The model is Llama 3.2 1B Instruct.
I used Plutchik's wheel, which pairs eight emotions into four opposites: joy and sadness, trust and disgust, anger and fear, anticipation and surprise.
For each pair I wrote 15 sentences per side.
I ran them through the model and took the hidden state at the last token, at every layer.
The steering vector is the mean of one side minus the mean of the other, normalised to length one.
At generation time a forward hook adds a scaled copy of that vector to one layer's output, at every new token.
I also added a safety monitor: if the steered text becomes much less likely under the unsteered model, it backs off the strength.
I scored outputs with an independent emotion classifier over 7,060 generations.
Joy steered best: 18 of 20 at layer 9, and 36% averaged over the whole sweep.
Anger, disgust and sadness barely moved, and that gap is the most interesting result.

### 3-minute version

**The question.** Can you change how a model feels, at inference time, by adding a direction to its hidden states? This is called activation steering. It costs no training and can be switched on per request.

**The vectors.** Plutchik's model gives four bipolar axes. For each axis I wrote 15 short sentences per pole, such as "I am on cloud nine!" against "I am in the depths of despair." For joy and sadness I rewrote the sadness set to mirror the joy set line by line, so the two sides differ mainly in emotion (`c3`, comment above the sadness list). Each sentence goes through the model once with `output_hidden_states=True`. I keep the last token's vector at all 17 hidden states (embeddings plus 16 layers). The steering vector for a layer is mean(positive) minus mean(negative), divided by its norm (`c4:21-56`). Because one forward pass gives every layer, all 68 vectors come from 120 forward passes, and switching layers is a dictionary lookup.

**Injection.** A forward hook on `model.model.layers[L]` adds `strength * v` to the hidden state at the last position (`c4:64-72`). With the KV cache on, every decode step has one position, so every generated token is pushed. The hook is removed in a `finally` block, so the model is never left steered by accident.

**Safety monitor.** Too much strength breaks the model before it changes the mood. It starts producing non-words or repeating itself. So `generate_guarded` (`c6:51`) first makes an unsteered baseline with the same seed. It then scores each steered continuation's perplexity under the unsteered model. If the ratio to the baseline goes above 1.75, it retries at 75%, 50% and 25% strength, and finally falls back to the baseline text.

**Evaluation.** I used the j-hartmann DistilRoBERTa emotion classifier. It has seven labels: anger, disgust, fear, joy, neutral, sadness and surprise. Success means its top label equals the target pole. Trust and anticipation have no label, so those two poles cannot be scored. Stage 1 swept all 16 layers at strength ±1 and ±5 on 2 neutral prompts. Stage 2 swept layers 5 to 12 at ±1 to ±5 on 4 neutral prompts with 5 samples each, 20 generations per cell.

**Results.** Middle layers work best. In stage 1, layer 9 scored 27% across all poles; layer 0 scored 4% and layer 14 scored 2%. In stage 2, joy reached 18/20 at layer 9, strength 5. Averaged over the stage-2 sweep, joy was 36%, surprise 23%, fear 14%, sadness 8%, disgust 6% and anger 3%. Success rises with strength: 4% at |1| to 27% at |5|, all poles pooled.

**What I learned.** Steering works well when it pushes in the direction the instruct model already likes, which is cheerful. It works poorly against that prior. The classifier is a weak judge: most failures are "neutral", and steering toward trust or anticipation mostly gets labelled "joy". The next steps would be a random-vector control, a prompting baseline, and human or LLM-judge scoring.

---

## 2. Architecture

### Diagram

```
                 ONE-TIME CALIBRATION
 c3:4  CALIBRATION_DATA = {axis: ([15 pos sentences], [15 neg sentences])} x 4 axes
         |
         v
 c4:21 AppraisalSteerer.calibrate_all_layers(data)
         |  for each of 120 sentences:
         |    c4:15 _get_all_layer_last_token_states(text)
         |       tokenizer(text)  (raw text, NO chat template, BOS added)
         |       model(**inputs, output_hidden_states=True)       c4:17
         |       keep h[:, -1, :] for all 17 states, as fp32 on CPU c4:19
         |  sum per pole per state, then
         |    direction = mean_pos - mean_neg                     c4:50
         |    direction = direction / ||direction||               c4:51
         v
 all_layer_vectors[state_idx 0..16][axis] -> unit vector, fp16, on GPU
         |
 c4:58 set_layer(L): vectors = all_layer_vectors[L + 1]           c4:61
         |                          (+1 skips the embedding state)
         v
                 PER REQUEST (Gradio)
 c5:4  run_demo(prompt, layer, joy, trust, anger, anticipation, guard_on, seed)
         |  vec = joy*v_JS + trust*v_TD + anger*v_AF + antic*v_AS  c5:12-15
         v
 c6:51 generate_guarded(steerer, prompt, vec, seed)
         |-- c4:74 generate(strength=0, seed)  -> baseline text + ids
         |      c6:38 continuation_perplexity(ids)  -> base_ppl
         |
         |-- for scale in [1.0, 0.75, 0.5, 0.25]                   c6:65
         |      c4:74 generate(manual_vec = vec*scale, seed)
         |          c4:82 apply_chat_template(user prompt)
         |          c4:93 layers[L].register_forward_hook(_hook_fn)
         |          c4:96 torch.manual_seed(seed)
         |          c4:98 model.generate(do_sample, T=0.8, 60 new tokens)
         |               every forward pass of layer L:
         |               c4:64 _hook_fn: hidden[:, -1, :] += vec   c4:69
         |          c4:103 finally: hook removed, _inject_vec = None
         |      c6:38 continuation_perplexity(ids)  (no hook -> unsteered)
         |      ratio = ppl / base_ppl                             c6:71
         |      ratio <= 1.75 or ratio is nan -> ACCEPT            c6:77
         |-- all rungs fail -> return baseline, collapsed=True     c6:83
         v
 c2:11 classify_emotion(base_text), classify_emotion(steered_text)
         |  j-hartmann distilroberta, top_k=None, text[:512]       c2:15
         v
 Gradio: unsteered box | steered box | safety-monitor table        c5:55-92

                 OFFLINE EVALUATION (code NOT in repo)
 sweep_stage1_final.csv  : 2 prompts x 16 layers x {±1,±5} x 4 axes x 2 samples (+4 baselines) = 1,028 rows
 sweep_stage2_final.csv  : 4 prompts x layers 5-12 x {±1..±5} x 4 axes x 5 samples
                           + Trust+/Anticipation+ at 7 and 9 (+20 baselines)   = 7,060 rows
 columns: sentence, axis, direction, layer, strength, sample, predicted_label,
          confidence, expected_label, success, text, prob_<7 labels>
```

### One request, end to end

Take the Gradio defaults: prompt "I am waiting for the results.", layer 12, Joy slider 5, other sliders 0, guard on, seed 0.

1. `run_demo` calls `steerer.set_layer(12)` (`c5:10`). That sets `steerer.vectors = all_layer_vectors[13]`, the output of decoder layer 12 at calibration time (`c4:61`).
2. It builds `vec = 5 * v_Joy_Sadness + 0 * ...` (`c5:12-15`). `vec` is a 2048-dim fp16 vector with norm 5. (2048 is Llama 3.2 1B's hidden size from its public config. The repo never prints it.)
3. `generate_guarded` (`c6:51`) first calls `generate(prompt, strength=0.0, seed=0, return_ids=True)` (`c6:58`). With strength 0 and no `manual_vec`, `_inject_vec` is `None` (`c4:90`). The hook is still registered but returns the output unchanged (`c4:65-66`).
4. Inside `generate`, the prompt is wrapped in the Llama chat template with `add_generation_prompt=True` (`c4:82`). `torch.manual_seed(0)` is set (`c4:96`). `model.generate` samples up to 60 tokens at temperature 0.8 (`c4:98-100`).
5. `continuation_perplexity(steerer, ids, prompt_len)` (`c6:38`) runs one more forward pass on the full sequence with no hook. It takes log-probs of each actual next token and averages only over the generated part (`c6:48`, slice from `prompt_len - 1`). `exp(-mean)` is the baseline perplexity.
6. Rung 1.0: `generate(prompt, manual_vec=vec, seed=0)` (`c6:67`). Now `_inject_vec = vec` (`c4:86`). The hook is attached to `model.model.layers[12]` (`c4:92-93`).
   - Prefill: the whole prompt passes through. At layer 12 the hook adds `vec` to the last prompt position only (`c4:69`). Earlier prompt positions stay unsteered.
   - Decode: each step feeds one new token. Its layer-12 output gets `+vec`. Its keys and values at later layers are computed from the steered state, so they are cached steered.
   - The same seed means the same random numbers are drawn per step. Where the steered and unsteered distributions are close, the same tokens come out.
7. The hook is removed (`c4:103`). The steered continuation is scored by the unsteered model. `ratio = ppl / base_ppl` (`c6:71`).
8. If `ratio <= 1.75`, the result is accepted at scale 1.0 (`c6:77-80`). Otherwise the loop tries `vec * 0.75`, then 0.5, then 0.25. If all fail, it returns the baseline text with `collapsed=True` (`c6:83-85`).
9. `classify_emotion` labels both texts (`c5:20-21`). `strip_response` first cuts the decoded text after the assistant header and removes `<|eot_id|>` (`c6:31-34`).
10. Gradio shows both texts with labels and a table of every rung tried, with its perplexity and ratio (`c5:27-46`).

Cost of this request: at least 2 generations and 2 scoring passes, at most 5 generations and 5 scoring passes.

---

## 3. Design decisions

Status key: **documented** = a notebook comment, docstring, figure title or metadata says it. **Reconstructed** = my reading of the code or CSVs; no doc gives the reason.

| # | Decision | Alternatives | Chosen, and why | Cost | Status |
|---|---|---|---|---|---|
| D1 | Model: Llama 3.2 1B Instruct, fp16 (`c1:4`, `c1:18`) | Llama 3 8B, Qwen, GPT-2 | Fits a free Colab T4 easily; instruct model gives chat-style replies; 16 layers make a full layer sweep cheap | Small model with weaker emotion features; strong cheerful RLHF prior; gated weights need an HF token | T4 is documented (notebook metadata `gpuType: T4`; `c7:11` "Colab T4"). The reason for 1B is reconstructed |
| D2 | Emotion model: Plutchik, 4 bipolar axes (`c3`) | Ekman's 6, valence–arousal, single emotions vs neutral | Plutchik gives natural opposite pairs, and difference-of-means needs a contrast. One vector covers two emotions by sign | Assumes opposites are antipodal. Trust and Anticipation have no classifier label. Joy 36% vs Sadness 8% on the same axis shows the sign flip is not symmetric | Plutchik is documented (Gradio text `c5:57`). The reason is reconstructed |
| D3 | Vector: difference of means (`c4:48-51`) | Logistic-regression probe, PCA of differences (RepE), CAA on A/B answer tokens, SAE features | No training, closed form, works with 15 examples, standard in ActAdd/CAA work | Picks up any feature that differs between the two sets, not just emotion | Reconstructed |
| D4 | Read the **last token** of the raw sentence (`c4:19`) | Mean over all tokens, chat-templated calibration | Last token sees the whole sentence under causal attention | The last token is mostly "." or "!". Anger has 6/15 sentences ending in "!", Fear 0/15. Calibration uses no chat template, but generation does (`c4:82`) | Reconstructed |
| D5 | Unit-normalise each vector (`c4:51`) | Raw difference, scale by mean residual norm | One "strength" unit means the same L2 size at every layer | The residual norm grows with depth, so strength 5 is a big push early and a small one late. The `c7:8-9` comment says exactly this | Consequence documented (`c7:8-9`); reason reconstructed |
| D6 | Calibrate **all layers in one forward pass** per sentence, cache them (`c4:21-24`, `c4:58-61`) | Re-run calibration per chosen layer | A forward pass already returns every layer; layer switches become lookups | Holds 17 x 4 fp16 vectors on the GPU, which is tiny | Documented (`c4:22-24` docstring, `c4:59` docstring, `c9` timing cell) |
| D7 | Inject at **one layer, last position, every step** via forward hook (`c4:64-72`, `c4:93`) | All positions, prompt positions only (ActAdd), several layers, edit weights | Simple, reversible, no model copy, steers every generated token | Prompt positions before the last stay unsteered. Only works with KV cache on (see exercise 3). Not batch-aware for mixed requests | Reconstructed |
| D8 | Sampling at T=0.8 with an explicit seed shared by baseline and steered run (`c4:76-80`, `c4:96`) | Greedy decoding, independent seeds | Sampling gives varied text; a shared seed makes the two runs differ mainly by the vector | Results depend on seed; "noise cancels" is only true until the token streams diverge | Documented (`c4:76-80` docstring) |
| D9 | Score with an **external classifier**, j-hartmann DistilRoBERTa (`c2`) | Human ratings, LLM judge, lexicon (NRC), the steered model's own logits | Automatic, independent of the steered model, cheap enough for 7,060 texts | Ekman labels, not Plutchik; "neutral" dominates; positive valence collapses to "joy"; input cut at 512 characters | Documented as "for automatic scoring" and "independent emotion classifier" (`c2:1`, `c5:58`) |
| D10 | Coherence guard: perplexity ratio vs same-seed baseline, threshold 1.75x, ladder 1.0/0.75/0.5/0.25, then fallback (`c6:26-27`, `c6:51-86`) | Fixed slider cap, repetition penalty, classifier for fluency, entropy monitor | Safe strength depends on layer and on how axes mix, so a fixed cap cannot be safe | At least 2x generation cost. 1.75 is never justified in the repo. Perplexity misses repetition loops (see Q26) | Problem, detector and policy documented (`c6:2-21` docstring). The value 1.75 is **not documented** |
| D11 | Score perplexity with the **unsteered** model, continuation only (`c6:38-48`) | Score with the steered model; score the whole sequence | Asks "how strange does the normal model find this text?". The prompt is context, not evidence | Unsteered model also finds genuinely emotional text a bit more surprising | Documented (`c6:9-15`) |
| D12 | Two-stage sweep: coarse all 16 layers at ±1/±5, then fine layers 5–12 at ±1..±5 | One big grid | Stage 1 finds the useful band cheaply; stage 2 spends samples there | Stage 2 layer range was picked using stage-1 results on overlapping prompts, so no held-out check | "Stage 1, coarse" is documented in the figure title `stage1_success_vs_layer.png`. The rest is reconstructed from CSVs |
| D13 | Neutral, short, first-person test prompts ("I looked at the results on my screen.") | Emotional prompts, long tasks | A neutral input means any emotion in the reply comes from the vector | Only 4 prompts, one style; the model often answers "you didn't tell me what you saw", which is easy to label neutral | Reconstructed |
| D14 | Joy/Sadness sets written as mirrored minimal pairs (`c3`) | Free-form sentences | Cancels length and syntax, so the difference is mostly affect | Only done for this axis. Joy still has 2 "!" endings vs 0 for sadness | Documented (`c3` comment: "rewritten to mirror the Joy set's structure") |
| D15 | Trust+ and Anticipation+ also swept at strength 7 and 9 (CSV) | Same grid as others | Not documented. Likely to see whether the unscorable poles move at all | Those cells cannot be scored; at 9 the text degrades ("that is, that is") | Reconstructed |
| D16 | Gradio UI with `share=True`, sliders ±15, default layer 12 (`c5:68-76`, `c5:95`) | Script only | Live demo for the course | Default layer 12 is a poor layer in stage 2 (5.7% success); sliders go far past the swept range of ±5 | Reconstructed |

---

## 4. Hardest problems, as STAR stories

There are no PROGRESS logs and the commit has no message. These stories come only from notebook comments. Where the comment does not state an outcome, I say so. Do not add numbers to them.

### S1. Over-steering broke the model before it changed the mood

- **Situation.** Pushing harder on a vector did not make the model "more joyful". Past some strength it produced non-words like "interploggrus" (`c6:3-5`). The breaking point changed with the layer and with how many axes were mixed. Max Joy plus max Anger pulled in conflicting directions (`c6:6-8`).
- **Task.** Let a user drag sliders freely without ever seeing broken output.
- **Action.** I added a detector and a policy (`c6`). The detector is the perplexity of the generated text under the unsteered model, divided by the perplexity of an unsteered baseline drawn with the same seed. The policy retries at 75%, 50% and 25% strength, then falls back to the baseline. Every rung is logged and shown in the UI. I wrote a smoke test at strength 14, layer 12, with the guard off and on (`c8`).
- **Result.** The guard is in the demo, with a toggle so you can see the collapse it prevents. **The smoke-test output was not saved**, so I cannot quote a perplexity before and after. Say "I built and demoed it", not "it reduced collapse by X".

### S2. Sampling noise made steered vs unsteered comparisons meaningless

- **Situation.** Generation samples at T=0.8. Two runs differ even with no steering. A perplexity rise could just be an unlucky sample.
- **Task.** Make the baseline and the steered run differ only by the vector.
- **Action.** I added a `seed` argument to `generate` and set `torch.manual_seed(seed)` right before `model.generate` (`c4:76-80`, `c4:96`). The guard and the sweep both pass the same seed to the baseline and the steered call (`c6:58`, `c6:67`, `c7:26`, `c7:40`).
- **Result.** The ratio now isolates the vector, at least until the two token streams diverge. The deterministic seeding is also why `sweep_stage2_checkpoint.csv` matches the final CSV exactly on all 5,870 shared rows.

### S3. Switching layers re-ran calibration

- **Situation.** The docstring says the earlier approach re-ran the model per layer (`c4:22-24`). Moving the layer slider meant recalibrating.
- **Task.** Make layer changes instant for the UI and the sweeps.
- **Action.** One forward pass with `output_hidden_states=True` returns all 17 states. I keep the last token of each, build all vectors at once, and make `set_layer` a dictionary lookup (`c4:21-61`). Cell `c9` times three layer switches.
- **Result.** Calibration cost is 120 forward passes total. The timing output of `c9` was not saved, so quote no number.

### S4. The sadness vector carried the wrong concept

- **Situation.** The first sadness sentences used loss and abandonment imagery. The joy sentences used direct affect words. So the Joy minus Sadness difference also encoded "loss stories" and sentence structure.
- **Task.** Make the two sides differ mostly in emotion.
- **Action.** I rewrote the sadness set to mirror the joy set line by line ("I am radiating positivity today." vs "I am radiating sadness today.") (`c3`, comment above the sadness list). Mean length is now 5.67 words on both sides.
- **Result.** Not measured in the repo. Stage 1 and stage 2 texts differ even for unsteered baselines, so I cannot attribute any change to this rewrite. Sadness stayed weak (8% in stage 2).

### S5. Stale and off-by-one vectors

- **Situation.** After editing the calibration data, reading `all_layer_vectors[13]` raised a `KeyError` (`c11` comment "should now print real numbers, not KeyError"). Also, `hidden_states[0]` is the embedding output, so layer i's output is index i+1 (`c4:18`).
- **Task.** Make sure the vectors in memory match the current data and the layer that gets hooked.
- **Action.** I added a recalibration cell (`c11`) and a check cell that prints the new sadness sentences and the first values of the layer-12 vector (`c10`). `set_layer` reads index `layer_id + 1` (`c4:61`), with a comment.
- **Result.** The check cells exist. They show I hit this problem. Their outputs were not saved.

### S6. Found during this prep, not during the project: a punctuation confound

Do not present this as something you found back then. Present it as "when I re-read my data, I noticed...".

- Calibration last tokens: Anger has 6/15 sentences ending in "!", Fear 0/15. Joy has 2/15, Sadness 0/15. Surprise has 2 "!" and 2 "?", Anticipation 0.
- Outputs at |strength| 5 in stage 2: "!" appears in 23% of joy-steered texts and 17% of anger-steered texts, but 1% of sadness-steered, 0% of fear-steered and 0% of baselines.
- Stage 1, layers 0–3, |strength| 5: anger-steered texts average 14.7 "!" each; joy-steered 2.75. One joy text at layer 0 is just "!!!!!!!!..." (37 marks).
- So part of the anger and joy vectors encodes "the last token is !". That fits the fact that early layers mostly hold token identity. It is evidence, not proof. The fix is in exercise 5.

---

## 5. Numbers sheet

**Setup for every sweep number.** Model Llama 3.2 1B Instruct in fp16. Sampling at T=0.8, 60 new tokens max. Unit-norm vectors from 15 sentences per pole, last token. Injection at the output of one decoder layer, last position, every step. Scored by j-hartmann/emotion-english-distilroberta-base; success = top label equals the target pole. Hardware for the sweeps is **not recorded**. The notebook targets a Colab T4. The sweep code is not in the repo, so the repo reproduces these numbers only by recomputing from the CSVs.

- Stage 1 (`sweep_stage1_final.csv`): prompts "I looked at the results on my screen." and "The meeting ended and everyone left the room." x 2 samples. All 16 layers. Strengths ±1, ±5. 4 generations per cell, 1,028 rows, 768 scorable.
- Stage 2 (`sweep_stage2_final.csv`): those two plus "I opened the door and stepped outside." and "The report was sent this afternoon." x 5 samples. Layers 5–12. Strengths ±1 to ±5 (plus 7 and 9 for Trust+ and Anticipation+). 20 generations per cell, 7,060 rows, 4,800 scorable, 240 scorable cells.
- Baseline: 20 unsteered stage-2 generations: 19 neutral, 1 joy. Mean P(joy) 0.042.

### Resume numbers

| Claim | What the repo shows | Supported? | Honest wording |
|---|---|---|---|
| "Llama 3.2 1B" | `c1:4` `meta-llama/Llama-3.2-1B-Instruct` | Yes | as is |
| "all 8 primary emotions via Plutchik's 4 bipolar axes" | Vectors exist for all 4 axes, so all 8 poles (`c3`). Only 6 poles can be scored. Of those, Anger 3.3%, Disgust 5.9%, Sadness 7.6% | Partly. Built yes; works for all 8, no | "spanning Plutchik's 4 bipolar axes (8 poles; 6 automatically scorable)" |
| "real-time hidden-state interventions" | Forward hook during generation (`c4:64-72`, `c4:93`). No latency measured. The guard costs 2x to 5x a plain generation | Mechanism yes; no timing number | "applied as forward-hook interventions during generation" |
| "validated with ablations" | Layer sweep (16 layers), strength sweep, sign flip, guard on/off demo (`c8`, output not saved). **No** random-vector control, prompting baseline, calibration-size ablation or saved perplexity numbers | Weak | "evaluated with layer x strength sweeps over 7,060 generations, scored by an independent classifier" |
| "up to 90% steering success" | Joy+, layer 9, strength 5: **18/20** in stage 2. Wilson 95% CI **70%–97%**. It is the best of 240 cells. Same cell in stage 1: 4/4 | True as a maximum; misleading as a headline | "reached 90% (18/20) for joy at the best layer and strength; joy averaged 36% across the sweep, other emotions less" |
| "entirely at inference time with no fine-tuning" | No training code at all | Yes | as is |
| "15 sentences per pole" (brief) | `c3`: every list has 15 | Yes | as is |

### Supporting numbers (stage 2 unless noted)

| Number | Value | Notes |
|---|---|---|
| Success by pole, whole sweep | Joy 285/800 = **35.6%**; Surprise 182/800 = 22.8%; Fear 114/800 = 14.2%; Sadness 61/800 = 7.6%; Disgust 47/800 = 5.9%; Anger 26/800 = **3.3%** | Pooled over layers 5–12 and |strength| 1–5. Joy CI 32%–39% |
| All scorable poles pooled | 715/4,800 = 14.9% | Stage 1: 99/768 = 12.9% |
| Best cell per pole | Joy 18/20 (L9, +5); Fear 14/20 (L5 and L9, −5); Surprise 13/20 (L9, −4); Disgust 7/20 (L6, −5); Sadness 6/20 (L8, −5); Anger 3/20 (L8, +5) | |
| Cells at ≥50% | 23 of 240: joy 14, surprise 6, fear 3 | No anger, disgust or sadness cell reaches 50% |
| Success by |strength|, all poles | 1: 3.7%; 2: 7.6%; 3: 13.8%; 4: 22.0%; 5: 27.4% | Monotonic. Positive sign higher at every strength (5: 32.2% vs 25.0%), but the positive sign is joy and anger only |
| Joy by |strength| | 1: 11.9%; 3: 36.9%; 5: 58.1% | |
| Success by layer, all poles (stage 2) | L5 12.8%, L6 19.3%, L7 18.5%, L8 18.3%, **L9 21.5%**, L10 13.3%, L11 9.7%, L12 5.7% | 600 generations per layer |
| Success by layer (stage 1) | L0 4.2%, L4 22.9%, L8 22.9%, **L9 27.1%**, L12 6.3%, L14 2.1%, L15 8.3% | 48 per layer, so ±13 points of noise |
| Joy at strength 5, by layer | L5 6/20, L6 13, L7 12, L8 11, **L9 18**, L10 14, L11 15, L12 4 | |
| Joy at L9 by sentence | 4, 5, 5, 4 out of 5 | The 90% holds across all 4 prompts |
| Joy at L9 over all strengths | 56% | |
| Formulaic joy | 19/20 replies in the 90% cell start "That's great" or "That's a great". "great" appears in 166/285 joy successes and 0/20 baselines | Relevant to Q19 |
| Main failure label | "neutral" for every pole: e.g. 485 of 515 joy failures, 589 of 774 anger failures | |
| Unscorable poles | Trust+ labelled joy 303/1,120, neutral 773. Anticipation+ labelled joy 261/1,120, neutral 808 | Positive valence collapses to "joy" |
| Mean target probability | Joy 0.331 vs 0.042 baseline; Surprise 0.234 vs 0.045; Fear 0.130 vs 0.011; Disgust 0.086 vs 0.023; Sadness 0.079 vs 0.015; Anger 0.048 vs 0.012 | Every pole moves above baseline, even the weak ones |
| Degradation proxy (mine, not the project's) | Share of texts with a word trigram repeated ≥3 times: 0% at |s|=1, 2.0% at |s|=5, 9.1% at 7, 21.6% at 9 | Stage 1 layer 0 at |s|=5: 25%, and 8 empty outputs at layers 0–1. Crude regex metric I computed for prep |
| Guard threshold | 1.75x baseline perplexity (`c6:26`) | Not calibrated anywhere in the repo |
| Guard ladder | 1.0, 0.75, 0.5, 0.25, then baseline (`c6:27`) | |
| Guard cost | 2 to 5 generations and 2 to 5 scoring passes per request | From the code, not timed |

---

## 6. Prerequisite concepts

**Residual stream.** In a pre-norm transformer each layer reads the running hidden state, computes an update, and adds it back. The stream is a shared bus that every later layer reads. Adding a vector to it at layer L changes what every later layer sees. In HF, `output_hidden_states` returns 17 tensors for a 16-layer model: index 0 is the embedding output, index i+1 is layer i's output, and the last one has the final RMSNorm applied.
Sources: Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits".

**Linear representation hypothesis.** Many high-level concepts (sentiment, truth, refusal) are encoded as directions in activation space. If so, adding the direction moves the concept. Sentiment in particular has been shown to be roughly one direction in several LLMs.
Sources: Park, Choe and Veitch 2023, "The Linear Representation Hypothesis and the Geometry of LLMs"; Tigges et al. 2023, "Linear Representations of Sentiment in Large Language Models".

**Difference-of-means steering (ActAdd).** Take activations for contrasting inputs, subtract the means, and add the difference during generation. No gradients, no training. It is the simplest member of a family that also includes probes and PCA-based directions.
Sources: Turner et al. 2023, "Activation Addition: Steering Language Models Without Optimization"; Zou et al. 2023, "Representation Engineering".

**Contrastive Activation Addition (CAA).** Builds the vector from many paired examples that differ only in the answer letter (A vs B) for a behaviour. It reads the activation at the answer token and adds the vector at all positions after the prompt. It controls for everything except the behaviour much better than free sentences.
Source: Rimsky (Panickssery) et al. 2024, "Steering Llama 2 via Contrastive Activation Addition" (ACL).

**PyTorch forward hooks and the KV cache.** `register_forward_hook(fn)` calls `fn(module, input, output)` after a module runs; returning a value replaces the output. With the KV cache on, decode steps pass one token, so `[:, -1, :]` is the new token. Keys and values computed from a steered state are cached, so the push persists through attention.
Source: PyTorch `nn.Module.register_forward_hook` docs; HF `generate` docs.

**Plutchik's wheel.** Eight primary emotions in four opposite pairs: joy–sadness, trust–disgust, fear–anger, anticipation–surprise. Ekman's basic six (anger, disgust, fear, joy, sadness, surprise) overlap on six of them but lack trust and anticipation. That gap is why two poles cannot be scored here.
Sources: Plutchik 1980, "Emotion: A Psychoevolutionary Synthesis"; Plutchik 2001, "The Nature of Emotions", American Scientist.

**Perplexity.** exp of the mean negative log-likelihood per token. Low means the model finds the text likely. It is a fluency signal, not a meaning signal. Repetitive text can have very low perplexity, which is why likelihood-maximising decoding degenerates into loops.
Source: Holtzman et al. 2020, "The Curious Case of Neural Text Degeneration".

**Classifier-based evaluation.** An external classifier gives cheap, automatic labels. Its label set, training domain and biases become your metric's biases. j-hartmann's model is DistilRoBERTa fine-tuned on six English emotion datasets (tweets, Reddit, self-reports, TV dialogue) with Ekman's six plus neutral. Check its model card for its reported accuracy before quoting one.
Sources: Hartmann 2022, model card `j-hartmann/emotion-english-distilroberta-base`; Pres et al. 2024, "Towards Reliable Evaluation of Behavior Steering Interventions in LLMs".

**What different layers do.** Early layers mostly hold token identity and local syntax. Middle layers hold more abstract features. Late layers prepare the next-token distribution. Logit-lens studies show predictions forming in the second half.
Sources: Tenney et al. 2019, "BERT Rediscovers the Classical NLP Pipeline"; nostalgebraist 2020, "interpreting GPT: the logit lens"; Belrose et al. 2023, "Eliciting Latent Predictions with the Tuned Lens".

**Sparse autoencoders (SAEs).** Train a wide, sparse autoencoder on activations. Its features are often single, readable concepts. You can steer by clamping a feature, as in "Golden Gate Claude".
Sources: Bricken et al. 2023, "Towards Monosemanticity"; Templeton et al. 2024, "Scaling Monosemanticity".

**Small-n statistics and multiple comparisons.** 18/20 has a Wilson 95% interval of 70%–97%. Picking the best of 240 cells inflates the maximum (winner's curse). Report the mean, the best cell with its CI, and ideally a held-out re-run of the chosen cell.
Source: Agresti and Coull 1998, "Approximate is Better than 'Exact' for Interval Estimation of Binomial Proportions".

**Prompting as a baseline.** Recent benchmarks find that plain prompting ("answer joyfully") often beats representation steering on concept control. A steering paper needs this baseline.
Source: Wu et al. 2025, "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders".

---

## 7. Fifty likely questions

**Method**

1. **What is activation steering, in one line?**
   Adding a fixed vector to a model's hidden state during generation to shift its behaviour, with no weight change.

2. **How exactly do you build the vector?**
   For each pole, 15 sentences. One forward pass each, last token, all 17 hidden states (`c4:15-19`). Mean of positive minus mean of negative, divided by its norm (`c4:48-51`). One unit vector per axis per layer.

3. **Why difference of means and not a trained linear probe?**
   With 15 examples per class a probe overfits and its weight vector points along whatever separates the classes best, which may be a rare feature. The mean difference is the maximum-likelihood direction under a shared-covariance Gaussian, needs no training, and is what ActAdd and CAA use. The cost is that it absorbs every confound between the two sets.

4. **Why the last token?**
   Under causal attention only the last token has seen the whole sentence. The cost: in my sets the last token is almost always "." or "!", so punctuation leaks into the vector (S6).

5. **Why normalise to unit length?**
   So "strength 5" means an L2 push of 5 everywhere. The catch is that the residual norm grows with depth, so the same push is relatively bigger early and smaller late. The `c7:8-9` comment notes this. A better unit is a fraction of the mean residual norm at that layer.

6. **Why Plutchik and not Ekman or valence–arousal?**
   Difference-of-means needs a contrast. Plutchik gives four natural opposite pairs, so one vector covers two emotions by sign. Cost: my classifier is Ekman-based, so Trust and Anticipation are unscorable.

7. **Is sadness really minus joy in activation space?**
   My data say not exactly. On the same vector, +5 at layer 9 gives joy 18/20 but −5 gives sadness at best 6/20 anywhere. The instruct model's cheerful prior resists the negative direction. A separate sadness-vs-neutral vector would test this.

8. **Why 15 sentences per pole? Is that enough?**
   The comment says it was expanded to 15 for "robust, noise-free" vectors (`c3:2`). I did not run a calibration-size ablation. To check, I would build vectors from random halves and report their cosine similarity per layer. High cosine means 15 is enough.

9. **Why Llama 3.2 1B?**
   It fits a free Colab T4 in fp16, and 16 layers make full sweeps cheap. It is an instruct model, so replies are chat-like. The cost is a small model with a strong RLHF prior.

10. **How does the hook work, and what happens with the KV cache?**
    `register_forward_hook` on `model.model.layers[L]` (`c4:93`). The hook adds the vector to `hidden[:, -1, :]` (`c4:69`). At prefill that is the last prompt token. At each decode step there is one token, so every generated token is steered. Its keys and values are cached in steered form.

11. **Why only the last position and not all positions?**
    Simplicity, and it makes the generated tokens carry the push. The cost is that earlier prompt tokens stay unsteered. CAA adds to all positions after the prompt; ActAdd adds at the prompt positions. I did not compare them.

12. **Why does the hook handle a tuple?**
    Older HF decoder layers return a tuple (hidden, cache, attn); newer ones return a tensor. The hook handles both (`c4:67-72`).

13. **Why is the hook removed in `finally`?**
    So an exception mid-generation never leaves the model permanently steered. The perplexity scorer relies on this: it must see the unsteered model (`c6:40-41` docstring).

**Layers and strength**

14. **Why do middle layers work best?**
    Data: stage 1 success is 4% at layer 0, 27% at layer 9, 2% at layer 14. Stage 2 peaks at layer 9 (21.5%) and falls to 5.7% at layer 12. Explanation, which I did not test directly: early layers hold token identity, so the early vector is partly "!" vs "." and adding it breaks syntax (layer-0 joy gives "!!!!!!"). Late layers have little computation left to turn the push into word choice, and the residual norm is larger, so a fixed push is relatively small. Middle layers hold abstract features like sentiment and still have layers to act on them.

15. **Why layer 9 specifically?**
    It was the best layer in both stages. I would not claim 9 is special; 6 to 10 form a band, and layer 9 is about 56% through a 16-layer model. Papers on 7B models report a similar mid-depth band.

16. **What happens as strength rises?**
    Success climbs steadily: 3.7% at |1| to 27.4% at |5|, all poles. Degradation also climbs. By my crude repetition metric, 2% of texts loop at |5|, 9% at 7 and 22% at 9. At strength 9 Trust+ replies look like "That is this is in this level."

17. **What is the "steering budget"?**
    The strength range where the emotion moves but coherence holds (`c7:6-7`). It differs per layer because of the norm issue. The `c7` sweep plots it, but its figure was not committed.

18. **Why does layer 15 behave oddly?**
    HF applies the final norm before storing the last hidden state. So the layer-15 vector comes from post-norm states but is injected pre-norm. That mismatch is a bug in principle. It only affects layer 15.

19. **Did you try several layers at once?**
    No. That is a natural next step, usually with smaller strength per layer.

**Evaluation**

20. **How did you measure success?**
    The j-hartmann classifier labels each reply. Success means its top label equals the target. 20 generations per cell in stage 2 (4 prompts x 5 samples).

21. **Why is classifier-based evaluation weak?**
    Five reasons, all visible in the data. Its labels are Ekman's, so 2 of 8 poles are unscorable. "Neutral" absorbs most failures (e.g. 589 of 774 anger failures). Positive valence collapses to "joy": Trust+ is labelled joy 303 times out of 1,120. Top-label success throws away graded change: every weak pole still raises its mean probability 3x to 12x over baseline. And it rewards surface words: 19/20 replies in the 90% cell start "That's great".

22. **Hostile: isn't 90% cherry-picked?**
    It is the best of 240 cells, so yes, it is a maximum. I report it with its context: 18/20, Wilson CI 70%–97%, joy mean 36%. Two things support it as real. It holds on all 4 prompts (4, 5, 5, 4 of 5). The same cell was 4/4 in the separate stage-1 run.

23. **Hostile: "That's great" every time. Is that joy, or sycophancy?**
    Fair. The vector mostly makes the assistant enthusiastic and approving. The classifier calls that joy. A human might call it cheerful assistant tone, not the emotion joy. I would need a human or LLM-judge rubric to separate "the speaker feels joy" from "the reply is upbeat".

24. **Why does joy work and anger not?**
    Joy agrees with the instruct model's cheerful prior; anger fights its safety training. The anger vector is also partly punctuation: 6 of 15 anger sentences end in "!". Many anger-steered replies just add exclamation marks or become confused, and the classifier calls them neutral or surprise.

25. **How would you evaluate Trust and Anticipation?**
    A classifier with Plutchik labels or a 28-label GoEmotions model, an LLM judge with a rubric, or a small human study. A cheap check is the NRC emotion lexicon, which has trust and anticipation.

26. **What baseline did you compare against?**
    Unsteered generation with the same prompts: 19/20 neutral, 1/20 joy. There was no prompting baseline and no random-vector baseline. Those are the two missing controls.

27. **Why is a random-vector control needed?**
    Any big push may make the model weird, and weird text may get labelled "surprise" or "fear". A random unit vector at the same strength shows how much success is just disruption. Without it, the surprise and fear numbers are hard to interpret.

28. **Hostile: you say "validated with ablations". Which ablations?**
    Layer sweep, strength sweep, sign flip and the guard on/off demo. No random-vector, calibration-size or prompting ablation. I would now call them "sweeps", not ablations.

29. **Is there leakage between choosing settings and reporting them?**
    Yes. Stage 2's layer range was chosen from stage 1 on overlapping prompts, and the best cell is reported on the same data that picked it. A clean version re-runs the chosen cell on fresh prompts.

**Safety monitor**

30. **What does the guard measure?**
    Perplexity of the steered continuation under the unsteered model, divided by the same for a same-seed unsteered baseline (`c6:38-48`, `c6:71`).

31. **Why score with the unsteered model?**
    It asks "how strange does the normal model find this text?". A real mood shift stays fluent and barely moves it. Non-words make it spike. Scoring with the steered model would measure its own confidence, which stays high even when it is broken.

32. **Why 1.75?**
    Not documented. It is a hand-set value. I would calibrate it: collect steered texts, label a few hundred as coherent or not, and choose the threshold that best separates them.

33. **Hostile: what does perplexity miss?**
    Loops. "That is, that is, that is" becomes very likely after the first repeat, so perplexity can fall. Also, empty or one-token outputs return NaN, and NaN is accepted (`c6:44`, `c6:77`). A repetition check (distinct n-grams) or an entropy check would catch loops.

34. **Is there a bug in the guard?**
    Yes, a small one. `ratio = ppl / base_ppl if base_ppl else nan` (`c6:71`). If the baseline is too short, `base_ppl` is NaN. NaN is truthy, so the ratio is NaN, and NaN is accepted at full strength. A degenerate baseline silently disables the guard. The fix is `if math.isfinite(base_ppl)`.

35. **Does the shared seed really cancel noise?**
    Partly. Both runs draw the same random numbers per step. While the two distributions are close, the same tokens come out. Once one token differs, the streams diverge. It is variance reduction, not exact pairing.

36. **Hostile: "real-time"? The guard runs up to five generations.**
    The hook itself adds one vector add per step, which is negligible next to a 1B forward pass. The guard costs 2 to 5 full generations plus scoring. I never timed either. Honest wording: "applied during generation".

**Alternatives**

37. **Why not CAA?**
    CAA needs many A/B-style paired prompts per behaviour. I had a course timeline and wanted emotions in free text. Difference of means on free sentences is the simpler version of the same idea. CAA's pairing would have removed my punctuation confound.

38. **Why not SAE features?**
    No public SAE existed for Llama 3.2 1B in my setup, and training one is a project by itself. An SAE feature could be cleaner and more specific than a mean difference. The cost is training compute and the search for the right feature.

39. **Why not fine-tune (LoRA or DPO)?**
    Fine-tuning needs labelled data, training compute and one adapter per emotion. It cannot mix emotions at a chosen strength from a slider at runtime. Steering is free and reversible. Fine-tuning would likely give more reliable emotion, at a much higher cost.

40. **Hostile: why not just prompt "respond joyfully"?**
    Prompting is probably stronger for joy, and I did not test it, which is a real gap. Steering's advantages: it is continuous and mixable, it does not use context, and it is hidden from the user prompt. Its other use is scientific: it tests whether emotion is a direction in the model.

41. **How does this relate to RepE and ITI?**
    RepE finds directions with PCA on paired differences and reads or controls them. ITI shifts specific attention heads along truthful directions. My method is the simplest variant: one mean difference, one layer, the whole residual stream.

**Engineering and scale**

42. **Would this work on a 70B model?**
    Likely better. Larger models tend to have cleaner linear features. The method costs the same: 120 forward passes to calibrate, one add per step. The strength scale would need retuning because residual norms differ.

43. **How would you serve this in production?**
    Per-request vectors in a batch: add a [batch, d] tensor instead of one vector. In vLLM, a custom layer hook or a steering-vector feature. Precompute vectors. Drop the guard's re-generation and use a cheaper online check, such as token entropy per step.

44. **Are the four axis vectors orthogonal? What happens when you mix them?**
    I did not measure cosine similarity. My sweeps are single-axis only. The Gradio UI mixes them, and the `c6` docstring notes that max Joy plus max Anger conflict. Measuring pairwise cosines per layer is a one-line check.

45. **Any numerical issues in fp16?**
    Vectors are computed in fp32 on CPU, then cast to fp16 (`c4:19`, `c4:52`). Adding norm-5 vectors to fp16 states is safe. Strengths of 15 at early layers can give extreme values, but that is collapse, not overflow.

46. **Could steering bypass safety training?**
    Yes, in principle. Refusal is known to be roughly one direction, and subtracting it removes refusals. That is why steering is a safety research tool and a risk for open-weight models.

**Reflection**

47. **Hostile: can I rerun your results from the repo?**
    Not fully. The notebook never builds the `steerer`, and the sweep code that wrote the CSVs is not committed. Every number in this sheet can be recomputed from the CSVs, but the CSVs cannot be regenerated from the repo as it stands.

48. **What would you do differently?**
    Calibrate with chat-templated inputs and strip final punctuation. Add a random-vector control and a prompting baseline. Use an LLM judge or human labels alongside the classifier. Log perplexity in the sweep. Commit the sweep code. Report the mean first and the best cell second.

49. **What comes next?**
    Compare SAE features to my mean-difference vectors for the same emotions. Test multi-layer injection. Measure whether the joy direction found from sentences matches the sentiment direction found by probes.

50. **What is the single most interesting finding?**
    The asymmetry. The same axis gives 36% joy but 8% sadness, and anger is the weakest at 3%. Steering is easy along the model's trained prior and hard against it.

---

## 8. Where the honest answer is "it didn't work" or "I didn't do that"

Say it plainly, give the reason, then say what you would do. Do not apologise at length.

- **"Did it work for all eight emotions?"** "No. It worked well for joy, moderately for surprise and fear, and barely for sadness, disgust and anger. Trust and anticipation I couldn't score, because the classifier has no such labels. The pattern itself is a finding: steering works with the model's cheerful prior and fights against it."
- **"Did you have a human evaluation?"** "No. Everything is classifier-scored. That's the weakest part of the evaluation, and the 'That's great' pattern shows why."
- **"Did you compare to prompting?"** "No, and I should have. Recent work suggests prompting is a strong baseline. I'd expect it to beat my vector on joy."
- **"Random-vector control?"** "No. Without it I can't fully separate 'moved the emotion' from 'disrupted the model'. That matters most for surprise, which disruption can mimic."
- **"How well did the guard work?"** "I built and demoed it, but I didn't save a quantitative evaluation, and the 1.75 threshold was hand-set. It also has a NaN edge case I'd fix."
- **"What's the latency overhead?"** "I didn't measure it. The hook is one vector add per step. The guard is 2 to 5 times a normal generation."
- **"Did you test mixing axes?"** "The UI allows it, but all my measured results are single-axis."
- **"Can I reproduce it?"** "The results can be recomputed from the saved CSVs. The sweep script isn't in the repo, so regenerating them needs that script restored."
- **"Did you check that 15 sentences is enough?"** "No. I'd check by building vectors from random halves and comparing their cosine similarity."
- **"Did you compare with CAA or SAEs?"** "No. I chose the simplest method for a course project. Those are the right comparisons for a follow-up."

---

## 9. Self-test exercises

Predict the effect before you read the answer. None of these were run.

### Exercise 1. Remove the normalisation

Change `c4:51` from `direction = direction / direction.norm()` to nothing (keep the raw difference).

<details><summary>Answer</summary>

The vector's length becomes ||mean_pos − mean_neg||, which differs per axis and per layer and generally grows with depth. "Strength 5" now means 5 times that length. That is far larger than before at most layers, so you hit collapse at much lower slider values, and the guard backs off more often. Strengths are no longer comparable across axes. The layer ranking may shift, because late layers now get proportionally larger pushes. The direction is unchanged, so at a tuned strength the kind of steering is the same.
</details>

### Exercise 2. Steer every position

Change the hook body (`c4:69` and `c4:71`) to `hidden += self._inject_vec` (all positions, not `[:, -1, :]`).

<details><summary>Answer</summary>

Prefill now steers every prompt token, not just the last one. Decode is unchanged, because each step has only one position. Every prompt token's keys and values at later layers are now steered, so the effect at the same strength is stronger. Collapse arrives at lower strength. This is closer to how ActAdd and CAA inject. The perplexity scorer is unaffected because the hook is removed before scoring.
</details>

### Exercise 3. Turn off the KV cache

Add `use_cache=False` to the `model.generate` call at `c4:98`.

<details><summary>Answer</summary>

Each step now reruns the whole sequence. The hook still adds only at the last position. So at step t only the newest position is steered. All earlier generated tokens are recomputed unsteered, which differs from the cached run, where their steered keys and values persisted. The effect becomes weaker, and generation becomes much slower (quadratic). This shows the method silently depends on the KV cache.
</details>

### Exercise 4. Fix the guard's NaN hole

At `c6:71`, replace `if base_ppl` with `if math.isfinite(base_ppl)`. Also change `c6:77` so a NaN steered ratio is rejected instead of accepted.

<details><summary>Answer</summary>

First change: no visible effect, because the result is still NaN when the baseline is too short. The real fix is the decision at `c6:77`. With NaN rejected, a steered output of fewer than 2 tokens is treated as failed. The guard backs off, and may end at the baseline. Side effect: if the baseline itself is too short, every rung now fails and you always fall back. So you also need a rule for a short baseline, such as accepting any steered rung whose perplexity is below a fixed absolute cap.
</details>

### Exercise 5. Strip final punctuation from the calibration sentences

Before calibrating, apply `s.rstrip('!.?')` to every sentence in `CALIBRATION_DATA`, then rerun `c11`.

<details><summary>Answer</summary>

The last token is now an emotion word or a content word, not punctuation. Anger loses its "!" component, so anger-steered outputs should contain far fewer "!". The anger and joy vectors at early layers change the most, since early layers are mostly token identity. Layer-0 joy should no longer emit "!!!!!". Whether anger success rises is uncertain. It might, because the vector now spends its length on meaning. It might not, if the instruct model's resistance to anger is the main cause. Measure it with the stage-2 grid before claiming either way.
</details>
