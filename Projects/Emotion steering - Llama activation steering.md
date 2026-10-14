---
type: project
folder: projects/emotion_steering
repo: github.com/jayavardhan-g/Inference-Time-Emotion-Steering
dates: course final project (notebook 2026-08-16, uploaded 2026-08-18)
status: done
kind: coursework — interpretability / activation steering
stack: [Llama-3.2-1B-Instruct, j-hartmann emotion-distilroberta classifier, Gradio, Colab]
tags: [project, interpretability, coursework]
---
# Emotion Steering Wheel — inference-time activation steering on Llama-3.2-1B

**What.** This was the final project of an ML course. It builds difference-of-means steering vectors for four Plutchik axes (Joy–Sadness,
Trust–Disgust, Anger–Fear, Anticipation–Surprise) from 15 calibration sentences per pole. Each vector is computed at every layer
and injected into the residual stream at generation time. Extras: a **perplexity-guarded "safety monitor"** that backs off strength
when the continuation stops being coherent (scored by the unsteered model), a steering-budget sweep and a Gradio UI.

**Key results**, recomputed from the sweep CSVs because the notebook has no write-up. Success means the
independent emotion classifier predicts the intended emotion:
- Stage 1 (all 16 layers, strength ±1/±5, 2 prompts): success is highest at **layers 4–9 (19–27%)** and near zero at layer 0 and at layers 14–15.
- Stage 2 (layers 5–12, strength −5…+9, 4 prompts, 7,060 generations): **Joy+ 36%**, Anticipation−/Surprise 23%,
  Anger 14%, Sadness 8%, Disgust 6%, Fear (Anger_Fear+) only **3%**. Success rises with |strength| (3% at ±1 → 32% at +5).
- Trust and Anticipation can't be scored, because the classifier has no such labels.

**Takeaway.** Difference-of-means steering works for joy and barely works for fear or disgust on a 1B model. Over-steering
collapses into non-words before the mood changes, which is why the guard exists. The [[Project strategy and resume plan]] judged a plain repeat of
this method redundant. Its suggested follow-ups were SAE feature discovery and speech-LLM interpretability.

**Docs in this vault** (copied 2026-10-14 from `/data/jayavardhan/projects/emotion_steering/`):
- figures (4): `Sources/emotion_steering/`

![[Sources/emotion_steering/stage1_success_vs_layer.png]]

**Source.** `Project.ipynb`, `sweep_stage1_final.csv`, `sweep_stage2_final.csv`, `stage1_*.png`.
