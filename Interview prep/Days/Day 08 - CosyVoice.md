---
type: study-day
day: 8
date: 2026-11-15
tags: [interview, day, cosyvoice, speech, lora]
---
# Day 8 (Sun Nov 15): CosyVoice Indic LoRA

Part of [[Interview plan (Nov 2026)]]. Repo `/data/jayavardhan/cosyvoice_rnd` (branch `main`). Data and checkpoints in
`/mnt/warm_storage/jayavardhan/cosy/` (`phase2`, `phase3`, `phase4`, `phase4_te`, `phase4_multi`, `phase4_cv3_*`). About 7 h.
Builds on [[Day 07 - Seminar and speech basics]] (the CosyVoice 2 pipeline). Project note: [[CosyVoice Indic LoRA (R&D)]].

**Code walkthrough:** [[Day 08 - code walkthrough]]: one small figure per pipeline, the exact code to read in order, and what each piece does.

**There is no prep pack for this project yet.** This note is written from the code and training logs. The upstream
`cosyvoice/` package is Alibaba's code, not yours: skim it only. Your code is `scripts/`.

**Resume bullets (both resumes):**
- "Adapted CosyVoice 2 to Hindi and Telugu via LoRA fine-tuning on its Qwen2 LLM component (**9M params, 1.36%** of the base model, 17 MB), trained on **35 hours** of speech across **4 H200 GPUs** with BF16 precision."
- "Trained a single adapter jointly on Hindi and Telugu with strong results and **no catastrophic forgetting**."
- "Reduced Hindi WER from **1.99** in the zero-shot setting to **0.11** after LoRA adaptation."

**What the repo and logs show, and what to fix:**
- **9M / 1.36% is right.** The training log (`phase4_multi/checkpoints/training.log`) reports `trainable_params 8,798,208` of `647,972,675` = **1.358%**, at r = 16, α = 32. The denominator is the **whole CosyVoice 2** (LLM + flow + vocoder, `phase4b_train_lora.py:172-177`); against the 0.5B Qwen alone it is ~1.8%.
- **"Across 4 H200 GPUs" is not data-parallel training.** `phase4b_train_lora.py` has no DDP; its docstring runs it on one GPU (`CUDA_VISIBLE_DEVICES=4`). The GPUs ran **separate jobs in parallel** (Hindi, Telugu, joint; CV2 and CV3) and **parallel generation** (`generate_all_parallel.sh`). Say "trained on H200s, running the Hindi, Telugu and joint adapters as parallel single-GPU jobs". The old plan's "DDP" block is dropped.
- **WER 1.99 → 0.11: the result CSVs are not on the server.** `compile_results.py` reads `data/phase5_outputs/wer_*_all.csv`, and `data/` is missing from the repo and from `cosy/`. **Find the R&D presentation or those CSVs before Day 15** and check the numbers, the test set and how WER was averaged.
- **WER above 1 needs explaining** (see A3): it means the zero-shot model produced many *inserted* words. Say it before they ask.

**By tonight you can:**
- explain LoRA and compute 8,798,208 by hand;
- walk one Hindi clip from audio to the training loss;
- explain WER, why it can exceed 1, and how "no catastrophic forgetting" was tested;
- give honest answers on the multi-GPU claim and the missing WER files.

## Schedule

| # | Time | Kind | What |
|---|---|---|---|
| 0 | 15 min | recall | Day 7: the CosyVoice 2 pipeline, 25 Hz tokens, bitrates |
| 1 | 20 min | read | README "5-Day R&D Strategy"; the project note; the five phases (below) |
| 2 | 45 min | theory + maths | LoRA (A1) |
| 3 | 30 min | skim | Upstream CosyVoice: `cli/cosyvoice.py` → `cli/model.py` → `llm/llm.py` (class and method names) |
| 4 | 45 min | code | `phase4a_pretokenize.py` (B1) |
| — | lunch | | |
| 5 | 75 min | code | `phase4b_train_lora.py` (B2) |
| 6 | 30 min | theory | BF16 vs FP16; catastrophic forgetting; ASR-based evaluation (A2, A3) |
| 7 | 45 min | code | `phase4c_generate.py`, `phase5_eval_wer.py` (B3) |
| 8 | 30 min | skim | Phases 1–3: baseline, codec benchmark, the distillation attempt (B4) |
| 9 | 45 min | rebuild | A LoRA layer from scratch, and a WER function (C) |
| 10 | 20 min | trace | One clip through pretokenize → loss; the parameter count (D) |
| 11 | 40 min | practice | The questions (E) |

**The five phases** (README; project note):
1. Baseline: zero-shot CosyVoice on Hindi (`step1_run_hindi_baseline.py`).
2. Benchmark neural codecs on Indic audio (`phase2_codec_benchmark.py`).
3. Try distilling mHuBERT-147 semantics into WavTokenizer (`phase3_*`). An attempt; know how it ended.
4. LoRA fine-tune the CosyVoice 2 LLM: pretokenize (4a), train (4b), generate (4c).
5. Evaluate with IndicConformer ASR WER (`phase5_eval_wer.py`).

**Data:** AI4Bharat IndicVoices-R, 18 h Hindi + 17 h Telugu = 35 h (14,662 clips) (project note).

## A. Theory

### A1. LoRA (block 2, 45 min)

**Read:** Hu et al., *LoRA* (2021), §1, §4 (method) and §7.1–7.2 (which weights, what rank). About 35 min.
**Optional:** Sebastian Raschka's blog post *"Practical Tips for Finetuning LLMs Using LoRA"*.
- Freeze `W` (d_out × d_in). Learn `ΔW = B·A`, with `A` (r × d_in) and `B` (d_out × r). Output = `Wx + (α/r)·BAx`.
- `B` starts at zero, so training starts exactly from the base model.
- Parameters per adapted matrix = `r·(d_in + d_out)`.
- **Your config:** r = 16, α = 32 (scale 2), dropout 0.05, on **all seven** projections: q, k, v, o, gate, up, down (`phase4b_train_lora.py:101-107`).

**Compute 8,798,208 by hand** (Qwen2.5-0.5B: 24 layers, hidden 896, MLP 4864, 2 KV heads × 64 = 128):
- q: 16·(896+896) = 28,672; o: 28,672; k: 16·(896+128) = 16,384; v: 16,384;
- gate: 16·(896+4864) = 92,160; up: 92,160; down: 92,160;
- per layer = 366,592; × 24 layers = **8,798,208**. ✓
- In bf16 that's 8.8M × 2 B ≈ **17.6 MB**, the "17 MB" on resume 5.

### A2. BF16 and forgetting (block 6, first 15 min)

- **BF16** keeps FP32's 8 exponent bits (same range) and has only 7 mantissa bits. FP16 has 5 exponent bits, so it needs loss scaling to avoid underflow. BF16 doesn't. Your training uses `torch.autocast(..., dtype=torch.bfloat16)` (`phase4b_train_lora.py:235`).
- **Catastrophic forgetting:** fine-tuning on new data degrades old skills. LoRA limits it, because the base weights never change and the adapter is small. **How you tested it:** look at `generate_crosslingual_parallel.sh` and `compile_results.py`, and write down which languages and conditions were scored with the adapter on. If no base-language (English or Chinese) score with the adapter exists, say "no forgetting on the languages I tested".

### A3. Evaluating TTS with ASR (block 6, last 15 min)

- **WER = (S + D + I) / N**: substitutions, deletions and insertions over the number of reference words.
- **It can exceed 1**, because insertions are unbounded. A zero-shot model reading Hindi it doesn't know can babble long, wrong speech, which the ASR transcribes as many extra words. So **1.99 means "mostly garbage, much longer than the reference"**, not a bug.
- Your ASR judge is **AI4Bharat IndicConformer** (`phase5_eval_wer.py:40`), with Hindi text normalisation (`:30`) and `jiwer` (`:111`). The reported WER is the **mean of per-clip WERs** (`:169`), not a corpus WER, so a few babbling clips can pull the mean above 1.
- The ASR has its own error rate on *real* Hindi speech. That floor bounds how low TTS WER can go.

## B. Code

### B1. Pretokenize (block 4, 45 min, deep)

`scripts/phase4a_pretokenize.py` (204 lines). It caches `(text tokens, speech tokens, speaker embedding)` per clip, so training never runs the frozen tokenisers.
- 46 `main`; 80 the CosyVoice frontend;
- **141 `_extract_speech_token`**: the S3 speech tokens (25 Hz FSQ, Day 7);
- **146 `_extract_spk_embedding`**: the speaker embedding. It's cached, but the CosyVoice 2 LLM's loss never reads it; the flow model uses it at synthesis;
- **153 `_extract_text_token`**: from the normalised transcript (152);
- 166–172: what's written to the cache and the manifest.

### B2. Train (block 5, 75 min, deep)

`scripts/phase4b_train_lora.py` (307 lines):
- 42 `CachedTokenDataset`; **55 `collate_fn`**: pads speech tokens and keeps true lengths (68–84). Why do lengths matter for the loss?
- **96 `apply_lora_to_qwen`**: `inject_adapter_in_model`, not `get_peft_model`. Why? (*Keeps the original class, so CosyVoice's hard-coded `self.llm.model.model.embed_tokens` path still works; docstring 97–98.*)
- 156: the flow-matching model and the HiFT vocoder are frozen. **Only the LLM gets adapters.**
- 168–178: apply LoRA, log the trainable count.
- 198: AdamW, lr 2e-4, weight decay 0.01. 235–240: bf16 autocast, gradient accumulation (batch 4 × accum 4 = 16).
- 253–283: validation every 300 steps, keep the best, early stopping after 4 non-improving checks, and a time budget (`--hours-budget`).
- Which loss is it? It comes from CosyVoice's `Qwen2LM.forward` (`out["loss"]`, 237): next-token cross-entropy on the **speech tokens**, conditioned on text and speaker. Confirm in `cosyvoice/llm/llm.py`.

### B3. Generate and evaluate (block 7, 45 min, deep)

- `phase4c_generate.py` (176): 39 apply LoRA again; **52 `load_lora_weights`** (only the adapter tensors are saved, so `lora.pt` is small); 139 `cv.inference_zero_shot` with a prompt wav and prompt text (the voice to clone).
- `phase5_eval_wer.py` (179): 40 IndicConformer; 30 normalisation; 111–142 per-clip WER and CER; 167 aggregation per label.

### B4. Phases 1–3 (block 8, 30 min, skim)

`step1_run_hindi_baseline.py`, `phase2_codec_benchmark.py` (533), `phase3_prepare_data.py`, `phase3_train_codec.py` (397), `compile_results.py`.
For each, write one line: what question did it answer, and what was the answer? For phase 3, the logs are in
`cosy/phase3/checkpoints*/training.log` (runs `lam05`, `lam10`, `v2`). Why did you move from codec work to LoRA on the LLM?

## C. Rebuild it yourself (block 9, 45 min)

1. **LoRA in PyTorch, about 30 lines:** `class LoRALinear(nn.Module)` wrapping a frozen `nn.Linear`, with `A` (Kaiming init), `B` (zeros), scale α/r and dropout. Check that at init the output equals the base layer's exactly. Count trainable parameters for one Qwen2.5-0.5B layer: you should get 366,592.
2. **Merge:** write `merge()` that folds `W + (α/r)·BA` into the weight. Check the merged layer gives the same output. (This is why LoRA adds no inference latency.)
3. **WER, about 20 lines:** word-level Levenshtein distance returning S, D and I. Make a case where WER > 1 (reference 3 words, hypothesis 8 wrong words).

## D. Trace (block 10, 20 min)

1. One Hindi clip, 4 s long, with a 12-word transcript: how many speech tokens at 25 Hz? What three things does pretokenize cache for it?
2. What does the LLM see as input and as target during training? Which parameters get gradients?
3. Batch 4 × accumulation 4: how many clips per optimiser step?
4. Redo the 8,798,208 count without looking.

## E. Questions

1. Give the 60-second pitch: problem, approach, result.
2. Why LoRA on the LLM, and not on the flow-matching model or the vocoder?
3. Explain LoRA. What do r and α do? Why initialise B to zero?
4. Compute the trainable parameter count.
5. Why r = 16 and all seven projections? (*Not documented beyond the config; give the LoRA paper's reasoning and say "reconstructed".*)
6. Your resume says 4 H200s. Was it distributed training? (*No: parallel single-GPU jobs. Say it plainly.*)
7. Why BF16 and not FP16?
8. What is WER? How can it be 1.99?
9. How did you test for catastrophic forgetting?
10. One adapter for Hindi and Telugu: why not two? What did joint training cost or gain?
11. Why did the codec and distillation work (phases 2–3) not make it into the final system?
12. Hostile: "Your WER is computed by another model. How do you know it measures speech quality?" (*It measures intelligibility only; the ASR's own error floor; voice similarity and naturalness need SIM or MOS, which you didn't run, unless the presentation says otherwise.*)

## My notes and fumbles

- 
