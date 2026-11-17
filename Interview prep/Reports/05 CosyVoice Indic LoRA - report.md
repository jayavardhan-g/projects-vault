# Project report: Adapting CosyVoice 2 to Hindi and Telugu with LoRA (CS692, R&D Project II)

> **For the LLM reading this.** This report describes a course project by G Jayavardhan (M.Tech CSE, IIT Bombay) for
> **CS692: R&D Project II**, Spring 2026, instructor **Prof. Ganesh Ramakrishnan**. He must defend it in interviews.
> It was written from his scripts (`/data/jayavardhan/cosyvoice_rnd/scripts/`), the training logs and data manifests on
> `/mnt/warm_storage/jayavardhan/cosy/`, and the upstream CosyVoice code.
>
> Labels:
> - **[CODE]**: what the scripts do, with `file:line`.
> - **[LOG]**: a value read from training logs or manifests.
> - **[RESUME-ONLY]**: a number whose result file was **not found** on the server.
> - **[BACKGROUND]**: standard knowledge.
> - **[RECON]**: the report writer's reasoning.
>
> The repo's `README.md` is an early *plan* (CosyVoice 3, DoRA, Tamil/Bengali/Marathi, Mimi/SNAC) and does **not**
> describe what was actually done. Trust the scripts and logs over it.

---

## 1. Summary

CosyVoice 2 (Alibaba FunAudioLLM, the "CosyVoice2-0.5B" release) is a zero-shot TTS system. Its architecture:
- a **Qwen2.5-0.5B-based LLM** that predicts discrete speech tokens (25 Hz, FSQ) from text and a voice prompt;
- a **conditional flow-matching** model that turns tokens into a mel spectrogram;
- a **HiFi-GAN (HiFT)** vocoder.

It was not trained on Hindi or Telugu, so zero-shot Indic speech is poor. He adapted it by **injecting LoRA adapters
into the inner Qwen2 transformer of the LLM only** (q, k, v, o, gate, up and down projections, r = 16, α = 32), with
everything else frozen: the flow model, the vocoder, and the LLM's speech embedding and speech-token output head.

Training data: filtered **IndicVoices-R** clips, **Hindi 18.1 h + Telugu 16.6 h = 34.8 h**, pre-tokenized once into
(text tokens, speech tokens, speaker embedding) caches.

Three adapters were trained on CosyVoice 2 (Hindi-only, Telugu-only, joint Hindi+Telugu), plus the same three on
CosyVoice 3 (Fun-CosyVoice3-0.5B). Each run took **8,798,208 trainable parameters (1.358% of CosyVoice2's 648M)**, used
bf16 autocast and AdamW with early stopping, and finished in 20–40 minutes on one H200.

Evaluation: generate fixed test sentences in zero-shot mode with Hindi, Telugu or English voice prompts, transcribe
with **AI4Bharat IndicConformer-600M**, and compute WER/CER. English WER was used to check forgetting. Before this
(phases 2–3) he benchmarked 7 neural codecs on Hindi and tried distilling mHuBERT-147 into WavTokenizer. **That
distillation did not work.**

Project facts:
- **Timeline:** baseline, codec and distillation work in late April 2026; LoRA runs on 2026-05-02 (CV2) and 2026-05-03 (CV3).
- **Code:** `/data/jayavardhan/cosyvoice_rnd` (an upstream CosyVoice clone plus his `scripts/`; git `a8a46d5`, 2026-08-18). Envs `cosyv0`, `cosyv1`, `codecbench`, `codec`.
- **Upstream code is not his.** `cosyvoice/` (the model, frontend, flow, hifigan) is FunAudioLLM's. His code is `scripts/*.py` and `scripts/*.sh`, about 4,500 lines.

---

## 2. Resume bullets vs evidence

| Bullet | Verdict | Evidence |
|---|---|---|
| "Adapted CosyVoice 2 to Hindi and Telugu via LoRA fine-tuning on its Qwen2 LLM component" | **Supported.** | `phase4b_train_lora.py:96-112, 155-169`: PEFT `inject_adapter_in_model` on `cv.model.llm.llm.model` (the inner Qwen2ForCausalLM). |
| "(9M params, 1.36% of the base model…)" | **Supported.** | [LOG] `trainable_params: 8798208, total_params: 647972675, trainable_pct: 1.358`. "Base model" = the *whole* CosyVoice2 (LLM + flow + HiFT), not the LLM alone. Checked by hand in §5.3. |
| "…17 MB" (resume 5) | **Wrong.** | [LOG] every `best/lora.pt` is **35,284,758 bytes (≈35 MB)**. The adapters are saved in **fp32** (8.8M × 4 bytes). 17 MB would be bf16. Say "35 MB fp32 (≈17.6 MB if stored in bf16)" or drop it. |
| "trained on 35 hours of speech" | **Supported.** | [LOG] manifests: Hindi 7,690 clips / 18.13 h / 341 speakers; Telugu 7,270 clips / 16.63 h / 337 speakers; total 34.76 h. The joint run used all 14,960 (14,662 train / 298 val). |
| "…across 4 H200 GPUs with BF16 precision" | **Partly.** | BF16: yes (`torch.autocast(dtype=torch.bfloat16)`, `:235`). **4 GPUs:** the training script is **single-GPU** (no DDP; run with one `CUDA_VISIBLE_DEVICES`). Separate runs went to different GPUs in parallel, and generation used **4 GPUs (4–7) for 4 concurrent jobs** (`generate_all_parallel.sh`). Honest wording: "experiments run in parallel across 4 H200s; each adapter trained on one GPU". |
| "Trained a single adapter jointly on Hindi and Telugu with strong results and no catastrophic forgetting" | **Joint adapter supported; results not verifiable here.** | `phase4_multi` (CV2) and `phase4_cv3_multi` trained on both languages [LOG]. Forgetting was tested by English WER of each adapter vs base CV2 (`phase5_eval_all.sh`, english pass). **The WER CSVs are not on the server.** |
| "Reduced Hindi WER from 1.99 in the zero-shot setting to 0.11 after LoRA adaptation" | **[RESUME-ONLY]** | `data/phase5_outputs/wer_*_all.csv` and `data/results/cv2_results.txt` are missing (`data/` is gitignored and absent). WER is **IndicConformer-600M CTC transcription, light Hindi normalisation, mean of per-sentence WER** (`phase5_eval_wer.py`). He must be able to explain how WER can be 1.99 (§8.2). |

---

## 3. Background (teach first)

### 3.1 CosyVoice 2 architecture [BACKGROUND + upstream code]
```
text ──Qwen BPE tokenizer──► text tokens ┐
prompt wav ─► S3 tokenizer (FSQ, 25 Hz) ─► prompt speech tokens ┤
prompt wav ─► CAM++ speaker encoder ─► 192-d speaker embedding  │ (used by flow; CV2 removed it from the LM)
                                                                ▼
             LLM (Qwen2.5-0.5B backbone, "Qwen2LM")  ─► speech tokens (autoregressive)
                                                                ▼
             Flow matching (conditional, chunk-aware) + prompt mel + spk emb ─► mel spectrogram
                                                                ▼
             HiFT vocoder (HiFi-GAN + iSTFT, f0 predictor) ─► 24 kHz waveform
```
- Upstream module path: `cv.model.llm` is the Qwen2LM wrapper, which has `forward(batch, device) → {loss}`.
  `cv.model.llm.llm.model` is the inner HF `Qwen2ForCausalLM`. `cv.model.flow` and `cv.model.hift` are the other stages.
- The LLM's training loss is next-token **cross-entropy over speech tokens**, given text tokens (teacher forcing).
- Speech tokens have their own embedding table and output head (`llm_decoder`), **outside** the inner Qwen. They
  stayed frozen here, because LoRA targets only the Qwen projection layers.

### 3.2 LoRA [BACKGROUND]
- For a frozen weight W (d_out × d_in), learn ΔW = **B·A**, with A (r × d_in) and B (d_out × r), and r ≪ d.
- The forward pass is y = Wx + (α/r)·B·A·x. B is initialised to 0, so training starts exactly at the base model.
- Trainable parameters per layer = r·(d_in + d_out).
- Advantages:
  - small checkpoints;
  - the base model is untouched (low forgetting risk);
  - adapters can be swapped per language;
  - can be merged into W at inference for zero extra latency.
- Here α/r = 32/16 = **2**, and dropout is 0.05.
- [RECON] Why LoRA over full fine-tuning: about 35 h of data per language risks overfitting and forgetting a 0.5B
  model. LoRA restricts the update to a low-rank subspace and keeps the base intact, so English is preserved.

### 3.3 BF16 [BACKGROUND]
- bfloat16 has 8 exponent bits (the same dynamic range as fp32) and 7 mantissa bits.
- Unlike FP16 it needs **no loss scaling**, because gradients don't underflow.
- `torch.autocast(bf16)` runs matmuls in bf16 while the master weights and the optimizer state stay fp32.

### 3.4 Catastrophic forgetting [BACKGROUND]
Fine-tuning on new data can destroy old abilities. The test here: does the adapter-equipped model still speak
**English** as well as base CV2? This was measured with English WER.

### 3.5 WER [BACKGROUND]
- WER = (S + D + I) / N: substitutions, deletions and insertions against N reference words.
- It is **unbounded above**: if the hypothesis has many extra words (insertions), WER > 1.
- CER is the same at character level.

---

## 4. Project phases (what was actually done)

| Phase | Script(s) | What | Outcome |
|---|---|---|---|
| 1 | `step1_run_hindi_baseline.py`, `step1_test_zeroshot_working.py`, `pick_prompt.py` | zero-shot Hindi and English generation with CV2 and CV3 from a Hindi prompt clip (prompt picked by SNR/CER filters) | the baseline (resume's "1.99" Hindi WER for CV2) |
| 2 | `phase2_codec_benchmark.py` | round-trip 50 high-SNR Hindi test clips through **7–8 codecs**: CosyVoice FSQ tokenizer, SNAC, DAC, EnCodec, WavTokenizer, XCodec2, Mimi (full and semantic). Metrics PESQ-wb, STOI, log-mel MCD, RTF | results CSV not on the server |
| 3 | `phase3_prepare_data.py`, `phase3_train_codec.py` | **distil mHuBERT-147** (multilingual HuBERT) features into WavTokenizer's quantised representation, in 2 stages (head warm-up, then unfreeze the encoder with the decoder frozen; λ up to 0.5 or 1.0) | **didn't work** [LOG]: val distill loss barely moved (0.0147 → 0.0144) while val reconstruction loss **got worse** once the encoder was unfrozen (0.44 → 0.94 in `checkpoints_v2`); the long λ = 0.5/1.0 runs sat at recon ≈ 1.05. Abandoned. |
| 4a | `phase4a_pretokenize.py` | build the LoRA training cache | Hindi 18.1 h, Telugu 16.6 h |
| 4b | `phase4b_train_lora.py` | LoRA training (6 runs: CV2/CV3 × hi/te/multi) | §6 |
| 4c | `phase4c_generate.py`, `generate_all_parallel*.sh`, `generate_crosslingual_parallel.sh` | generate test sentences with each adapter × prompts (hi ×2, te ×2, en ×1) and cross-lingual combinations | 13 configs, 4 jobs at a time on GPUs 4–7 |
| 5 | `phase5_eval_wer.py`, `phase5_eval_wer_whisper.py`, `phase5_eval_all.sh` | IndicConformer WER/CER per sentence; Hindi, Telugu and English passes; per-category pivots | CSVs not on the server |
| — | `compile_results*.py`, `plot_*` | tables and training curves | outputs not on the server |

(Separately, `/mnt/warm_storage/jayavardhan/cosyvoice3_exp/` has later May 17–19 experiments with the upstream CV3
training recipe on a "rasa_hindi" set: overfit tests and a fine-tune. They are not part of the resume claims.)

---

## 5. Data pipeline (`phase4a_pretokenize.py`)

### 5.1 Source and filtering
- **IndicVoices-R** (AI4Bharat, Hugging Face `ai4bharat/indicvoices_r`), stored as parquet shards with audio bytes and
  metadata: duration, SNR, C50, CER, speaker_id, gender, normalized text. Telugu: 35 train shards and 2 test shards
  downloaded (`Instructions`).
- **Filters** (`:46-52`, mask in the shard loop):
  - duration **2–15 s**;
  - **SNR ≥ 20 dB**;
  - **C50 ≥ 15 dB** (speech clarity / low reverberation);
  - dataset-provided **CER ≤ 0.10** (the transcript is trustworthy);
  - stop at `--max-hours` (default 20 h).
- [RECON] Why these filters: TTS learns whatever is in the audio, so noisy or reverberant clips and wrong transcripts
  teach the model noise and misalignment. High SNR/C50 and low CER mean clean, correctly labelled targets. 2–15 s keeps
  sequences bounded for memory.

### 5.2 Per-clip caching
For each kept clip, CosyVoice's own frontend produces:
- `speech_tokens` = `fe._extract_speech_token(wav)` (S3 tokenizer, **25 tokens/s**);
- `spk_emb` = `fe._extract_spk_embedding(wav)` (**192-d**);
- `text_tokens` = `fe._extract_text_token(normalized_text)` (Qwen BPE).

These are saved as one `.pt` per clip, together with a manifest CSV and a train/val split (val fraction 0.02).
[RECON] Why pre-tokenize: the tokenizer and speaker model run once instead of every epoch. That makes training
I/O-light and fast (each run took under an hour), and it guarantees the targets come from **the exact tokenizer the
frozen flow decoder expects**.

### 5.3 Parameter count, checked by hand [RECON, matches the log exactly]
Qwen2.5-0.5B: 24 layers, hidden 896, intermediate 4,864, 14 query heads and 2 KV heads with head_dim 64, so k/v outputs are 128 wide. LoRA r = 16 adds r·(in + out) per matrix:

| matrix | in → out | params |
|---|---|---|
| q, o | 896 → 896 | 16 × 1,792 = 28,672 each |
| k, v | 896 → 128 | 16 × 1,024 = 16,384 each |
| gate, up | 896 → 4,864 | 16 × 5,760 = 92,160 each |
| down | 4,864 → 896 | 92,160 |

- Per layer: 2·28,672 + 2·16,384 + 3·92,160 = **366,592**. × 24 = **8,798,208**, which is exactly the logged count.
- 8,798,208 / 647,972,675 = **1.358%**.
- In fp32: 8.8M × 4 B = **35.2 MB**, which matches `lora.pt` at 35,284,758 B.
- On CosyVoice 3 (868M total) the same adapter is 1.014%.

---

## 6. Training (`phase4b_train_lora.py`)

### 6.1 Code walk
- `CachedTokenDataset` loads the cached dicts. `collate_fn` (`:55-89`) zero-pads `text_token` and `speech_token` to the
  batch maximum and builds the batch dict that the upstream `Qwen2LM.forward(batch, device)` expects: `text_token`,
  `text_token_len`, `speech_token`, `speech_token_len`, `embedding` [B, 192], plus empty `instruct_token*` keys so the
  same script works for CV3. The lengths let the model mask padding.
- `AutoModel(model_dir=…)` loads CosyVoice. **Everything is frozen**: `llm`, `flow`, `hift` (`:155-158`).
- `apply_lora_to_qwen` (`:96-112`): `LoraConfig(r, lora_alpha, lora_dropout, target_modules=[q,k,v,o,gate,up,down], bias="none")`
  with `inject_adapter_in_model`. That is **in-place injection, not `get_peft_model`**.
  Why [DOC, `:97-98`]: wrapping would change the module class and break CosyVoice's hard-coded attribute path
  `self.llm.model.model.embed_tokens`.
- CV2 vs CV3 is auto-detected by class name (`:163`).
- Logs trainable and total parameter counts (the 1.358%) and a sample of trainable names as a sanity check (`:171-193`).
- Optimiser: **AdamW**, lr 2e-4, betas (0.9, 0.999), weight decay 0.01, LoRA parameters only (`:196-199`).
- Loop (`:228-300`):
  - `torch.autocast(bfloat16)` forward with `loss = out["loss"] / grad_accum`, then backward;
  - every 4 micro-batches: `clip_grad_norm_(1.0)` → `opt.step()` → `zero_grad()`;
  - log every 50 steps;
  - **validate every 200–300 steps**, save `best/lora.pt` when val loss improves;
  - **early stopping after 4 validations without improvement**;
  - wall-clock deadline (`--hours-budget`);
  - periodic `step_N/lora.pt` snapshots.
- Only `requires_grad` parameters (the LoRA A and B matrices) are saved.
- Effective batch = **4 × 4 = 16 utterances**.

### 6.2 Runs [LOG]
| Run | Base | Train / val clips | Steps (early stop) | Best val loss | Time |
|---|---|---|---|---|---|
| `phase4` (Hindi) | CosyVoice2-0.5B | 7,537 / 153 | 2,200 | 4.400 | 0.36 h |
| `phase4_te` (Telugu) | CV2 | 7,125 / 145 | 2,400 | 4.457 | 0.39 h |
| `phase4_multi` (Hindi + Telugu) | CV2 | 14,662 / 298 | 4,800 | 4.418 | 0.68 h |
| `phase4_cv3_hi` | Fun-CosyVoice3-0.5B | 7,485 / 152 | 2,200 | 3.977 | 0.37 h |
| `phase4_cv3_te` | CV3 | 7,125 / 145 | 2,000 | 4.095 | 0.34 h |
| `phase4_cv3_multi` | CV3 | 14,610 / 297 | 3,900 | 4.027 | 0.66 h |

All runs: r = 16, α = 32, dropout 0.05, lr 2e-4, batch 4 × accumulation 4, bf16, one GPU each.

[RECON] Reading the losses:
- **The joint adapter's val loss (4.418) sits between the single-language ones** (4.400 hi, 4.457 te) on a mixed val set.
  That is consistent with one adapter learning both languages without a big penalty.
- About 2,200 optimizer steps × 16 ≈ 35k samples ≈ 4.7 epochs over the Hindi set before early stopping.
- CV2 and CV3 losses are **not comparable**: different speech tokenizers, so the target vocabularies differ.

---

## 7. Generation and evaluation

### 7.1 Generation (`phase4c_generate.py`)
- Load base CV → inject the same LoRA structure → **copy saved tensors into the matching parameter names**, with shape checks and skipped names reported (`load_lora_weights`).
- Zero-shot synthesis: `cv.inference_zero_shot(text, prompt_text, prompt_wav, stream=False)`. The prompt is a short
  Hindi, Telugu or English clip plus its transcript, which sets the voice. CV3 needs `<|endofprompt|>` appended to the
  prompt text; CV2 must not have it.
- Writes `<id>_<lang>.wav` and a CSV with generation time and audio length.
- **Configurations** (`generate_all_parallel.sh`):
  - `lora_hi` on hi (2 prompts), te (2), en (1);
  - `lora_te` on te (2), en (1);
  - `lora_multi` on hi (2), te (2), en (1);
  - plus **cross-lingual** runs (e.g. Telugu or English voice speaking Hindi text).
  - 13 jobs, run 4 at a time on GPUs 4–7.

### 7.2 WER evaluation (`phase5_eval_wer.py`)
- ASR: **`ai4bharat/indic-conformer-600m-multilingual`** (`trust_remote_code`). Audio is resampled to 16 kHz and decoded with **CTC** (faster than RNNT, "fine for WER").
- Normalisation (`normalize_hindi`): collapse whitespace and strip punctuation, including the danda `।`.
- `jiwer.wer` and `jiwer.cer` per sentence. **Reported WER = mean over sentences** (macro average).
- Output: a long CSV (label, id, category, ref, hyp, wer, cer), a summary per label, and a **per-category pivot**.
- Passes in `phase5_eval_all.sh`:
  - **Hindi:** base CV2 vs `lora_hi` (p0, p1) vs `lora_multi` (p0, p1) vs cross-lingual voices;
  - **Telugu:** `lora_hi` used cross-lingually, `lora_te`, `lora_multi`, cross-lingual voices;
  - **English** (forgetting check): base CV2 vs `lora_hi`, `lora_te`, `lora_multi`.
- `phase5_eval_wer_whisper.py` is an alternative Whisper-based scorer.

---

## 8. Results and how to talk about them

### 8.1 What can be verified
Parameter count, data hours, the training curves and early stopping, file sizes, and the experiment matrix. All are in §5–7.

### 8.2 Hindi WER 1.99 → 0.11 [RESUME-ONLY]
The result files are not on the server. If he has them on a laptop or in a presentation, those are the source. How to explain the numbers:
- **Why can zero-shot WER be 1.99 (above 1)?**
  - Base CosyVoice 2 never learned to speak Hindi. Its LLM produces speech tokens that do not correspond to the Hindi
    text, so the result is babble, other-language phonetics, or long run-on audio.
  - IndicConformer then transcribes **many words that aren't in the reference**. Insertions push (S + D + I)/N past 1.
  - Because the reported number is a **mean of per-sentence WER**, a few short sentences with heavy insertions can
    dominate it.
  - [RECON] He should check (a) whether some outputs were much longer than expected (looping generation) and
    (b) the median WER and the corpus-level WER (total edits ÷ total words) alongside the mean.
- **0.11 after LoRA** means about 1 word in 9 wrong, as judged by an ASR model that has its own error rate on natural
  Hindi. A fair comparison would also report the ASR's WER on **real** IndicVoices-R recordings of similar text (a
  "ground-truth floor"). [RECON]
- Category pivots exist in the eval script (sentence categories from the test CSV), so per-category behaviour can be discussed if he has the CSV.

### 8.3 "No catastrophic forgetting" [RECON on the method]
- Evidence would be that the English WER of `lora_hi`, `lora_te` and `lora_multi` is close to base CV2's English WER (the english pass in `phase5_eval_all.sh`).
- Why it is plausible:
  - the base weights are frozen and the update is low-rank;
  - the speech-token embedding and output head, the flow model and the vocoder are untouched;
  - the update is small (r = 16) and early-stopped.
- LoRA adapters can also be **removed** to recover the exact base model.

### 8.4 The joint adapter
- A single adapter trained on the union (14,662 clips) reached val loss 4.418, in the same range as the separate adapters.
- One adapter for both languages is simpler to deploy and lets Hindi and Telugu share learned Indic phonetic structure.
  Both are Indian languages but from **different families** (Indo-Aryan vs Dravidian).

---

## 9. Hardest problems and what didn't work

1. **Fitting PEFT into CosyVoice's code.** `get_peft_model` wraps the model and breaks the hard-coded path
   `self.llm.model.model.embed_tokens`. Fix: `inject_adapter_in_model` in place, which keeps the class (`:96-112`).
   A matching injection plus a name-by-name copy is used to load adapters at inference.
2. **One script for CV2 and CV3.** The batch dict needs CV3-only keys (`instruct_token*`) filled with empty tensors.
   CV3 prompts need `<|endofprompt|>`. The version is detected by LLM class name.
3. **The semantic distillation attempt failed** (Phase 3). Aligning WavTokenizer's representation with mHuBERT-147
   barely reduced the distillation loss and damaged reconstruction when the encoder was unfrozen. The conclusion was
   that a better approach was to adapt the **LLM** on top of CosyVoice's existing tokenizer rather than change the
   codec, and that is what Phase 4 did. [RECON on the motivation]
4. **Data quality.** IndicVoices-R contains noisy, reverberant and mis-transcribed clips, hence the SNR, C50 and CER
   filters. Resumable caching (`--start-shard`, skip existing `.pt`) handles crashes during long extraction.
5. **Shared machine.** Storage was split between `/data` and warm storage. Runs were spread across GPUs 4–7.

---

## 10. Design decisions

| Decision | Alternatives | Why | Source |
|---|---|---|---|
| Adapt the LLM only | adapt flow/vocoder; full fine-tune | the LLM decides *what* tokens are spoken (pronunciation and content); flow and vocoder render acoustics, which are largely language-agnostic | [RECON] |
| LoRA r = 16, α = 32 on all attention and MLP projections | attention-only; DoRA (as in the README plan) | MLP layers hold much of the language knowledge; r = 16 is a common default; DoRA was planned but not used | [CODE] + [RECON] |
| Freeze the speech embedding and output head | train them | the speech-token vocabulary is unchanged; keeps the decoder's expectations intact | [CODE] |
| Pre-tokenize with CosyVoice's own frontend | tokenize on the fly | speed; identical tokenizer | [CODE] |
| Quality filters (SNR ≥ 20, C50 ≥ 15, CER ≤ 0.1, 2–15 s) | no filtering | clean targets, bounded sequences | [CODE] |
| bf16 autocast, AdamW, grad clip 1.0, grad accumulation 4 | fp16, SGD | stability without loss scaling; effective batch 16 on one GPU | [CODE] |
| Early stopping (patience 4) on val loss | fixed epochs | prevents overfitting a small dataset | [CODE] |
| IndicConformer for WER | Whisper | Indic-specialised ASR (a Whisper scorer exists as an alternative) | [CODE] |
| English WER as the forgetting test | none | direct test of retained ability | [CODE] |
| Joint hi+te adapter | one per language | a single deployable adapter | [CODE] |

---

## 11. Limitations
- The WER result files are not on the server, so the headline numbers can't be re-derived here.
- WER from an ASR model measures intelligibility, not naturalness or speaker similarity. No MOS/UTMOS or speaker-similarity results were found.
- Small test set (sentence CSV size unknown here), mean-of-ratios WER, one or two prompts per language.
- Single-GPU training, short runs (under 1 h), no hyperparameter sweep over r or α.
- The repo README describes an unexecuted plan and should not be shown as the project description.

---

## 12. Likely interview questions
1. *Walk me through CosyVoice 2's pipeline.* → §3.1.
2. *Why LoRA on the LLM and not on the flow model?* → §10, row 1.
3. *How many parameters does your LoRA add? Derive it.* → §5.3.
4. *What is α/r, and what does it do?* → scales the update; here 2. With B initialised to 0 the model starts as the base model.
5. *How did you prepare the data?* → §5: filters, frontend extraction, caching.
6. *How can WER be 1.99?* → §8.2: insertions, mean of per-sentence WER.
7. *How did you show there was no catastrophic forgetting?* → English WER of each adapter vs base. Frozen base weights.
8. *Why bf16 and not fp16?* → §3.3.
9. *Did you use 4 GPUs with DDP?* → No. Single-GPU training runs, parallel experiments and 4-way parallel generation. Be precise.
10. *Why one joint adapter?* → simpler deployment, shared Indic structure; its val loss is between the single-language ones.
11. *What would you do next?* → MOS and speaker-similarity evaluation; ASR-on-real-speech floor; more data; try DoRA or higher r; adapt the speech-token head; streaming tests; Tamil and other languages. [RECON]
12. *What didn't work?* → mHuBERT distillation into WavTokenizer (§9.3).
13. *Why did you need `inject_adapter_in_model`?* → §9.1.
14. *How does zero-shot prompting set the voice?* → prompt speech tokens + prompt text go to the LM; prompt mel + speaker embedding go to flow matching.
15. *What does early stopping on val loss risk for TTS?* → val CE loss doesn't equal perceptual quality. Ideally select by WER or MOS on a dev set. [RECON]

---

## 13. Numbers cheat sheet
| Number | Meaning |
|---|---|
| 8,798,208 (≈9M) | LoRA trainable params (r = 16, 7 projections × 24 layers) |
| 1.358% / 1.014% | of CosyVoice2 (648.0M) / CosyVoice3 (868.0M) total params |
| 35.3 MB | adapter file size (fp32); not 17 MB |
| 18.13 h + 16.63 h = 34.8 h | Hindi + Telugu training audio (7,690 + 7,270 clips; 341 + 337 speakers) |
| 2–15 s, SNR ≥ 20, C50 ≥ 15, CER ≤ 0.10 | data filters |
| r 16, α 32, dropout 0.05, lr 2e-4, bs 4 × 4 | LoRA and training hyperparameters |
| 4.400 / 4.457 / 4.418 | best val loss: hi / te / joint (CV2) |
| 0.36–0.68 h | wall-clock per training run (early-stopped) |
| 25 Hz, 192-d | CV2 speech-token rate; speaker-embedding size |
| 1.99 → 0.11 | Hindi WER zero-shot → LoRA [RESUME-ONLY; files missing] |
| IndicConformer-600M, CTC | ASR used for WER |
