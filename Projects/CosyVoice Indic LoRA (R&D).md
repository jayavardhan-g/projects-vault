---
type: project
folder: not in projects/ — likely /mnt/warm_storage/jayavardhan/{cosy, cosyvoice3_exp}
dates: before 2026-06 (warm-storage folders dated May 2026)
status: done (summarised from Intro.md only)
kind: speech R&D — TTS adaptation
stack: [CosyVoice 2/3 0.5B, WavTokenizer-large, mHuBERT-147, LoRA, IndicConformer, 4× H200]
tags: [project, speech]
---
# CosyVoice Indic LoRA adaptation (speech R&D)

**Not read directly.** This note is built only from the summary in `init/Intro.md` §4. The code and data are probably the
warm-storage folders `cosy/` and `cosyvoice3_exp/` (dated May 2026), which were not opened for this vault.

**What.** The project adapted CosyVoice to Hindi and Telugu. Phases: (1) baseline CosyVoice on Indic text, (2) benchmark neural
audio codecs on Indic audio, (3) attempt mHuBERT-147 semantic distillation into WavTokenizer, (4) LoRA fine-tune of the
CosyVoice 2 LLM (~9M trainable parameters), (5) evaluate adapters with IndicConformer ASR WER across languages and conditions.

**Data.** AI4Bharat IndicVoices-R: 18 h Hindi and 17 h Telugu (35 h combined, 14,662 clips). Trained in BF16 with FlashAttention-2 on 4 GPUs.

**Results.** None are recorded in `Intro.md`. They are in the R&D presentation it refers to.

**To fill in:** results, and paths to the presentation and code in warm storage.
