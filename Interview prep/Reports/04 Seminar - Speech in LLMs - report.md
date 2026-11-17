# Project report: M.Tech seminar, "Speech in Large Language Models and Controlled Text-to-Speech"

> **For the LLM reading this.** This report describes a literature-review seminar by G Jayavardhan (M.Tech CSE, IIT Bombay),
> course **CS694 (M.Tech Seminar)**, Spring 2026, guide **Prof. Ganesh Ramakrishnan**, mentor Iisha Pandey. He must
> defend it in interviews. The seminar has no code. Its content is the written report
> (`/data/jayavardhan/mtp/qwen/report/Seminar Report.docx`, May 2026, about 7,000 words) and a talk.
>
> Labels:
> - **[REPORT]**: what his seminar report says.
> - **[BACKGROUND]**: standard knowledge added by the report writer so the material can be taught from zero. Specific
>   paper numbers under this label should be double-checked against the papers before he quotes them.
>
> Use this to teach the topic from basics and to quiz him on it.

---

## 1. Summary

The seminar studies how speech moved from a **cascaded pipeline** (ASR → text LLM → TTS) to **token-based
speech-language models**, in which speech is turned into discrete tokens and modelled like text. The focus is on
**controlled text-to-speech (TTS)**: generating speech whose content, voice identity, emotion, prosody, speed and style
can be controlled.

It covers:
- **tokenization methods:** VQ, RVQ, FSQ, single-codebook codecs;
- **SpeechGPT:** HuBERT units in an LLM vocabulary, plus instruction tuning;
- **CosyVoice:** supervised semantic tokens from an ASR encoder, an LLM, flow matching and a vocoder;
- **CosyVoice 2:** FSQ at 25 Hz, Qwen2.5-0.5B as the LM, chunk-aware streaming flow matching;
- **WavTokenizer:** a single-quantizer acoustic codec at 40–75 tokens/s;
- **Spark-TTS:** BiCodec with global and semantic tokens and attribute control;
- **CosyVoice 3:** about 1M hours, a multi-task supervised tokenizer, DiffRO post-training, CV3-Eval.

It ends with the central trade-offs (content vs voice vs control, compression vs fidelity, modular vs end-to-end), open
problems, evaluation gaps and safety.

**Resume bullets (resumes 4 and 5):**
- "Studied the shift from cascaded pipelines to natively multimodal architectures for speech in LLMs, covering tokenization (VQ, FSQ, WavTokenizer), unified frameworks (SpeechGPT), and generation systems (CosyVoice)."
- "Analyzed trade-offs between compression and fidelity in speech tokenization, and between modular vs. end-to-end system design for latency, coherence, and multilingual scalability."

Both match the report's content. The report's title is "Speech in Large Language Models and Controlled
Text-to-Speech", while the resume calls it "Speech in Large Language Models: A Literature Review". **This seminar is
the theory behind his CosyVoice project** (see the CosyVoice report).

---

## 2. Audio and speech basics (teach first) [BACKGROUND]

- **A waveform** is air-pressure samples over time. **Sample rate** is samples per second: 16 kHz for ASR, 22.05 or 24 kHz for TTS. One second at 16 kHz is 16,000 numbers, far too long a sequence for a transformer.
- **Frames and hops.** Audio is cut into short overlapping windows, e.g. 25 ms windows every 10 ms (hop). That gives 100 frames per second.
- **STFT** (short-time Fourier transform) gives per-frame frequency content.
- **Mel spectrogram:** the STFT magnitude mapped onto the mel scale (perceptual frequency), usually 80 or 128 mel bins, log-compressed. It is the standard intermediate representation for ASR inputs and TTS outputs.
- **Vocoder:** turns a mel spectrogram back into a waveform. Examples: HiFi-GAN, a GAN-trained convolutional upsampler; Vocos, which predicts STFT coefficients and applies an inverse FFT.
- **Token rate (Hz):** discrete tokens per second of audio. Text is roughly 3–4 tokens per second of speech. Speech codecs produce 25–600+ tokens/s, which is why sequence length matters.
- **Bitrate** = tokens per second × log₂(codebook size) × number of codebooks. Example: 75 tokens/s × 12 bits (4096 codes) × 1 codebook = 900 bps = 0.9 kbps (WavTokenizer).
- **Three layers of information in speech** [REPORT §2.2]:
  - **semantic:** words, phonemes;
  - **acoustic:** speaker timbre, pitch, loudness, room and channel;
  - **paralinguistic:** emotion, intent, emphasis, sarcasm, laughter, hesitation.
- **ASR encoders:** Whisper, SenseVoice. They map audio to text. Early layers are acoustic and later layers semantic, which is why CosyVoice takes tokens from a **middle** layer.
- **Self-supervised speech models:** HuBERT, wav2vec 2.0. They are trained without text, e.g. HuBERT predicts k-means cluster ids of masked frames. Their discrete units capture *what is said* well and *how it is said* poorly [REPORT §2.2, "emotion problem"].
- **Evaluation metrics:**
  - **WER / CER:** ASR on the generated speech vs the input text, (S+D+I)/N. It can exceed 1 when there are many insertions.
  - **Speaker similarity (SS / SIM):** cosine similarity of speaker-verification embeddings.
  - **MOS:** human rating 1–5, or predicted MOS (UTMOS, DNSMOS).

---

## 3. Why speech must be tokenized [REPORT ch. 2]

- An LLM predicts the next discrete token, but a waveform is continuous and very long. So speech must be compressed into a discrete vocabulary. The pipeline is: waveform → encoder → dense vectors → **quantizer** → token ids.
- **The token design decides what the LLM can see:**
  - content-focused (semantic) tokens give intelligible words but lose texture and emotion;
  - reconstruction-focused (acoustic) tokens preserve sound but are hard to align with text.
  - So systems mix types: **semantic tokens for content, acoustic tokens for detail, global tokens for utterance-level speaker attributes.**
- **Control as conditional generation:** TTS is p(speech | text). Controlled TTS adds conditions: speaker, emotion, pitch, rate, accent, language, events such as [laughter]. They can be given as prompt text, special tokens, speaker embeddings, reference speech tokens or attribute tokens.
  - The hard part is **conditioning without leakage.** Reference audio also carries its language, rhythm and emotion, which can contaminate the output.
  - CosyVoice drops prompt speech tokens in cross-lingual cloning and uses a language id instead.
  - CosyVoice 2 removes the speaker embedding from the LM because it leaked language and paralinguistic information.
  - His phrasing: "controllability is partly about deciding what not to condition on."

---

## 4. Quantizers and codecs [REPORT ch. 3 + BACKGROUND]

### 4.1 Vector quantization (VQ)
- An encoder outputs z, and the quantizer picks the nearest codebook vector: k = argmin_j ‖z − e_j‖₂. The token is k, and the decoder receives e_k.
- [BACKGROUND] Training (VQ-VAE):
  - argmin is not differentiable, so the **straight-through estimator** copies gradients from e_k to z;
  - loss = reconstruction + ‖sg[z] − e‖² (codebook loss) + β‖z − sg[e]‖² (commitment loss), where sg is stop-gradient;
  - alternatively the codebook is updated by an exponential moving average (EMA).
- **Weaknesses** [REPORT]:
  - **codebook collapse:** only a few codes get used, giving a long-tailed usage distribution;
  - nearest-neighbour search costs grow with codebook size.
  - Fixes [BACKGROUND]: k-means initialisation, dead-code restart ("random awakening" in WavTokenizer), lower-dimensional factorised codes, L2-normalised codes.
- **CosyVoice v1:** a single **4,096-entry** VQ codebook inserted in the **middle of a supervised ASR encoder** [REPORT §3.1].

### 4.2 Residual vector quantization (RVQ)
- Stages are applied in sequence: q₁ = VQ₁(z), r₁ = z − q₁, q₂ = VQ₂(r₁), … and z ≈ Σ qᵢ. Each frame becomes **several tokens**, one per codebook.
- [BACKGROUND] Used by SoundStream, EnCodec and DAC. Example: EnCodec at 24 kHz uses 75 frames/s with up to 32 codebooks of 1,024 entries. 8 codebooks give 75 × 8 × 10 bits = 6 kbps, and **600 tokens per second**.
- **Upside:** high-fidelity reconstruction, and bitrate can be scaled by dropping later codebooks.
- **Downside for LLMs** [REPORT §3.2]: the LM must predict multiple correlated tokens per frame. This raises token rate, memory and long-range modelling difficulty, and needs special architectures: AR + NAR splits (VALL-E), parallel or delay patterns (MusicGen), or local–global transformers.
- **A single flat stream is preferable for LLMs, because it resembles text.**

### 4.3 Finite scalar quantization (FSQ)
- Project the latent to a **low-dimensional** space (d dimensions). Bound each dimension (e.g. with tanh) and **round** it to one of L levels. The tuple of levels is read as a number in a mixed-radix system, which gives one token id. Codebook size = ∏ Lᵢ.
  - [BACKGROUND] e.g. d = 8 with 3 levels each gives 3⁸ = 6,561 codes.
  - Gradients use the straight-through estimator through rounding.
- **Advantages** [REPORT]:
  - no learned codebook, so **no codebook collapse**;
  - near-uniform code usage.
- **Weakness:** each dimension is quantized independently, so correlations between features are not modelled the way a learned VQ codebook models them. The surrounding encoder and projector layers compensate.
- Used in the **CosyVoice 2** tokenizer (inside SenseVoice-Large), **CosyVoice 3** (inside a MinMo-based model) and Spark-TTS's global tokenizer.

### 4.4 WavTokenizer: single-codebook acoustic codec [REPORT §3.4]
- **One quantizer**, **40 or 75 tokens/s** for 24 kHz audio, compared with hundreds per second for RVQ codecs.
- Reconstruction tricks:
  - a broader VQ space;
  - k-means codebook initialisation;
  - random awakening of unused codes;
  - an extended context window;
  - attention in the decoder;
  - a multi-scale discriminator;
  - an inverse-Fourier-transform decoder (Vocos-style).
- Result: **0.9 kbps (1 quantizer, 75 tokens/s)** with strong UTMOS, beating higher-bitrate DAC configurations on some test sets.
- **Limitation:** it is still an **acoustic** codec without ASR-like understanding. It complements semantic (S3) tokenizers rather than replacing them.

### 4.5 The compression vs fidelity trade-off (resume bullet 2)
- Fewer tokens per second or fewer bits means shorter LM sequences, faster generation, and an easier time for long context. The cost is less acoustic detail and possible loss of paralinguistic cues.
- More codebooks give better reconstruction (higher UTMOS/PESQ) but make LM modelling harder.
- Semantic tokens compress more and align with text (good for LMs). Acoustic tokens preserve timbre (good for reconstruction).
- The field's answer is a **two-stage design**: the LM generates low-rate semantic tokens, and a separate renderer (flow matching + vocoder, or a codec decoder) adds acoustic detail.

---

## 5. SpeechGPT (Zhang et al., EMNLP Findings 2023) [REPORT ch. 4]

- **Goal:** give an LLM *intrinsic* cross-modal conversation, with no external ASR or TTS modules.
- **How:**
  - extend the LLM's vocabulary with **discrete speech units** from a HuBERT-style extractor;
  - generate speech by producing units, which a **unit vocoder** turns into audio.
  - [BACKGROUND] Backbone LLaMA-13B; HuBERT features k-means-clustered into 1,000 units.
- **SpeechInstruct dataset:**
  - cross-modal instructions (speech → text, text → speech);
  - **chain-of-modality** instructions (speech in → text reasoning → speech out).
- **Three training stages:**
  1. modality-adaptation pre-training: continue unit sequences, so the model learns the unit distribution;
  2. cross-modal instruction fine-tuning, which aligns speech and text;
  3. chain-of-modality instruction fine-tuning with **LoRA**.
- **Limitations for controlled TTS:**
  - no paralinguistic modelling (emotion);
  - **text-first** responses (it generates text before speech, which adds latency);
  - context length limits multi-turn spoken dialogue;
  - HuBERT units are semantic and insufficient for fine acoustic rendering.
- **Lesson:** adding speech tokens to the vocabulary is not enough. Alignment needs staged training. Later systems split the job: the **LM generates semantic tokens, and a flow-matching or codec decoder renders acoustics.**

---

## 6. CosyVoice (Du et al. 2024) [REPORT ch. 5]

- **Central claim:** speech tokens for LLM-based TTS should be **semantically aligned with text**. Unsupervised tokens are not explicitly aligned.
- **Supervised semantic speech (S3) tokenizer:** a VQ layer is placed **inside the encoder of a supervised multilingual ASR model**. The ASR loss forces the tokens to carry content, and the middle-layer placement keeps some acoustic context.
  - [BACKGROUND] CosyVoice 1 tokens: 50 Hz, 4,096 codes.
- **Four components:**
  1. text encoder (aligns text and speech semantic spaces);
  2. S3 speech tokenizer;
  3. **autoregressive LM** that predicts speech tokens from text, a speaker embedding and prompt tokens;
  4. **conditional flow matching (CFM)** that turns tokens into a mel spectrogram, followed by a neural **vocoder** that produces the waveform.
- **Zero-shot voice cloning:**
  - a speaker embedding comes from a pretrained voice-print (speaker-verification) model;
  - the LM gets the speaker embedding, text and prompt speech tokens;
  - flow matching gets the generated tokens, the speaker embedding and the prompt's mel features.
  - **Cross-lingual:** prompt speech tokens are omitted and a language id is added, to avoid leaking the prompt's language and prosody.
- **CosyVoice-Instruct:** natural-language style descriptions ("a happy girl with high tone and quick speech") and markers such as `[laughter]`, `[breath]` and emphasis tags. Instructions are ambiguous: "happy" maps to pitch, rhythm, loudness, brightness and emphasis together.
- **Result:** supervised semantic tokens beat unsupervised tokens on content consistency (WER) and speaker similarity. **Token supervision matters as much as scale.**
- **Training data** (table in his report): Chinese 130k h, English 30k h, Cantonese 5k h, Japanese 4.6k h, Korean 2.2k h.
- [BACKGROUND] **Flow matching** learns a velocity field v_θ(x_t, t, cond) that transports Gaussian noise to the data distribution along (usually straight) paths: x_t = (1−t)·x₀ + t·x₁. The training target is x₁ − x₀. At inference you integrate an ODE in a few steps. It is cheaper and more stable than diffusion with many steps. Classifier-free guidance improves adherence to the condition.

---

## 7. CosyVoice 2 (Du et al. 2024): streaming [REPORT ch. 6]

- **Motivation:** voice chat needs speech to start before all the text exists. Offline TTS waits for the full text.
- **A simplified LM:**
  - **removes the text encoder**: the pretrained LLM backbone, **Qwen2.5-0.5B**, aligns text and speech directly;
  - **removes the speaker embedding from the LM**: an utterance-level embedding leaked language and paralinguistic information.
  - Speaker identity now enters through prompt tokens and the flow-matching stage.
- **Sequences:**
  - non-streaming: `[S] text… [T] speech… [E]`;
  - **streaming:** text and speech tokens interleaved at a fixed **N:M = 5:15** ratio, so speech tokens can be emitted before the text is complete.
- **FSQ tokenizer** inside SenseVoice-Large: project → bound and round → project back → id. **25 tokens/s**, half CosyVoice 1's rate. Better codebook use and an easier LM job.
  - [BACKGROUND] Codebook 6,561 = 3⁸.
- **Chunk-aware causal flow matching:** one model trained with **four attention masks**, sampled at random:
  - non-causal (offline quality);
  - full-causal (lowest latency);
  - Chunk-M (trade-off);
  - Chunk-2M (near offline).
  - One model serves both streaming and offline, and wider-context masks act as implicit teachers for narrower ones.
- **The latency vs quality trade-off:** with less context, pronunciation and prosody can be wrong. His example: "lead" (to lead a movie) vs "lead" (the metal pipe). More context means better quality but higher first-packet latency.
- [BACKGROUND] The CosyVoice 2 paper reports about 150 ms first-packet latency in streaming mode with quality close to offline.

---

## 8. Spark-TTS (Wang et al. 2025) [REPORT §7.1–7.2]

- **Two problems it targets:**
  - architectural complexity (multi-stage pipelines or parallel token streams that are hard to fit into a text LLM);
  - coarse control (it can mimic a voice but cannot "raise pitch by X").
- **BiCodec:** speech decomposed into
  - **semantic tokens** (50 tokens/s, *what* is said): a single-codebook VQ with factorised low-dimensional codes (as in DAC), from averaged wav2vec 2.0 layers 11, 14 and 16 (11 and 14 track words, 16 tracks phonemes), with ConvNeXt encoder and decoder;
  - **global tokens** (a **fixed-length** sequence, *who* is speaking: timbre, pitch range): an ECAPA-TDNN encoder (Wespeaker), cross-attention with learnable queries, and **FSQ** to avoid collapse.
  - Keeping the streams separate prevents speaker information leaking into content and vice versa.
- **The LM is Qwen2.5-0.5B.** It outputs global tokens first, then semantic tokens. The BiCodec decoder reconstructs the waveform **directly**, with no separate flow-matching or vocoder stage.
- **Attribute control:**
  - coarse: gender, 5-level pitch and 5-level speed labels;
  - fine: numeric pitch and speaking rate quantised to integers;
  - done **chain-of-thought style**: the LM first predicts the attribute values, then generates speech tokens conditioned on them.
- **VoxBox dataset:** about 100k hours annotated with gender, pitch and speed (English 20k+, Mandarin about 35k, other about 45k).
- **Limitation** (his table): optimised mainly for shorter sentences; latency concerns.

---

## 9. CosyVoice 3 (Du et al. 2025) [REPORT §7.3–7.4]

- **Goal:** in-the-wild generation (languages, domains, text formats, noise).
- **Tokenizer:** FSQ inside a large audio-understanding model (MinMo-based), trained **multi-task**: ASR, language id, speech emotion recognition, audio event detection, speaker analysis. Tokens therefore carry paralinguistics as well as content.
- **Scale:**
  - about **1M hours** across 9 languages plus many Chinese dialects and accents;
  - the LM grows from **0.5B to 1.5B**;
  - a larger CFM (DiT-based) renderer.
- **DiffRO (Differentiable Reward Optimization):** post-training that optimises speech tokens directly using rewards from ASR-like and other downstream models, without synthesising waveforms during training.
- **Instructions:** 100+ speaking styles (emotion, speed, tone, role, dialect, accent), plus markers.
- **CV3-Eval benchmark:** multilingual, cross-lingual, emotion, expressive and accent voice cloning, and continuation. Its motivation is that audiobook test sets are too clean.
- **Remaining weakness:** emotion control works better when the emotion matches the text's sentiment. Emotion that contradicts the text is hard, so models infer emotion from text instead of treating it as an independent dimension.

---

## 10. Comparative view [REPORT Table 8.1]

| System | Token design | Renderer | Control | Main limitation |
|---|---|---|---|---|
| SpeechGPT | HuBERT units added to LLM vocab | unit vocoder | cross-modal + chain-of-modality instruction tuning | weak paralinguistics; text-first response |
| CosyVoice | supervised S3 tokens, VQ in ASR encoder | CFM + vocoder | reference audio, speaker embedding, instructions | non-streaming; prompt-language leakage |
| CosyVoice 2 | S3 + FSQ at 25 Hz | chunk-aware CFM + vocoder | unified streaming/offline; instructions | latency–quality trade-off depends on context |
| Spark-TTS | BiCodec global + semantic | BiCodec decoder | cloning + attribute-based voice creation | short sentences; latency |
| CosyVoice 3 | multi-task supervised FSQ | larger CFM/DiT | natural language + fine-grained; DiffRO | emotion, in-the-wild robustness |

**The progression in one line:** intrinsic cross-modality (SpeechGPT) → text-aligned tokens (CosyVoice) → streaming (CosyVoice 2) → explicit attributes (Spark-TTS) → scale and post-training (CosyVoice 3).

---

## 11. His four research questions and his answers [REPORT §1.3 + synthesis]

1. **What discrete representation suits LLM-based TTS?** Low-rate, **single-stream, text-aligned (supervised) semantic tokens** for the LM, with acoustic detail delegated to a renderer. FSQ is preferred over VQ for code utilisation.
2. **How should content, identity and style be separated?**
   - semantic tokens for text alignment;
   - speaker embeddings or *global tokens* for identity;
   - flow matching or codec decoders for acoustics;
   - natural-language instructions for style.
   - Keep the utterance-level speaker embedding out of the LM to avoid leakage (CosyVoice 2).
   - Separation is still imperfect, because identity and style are entangled in real speech.
3. **How can one system do offline and streaming?** Interleaved text and speech tokens (5:15), plus chunk-aware flow matching trained with randomly sampled masks (CosyVoice 2).
4. **What limits remain going from cloning to explicit control?** Instruction ambiguity, emotion that depends on the text, limited annotated data (hence VoxBox), and evaluation that does not measure control fidelity.

**The central trade-off** [REPORT §8.2]: content consistency vs speaker similarity vs expressive control. Push one and the others tend to degrade:
- prioritise content and you get flat speech;
- prioritise imitation and you copy unwanted prompt traits;
- prioritise style and pronunciation or identity degrade.

---

## 12. Modular (cascaded) vs end-to-end (resume bullet 2) [REPORT §1.1 + BACKGROUND]

| | Cascaded ASR → LLM → TTS | End-to-end / token-based speech LM |
|---|---|---|
| Latency | adds up across stages; each waits for the previous one | can stream; interleaved generation |
| Information | ASR drops paralinguistics (sarcasm, urgency, laughter) | tokens can carry prosody and emotion (depending on the tokenizer) |
| Error propagation | ASR errors flow into the LLM | fewer hand-offs, but harder to debug |
| Modularity | each part is trained and upgraded independently; strong text LLMs reused | needs paired speech data; risk of losing text ability (catastrophic forgetting) |
| Coherence | the TTS doesn't know the conversational intent | one model sees the context and generates the voice |
| Multilingual scaling | swap or extend modules per language | needs multilingual speech-token data; low-resource languages suffer |

Hybrid designs such as CosyVoice (LLM for tokens, separate renderer) sit in the middle.
[BACKGROUND] Fully duplex models such as Moshi model user and system audio streams together. His M.Tech project (excluded here) builds on this text–speech gap.

---

## 13. Evaluation, safety and future work [REPORT ch. 8]

- **Evaluation gaps:** WER/CER (content), speaker-verification similarity (identity) and MOS predictors (quality) don't measure whether the intended emotion, accent, rate or laughter happened. CosyVoice 3's CV3-Eval is a step toward that.
- **Safety:** zero-shot cloning enables impersonation and fraud. Attribute-created voices reduce identity risk but still allow persuasive synthetic speech. Mitigations: consent, **watermarking**, provenance, detection, and restrictions on cloning.
- **Future directions:**
  - paralinguistics-preserving tokenizers (multi-task);
  - low-resource languages, where quality matters more than quantity (this links to his CosyVoice Hindi/Telugu work);
  - context-aware streaming pronunciation with low first-packet latency;
  - precise, measurable control interfaces;
  - multidimensional evaluation.

---

## 14. Likely interview questions (with answer pointers)

1. *Why can't an LLM take raw audio?* → continuous, too long; needs a discrete vocabulary (§3).
2. *Compare VQ, RVQ and FSQ.* → §4: collapse, tokens per frame, codebook utilisation, correlation modelling.
3. *What is codebook collapse and how is it fixed?* → few codes used; k-means initialisation, dead-code restarts, FSQ.
4. *Compute the bitrate of a codec with 8 codebooks of 1,024 entries at 75 Hz.* → 75 × 8 × 10 = 6 kbps, 600 tokens/s.
5. *Why do RVQ codecs make LLM modelling hard?* → several correlated tokens per frame; needs AR+NAR or delay patterns.
6. *What are supervised semantic tokens and why do they beat HuBERT units for TTS?* → ASR supervision gives text alignment; middle-layer placement keeps acoustics.
7. *Why did CosyVoice 2 remove the speaker embedding from the LM?* → leakage of language and paralinguistics.
8. *How does CosyVoice 2 stream?* → 5:15 interleaving + chunk-aware CFM with four masks.
9. *What does flow matching do in CosyVoice?* → tokens (+ speaker, prompt mel) → mel, via an ODE from noise; then a vocoder.
10. *What is BiCodec's key idea?* → fixed-length global tokens (who) + 50 Hz semantic tokens (what), one stream; FSQ for global tokens.
11. *How does Spark-TTS give fine-grained control?* → predict attribute values first (CoT), then tokens.
12. *What is DiffRO?* → reward optimisation on tokens via differentiable downstream models, no waveform synthesis.
13. *Cascaded vs end-to-end: when would you choose cascaded?* → quick deployment with a strong text LLM, modular upgrades, no paired data, debuggability.
14. *Why is evaluating controllable TTS hard?* → standard metrics ignore control fidelity; CV3-Eval.
15. *Why can WER exceed 1?* → insertions: (S+D+I)/N with I large (relevant to his CosyVoice zero-shot Hindi WER of 1.99).
16. *What is the "emotion problem"?* → HuBERT-style units keep what was said, not how.
17. *Your conclusion in one sentence?* → controllable LLM-based TTS isn't solved by scale. It needs coordination of representation, conditioning, decoding and evaluation.

---

## 15. Key facts cheat sheet

| Fact | Value |
|---|---|
| CosyVoice 1 tokenizer | VQ, 4,096 codes, middle of a supervised ASR encoder ([BACKGROUND] 50 Hz) |
| CosyVoice 2 tokenizer | FSQ in SenseVoice-Large, **25 tokens/s** ([BACKGROUND] 6,561 codes) |
| CosyVoice 2 LM | Qwen2.5-0.5B; no text encoder, no speaker embedding in the LM |
| CosyVoice 2 streaming | text:speech interleave **5:15**; 4 masks (non-causal, full-causal, Chunk-M, Chunk-2M) |
| CosyVoice 3 | ~1M h, 9 languages + dialects, LM 1.5B, DiffRO, CV3-Eval, 100+ styles |
| CosyVoice training data | zh 130k h, en 30k h, yue 5k h, ja 4.6k h, ko 2.2k h |
| WavTokenizer | 1 quantizer, 40 or 75 tokens/s at 24 kHz, 0.9 kbps |
| Spark-TTS | BiCodec: 50 tokens/s semantic + fixed-length global; Qwen2.5-0.5B; VoxBox ~100k h |
| Spark-TTS semantic features | wav2vec 2.0 layers 11, 14, 16 averaged |
| SpeechGPT | HuBERT units in vocab; SpeechInstruct; 3 stages (last with LoRA) |


---

## Appendix A: Code

None. This seminar is a literature review with no implementation. The code that applies these ideas is in the CosyVoice project; see its report's Appendix A.
