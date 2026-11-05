> Prep pack for **Seminar**, written 2026-11-05 from `interview_materials/seminar/INTERVIEW_PREP.md`. The vault copy is the one to study and edit.

# Interview prep: Speech in LLMs (M.Tech seminar, CS694, Spring 2026)

This pack is for defending the seminar as its sole author. The seminar is a literature review, so it has no code.
Two template sections are adapted. Section 2 is a concept map of the field instead of a code trace. Section 9 has
explain-it-back exercises instead of code changes.

## How to read the source tags

Every claim carries a tag. The tag says how safe it is to repeat.

| Tag | Meaning | Safe to quote? |
|---|---|---|
| **[R §x.y]** | Your report, `/data/jayavardhan/mtp/qwen/report/Seminar Report.docx`, section x.y | Yes. You wrote it. |
| **[N file:line]** | Your notes in `/data/jayavardhan/vaults/personal/Seminar/` | Yes, but some notes have errors (Section 5.3). |
| **[T line]** | Your talk script and slide text, `/data/jayavardhan/vaults/personal/SprintSeminar.md` | Yes, but some slides overstate (Section 5.3). |
| **[P]** | In the paper, but not in your materials. Written from memory. | Check the paper before you quote a number. |
| **[M]** | Reasoning in this pack. No paper says it. | Say "my reading is..." |

`SprintSeminar.md` is outside the folder you named. It holds your sprint-talk script and the text of the 25 final
slides, so this pack uses it.

## Read these first: three things that matter most

1. **Your own materials contradict each other in a few places. An interviewer who has read your slides or report can
   catch this.** Fix your story before the interview. The worst cases:
   - WavTokenizer: slide 9 says the tokens keep "rich semantic content" [T 211]. The report says WavTokenizer "does not
     have the ASR-like understanding capability of supervised semantic tokenizers" [R §3.4]. Your CosyVoice note says it
     "couldn't retain the semantic information" [N CosyVoice.md:238]. Use the report's version.
   - Emotion in CosyVoice tokens: the talk script says CosyVoice "made these systems emotion-aware" by taking tokens
     from between layers 6 and 7 [T 13]. CosyVoice's S3 tokens come from an ASR objective. They carry content and some
     prosody, but little speaker identity or emotion. That is why CosyVoice needs a speaker embedding and a prompt
     mel, and why CosyVoice 3 moved to a multi-task tokenizer [R §7.3]. The "middle layer keeps acoustics" idea is your
     intuition. The paper does not give that reason (Section 3, decision F3).
   - The same script says Spark-TTS used the ASR middle-layer trick [T 13]. It did not. BiCodec averages layers 11, 14
     and 16 of wav2vec 2.0, a self-supervised model [N SparkTTS.md:64] [R §7.2].
   - Slide 5 says more RVQ layers give "exponentially larger token sequences" [T 125]. The sequence grows linearly with
     the number of codebooks. Only the number of code combinations grows exponentially.
2. **The resume and the report do not quite match.** The report's title is "Speech in Large Language Models and
   Controlled Text-to-Speech" (CS694, May 2026). "A Literature Review" was the ChatGPT outline's title
   [N Seminar Outline ChatGPT.md:1]. The report also covers Spark-TTS and CosyVoice 3, which the resume omits, so
   expect questions on them. "Natively multimodal architectures" fits only SpeechGPT; CosyVoice is a TTS system.
   "Coherence" and "multilingual scalability" are on the resume, but the report treats them only lightly. Section 7
   gives you answers for both.
3. **The resume has no numbers, and that is safe. But the report's Acknowledgment thanks the department "to study and
   conduct meaningful experiments" [R Acknowledgment], and the report contains no experiments.** If asked, say plainly
   that the seminar was a reading project. Your hands-on speech work is the separate CosyVoice Indic LoRA R&D project,
   which has its own prep pack. Do not mix that project's results into seminar answers.

---

## 1. Pitches

### 60-second pitch

> My seminar studied how large language models learned to listen and speak. The old way was a cascade. Speech goes
> to an ASR model, the text goes to the LLM, and the reply goes to a TTS model. It works, but it is slow, and it throws
> away how things were said: tone, emotion, sarcasm.
>
> The newer way is to turn speech into discrete tokens, so a language model can treat speech like text. So the first
> half of my review was about tokenizers: vector quantization, residual VQ and finite scalar quantization, and
> WavTokenizer, which fits audio into one codebook at 40 or 75 tokens a second.
>
> Then I traced the systems built on these tokens. SpeechGPT put speech units into an LLM's vocabulary. CosyVoice
> made the tokens supervised by an ASR model, so they line up with text, and used flow matching to turn them back into
> audio. CosyVoice 2 swapped VQ for FSQ, used a pretrained Qwen backbone, and added streaming.
>
> The main lesson is that the tokenizer decides what the model can say, and how. It is a design choice, not a
> preprocessing detail.

### 3-minute deep explanation

> **The problem.** LLMs work on discrete tokens. Speech is a continuous waveform. At 24 kHz, one second is 24,000
> samples. You cannot feed that to a Transformer as it is. You need a compact, discrete representation.
>
> **Why not just use a cascade?** A cascade of ASR, then an LLM, then TTS does work. It has two problems. First,
> latency. Each stage waits for the one before it, and the system must also wait for the user to stop talking.
> Second, information loss. The transcript keeps the words but drops the prosody. A sarcastic "great, thanks" reads as
> positive. The TTS at the end also gets only plain text, so it cannot match the mood of the conversation.
>
> **Tokenization.** The basic tool is vector quantization. An encoder makes a vector per frame. We replace it with
> the nearest entry in a learned codebook and keep the index. VQ has a well-known failure called codebook collapse:
> only a few codes get used, so most of the vocabulary is wasted. Residual VQ stacks several codebooks, each one
> coding the error of the last. It gives high fidelity, but it produces several tokens per frame. An LLM then has to
> predict a grid of tokens, not a flat sequence. Finite scalar quantization drops the learned codebook. It projects to
> a few dimensions, bounds each one, and rounds it to a few levels. The code is just the combination of levels. There
> is nothing to collapse, so all codes get used.
>
> **Two kinds of tokens.** Acoustic tokens, from codecs like EnCodec or WavTokenizer, are trained to rebuild the
> waveform. They keep the voice and the recording quality, but they do not line up well with text. Semantic tokens,
> from HuBERT or from an ASR encoder, keep the content and drop much of the voice. WavTokenizer pushes acoustic tokens
> hard: one codebook of 4,096 codes at 75 tokens a second, which is 0.9 kbps.
>
> **SpeechGPT** was an early unified model. It added 1,000 HuBERT unit tokens to a LLaMA vocabulary. It trained in
> three stages: continue speech units, then align speech with text through instructions, then chain-of-modality
> tuning. In that last stage the model writes the text answer first and then speaks it. It proved that one LLM can take
> speech in and give speech out. But the units carry no emotion, and the text-first path keeps much of the cascade's
> latency.
>
> **CosyVoice** is a TTS system. Its key idea is the supervised semantic token. You take a multilingual ASR model and
> insert a VQ layer inside its encoder, then train with the ASR loss. Now each token must help recognize the words,
> so the tokens line up with text. An LLM predicts these tokens from text. A conditional flow-matching model turns them
> into a mel spectrogram, with the speaker's embedding and a prompt mel as conditions. A vocoder makes the waveform.
> The LLM handles content and rhythm. The flow model handles timbre and the recording environment.
>
> **CosyVoice 2** made four changes. It replaced VQ with FSQ, which raised codebook use. It dropped the separate text
> encoder and used a pretrained Qwen2.5-0.5B as the backbone. It removed the speaker embedding from the LM, because
> the embedding leaked language and style. And it added streaming. Text and speech tokens are interleaved 5 to 15, and
> the flow-matching model is trained with chunk-aware masks. So one model can run offline or streaming.
>
> **Trade-offs.** More compression means shorter sequences and cheaper models, but less fidelity. Modular systems are
> easier to build and debug, and each part can use its own data. End-to-end systems can be faster and keep
> paralinguistic cues, but only if the tokens keep those cues. A semantic-only token loses emotion just as surely as
> an ASR transcript does. That is the point I would stress most.

---

## 2. Concept map of the field (replaces the architecture and code trace)

### 2.1 The map

```
                         SPEECH IN LLMs: THE DESIGN SPACE
                         ================================

  WAVEFORM  24 kHz x 16 bit = 384 kbps, 24,000 samples/s
     |
     v
  [A] REPRESENTATION  (what the LM sees)                         [R ch.2-3]
     |
     +-- continuous features (log-mel, Whisper/HuBERT states) --> used for INPUT (understanding)
     |        e.g. Whisper encoder + adapter into an LLM            [M: not covered in report]
     |
     +-- discrete tokens -------------------------------------------------------------+
            |                                                                         |
            | HOW to discretise (quantiser)            WHAT the tokens keep (objective)|
            |  VQ   : nearest of K learned codes        acoustic : trained to rebuild   |
            |         risk: codebook collapse                     the waveform         |
            |         [R 3.1][N FSQ vs RVQ.md:1-7]                (EnCodec, DAC,       |
            |  RVQ  : N codebooks on residuals                     WavTokenizer)       |
            |         N tokens/frame [R 3.2]           semantic : content only         |
            |  FSQ  : bound + round each dim                      - self-supervised     |
            |         codes = prod(levels), no collapse             (HuBERT k-means:    |
            |         [R 3.3][N FSQ vs RVQ.md:32-42]                 SpeechGPT)         |
            |                                                     - supervised by ASR   |
            |                                                       (S3: CosyVoice 1/2) |
            |                                          hybrid   : both, split by stream |
            |                                                     (BiCodec: Spark-TTS;  |
            |                                                      SpeechTokenizer,     |
            |                                                      Mimi [P])            |
            +--------------------------------------------------------------------------+
     |
     v   token RATE sets sequence length: 25 Hz (S3 v2) ... 75 Hz (WavTokenizer) ... 600/s (EnCodec 6 kbps)
     |
  [B] MODELLING  (who reasons and who generates)
     |
     +-- cascade:  ASR --> text LLM --> TTS          latency adds up; prosody lost at ASR    [R 1.1]
     +-- unified LM, speech units in vocabulary:     SpeechGPT (chain of modality: text first) [R ch.4]
     +-- TTS LM, text --> speech tokens:             CosyVoice 1 (own LM, x-vector)            [R ch.5]
     |                                               CosyVoice 2 (Qwen2.5-0.5B, streaming)     [R ch.6]
     |                                               Spark-TTS (Qwen2.5-0.5B, attribute CoT)   [R 7.1]
     |                                               CosyVoice 3 (1.5B, 1M h, DiffRO)          [R 7.3]
     +-- full-duplex speech LM (listen while talking): Moshi, GPT-4o            [P: beyond your report]
     |
     v
  [C] RENDERING  (tokens --> waveform)
     |
     +-- unit vocoder (HiFi-GAN on units)            SpeechGPT
     +-- flow matching --> mel --> vocoder           CosyVoice 1; chunk-aware causal in CosyVoice 2
     +-- codec decoder, straight to waveform         WavTokenizer, BiCodec (Spark-TTS)
     |
     v
  [D] EVALUATION
        content: WER/CER by an ASR model | voice: speaker-embedding cosine (SIM)
        quality: MOS, UTMOS | control: emotion/rate accuracy (weak) | latency: first-packet  [R 8.3]

  CROSS-CUTTING TRADE-OFFS
    compression <-> fidelity      : fewer tokens/s = shorter sequences, less detail            [R 2.1, 3.4]
    content <-> voice <-> control : the triangle in [R 8.2]
    latency <-> context           : streaming chunk size; the "lead" example                   [R 6.5]
    modular <-> end-to-end        : debuggability and data reuse vs latency and paralinguistics [R 1.1, 8.1]
```

### 2.2 The lineage in one line each

```
VQ-VAE (2017) --> SoundStream/EnCodec RVQ (2021-22) --> WavTokenizer: 1 codebook, 40/75 tok/s (ICLR 2025)
HuBERT k-means units (2021) --> SpeechGPT: units in LLM vocabulary (2023)
ASR loss on the quantiser --> CosyVoice S3 tokens + LM + flow matching (2024)
   --> CosyVoice 2: FSQ, Qwen backbone, streaming (Dec 2024)
   --> CosyVoice 3: multi-task tokenizer, 1M h, 1.5B LM (2025)
BiCodec global + semantic tokens --> Spark-TTS: attribute control (2025)
```

Your slide 23 tells the same story [T 369-378]. Its theme line is "discrete tokenization is the bridge".

### 2.3 Trace one utterance end to end, through three designs

Take one user turn: about 10 seconds of speech, about 25 words. The reply is also about 10 seconds. Numbers marked
[M] are rough arithmetic, not measurements.

**(a) Cascade** [R §1.1, Fig 1.1]

```
user audio 10 s
  -> VAD waits for silence (endpointing)                    adds a fixed wait  [M]
  -> ASR -> text, about 25 words, about 33 BPE tokens      prosody dropped here
  -> text LLM -> reply text                                 must wait for the first clause before TTS
  -> TTS -> mel -> vocoder -> audio                         TTS sees plain text, no conversation mood
```

Information lost: tone, sarcasm, hesitation, laughter [R §1.1]. Errors are hard decisions: a misheard name stays
misheard.

**(b) SpeechGPT** [R §4.1-4.2] [P for the numbers]

```
user audio 10 s @16 kHz
  -> mHuBERT features, 50 Hz -> k-means, 1,000 clusters -> about 500 unit tokens
     (repeated units are usually merged, so fewer)          [P]
  -> wrapped as <sosp> <u123> <u7> ... <eosp>, inside one LLaMA-13B vocabulary   [P]
  -> chain of modality: model writes [text of the question] -> [text answer] -> [answer units]
  -> unit HiFi-GAN vocoder -> audio
```

One model, but the model still writes text before it speaks. So latency stays close to a cascade's [R §4.3].
HuBERT units hold little emotion [T 10], so the "sarcasm problem" remains.

**(c) CosyVoice 2, streaming TTS** [R ch.6] [N CosyVoice 2.md:233-313]

```
reply text arrives from a chat LLM, token by token
  -> BPE tokens (Chinese multi-character tokens split into characters)  [N CosyVoice 2.md:25]
  -> Qwen2.5-0.5B text-speech LM. Sequence:
       [S] prompt text, prompt speech tokens, then
       5 text tokens, 15 speech tokens, 5 text, 15 speech, ...
     (the model emits a filling token when it needs the next 5 text tokens)  [N CosyVoice 2.md:241-243]
  -> each 15 tokens @25 Hz = 0.6 s of speech
  -> look-ahead conv, upsample x2 to 50 Hz mel rate                      [N CosyVoice 2.md:264-269]
  -> chunk-aware causal flow matching (conditions: tokens, speaker embedding, prompt mel)
  -> mel chunk -> vocoder -> 0.6 s of audio plays while the next chunk is made
```

The full 10-second reply is 250 speech tokens. The same 10 seconds as EnCodec at 6 kbps would be 6,000 tokens
(Section 5.2).

### 2.4 Where things are in your materials

| Topic | Location |
|---|---|
| VQ drawbacks, FSQ size k^d | `Seminar/FSQ vs RVQ.md:1-7` |
| RVQ worked example | `Seminar/FSQ vs RVQ.md:13-30` |
| FSQ worked example (index 16) | `Seminar/FSQ vs RVQ.md:32-42` |
| RVQ/VQ long tail, "hard to predict" | `Seminar/FSQ vs RVQ.md:45-59` |
| CosyVoice 4 components | `Seminar/CosyVoice.md:100-106`, `:276-281` |
| S3 tokenizer, VQ after 6 layers, 4,096 codes | `Seminar/CosyVoice.md:176-179`, `:259-260` |
| Tokenizer WER 3.18 / 7.56 | `Seminar/CosyVoice.md:380` |
| Zero-shot ICL, cross-lingual | `Seminar/CosyVoice.md:135-147`, `:341-358` |
| Your CosyVoice talk script | `Seminar/CosyVoice.md:387-459` |
| CosyVoice 2 changes | `Seminar/CosyVoice 2.md:11-15`, `:184-192` |
| FSQ inside S3, 25 Hz | `Seminar/CosyVoice 2.md:37-43`, `:211-220` |
| Why the speaker embedding was removed | `Seminar/CosyVoice 2.md:226-231` |
| Streaming N:M, sequence layouts | `Seminar/CosyVoice 2.md:233-260` |
| Chunk masks, look-ahead conv | `Seminar/CosyVoice 2.md:262-313` |
| CosyVoice 2 limitations | `Seminar/CosyVoice 2.md:104-109`, `:320-323` |
| Spark-TTS BiCodec, wav2vec layers | `Seminar/SparkTTS.md:35-78` |
| Spark-TTS open doubts | `Seminar/SparkTTS.md:109-113` |
| Talk script (sprint seminar) | `SprintSeminar.md:1-21` |
| Final slide text, 25 slides | `SprintSeminar.md:23-405` |

---

## 3. Design decisions

There are two kinds. Part A covers your own choices as the author. Part B covers the design choices inside the
papers, which interviewers will ask you to justify.

### Part A: your decisions as the author

**A1. Scope: tokenizers, then systems, then control.**
- Alternatives: a survey of every speech-LLM paper; a survey of ASR-side models (understanding); a deep dive on
  one paper.
- Chosen: six papers on the generation side, ordered as one line of development [R §1.2].
- Why: "The papers are not treated as separate entities. They are organized as a development in a direction"
  [R §1.2]. The mentor helped connect them into one topic [R Acknowledgment]. **Documented.**
- Cost: speech understanding (Whisper-style encoders, Qwen-Audio, SALMONN) and full-duplex systems (Moshi, GPT-4o)
  are out of scope. Expect "what about Moshi?" (Q49).

**A2. Framing shift, from "literature review" to "controlled TTS".**
- The ChatGPT outline was a general review titled "...: A Literature Review" [N Seminar Outline ChatGPT.md:1]. The
  Gemini outline was framed around generative speech and tokenization [N Seminar Outline Gemini.md:1]. The final
  slides were titled "Multilingual and Controlled Text-to-Speech" [T 27]. The report became "Speech in LLMs and
  Controlled TTS".
- Why the shift happened is **not documented**. The record shows only the outlines and the final title. Do not
  invent a reason. If asked, give the reason in the content: by the end, most papers were TTS systems, and control
  was the common open problem [R §1.3, research question 4].
- Cost: the resume bullet says "natively multimodal", but the report is mostly about TTS.

**A3. Adding Spark-TTS and CosyVoice 3.**
- Neither outline lists them. The report adds them as "recent systems that move the field toward stronger
  controllability" [R Abstract, ch.7]. **Documented** (the final structure). The timing is not documented.
- Cost: they are not on the resume, but an interviewer reading the report will ask.

**A4. Four research questions as the spine** [R §1.3].
- (1) Which discrete representation suits LLM-based TTS? (2) How should content, identity and style be separated?
  (3) How can one system do offline and streaming? (4) What limits remain from cloning to controllable voice
  creation? **Documented.** Use these to structure any long answer.

**A5. A comparison table instead of a ranking** [R §8.1, Table 8.1].
- The report compares token design, renderer, control mechanism and main limitation. It does not rank systems on
  numbers. **Documented.** Why: the papers use different test sets, so a numeric ranking would mislead
  [M, consistent with R §8.3].

**A6. Concrete examples for a mixed audience.**
- The sarcasm example for paralinguistic loss [T 11] [R §1.1]. "project" vs "project" for context-dependent
  pronunciation [N CosyVoice.md:392]. "lead" for streaming context [R §6.5]. Hand-worked VQ/RVQ/FSQ examples
  [N FSQ vs RVQ.md:10-42]. **Documented.**

### Part B: the field's decisions, which you must defend

Each item gives the alternatives, the choice, the reason, and the cost.

**F1. Discrete tokens rather than continuous features.**
- Alternative: feed continuous encoder features to the LLM (common for understanding), or predict continuous mels.
- Choice in your papers: discrete tokens.
- Why: the LLM can then use its normal machinery: a vocabulary, softmax, cross-entropy, sampling [R §2.1].
- Cost: quantization loses information, and a separate decoder must restore detail [R §2.1].
- Status: **documented** for the reason. The continuous alternative is [M].

**F2. Semantic tokens for the LM, with acoustics left to a decoder.**
- Alternative: model acoustic codec tokens directly. VALL-E does this with EnCodec [P].
- Choice (CosyVoice, SpeechGPT): semantic tokens in the LM. CosyVoice puts acoustic detail in flow matching.
- Why: semantic tokens align with text, so the LM learns the text-to-token mapping easily. In CosyVoice, the LM
  models content and prosody, and the flow model models timbre and environment [N CosyVoice.md:92-94].
- Cost: two stages, and the tokens lose identity and emotion. CosyVoice 3 tries to fix this with multi-task
  supervision [R §7.3].
- Status: **documented.**

**F3. Supervised semantic tokens: VQ inside an ASR encoder (CosyVoice).**
- Alternative: unsupervised tokens, from k-means on HuBERT or from codecs.
- Choice: put a single 4,096-entry VQ after the first 6 encoder layers of an ASR model. Train with the ASR loss and
  update the codebook by EMA [N CosyVoice.md:176-179, :297].
- Why (paper): tokens trained with text supervision align with text. This gives better content consistency and
  speaker similarity than unsupervised tokens [R §5.1, §5.4]. Inserting VQ costs only a little ASR accuracy: WER
  3.18% on test-clean and 7.56% on test-other [N CosyVoice.md:380].
- Why the middle (**your interpretation**): early layers are acoustic and late layers are semantic, so the middle
  keeps both [R §3.1, §5.1] [N CosyVoice.md:411]. The paper places the VQ there, but as far as I know it does not
  argue the point this way. Present it as "my intuition, consistent with layer-probing results on speech encoders".
- Cost: the tokens depend on ASR data, so low-resource languages suffer [T 383]. An ASR objective rewards
  speaker-invariance, so identity must come from elsewhere.

**F4. Flow matching, not diffusion and not AR acoustic tokens (CosyVoice).**
- Alternatives: DDPM-style diffusion (TorToise TTS); an AR model over codec tokens; phoneme-duration flow models.
- Choice: optimal-transport conditional flow matching (OT-CFM) from tokens to mel, then a vocoder.
- Why: OT-CFM "can achieve better performance compared to DPMs with simpler gradients, easier training and faster
  generation" [N CosyVoice.md:132]. Unlike earlier flow-matching TTS, it needs no phoneme duration predictor, no
  phonemizer and no forced aligner [N CosyVoice.md:80-82].
- Cost: a second model. It also blocked streaming in v1, because it was non-causal. Spark-TTS criticizes this
  multi-stage design [N SparkTTS.md:30-33].
- Status: **documented.**

**F5. Speaker embedding in the LM (v1), then removed (v2).**
- v1: an x-vector from a pretrained voiceprint model (CAM++) conditions the LM [N CosyVoice.md:122, :314].
- v2 removed it: "such utterance-level vector contains not only speaker identity but also language and paralanguage
  information, which harms the prosody naturalness and cross-lingual capability" [N CosyVoice 2.md:226-227].
  **Documented.**
- Where identity comes from in v2: the prompt speech tokens in the LM (in-context learning), and the flow-matching
  stage, which still conditions on a speaker embedding and the prompt mel [P, verify in the CosyVoice 2 paper].
- Cost: the LM can no longer be told "this voice" by a vector alone. SFT mode needs speaker tags such as
  `SpeakerA<|endofprompt|>` [N CosyVoice 2.md:72-75].

**F6. Text encoder removed; a pretrained LLM as the backbone (v2).**
- v1 had its own text encoder and its own LM, about 300M parameters in total [N CosyVoice.md:430].
- v2 uses Qwen2.5-0.5B and drops the text encoder, because "Qwen2.5-0.5B is powerful enough to align the text and
  speech tokens" [N CosyVoice 2.md:230-231]. **Documented.**
- Cost: a larger LM per step, and dependence on that LLM's tokenizer. Hence the mask on one-to-many Chinese BPE
  tokens [N CosyVoice 2.md:25].

**F7. Raw text with BPE, no G2P front end (v2).**
- Alternative: phonemes via grapheme-to-phoneme conversion.
- Why: simpler preprocessing, and pronunciation is learned in context, end to end [N CosyVoice 2.md:206-208].
  **Documented.**
- Cost: rare words and polyphones depend on what the data covered. Languages with overlapping character sets
  degrade [N CosyVoice 2.md:107].

**F8. FSQ replaces VQ in the S3 tokenizer (v2).**
- How: project down to D dimensions (your note says D is usually 8), bound to [-K, K], round, then write the index in
  base (2K+1) [N CosyVoice 2.md:213-220].
- Why: better codebook use, and more speech information captured [N CosyVoice 2.md:14]. **Documented.**
- Number [P, verify]: as I recall, CosyVoice 2 reports VQ using about 23% of 4,096 codes and FSQ using 100% of 6,561
  codes (3 levels on 8 dimensions, 3^8 = 6,561).
- Cost: FSQ treats each dimension independently, so correlations must be learned by the projections around it
  [R §3.3].

**F9. 25 Hz tokens (v2), down from 50 Hz in v1.**
- The v2 tokenizer runs at 25 Hz [N CosyVoice 2.md:42]. The mel runs at 50 Hz, so tokens are upsampled by 2
  [N CosyVoice 2.md:264-267]. That v1 ran at 50 Hz is [P, verify].
- Why: half the LM steps per second of audio, which helps streaming [R §6.3]. **Documented** (the effect on the LM).
- Cost: coarser time resolution. Fine prosody must come from the flow stage.

**F10. Streaming by interleaving text and speech at N:M = 5:15 (v2).**
- Why: the LM can start speaking before the full text exists [R §6.2]. **Documented.**
- Why 5:15 in particular is **not documented** in your notes. My reading [M]: speech needs roughly 6 to 8 speech
  tokens per text token at 25 Hz. Emitting only 3 per text token keeps text ahead of speech, so the LM always has
  some look-ahead. Exercise 4 works this through.

**F11. One flow model with four masks, not separate streaming and offline models (v2).**
- Masks: non-causal, full-causal, chunk-M, chunk-2M, sampled at random per training example
  [N CosyVoice 2.md:59-66, :313].
- Why: one model serves every latency budget, and masks with more context act as implicit teachers for masks with
  less [R §6.4]. **Documented.**
- Cost: each mask gets only part of the training signal [M].

**F12. WavTokenizer: one quantizer, large codebook.**
- Alternative: an RVQ codec with 8 to 12 layers [T 169].
- Choice: one codebook of 4,096 codes, at 40 or 75 tokens/s for 24 kHz audio [R §3.4]. K-means initialization,
  random awakening of unused codes, a longer context window, attention in the decoder, multi-scale discriminators,
  and an inverse-Fourier (iSTFT) decoder [R §3.4].
- Why: more than one quantizer forces extra architecture in the downstream LM [R §3.2]. One flat stream looks like
  text. **Documented.**
- Cost: an acoustic codec with no text supervision. It suits reconstruction better than text alignment [R §3.4].

**F13. SpeechGPT: three training stages, with LoRA in the last.**
- Stages: modality-adaptation pre-training on speech units; cross-modal instruction fine-tuning; chain-of-modality
  fine-tuning with LoRA [R §4.2]. **Documented.**
- Why: "simply appending speech tokens to a text vocabulary is not sufficient" [R §4.2].
- Cost: the text-first response path, weak paralinguistics, and context length limits on multi-turn dialogue
  [R §4.3].

**F14. Spark-TTS: FSQ for global tokens, VQ for semantic tokens.**
- Why FSQ for the global tokens: to "mitigate the potential risk of training collapse associated with VQ"
  [N SparkTTS.md:76-77] [R §7.2]. **Documented.**
- Note: the same codec uses both quantizers, so it is not "FSQ beats VQ everywhere". FSQ is safest where the codes
  are few and fixed-length [M].

---

## 4. Hard problems, as STAR stories

These come only from your documents. A literature review has no bugs, so the stories are about understanding and
communication. Each result is the artefact you produced, not a measured outcome. Do not dress them up as more.

**Story 1: Six papers into one argument.**
- **S.** You had separate papers: SpeechGPT, WavTokenizer, CosyVoice 1 and 2, then Spark-TTS and CosyVoice 3. Early
  outlines from ChatGPT and Gemini listed them side by side [N Seminar Outline ChatGPT.md, Gemini.md].
- **T.** Write a report that reads as one development, not as six summaries.
- **A.** You chose tokenization as the thread: "discrete tokenization is the bridge" [T 378]. You ordered the
  papers by the problem each one fixes: SpeechGPT proves the idea, CosyVoice fixes the semantic gap, CosyVoice 2
  fixes latency, Spark-TTS adds explicit control [T 369-378] [R §8.1]. You wrote four research questions [R §1.3] and
  a comparison table [R Table 8.1]. Your mentor helped you connect the papers [R Acknowledgment].
- **R.** A 25-slide seminar [T 23-405] and the report. The conclusion, "not solved just by scale alone", follows
  from the comparison [R ch.9].

**Story 2: Why supervised tokens should help TTS at all.**
- **S.** CosyVoice cites earlier work showing that ASR-supervised quantizers beat k-means for ASR and translation. It
  also says their effect on TTS "remains unclear" [N CosyVoice.md:36-49].
- **T.** Understand the gap the paper fills, well enough to explain it.
- **A.** You worked through it in your notes. The old tokens group sounds by acoustic similarity. Supervised tokens
  must predict text, so they align with words. The risk is losing the acoustic detail TTS needs
  [N CosyVoice.md:51-64].
- **R.** The report's argument in §5.4: "the language model can only generate what the token space makes available".

**Story 3: Making streaming understandable.**
- **S.** The chunk-aware flow-matching masks and the look-ahead convolution are hard to picture from the paper's
  figure.
- **T.** Explain the latency-quality trade-off to a seminar audience.
- **A.** You worked the masks out by hand on a four-token example. You worked the look-ahead convolution with
  kernel P+1 and right padding P [N CosyVoice 2.md:271-311]. You found the "lead" example for missing context
  [R §6.5]. You also wrote the playback overlap into the talk: one chunk plays while the next is made [T 18].
- **R.** Section 6.4-6.5 of the report and slide 20.
- **Caveat.** Your chunk-M example for M=1 shows a sliding window: "A,B / A,B,C / B,C,D / C,D"
  [N CosyVoice 2.md:299-307]. The paper's chunk-M mask sees all past frames plus M future frames. Your own note
  heading says "M - future and all past tokens", so the example disagrees with the heading. Use the heading's
  version.

**Story 4: Questions you could not resolve (Spark-TTS).**
- **S.** Reading Spark-TTS, you wrote three doubts: the decoder size, how attribute control is measured, and why it
  fails on passages of 3-4 sentences but works on short ones [N SparkTTS.md:109-113].
- **T.** Report the system fairly without overclaiming.
- **A.** The report lists "optimized mainly for shorter sentences, latency concerns" as the main limitation
  [R Table 8.1]. Evaluation of control is named as an open problem [R §8.3].
- **R.** An honest limitation in the report. If asked, say the doubts stayed open. This is a good answer to "tell me
  about something you didn't understand".

**Story 5: Building FSQ and RVQ intuition with numbers.**
- **S.** The FSQ and RVQ definitions are abstract.
- **A.** You worked one 2-D vector, z = [1.2, -0.7], through two-stage RVQ (C3 + D1 = [1.3, -1]) and through 5-level
  FSQ (index 3·5 + 1 = 16) [N FSQ vs RVQ.md:10-42].
- **R.** Figures 3.1-3.3 in the report. Exercise 2 uses this example.

---

## 5. Numbers sheet

### 5.1 Resume numbers

**The resume bullets contain no numbers.** So nothing on the resume is unsupported in the numeric sense. Two word
choices need care.

| Resume phrase | What the report supports | Honest wording, if pressed |
|---|---|---|
| "natively multimodal architectures" | Only SpeechGPT is a unified speech-text LLM. CosyVoice 1/2 are TTS systems with an LLM inside. The report calls the direction "token-based speech-language models" [R Abstract]. | "...the shift from cascaded pipelines toward token-based speech-language models". Or keep it, and say SpeechGPT is the unified example and CosyVoice the generation side. |
| "Analyzed trade-offs... for latency, coherence, and multilingual scalability" | Latency: yes [R §1.1, ch.6]. Compression vs fidelity: yes [R §2.1, §3.2, §3.4]. Coherence: only implied (§8.2; the outline's "improve overall coherence" [N Seminar Outline ChatGPT.md, Comparative Analysis]). Multilingual: cross-lingual leakage and low-resource limits [R §2.3, §5.2, §8.5], but no modular-vs-E2E analysis of it. | "Analysed by comparing the papers' designs and reported results. No experiments of my own." Prepare Q44-Q45. |
| "Studied" / "Analyzed" | A reading project; no reproduction | Say so early. It builds trust. |

### 5.2 Field numbers you should know

Tag [R]/[N] numbers are in your materials. Tag [P] numbers are from memory; verify them before quoting. The
arithmetic is exact given the stated configuration.

**Bitrate formula.** bitrate = frames per second × codebooks per frame × log2(codebook size).

| Representation | Config | Tokens/s | Bitrate | 10 s of audio = tokens | Source |
|---|---|--:|--:|--:|---|
| Raw PCM | 24 kHz, 16-bit | 24,000 samples | 384 kbps | 240,000 samples | arithmetic |
| EnCodec 24 kHz | 75 Hz × 8 codebooks × 1,024 | 600 | 6.0 kbps | 6,000 | [P] |
| EnCodec 24 kHz | 75 Hz × 2 codebooks | 150 | 1.5 kbps | 1,500 | [P] |
| DAC 44.1 kHz | 86 Hz × 9 × 1,024 | about 775 | about 7.75 kbps | about 7,750 | [P] |
| Mimi (Moshi) | 12.5 Hz × 8 × 2,048 | 100 | 1.1 kbps | 1,000 (125 frames) | [P] |
| HuBERT / mHuBERT units | 50 Hz, k = 1,000 | 50 | about 0.5 kbps | 500 before merging repeats | [P] |
| WavTokenizer | 75 Hz × 1 × 4,096 | 75 | 0.9 kbps | 750 | [R §3.4] (75 tok/s, 0.9 kbps) |
| WavTokenizer | 40 Hz × 1 × 4,096 | 40 | 0.48 kbps | 400 | [R §3.4] (40 tok/s) |
| CosyVoice 1 S3 | 50 Hz × 4,096 | 50 | 0.6 kbps | 500 | 4,096: [N]; 50 Hz: [P] |
| CosyVoice 2 S3 (FSQ) | 25 Hz × 6,561 | 25 | about 0.32 kbps | 250 | 25 Hz: [N]; 6,561: [P] |
| BiCodec semantic | 50 Hz | 50 | about 0.65 kbps | 500 + fixed global tokens | 50 Hz: [N]; bitrate: [P] |
| Text | about 2.5 words/s | about 3-4 BPE tokens/s | under 0.1 kbps | about 33 | [M] |

Worked examples you should be able to do on a whiteboard:
- EnCodec 6 kbps: 75 × 8 × 10 bits = 6,000 bps.
- WavTokenizer: 75 × 12 bits = 900 bps. 24,000 / 75 = a hop of 320 samples. At 40 tok/s the hop is 600 samples.
- CosyVoice 2: log2(6,561) = 12.68 bits. 25 × 12.68 ≈ 317 bps.
- Compression: 384 kbps / 0.9 kbps ≈ 427 times smaller than raw 16-bit PCM.

**Sequence-length cost** [M arithmetic]. 6,000 EnCodec tokens vs 250 S3 v2 tokens is a 24× longer sequence. Attention
scores grow with the square, so about 576× the attention FLOPs. MLP FLOPs and the KV cache grow 24×. The real-time
condition is the sharpest point. An AR model must generate at least as many tokens per second as the token rate.
25 tok/s is easy at batch 1. 600 tok/s in one flat stream is not. That is why RVQ systems use tricks like VALL-E's AR
plus NAR split or delay patterns [R §3.2].

**KV cache example** [M, Qwen2.5-0.5B config from memory: 24 layers, 2 KV heads, head dim 64, bf16].
2 × 24 × 2 × 64 × 2 bytes = 12,288 bytes per token. 250 tokens is about 3 MB. 6,000 tokens is about 74 MB per
sequence.

**Paper numbers in your materials**

| Number | Meaning | Source |
|---|---|---|
| 4,096 | CosyVoice 1 S3 codebook size | [N CosyVoice.md:179] [R §3.1] |
| after 6 encoder layers | VQ position in the ESPnet Conformer ASR | [N CosyVoice.md:178] |
| 3.18% / 7.56% | WER of the VQ-inserted Conformer ASR on LibriTTS test-clean / test-other | [N CosyVoice.md:380] |
| about 300M | CosyVoice 1 text encoder + LM parameters | [N CosyVoice.md:374, :430] |
| about 170k h | CosyVoice 1 training data: zh 130k, en 30k, yue 5k, ja 4.6k, ko 2.2k | [R §5.4 table] [N CosyVoice.md:432] |
| 585 h, 2,456 speakers | LibriTTS (small-scale experiments) | [N CosyVoice.md:166] |
| 210,000 steps, 8 A800 | Training run in your notes. Which component is unclear, so don't quote it. | [N CosyVoice.md:420] |
| 25 Hz | CosyVoice 2 token rate | [N CosyVoice 2.md:42] [R §6.3] |
| 5 : 15 | Text : speech interleave ratio | [N CosyVoice 2.md:241] [R §6.2] |
| 0.5B | Qwen2.5-0.5B backbone (CosyVoice 2, Spark-TTS) | [R §6.2, §7.2] |
| about 211k h | CosyVoice 2 tokenizer data: zh 110,884 h + en 99,918 h | [N CosyVoice 2.md:85-87] |
| 1,000 | SpeechGPT unit vocabulary (k-means clusters) | [P] |
| LLaMA-13B | SpeechGPT backbone | [P] |
| 40 / 75 tok/s, 0.9 kbps | WavTokenizer | [R §3.4] [T 175] |
| 8-12 RVQ layers | typical prior codecs | [T 169] |
| 100k h | VoxBox (Spark-TTS): en 20k+, zh 35k+, other 45k | [N SparkTTS.md:20-26] [R §7.2] |
| 1M h, 9 languages, 1.5B LM | CosyVoice 3 | [R §7.3] |
| 100+ styles | CosyVoice 3 instruction styles | [R §7.4] |

**Latency anchors** [P]:
- Human turn-taking gaps average about 200 ms across languages (Stivers et al., PNAS 2009).
- OpenAI's GPT-4o launch post (May 2024): the older cascaded Voice Mode averaged 2.8 s (GPT-3.5) and 5.4 s (GPT-4).
  GPT-4o answers audio in as little as 232 ms, 320 ms on average. These are vendor numbers.
- Moshi reports a theoretical latency of 160 ms and about 200 ms in practice.

**Error propagation, rough maths** [M]. If ASR errors were independent at 10% WER, a 20-word question would come
through fully correct only 0.9^20 ≈ 12% of the time. At 5% WER, 0.95^20 ≈ 36%. Errors are not independent, and many
hit harmless words, so this is only a sketch of why one wrong entity can wreck the answer.

### 5.3 Claims in your own materials to fix or not repeat

| Claim | Where | Problem | Say instead |
|---|---|---|---|
| "only 200 codes used out of 1024" | [N FSQ vs RVQ.md:3] | No source given | "VQ often uses a small fraction of its codes. CosyVoice 2 reports low VQ use vs full FSQ use." |
| "token 45 → 12%, token 18 → 9%..." | [N FSQ vs RVQ.md:49-52] | Illustrative, no source | Call it an illustration |
| "more RVQ layers → exponentially larger token sequences" | [T 125] | Linear in N; combinations are K^N | "N codebooks give N tokens per frame, so N times longer flattened" |
| WavTokenizer has "rich semantic content" | [T 211] | Contradicts [R §3.4] and [N CosyVoice.md:238] | The paper claims semantic richness from probing; its training has no text supervision, so it aligns with text worse than S3 tokens |
| CosyVoice made tokens "emotion-aware" by taking layers 6-7 | [T 13] | Overstated; S3 tokens are mostly content | "S3 tokens carry content and some prosody; voice comes from the speaker embedding and prompt mel" |
| Spark-TTS used the ASR middle-layer trick | [T 13] | BiCodec uses wav2vec 2.0 layers 11/14/16 | As in [R §7.2] |
| "middle" placement and its reason | [R §3.1, §5.1] | The position is in the paper; the reason is your intuition | "My intuition is..." |
| SpeechGPT tokens from a "pre-trained speech codec (HuBERT-based)" | [T 249] | HuBERT is a self-supervised encoder, not a codec | "k-means units over mHuBERT features" |
| Chunk masks "process tokens in overlapping chunks" | [T 346] | They are attention masks over chunks, not overlapping windows | Describe the four masks |
| CosyVoice 2 "fine-grained speaker control: rate, pitch contour, energy" | [T 354] | The paper lists emotion, accent, role style and fine-grained tags; it says timbre can't be set by text instruction [N CosyVoice 2.md:108] | Use the paper's list |
| Chunk-M example as a sliding window | [N CosyVoice 2.md:299-307] | Paper: all past + M future | As in Story 3 |
| Report Figure 3.1 appears twice | [R §3.1 and §3.4]; the list of figures calls the second one Figure 8.1 | Numbering slip | Know it, in case someone has the PDF |
| Acknowledgment: "conduct meaningful experiments" | [R Acknowledgment] | The report has no experiments | Section 8, Q-D2 |

---

## 6. Prerequisite concepts

Each is 3-5 lines, plus the canonical source.

1. **Sampling, STFT and log-mel.** Audio is sampled at 16-48 kHz. The STFT cuts it into overlapping windows (about
   25 ms, hop 10-20 ms) and takes the spectrum of each. A mel filterbank warps frequency to match human pitch
   perception (80-128 bins), and a log compresses the range. ASR uses 100 frames/s. CosyVoice 2's mel is 50 Hz.
   *Source:* Davis & Mermelstein 1980; the Whisper paper for the modern ASR setup.
2. **Whisper / ASR encoders.** Log-mel goes in. A conv stem downsamples 2×, so 100 frames/s becomes 50 Hz. Then
   Transformer (or Conformer) layers follow. Many 50 Hz speech tokenizers inherit this rate. Lower layers are more
   acoustic, higher layers more phonetic and lexical. *Source:* Radford et al. 2023 (Whisper); Gulati et al. 2020
   (Conformer).
3. **Self-supervised speech units.** HuBERT masks parts of the input and predicts k-means cluster IDs of earlier
   features, then refines them. K-means on a middle layer gives "units" that track phones. wav2vec 2.0 uses a
   contrastive loss over quantized targets. *Source:* Hsu et al. 2021 (HuBERT); Baevski et al. 2020 (wav2vec 2.0).
4. **VQ-VAE.** The encoder output z_e is replaced by its nearest codebook vector e_k. The loss is reconstruction
   plus ||sg[z_e] − e||² (moves the code) plus β||z_e − sg[e]||² (commitment, β about 0.25). The gradient skips the
   argmin through the straight-through estimator: z_q = z_e + sg(e − z_e). An EMA update of the codebook can replace
   the codebook loss. *Source:* van den Oord et al. 2017; Razavi et al. 2019 (VQ-VAE-2).
5. **Codebook collapse.** Only the selected code gets updated. Codes that start far from the data are never chosen,
   so they never move, and the rich get richer. Detect it with usage counts or the perplexity of code usage. Fixes:
   k-means initialization, resetting dead codes to random encoder outputs, low-dimensional L2-normalized codes, EMA,
   or FSQ. *Source:* Dhariwal et al. 2020 (Jukebox, code resets); Yu et al. 2022 (ViT-VQGAN, factorized codes).
6. **RVQ.** Quantize, subtract, quantize the residual, repeat N times. You store N·K vectors but can express K^N
   combinations. Quantizer dropout during training gives one model with many bitrates. The cost is N tokens per
   frame. *Source:* Zeghidour et al. 2021 (SoundStream); Défossez et al. 2022 (EnCodec).
7. **FSQ.** Project to a few dimensions. Bound each with tanh, scale to L levels, round with the straight-through
   estimator. The implicit codebook is the product of the levels, for example [8,5,5,5] = 1,000. There is no codebook
   loss and no commitment loss, and usage is near 100%. *Source:* Mentzer et al. 2023, "Finite Scalar Quantization:
   VQ-VAE Made Simple".
8. **Neural audio codecs.** A conv encoder, a quantizer and a decoder, trained with reconstruction losses plus GAN
   discriminators (multi-scale, multi-period, STFT). Vocos-style decoders predict STFT frames and apply an inverse
   STFT, which is cheaper than upsampling convolutions. *Source:* EnCodec 2022; Kumar et al. 2023 (DAC); WavTokenizer
   (Ji et al., ICLR 2025).
9. **Acoustic vs semantic tokens.** Acoustic tokens rebuild the sound: voice, channel, room. Semantic tokens keep
   content and drop much of the voice. AudioLM made this split explicit: semantic tokens first, then acoustic tokens.
   *Source:* Borsos et al. 2022 (AudioLM); Zhang et al. 2023 (SpeechTokenizer).
10. **Autoregressive LMs, teacher forcing, BPE.** The model predicts the next token with cross-entropy. In training,
    the true previous tokens are the inputs (teacher forcing). BPE merges frequent byte pairs into subwords, so
    "watermelon" becomes "water" + "melon" [N CosyVoice 2.md:36]. *Source:* Sennrich et al. 2016 (BPE); Radford et
    al. 2019 (GPT-2).
11. **Vocabulary extension and LoRA.** Add new token rows to the embedding and output layers. They start untrained,
    so a pre-training stage is needed (SpeechGPT stage 1). LoRA trains a low-rank update BA beside frozen weights,
    with few parameters. *Source:* Hu et al. 2021 (LoRA); Zhang et al. 2023 (SpeechGPT).
12. **Speaker embeddings.** A fixed-size vector for a whole utterance, trained for speaker verification. Examples are
    x-vector (TDNN), ECAPA-TDNN and CAM++. It captures identity, but also language, channel and some style, which is
    the leakage CosyVoice 2 removed. *Source:* Snyder et al. 2018 (x-vectors); Desplanques et al. 2020 (ECAPA-TDNN).
13. **Diffusion vs flow matching.** Diffusion learns to undo noise over many steps. Flow matching regresses a velocity
    field along a chosen path from noise to data. With the optimal-transport path x_t = (1 − (1 − σ_min)t)·x0 + t·x1,
    the target is u = x1 − (1 − σ_min)·x0. The paths are nearly straight, so about 10 ODE steps suffice. Your notes
    say: diffusion removes noise step by step, flow matching takes a straight path [N CosyVoice.md:84-86]. *Source:*
    Lipman et al. 2023; Mehta et al. 2024 (Matcha-TTS, the decoder CosyVoice builds on [P]).
14. **Classifier-free guidance.** Drop the condition at random during training. At inference, push the output away
    from the unconditional prediction: v = (1 + w)·v_cond − w·v_uncond. This trades diversity for adherence.
    *Source:* Ho & Salimans 2022.
15. **Vocoders.** These turn a mel into a waveform. HiFi-GAN uses upsampling convolutions and GAN discriminators.
    iSTFT vocoders predict the spectrum and invert it. *Source:* Kong et al. 2020 (HiFi-GAN); Siuzdak 2023 (Vocos).
16. **Zero-shot TTS by in-context learning.** Put a few seconds of prompt speech tokens before the target, and the LM
    continues in that voice. VALL-E did this with EnCodec tokens and an AR plus NAR split. *Source:* Wang et al. 2023
    (VALL-E).
17. **Streaming and causal masks.** A causal mask lets frame t see only frames up to t. Chunked masks allow a fixed
    look-ahead. Latency then depends on chunk size plus compute per chunk. *Source:* CosyVoice 2 (Du et al. 2024).
18. **Metrics.** WER/CER of an ASR model run on the generated audio measures content. Speaker-embedding cosine (SIM)
    measures voice. MOS is human-rated, and UTMOS predicts it. Codecs also use PESQ, STOI and mel distance. None
    measures whether control was followed [R §8.3]. *Source:* Saeki et al. 2022 (UTMOS).
19. **Cascade vs end-to-end, and full duplex.** A cascade passes text between modules. End-to-end shares one model.
    Full duplex listens and speaks at once, with no explicit turn boundary. *Source:* Défossez et al. 2024 (Moshi);
    Zhang et al. 2023 (SpeechGPT).
20. **Inference basics that link to speech** (your Day 1 topics). Decode at batch 1 is memory-bound. Time per output
    token sets the maximum token rate the LM can sustain. The real-time factor is the generation time divided by the
    audio duration. Token rate therefore directly sets streaming feasibility.

---

## 7. Fifty likely questions, with crisp answers

[H] marks a hostile follow-up. Keep answers to 3-5 sentences out loud.

### Big picture

**Q1. Why tokenize speech at all? Why not give the LLM continuous features?**
For input, continuous features work well, and many speech-understanding models do exactly that. For output, an
LLM's machinery (softmax over a vocabulary, cross-entropy, sampling, in-context continuation) needs discrete
targets. Tokens let speech share the same sequence and loss as text [R §2.1]. The cost is quantization loss, which a
decoder must restore.

**Q2. What exactly does a cascade lose?**
Two things [R §1.1]. Latency: each stage waits for the previous one, and the system must also wait for end of
speech. Information: the transcript drops hesitation, sarcasm, laughter, accent and rhythm. The final TTS gets plain
text, so it cannot match the conversation's mood.

**Q3. [H] Modern cascades stream every stage. Isn't the latency argument weak?**
Partly, yes. With streaming ASR, LLM token streaming and sentence-level TTS, a cascade can get under a second. But
some costs stay: the endpointing wait, and the TTS needing a clause before it can phrase it well. OpenAI's numbers
show the size of the gap: their cascaded Voice Mode averaged 2.8-5.4 s, GPT-4o about 320 ms [P]. Latency is not the
deepest problem anyway. Paralinguistic loss at the text bottleneck is.

**Q4. How does error propagation show up?**
ASR makes hard decisions. A misheard entity becomes a wrong fact the LLM reasons over confidently, and nothing
downstream can recover it. Under a rough independence sketch, 10% WER leaves a 20-word query fully correct only
about 12% of the time [M]. Cascades soften this with N-best lists or confidence scores. End-to-end models avoid the
hard decision, but they are harder to debug.

**Q5. Does end-to-end automatically keep paralinguistics?**
No. That is the key point of my review. If the tokens are semantic-only, like HuBERT units in SpeechGPT, emotion is
lost at the tokenizer, just as in ASR [T 10-11] [R §4.3]. End-to-end is necessary, but the representation must also
keep those cues. That is why CosyVoice 3 trains its tokenizer on emotion and audio events too [R §7.3].

**Q6. Which of your papers are "natively multimodal"?**
Strictly, SpeechGPT: one LLM with speech and text in one vocabulary. CosyVoice 1, 2 and 3 and Spark-TTS are TTS
systems. They put an LLM inside the speech generator, but they do not listen. They are the generation half of a
native speech LLM. Qwen2.5-Omni's thinker-talker design joins the two halves [N CosyVoice 2.md:328].

### Tokenization

**Q7. Explain VQ and how it trains through the argmin.**
The encoder gives z. We pick the nearest of K codebook vectors and output its index. The argmin has no gradient, so
the straight-through estimator copies the decoder's gradient onto z. Two extra loss terms pull the code toward z and
make z commit to its code. Alternatively, the codebook is updated by EMA, as in CosyVoice's S3 tokenizer
[N CosyVoice.md:297].

**Q8. What is codebook collapse? How do you detect it and fix it?**
Only a few codes get used. Unused codes never win, so they never update [R §3.1]. You detect it by counting code
usage on a dataset, or by computing the perplexity of usage: log2 of the perplexity is the effective bits per token.
Fixes include k-means initialization and reviving dead codes (both in WavTokenizer [R §3.4]), low-dimensional
normalized codes, EMA, or switching to FSQ.

**Q9. Why does collapse matter for an LLM, beyond wasted capacity?**
The usage distribution gets a long tail [N FSQ vs RVQ.md:47-54]. Rare tokens get few training updates, so their
embeddings are poor and their predictions unreliable. The effective vocabulary is also smaller than the nominal one,
so the bitrate is lower than it looks.

**Q10. What is RVQ, and what does it cost an LLM?**
Each stage quantizes the previous stage's error. N codebooks of size K express K^N combinations with only N·K stored
vectors, so fidelity is high. But each frame becomes N tokens. The LLM must model a grid, which needs extra designs:
AR plus NAR splits, parallel heads, or local-global attention [R §3.2]. Flattening makes the sequence N times
longer, not exponentially longer.

**Q11. Do the EnCodec 6 kbps bitrate maths.**
24 kHz with a hop of 320 gives 75 frames/s. 8 codebooks of 1,024 entries give 10 bits each. 75 × 8 × 10 = 6,000 bps.
That is 600 tokens per second, or 6,000 tokens for 10 s [P config].

**Q12. Explain FSQ, and why it cannot collapse.**
Project to a few dimensions. Bound each with tanh to [-K, K] and round to an integer level. The token is the level
vector written in base (2K+1) [N CosyVoice 2.md:213-220]. There is no learned codebook, so no code can be starved.
Every grid point is reachable just by moving the encoder's output, which gives near-100% usage.

**Q13. CosyVoice 2's FSQ codebook size, and its bits per token?**
With 3 levels on 8 dimensions, 3^8 = 6,561 codes [P, verify]. log2(6,561) ≈ 12.7 bits. At 25 Hz that is about
317 bps, roughly half of v1's 600 bps (4,096 codes at 50 Hz).

**Q14. When is VQ better than FSQ?**
VQ places its codes where the data is, with full vectors, so it captures correlations between dimensions directly.
FSQ uses an axis-aligned grid, and the projections around it must learn the correlations [R §3.3]. For small
codebooks, a learned VQ can be more efficient. In practice, for LLM tokens, full usage tends to matter more
[R §3.3].

**Q15. [H] Your note says FSQ tokens are "easier to predict" because nearby vectors give nearby codes. But the LLM uses
a learned embedding table, so it never sees that structure. Isn't that wrong?**
Fair point. The LLM treats token IDs as arbitrary symbols, so it does not exploit the grid directly
[N FSQ vs RVQ.md:56-59]. What FSQ gives the LLM is a more uniform usage distribution, so every embedding gets trained,
and a more stable tokenizer. A small encoder change moves FSQ by one level on one axis. In VQ it can jump to an
unrelated code. Stability helps the tokenizer and its decoder, not the LLM's embedding lookup.

**Q16. Acoustic vs semantic tokens: what does each keep and lose?**
Acoustic tokens (EnCodec, WavTokenizer) are trained to rebuild the waveform. They keep the voice, channel and
prosody, but align with text poorly, so an LM finds content harder to control. Semantic tokens (HuBERT units, S3) keep
content and align with text, but drop much of the voice and emotion [R §2.1-2.2]. Hybrid designs keep both in separate
streams, like BiCodec's global plus semantic tokens [R §7.1].

**Q17. [H] Are HuBERT units semantic? They are trained with no text.**
They are self-supervised, which is why CosyVoice calls them unsupervised. "Semantic" here means phonetic content,
not meaning. They track phones better than voice, but they still carry some speaker and channel information, and
they are not aligned to text. Supervised S3 tokens make the alignment explicit, by training through an ASR loss
[R §2.2-2.3].

**Q18. How does the token rate affect sequence length and cost?**
Sequence length is token rate × duration × streams. Going from 6,000 tokens (EnCodec 6 kbps) to 250 (S3 at 25 Hz)
for 10 s cuts attention FLOPs by about 576× and the KV cache by 24× [M]. Above all, an AR model must generate as
fast as the token rate to be real-time. 25 tok/s is easy, and 600 tok/s in one stream is not.

**Q19. Why not go down to 5 Hz?**
Each token would have to cover 200 ms, which is longer than many phones. So either the codebook must grow
enormously, or content is lost. The decoder must then invent more detail, which hurts intelligibility and prosody.
Very low rates also make streaming chunks coarse. Current systems sit at about 12.5-25 Hz for LM tokens (Mimi
12.5 Hz [P]; CosyVoice 2 25 Hz).

**Q20. WavTokenizer: what is new, and how does one codebook get good quality?**
One quantizer with a larger codebook of 4,096 codes. K-means initialization and random awakening of unused codes.
A longer context window in training, attention in the decoder, multi-scale discriminators, and an iSTFT decoder
[R §3.4]. The result is 0.9 kbps at 75 tokens/s, with strong UTMOS, beating higher-bitrate DAC on some test sets
[R §3.4].

**Q21. [H] Your slides say WavTokenizer has rich semantics. Your report says it doesn't. Which is it?**
The paper claims semantic richness, measured with probing on audio classification tasks [P]. But its tokens are
trained only to rebuild audio, with no text supervision. So for TTS content alignment, they are weaker than ASR-
supervised tokens. The report's wording is my considered view [R §3.4]. The slide overstated it.

**Q22. How do you measure compression vs fidelity?**
Fix a token rate or bitrate, then measure reconstruction: UTMOS (predicted MOS), PESQ, STOI, mel distance, and
speaker similarity. For an LLM use, also measure downstream TTS WER and SIM. A codec that sounds good can still be
hard for an LM to model. UTMOS rewards clean-sounding speech, so it can hide content errors. Pair it with WER.

**Q23. What are hybrid tokenizers, and why have them?**
They split information by stream, so the LM sees content cleanly and identity separately. BiCodec has 50 Hz semantic
tokens and fixed-length global tokens for timbre [N SparkTTS.md:35-38]. SpeechTokenizer and Mimi distil a semantic
teacher into the first RVQ level [P]. They address the content-voice-control triangle [R §8.2].

### SpeechGPT

**Q24. Describe SpeechGPT's architecture and training.**
A discrete unit extractor (HuBERT-style), an LLM with speech units added to its vocabulary, and a unit vocoder
[R §4.1]. Three stages: modality-adaptation pre-training on unit sequences, cross-modal instruction fine-tuning on
SpeechInstruct, and chain-of-modality fine-tuning with LoRA [R §4.2]. Backbone: LLaMA-13B [P].

**Q25. Why a modality-adaptation stage first?**
The new unit embeddings start untrained, and the LLM has never seen unit statistics. Next-unit prediction on
unlabelled speech teaches the distribution of units before any alignment with text [R §4.2]. Without it, instruction
tuning would have to learn unit statistics and cross-modal mapping at once.

**Q26. [H] Chain of modality writes the text answer before speaking. Isn't that a cascade inside one model?**
Largely, yes, and the paper lists it as a limitation [R §4.3]. What it gains over a cascade is shared parameters and
knowledge across modalities, and one model to deploy. What it does not gain is latency, because text must come
first, or paralinguistics, because the units drop them. Later systems keep the text-first idea for reasoning quality
but interleave it with speech, as in CosyVoice 2's N:M streaming.

**Q27. [H] How was SpeechGPT evaluated? Was it convincing?**
Mostly qualitatively and by human ratings on cross-modal tasks, not on standard benchmarks with strong baselines [P,
verify]. So it is best read as a proof of concept: one LLM can take speech in and give speech out [T 259-271]. I would
not cite it for quality numbers.

**Q28. Why does SpeechGPT use LoRA in the last stage?**
To adapt the model to chain-of-modality behaviour without overwriting what the first two stages learned, and to keep
training cheap on a 13B model [R §4.2]. LoRA trains only low-rank updates. The design reason is my reading; the use of
LoRA is documented.

### CosyVoice 1

**Q29. Walk me through CosyVoice end to end.**
BPE text goes through a text encoder. An x-vector speaker embedding is added. The AR LM generates S3 tokens. A
conditional flow-matching model maps the tokens to a mel spectrogram, conditioned on the speaker embedding and the
prompt mel. A vocoder (HiFi-GAN type) makes the waveform [R §5.1] [N CosyVoice.md:276-281]. During training, only
the speech-token cross-entropy is used [N CosyVoice.md:128].

**Q30. What is a supervised semantic token, and why is it better for TTS?**
Take a multilingual ASR model. Insert VQ inside its encoder, and keep training with the ASR loss. Each token must now
help recognize the words, so the tokens align with text [R §5.1]. The LM then learns text-to-token mapping more
easily, and content consistency and speaker similarity improve over unsupervised tokens [R §5.4]. The LM can only
generate what the token space makes available.

**Q31. [H] Why put the VQ after the 6th layer? Did the paper justify that?**
The paper places it after the first 6 encoder layers [N CosyVoice.md:178]. My explanation is intuition: lower layers
are more acoustic, higher layers more linguistic, so a middle cut keeps content while some prosody survives
[R §3.1]. I did not see an ablation over positions in the paper, so I treat that reason as a hypothesis, not a
finding.

**Q32. If S3 tokens are mostly speaker-invariant, where does the voice come from?**
From two places. The LM gets the x-vector, plus prompt speech tokens for in-context continuation. The flow model gets
the speaker embedding and the prompt's mel spectrogram, which carry timbre and environment [N CosyVoice.md:144-147].
The LM models content and prosody. The flow model models timbre and acoustics [N CosyVoice.md:92-94].

**Q33. How does zero-shot and cross-lingual cloning work?**
Same language: prepend the prompt's text and speech tokens, and treat them as already generated, so the LM continues
in that style. Different language: drop the prompt's text and tokens, so the source language's rhythm doesn't leak
into the target [N CosyVoice.md:137-140] [R §5.2]. Controllability is partly about choosing what not to condition on
[R §5.2].

**Q34. Why flow matching rather than diffusion? Write the loss.**
OT-CFM learns a velocity field along a near-straight path from noise to data, so it needs few ODE steps and trains
with a simple regression loss. It is faster and simpler than diffusion [N CosyVoice.md:132]. The path is
x_t = (1 − (1 − σ)t)·x0 + t·x1, and the loss is ||v_θ(x_t, t, cond) − (x1 − (1 − σ)·x0)||², with the tokens, speaker
embedding and prompt mel as conditions. It also needs no phoneme duration predictor [N CosyVoice.md:80-82].

**Q35. Why go through mel and a vocoder, not straight to waveform?**
Mel at 50 Hz with 80 bins is far smaller than 24,000 samples a second, so the generative model works in a compact
space. Mature GAN vocoders then handle phase and fine detail. Spark-TTS shows the other route: a codec decoder that
outputs waveform directly and skips the flow stage [R §7.2]. The trade is fewer stages against the flexibility of a
strong acoustic model.

**Q36. Why compute loss only on speech tokens?**
The task is text-to-speech. The text is the condition, not something to predict. Loss on text tokens would spend
capacity on modelling the input [N CosyVoice.md:128]. The reason is my reading; the practice is documented.

**Q37. How does instruction-based generation work, and what is hard about it?**
CosyVoice-Instruct is fine-tuned on data with style descriptions before `<endofprompt>`, and with tags like
[laughter] and [breath], without the speaker embedding [N CosyVoice.md:150-160, :363] [R §5.3]. The hard part is that
"happy" maps to pitch, rhythm, loudness and brightness together. Control must also not hurt content or voice
[R §5.3].

### CosyVoice 2

**Q38. What changed from CosyVoice 1 to 2, and why?**
Four things [N CosyVoice 2.md:184-192]. FSQ replaced VQ, for better codebook use. The text encoder and LM speaker
embedding were removed, and a pretrained Qwen2.5-0.5B became the backbone, for better context understanding and no
leakage. Streaming and offline were unified through N:M interleaving and chunk-aware flow matching. Instructions
were extended to emotion, accent, role style and fine-grained tags, in the same model as zero-shot cloning.

**Q39. What leaked through the speaker embedding?**
The utterance-level vector carried language and paralinguistic information as well as identity. That hurt prosody
naturalness and cross-lingual cloning [N CosyVoice 2.md:226-227] [R §6.2]. A vector trained only to tell speakers
apart picks up anything that correlates with speaker in the data, including language and speaking style [M].

**Q40. Explain the N:M streaming layout. Why 5:15?**
The LM sees 5 text tokens, emits 15 speech tokens, then a filling token that means "give me more text", and repeats.
When the text ends, it emits [T] and finishes the speech [N CosyVoice 2.md:241-251]. The paper uses 5:15 [R §6.2]. My
reading of why [M]: speech needs about 6-8 tokens per text token at 25 Hz, so 3 per text token keeps the text ahead
of the speech. The LM never has to speak words it hasn't seen yet.

**Q41. Explain chunk-aware flow matching and its four masks.**
Non-causal: every frame sees everything (offline). Full-causal: only the past (lowest latency). Chunk-M: the past
plus M future frames. Chunk-2M: the past plus 2M future frames, near offline quality [N CosyVoice 2.md:59-66]. One mask
is sampled at random per training example, so one model serves all budgets. The larger-context masks act as
implicit teachers [R §6.4]. The flow's ODE steps are treated as one deep stacked network, which is why the masks must
hold across steps [R §6.4].

**Q42. What does the look-ahead convolution do?**
Before upsampling 25 Hz tokens to the 50 Hz mel rate, a 1-D conv with kernel P+1 and P frames of right padding lets
each position see P future tokens. Later causal layers then get a little future context [N CosyVoice 2.md:264-280].

**Q43. Where does first-packet latency come from in streaming TTS?**
The LM time for the first N text tokens and M speech tokens, plus the flow model and vocoder for the first chunk.
In a chat, add the upstream LLM's time to produce the first N text tokens [P: the paper has an explicit latency
formula of this shape; check it]. Each chunk is 15 tokens = 0.6 s of audio, and it plays while the next is made
[T 18].

### Trade-offs, evaluation, and the rest

**Q44. Modular vs end-to-end for multilingual scalability?**
Modular: each module trains on its own data. The text LLM uses huge text corpora, and you can plug in a per-language
ASR or TTS model. So coverage is the overlap of the three modules, and a new language can be added piecemeal.
End-to-end needs paired or in-domain speech for each language. Its tokenizer must cover the language: S3 tokens come
from an ASR model, so they inherit its language coverage [T 383] [R §8.5]. Leakage across languages appears too,
which is why CosyVoice drops cross-lingual prompts and CosyVoice 2 drops the speaker vector [R §2.3]. So for
low-resource languages, modular is often more practical today.

**Q45. [H] Your resume says you analysed "coherence". What do you mean, and where is it in the report?**
Two senses. Prosodic coherence: an end-to-end model keeps one context, so its speech style can follow the
conversation. A cascade's TTS reads each sentence fresh. Content coherence: speech-in models often reason worse than
text-in models, which is why SpeechGPT generates text first. The report treats this only briefly, through SpeechGPT's
text-first path and context limits [R §4.3] and the content-voice-control trade-off [R §8.2]. It is not a measured
analysis, and I would say so.

**Q46. [H] Content consistency is measured by an ASR model's WER, and S3 tokens come from an ASR model. Isn't that
circular?**
It is a real risk. Tokens shaped by ASR may favour speech that ASR systems find easy, so WER may flatter them. It is
partly reduced when the evaluation ASR differs from the tokenizer's ASR (the papers use Whisper or Paraformer for
evaluation [P]). It is still a reason to add human ratings and different ASR evaluators. The report flags metric
limits in §8.3.

**Q47. How would you evaluate controllable TTS properly?**
Content: WER/CER. Voice: SIM. Quality: MOS or UTMOS. Then control-specific checks: does an emotion classifier detect
the requested emotion, does the measured pitch or speaking rate match the instruction, did the [laughter] tag
produce laughter [R §8.3]. Add test sets that look like real use (noisy, code-switched, expressive), as CV3-Eval
starts to do [R §8.3].

**Q48. What safety issues come with this?**
Zero-shot cloning enables impersonation and fraud. Attribute-based voices avoid a real identity but still allow
persuasive synthetic speech. Mitigations are consent, watermarking, provenance tracking and detection [R §8.4] [T 386].

**Q49. [H] You didn't cover Moshi or GPT-4o. Isn't full duplex the real frontier?**
It is outside my scope, which was token design and TTS [R §1.2]. The link is direct, though. Moshi uses a 12.5 Hz codec
with a semantic first level, an "inner monologue" of text tokens, and two parallel audio streams [P]. Those are the
same ideas: low token rates, semantic plus acoustic levels, and text guidance. Full duplex adds the turn-taking
problem. If I extended the review, that would be the next chapter.

**Q50. If you built a voice assistant today, which design would you pick?**
For production with many languages, a streaming cascade with a paralinguistic side channel: streaming ASR plus
emotion tags into the LLM, then a CosyVoice 2-style streaming TTS that conditions on the dialogue. It is debuggable,
reuses the best text LLM, and covers languages. For a research product where expressiveness matters, a unified
model with semantic plus acoustic tokens and text guidance. My review supports this split: the representation, not
the label "end-to-end", decides what is kept [M, consistent with R ch.9].

**Bonus follow-ups to rehearse**
- "What would you do differently in the review?" Add the understanding side (Whisper-adapter LLMs) and full duplex
  (Moshi). Run at least one small experiment, such as codebook usage of a public codec, so one claim rests on my own
  measurement.
- "What is the open problem?" Paralinguistic tokens that keep emotion without hurting content. Measurable control.
  Low-resource languages. Better evaluation [R §8.5].
- "Spark-TTS in one breath?" BiCodec splits speech into 50 Hz semantic tokens (VQ, from wav2vec 2.0 layers 11/14/16)
  and fixed-length global tokens (FSQ, from ECAPA-TDNN plus learnable queries). Qwen2.5-0.5B predicts attributes,
  then global tokens, then semantic tokens, chain-of-thought style. The BiCodec decoder makes the waveform directly,
  with no flow stage [R §7.1-7.2].
- "CosyVoice 3 in one breath?" A multi-task tokenizer (ASR, language ID, emotion, audio events, speaker) on a MinMo
  base. 1M hours, 9 languages, a 1.5B LM, and DiffRO post-training with rewards from downstream models [R §7.3].

---

## 8. Questions where the honest answer is "I didn't do that"

**D1. "Did you implement or reproduce any of these models?"**
No. Say: "This was a literature seminar. I read the papers closely, worked the quantizers by hand, and compared the
designs. I didn't reproduce their numbers. My hands-on speech work is a separate R&D project, where I fine-tuned
CosyVoice 2 with LoRA. I'm happy to go into that separately." Then stop. Use that project's own prep pack for its
details.

**D2. "Your acknowledgment mentions experiments. What experiments?"**
Say: "That line is generic. The seminar report itself has no experiments. It's a review. The experimental work was
in my separate R&D project."

**D3. "Did you measure the latency of cascaded vs end-to-end systems?"**
No. Say: "No. My latency points are the papers' design arguments, plus a few published anchors such as OpenAI's
GPT-4o numbers and the 200 ms human turn gap. I'd present them as reported, not measured."

**D4. "Did you compare VQ and FSQ codebook usage yourself?"**
No. Say: "No. I rely on CosyVoice 2's reported comparison. A quick check I'd run is the code-usage histogram of an
open codec over LibriSpeech."

**D5. "How exactly is Spark-TTS's attribute control evaluated? Why does it fail on long passages?"**
You noted these as open doubts [N SparkTTS.md:109-113]. Say: "I didn't resolve that from the paper. I listed short
utterances as its main limitation. My guess for long passages is that the global tokens are fixed-length and the
training clips are short, but that is a guess."

**D6. "What about speech understanding: Whisper-based LLMs, Qwen-Audio, SALMONN?"**
Not covered. Say: "My scope was the generation side. For understanding, continuous encoder features through an
adapter are the main alternative to discrete tokens."

**D7. "Did you cover full-duplex systems?"**
No. Use Q49.

**D8. "Your resume says 'natively multimodal', but most of your papers are TTS systems."**
Say: "Fair. SpeechGPT is the unified model. The CosyVoice line is the generation half of such a system. A better
wording is 'token-based speech-language models', which is the report's own phrase."

**How to say "I didn't" well.** State it in the first sentence. Give the reason in one clause. Offer what you would
do, or what you did instead. Never pad it with numbers you didn't measure.

---

## 9. Five explain-it-back exercises

For each one: predict the answer out loud or on paper first. Then read the answer. Then explain the idea in 60
seconds, as if to an interviewer.

### Exercise 1: change the codec, predict the cost

A codec runs 24 kHz audio at a hop of 320 samples, with 8 RVQ codebooks of 1,024 entries. You replace it with one
codebook of 4,096 entries at a hop of 600. Predict the frames per second, the tokens per second, the bitrate, the
tokens for a 30-second reply, and the change in attention FLOPs.

<details><summary>Answer</summary>

- Old: 24,000 / 320 = 75 frames/s. 75 × 8 = 600 tokens/s. 600 × 10 bits = 6 kbps. 30 s gives 18,000 tokens.
- New: 24,000 / 600 = 40 frames/s. 40 tokens/s. 40 × 12 bits = 480 bps. 30 s gives 1,200 tokens.
- The sequence is 15× shorter, so attention FLOPs fall by about 225× and the KV cache by 15×.
- The real-time requirement drops from 600 to 40 tok/s.
- Cost: 12.5× lower bitrate, so less fidelity. This is WavTokenizer's 40 tok/s setting [R §3.4].

**Explain back:** why does the number of codebooks hurt the LM more than the codebook size does?
(Codebooks multiply the sequence length. Codebook size only adds rows to the embedding table and adds log2 bits.)
</details>

### Exercise 2: quantize by hand, then nudge the input

Use z = [1.2, -0.7] and the codebooks in your note [N FSQ vs RVQ.md:13-42].
(a) Run 2-stage RVQ and give the final error. (b) Run FSQ with levels {-2..2} and give the index.
(c) Change z to [1.2, -0.49]. Predict which outputs change, and by how much.

<details><summary>Answer</summary>

- (a) Stage 1 squared distances: C1 0.53, C2 1.53, C3 0.13, so pick C3 = [1, -1]. The residual is [0.2, 0.3]. Stage 2:
  D1 0.10, D2 0.40, D3 0.25, so pick D1 = [0.3, 0]. The sum is [1.3, -1]. The error is [-0.1, 0.3], squared 0.10.
  (After stage 1 alone it was 0.13.)
- (b) round(1.2) = 1, which is level index 3. round(-0.7) = -1, which is index 1. The token is 3·5 + 1 = 16.
- (c) FSQ: -0.49 rounds to 0, which is index 2. The token becomes 3·5 + 2 = 17, one step on one axis. RVQ: the stage 1
  distances become C1 0.28, C2 2.70, C3 0.30, so C1 now wins. The first code jumps to a different vector. The residual
  becomes [0.2, -0.49], and stage 2 picks D2. Both RVQ tokens change.
- Lesson: FSQ moves one digit at a time. VQ can jump to an unrelated code near a Voronoi boundary. This is the
  stability point in [N FSQ vs RVQ.md:56-59]. But remember Q15: the LLM's embedding table does not see FSQ's ordering.

**Explain back:** why can't FSQ collapse, and what does it give up in exchange?
</details>

### Exercise 3: move the VQ layer in the ASR encoder

CosyVoice places the VQ after layer 6 of the ASR encoder. Predict what happens if you move it to layer 1, or to the
last layer. Consider the tokenizer's own ASR WER, the TTS content WER, speaker similarity, and naturalness.

<details><summary>Answer</summary>

This is reasoning [M], not a result from the paper.
- **Layer 1:** the tokens are close to acoustics. The VQ must squeeze rich, speaker-dependent features into 4,096
  codes, so the tokenizer's ASR WER likely rises. The tokens align less with text, so the LM's job gets harder, and
  TTS content WER likely rises. They may keep more prosody.
- **Last layer:** the tokens are close to text. The tokenizer's ASR WER stays good, so the alignment is strong. But
  prosody and timing cues fade, so the speech can sound flat. More of the expressiveness burden moves to the flow
  model and the prompt.
- Speaker similarity mostly comes from the flow stage's conditions, so it is less affected in either case.
- The paper's measured fact: at its chosen position, VQ costs little ASR accuracy (3.18% / 7.56%)
  [N CosyVoice.md:380].

**Explain back:** why is "the middle keeps both" a hypothesis, not a finding, in your report?
</details>

### Exercise 4: change the streaming ratio

CosyVoice 2 interleaves N:M = 5:15. Assume speech needs about 7 speech tokens per text token at 25 Hz [M]. Predict
what happens at 5:5, and at 5:40.

<details><summary>Answer</summary>

After k blocks, the LM has seen 5k text tokens and emitted Mk speech tokens. That speech covers about Mk/7 text
tokens. Text look-ahead is therefore 5k − Mk/7.
- **5:15:** the look-ahead grows by 5 − 2.1 ≈ 2.9 text tokens per block. Text stays ahead, so the LM always knows
  some upcoming words. (Good for the "lead" problem [R §6.5].)
- **5:5:** the look-ahead grows by about 4.3 per block, giving more context. But each block yields only 0.2 s of
  audio, so the flow model processes tiny chunks, overhead per chunk rises, and playback is more likely to stall.
- **5:40:** 40/7 ≈ 5.7 text tokens' worth of speech per block, which is more than the 5 text tokens given. Speech
  runs ahead of the text. The LM would need to speak words it hasn't seen. Expect pauses, filler, or hallucinated
  words, unless it learns to emit the filling token early.
- Rule: M/N must stay below the speech-per-text rate, for every language served. Chinese has a different rate from
  English, which is another reason to keep a margin.

**Explain back:** why can text ahead of speech help prosody, but text far ahead not reduce latency?
</details>

### Exercise 5: a latency and error budget for a cascade

Use these illustrative stage costs: end-of-speech wait 500 ms; ASR finalize 100 ms; LLM time to first token 300 ms;
10 tokens for the first clause at 30 ms per token; TTS first chunk 200 ms. (a) Predict the time to first audio.
(b) Which two changes cut the most? (c) The ASR has 8% WER. Predict how often a 15-word question comes through with no
error, under independence.

<details><summary>Answer</summary>

- (a) 500 + 100 + 300 + 300 + 200 = 1,400 ms. That is seven times the ~200 ms human gap [P].
- (b) The end-of-speech wait (use a semantic endpointer, or start the LLM speculatively on partial transcripts). And
  the first-clause wait plus TTS (use streaming TTS that starts on a few text tokens, like CosyVoice 2's N = 5). An
  end-to-end model removes the ASR hand-off and can start before the turn ends.
- (c) 0.92^15 ≈ 0.29. So about 71% of such questions contain at least one ASR error. Most are harmless, but each error
  is a hard decision the LLM cannot undo.
- Paralinguistic note: none of these changes restores tone. Only the representation can [Q5].

**Explain back:** "End-to-end is faster" is true for which stages, and not for which?
</details>

---

## Appendix: one-page cheat sheet

- **Bitrate** = frames/s × codebooks × log2(K). EnCodec 6 kbps = 75 × 8 × 10. WavTokenizer 0.9 kbps = 75 × 12.
- **Sequence for 10 s:** EnCodec 6k = 6,000 · WavTokenizer = 750/400 · HuBERT = 500 · S3 v1 = 500 · S3 v2 = 250 ·
  text ≈ 33.
- **VQ:** nearest code, STE, commitment β ≈ 0.25, EMA; collapse leads to dead codes and a long tail.
- **RVQ:** N tokens/frame, K^N combinations, N·K stored vectors; quantizer dropout gives variable bitrate.
- **FSQ:** project, tanh-bound, round, base-(2K+1) index; 3^8 = 6,561 in CosyVoice 2 [P]; no codebook loss.
- **WavTokenizer:** 1 codebook, 4,096 codes, 40/75 tok/s, k-means init, random awakening, attention + iSTFT
  decoder, multi-scale discriminator.
- **SpeechGPT:** mHuBERT k-means 1,000 units [P], LLaMA-13B [P], 3 stages, chain of modality (text first), unit
  vocoder.
- **CosyVoice 1:** S3 = VQ (4,096) after 6 ASR encoder layers, ASR loss, EMA; text encoder + LM (about 300M) +
  x-vector; OT-CFM to mel to vocoder; LM = content and prosody, flow = timbre and environment.
- **CosyVoice 2:** FSQ, 25 Hz; Qwen2.5-0.5B; no text encoder; no LM speaker vector (leakage); 5:15 interleave; four
  masks; upsample ×2 to 50 Hz mel; mSFT speaker tags.
- **Cascade vs E2E:** latency (endpointing + stage hand-offs), hard-decision errors, paralinguistics. E2E only keeps
  tone if the tokens keep tone.
- **Anchors:** human gap ≈ 200 ms; GPT-4o 320 ms average vs 2.8-5.4 s for the old cascade [P, vendor].
