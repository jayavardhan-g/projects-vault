# Project report: LeNet-5 compression vs robustness (CS725, Foundations of Machine Learning)

> **For the LLM reading this.** This report describes the CS725 course project (instructor **Prof. Abir De**, Autumn
> 2025) that G Jayavardhan lists on resume 4. Sources:
> - the repository `github.com/harshith2302/cs725_Project`, cloned at `/data/jayavardhan/projects/interview_materials/lenet` (`main` @ `3b2584b`);
> - `fml_report.pdf`, the extension report (Phase 2–3; every resume number comes from it);
> - `legacy/docs/report.pdf`, the original 5-person reproduction report.
>
> Labels:
> - **[REPORT]**: from `fml_report.pdf`;
> - **[CODE]**: from the repository;
> - **[CHECKED]**: arithmetic verified by the report writer;
> - **[RECON]**: the report writer's reasoning.
>
> **Read §2 (authorship) before anything else.**

---

## 1. Summary

The project has two parts.

**Part 1, reproduction** (5-person team, Nov 2025):
- LeNet-5 on MNIST (tanh, average pooling, dense C3), best test accuracy 99.01%;
- side experiments: SGD with a high learning rate, Adam + ReLU, an MLP comparison, augmentation, training-set size.

**Part 2–3, extension: "Is compression that is free on clean MNIST also free under corruption and attack?"**
1. **Methodological audit.** The Part 1 numbers selected the epoch on the *test set*. A 55k / 5k / 10k split with
   validation-only selection, 5 seeds, and normalisation moved *inside* the model fixes this.
2. **Compression pipeline.**
   - **iterative magnitude pruning (IMP)**, 20% per round for 20 rounds, lottery-ticket style;
   - **knowledge distillation** from a double-width teacher;
   - **INT8 quantization-aware training (QAT)**.
   - Each stage is evaluated on clean accuracy, **MNIST-C** (15 corruptions), **PGD-L∞** adversarial accuracy, on-disk size and CPU latency.
3. **A control experiment.** At 3.75% of weights, layer-wise pruning costs **0.56 pp clean but 3.89 pp MNIST-C (7×)**.
   This is traced to a **starved first conv layer** (6 of 150 weights left). **Global** magnitude pruning at matched
   sparsity removes the penalty (−0.13 pp, p = 0.83) and beats layer-wise by **3.76 pp** on MNIST-C.

Every number is a mean ± std over 5 seeds, built from 260 JSON-recorded runs (plus 27 failed configurations kept).
Statistics: 59 Welch t-tests with Holm-Bonferroni correction within families. Compute: 44 GPU-hours on NVIDIA A40s
(SLURM); latency measured on CPU.

---

## 2. Authorship: decide how to describe your role (important)

- **Part 1 (reproduction)** was a **5-person CS725 team**: Ajay Rangoji, **Harshith Matta**, **G Jayavardhan**, Vineeth
  Babu Ch, Pujith Sai Kumar K. Its code is `github.com/pujith22/LeNet-5-on-MNIST`, kept as `legacy/` in this repo.
- **Part 2–3 (the extension, the source of every resume number):**
  - `fml_report.pdf` lists **Harshith Matta as the only author** (roll 25M0834);
  - every commit in `harshith2302/cs725_Project` is by `harshith2302`;
  - the pruning, distillation and quantisation scripts are on an **unpushed branch** (`feat/compression-robustness`); `main` has only the Phase 0 baseline and the evaluation suite.
- Resume 4 presents the compression results as his work. **He must decide, and be ready to say, exactly what he did.**
  Interviewers can ask for code, and the only public code is someone else's. An honest framing, if accurate:
  "CS725 team project; I worked on X; the compression study's write-up was led by a teammate."
- Without the branch, the pruning code below is the report's **Listing 1** (pseudocode) plus the repo's evaluation code.

---

## 3. Resume bullets vs evidence

| Bullet | Verdict | Evidence |
|---|---|---|
| "Compressed LeNet-5 to 3.75% of weights via iterative magnitude pruning, retaining 98.52% clean accuracy and 69.83% MNIST-C accuracy, selected only on a held-out validation split across 5 seeds" | **Supported** [REPORT Table 2]. | "Standard, 3.75% weights: 98.52 ± 0.14 clean, 69.83 ± 2.41 MNIST-C." Dense model: 99.08 / 73.72. Validation-only selection and 5 seeds are the stated protocol. |
| "Diagnosed a 7× larger corruption loss under layer-wise pruning (3.89 pp on MNIST-C) traced to a starved conv1 layer; global pruning at matched sparsity closed the gap to 0.13 pp (p = 0.83), a 3.76 pp gain" | **Supported** [REPORT Tables 4, 5]. | Layer-wise at 3.75%: Δclean −0.56, ΔMNIST-C −3.89 (p = 0.019), ratio 7.0×. Global at 3.52%: −0.18 / −0.13 (p = 0.83). Head to head, global is +3.76 pp MNIST-C (p = 0.021) and +0.38 pp clean (p = 0.002). Note the sparsities are 3.75% vs 3.52%, "matched" but not identical. |
| "Improved PGD robustness from 51.98% to 82.18% through iterative magnitude pruning" | **Needs its conditions** [REPORT Table 3]. | This is at **ε = 0.05** (not the standard ε = 0.3, where both are 0%), for the **13.8%-weights** model (not 3.75%). At ε = 0.1 the numbers are 2.33 → 14.74. The report itself says "it is not a defence" and that it may be a first-layer effect. Say "at small ε (0.05)". |
| "…while reducing model size 3.28× (245.6 KB to 75.0 KB) via INT8 quantization, at a cost of just 0.17 pp clean accuracy" | **Supported, with nuances.** | 245.6 / 75.0 = **3.27×** [CHECKED] (the report rounds to 3.28). INT8 was **QAT**, and it is **1.07× slower** on CPU at batch 1 (0.359 → 0.383 ms, p = 0.0005). The 0.17 pp INT8 accuracy drop is **not statistically significant** at n = 5 [REPORT §5]. Pruning alone does not shrink the file, because zeros are still stored. |

---

## 4. Background (teach first)

### 4.1 LeNet-5 (LeCun et al., 1998)
- Input: 32 × 32. Here 28 × 28 MNIST is zero-padded by 2.
- Layers:
  - **C1** conv 6@5×5 → **S2** avg-pool 2×2;
  - **C3** conv 16@5×5 → **S4** avg-pool 2×2;
  - **C5** conv 120@5×5 (output 1×1);
  - **F6** FC 84 → output FC 10.
- Activation: tanh (the original used a scaled tanh 1.7159·tanh(2x/3)).
- The original used sparse C3 connections and RBF output units. Here C3 is **dense** and the output is a **linear layer + softmax/cross-entropy**.
- Parameters [CHECKED]:
  - conv1: 150 weights + 6 biases;
  - conv3: 2,400 + 16;
  - conv5: 48,000 + 120;
  - fc6: 10,080 + 84;
  - fc7: 840 + 10;
  - total **61,706** (61,470 weights).

### 4.2 Iterative magnitude pruning and the Lottery Ticket Hypothesis (Frankle & Carbin, ICLR 2019)
- Train, remove the p% smallest-magnitude weights (by a mask), **rewind** the surviving weights to their original
  initial values, retrain, and repeat. The sparse subnetwork that trains to full accuracy is a "winning ticket".
- Later work (Frankle et al. 2020, "linear mode connectivity") rewinds to an **early training iteration** ("late
  rewinding") instead of initialisation. This works better for larger networks.
- **Layer-wise** pruning removes p% within each layer. **Global** pruning ranks all weights network-wide with one threshold.
- Standard LTH practice prunes the output layer at **half** the rate.

### 4.3 Knowledge distillation (Hinton et al., 2015)
Train the student on loss = α·T²·KL(softmax(z_t/T) ‖ softmax(z_s/T)) + (1−α)·CE(y, z_s). A higher temperature T
exposes "dark knowledge": the relative probabilities of the wrong classes.

### 4.4 INT8 quantization
- Weights and activations are stored as 8-bit integers with a scale (and zero point): x ≈ s·(q − z).
- **Post-training quantization (PTQ)** calibrates scales after training. **Quantization-aware training (QAT)** inserts
  fake-quant operations during training so the network adapts to rounding.
- PyTorch eager mode (`torch.ao.quantization`) uses QuantStub/DeQuantStub, fuses conv + activation, applies
  `prepare_qat`, trains, then `convert`. The fbgemm backend uses per-channel weight quantization on x86.

### 4.5 MNIST-C (Mu & Gilmer, 2019)
15 corruptions of the MNIST test set (10,000 images each): shot noise, impulse noise, glass blur, motion blur, shear,
scale, rotate, brightness, translate, stripe, fog, spatter, dotted line, zigzag, canny edges. The reported number is
the mean accuracy over the 15.

### 4.6 Adversarial attacks
- **FGSM:** x' = clip(x + ε·sign(∇ₓL)).
- **PGD-L∞ (Madry et al., 2018):** start at a random point in the ε-ball, then take k steps of x ← Π_{ε-ball}(x + α·sign(∇ₓL)), clipping to [0, 1] each step. The MNIST standard is ε = 0.3, α = 0.01, 40 steps.
- **Adversarial training:** train on PGD examples (min-max).
- **Gradient masking:** a defence that only breaks the gradient looks robust but isn't. Checks:
  - accuracy decreases monotonically in ε;
  - PGD is at least as strong as FGSM;
  - more steps and restarts make the attack no weaker;
  - black-box transfer attacks are *weaker* than white-box.

### 4.7 Statistics
- **Welch's t-test** compares two means without assuming equal variances.
- **Holm-Bonferroni:** sort the p-values, compare the i-th smallest against α/(m − i + 1), and stop at the first failure. It controls the family-wise error rate.
- With n = 5 seeds the power is low: "not significant" means "not resolvable", not "no effect".

---

## 5. Phase 1: reproduction audit [REPORT §2, CODE]

1. **Test-set model selection** (the leak). Part 1 picked its reporting epoch by test accuracy, which inflated results by about 0.05–0.08 pp.
   - The fix: a **55,000 / 5,000 / 10,000** split with a fixed `SPLIT_SEED = 1234` (`src/data.py:23-70`). Select on validation, touch the test set once.

   | 20 epochs, 5 seeds | standard | augmented |
   |---|---|---|
   | correct (best epoch on val) | 99.08 ± 0.09 | 99.16 ± 0.04 |
   | leaky (best epoch on test) | 99.13 ± 0.07 | 99.24 ± 0.03 |

   - The augmentation gain is **budget-dependent**: significant at 10 epochs (+0.20 pp, p = 0.003) but not at 20 epochs under honest selection (+0.08 pp, p = 0.115). Leaky selection *manufactured* significance (p = 0.023).
   - The legacy "99.01%" was epoch 9 picked on test. No seed was set in the legacy path.
2. **Normalisation inside the model.** `InputNormalize` pads 28 → 32 and standardises with mean 0.1307 and std 0.3081 as **layer 0** (`src/models.py:22-40`).
   - Why: PGD's ε = 0.3 is defined in **pixel space**. If normalisation lived in the dataloader, an ε = 0.3 attack on standardised tensors would be ε ≈ 0.3 × 0.3081 ≈ 0.092 in pixels, and every robustness number would be wrong.
   - `assert_pixel_space_model` refuses to attack a model without `InputNormalize` (`src/attacks.py`).
3. **Every run writes a JSON record:** config hash, seed, git SHA, machine spec, metrics, wall time. Tables are generated by scripts and never hand-edited (`scripts/make_phase0_table.py`).

---

## 6. Phase 2: the compression pipeline [REPORT §3]

**Setup:**
- Training: tanh LeNet-5, SGD (lr 0.01, momentum 0.9), batch 64, **20 epochs**, the same budget for every condition.
- **Pruning:** IMP, removing **20% of the surviving weights per round for 20 rounds**, layer-wise by default, rewind to init. Variants compared: rewind-to-init, late rewinding, fine-tune without reset.
- **Distillation:** a double-width LeNet teacher (`LeNet5(width=2)`). T and α were chosen on validation from a 3 × 3 grid.
- **Quantization:** eager-mode INT8 QAT, really converted. Size measured on disk, latency on CPU at batch 1, single-threaded, median of 1,000 runs (`src/evaluate.py:48-115`).

**Pruning pseudocode** [REPORT Listing 1]:
```python
def prune_step(model, scope, rate=0.20):
    if scope == "layerwise":
        for layer in prunable_layers(model):
            threshold = magnitude_quantile(layer.weight, rate)      # per-layer threshold
            layer.mask &= (layer.weight.abs() >= threshold)
    elif scope == "global":
        all_weights = concat_magnitudes(model)                     # one threshold for all layers
        threshold = magnitude_quantile(all_weights, rate)
        for layer in prunable_layers(model):
            layer.mask &= (layer.weight.abs() >= threshold)
    apply_masks(model)
```
- Prunable modules are conv1, conv3, conv5, fc6, fc7 (`src/models.py:81-90`).
- [RECON] The real loop masks only surviving weights (the quantile is over non-zero weights), rewinds survivors to their initial values, retrains 20 epochs, and repeats.

**Where 3.75%, 13.8% and 1.30% come from** [CHECKED]: layer-wise, 20% per round, with **fc7 pruned at half rate (10% per round)**:
- **round 9:** non-output layers keep 0.8⁹ = 13.4%, fc7 keeps 0.9⁹ = 38.7%, so (60,630 × 0.134 + 840 × 0.387) / 61,470 = **13.8%**;
- **round 15:** 0.8¹⁵ = 3.52%, 0.9¹⁵ = 20.6%, so **3.75%**;
- **round 20:** 0.8²⁰ = 1.15%, 0.9²⁰ = 12.2%, so **1.30%**.

Global pruning's 3.52% and 1.15% are exactly 0.8¹⁵ and 0.8²⁰. Under layer-wise pruning conv1 keeps 150 × 0.8ⁿ: about **21 / 6 / 2** weights at rounds 9 / 15 / 20, which matches Table 4.

**Headline table** [REPORT Table 2] (Robust = worst-case PGD at ε = 0.3; size = on-disk KB):

| Configuration | Clean | MNIST-C | Robust | Size | Latency (ms) |
|---|---|---|---|---|---|
| Standard, dense | 99.08 ± 0.09 | 73.72 ± 0.92 | 0.00 | 245.6 | 0.366 |
| **Standard, 3.75% weights** | **98.52 ± 0.14** | **69.83 ± 2.41** | 0.00 | 245.6 | 0.365 |
| Standard, 1.30% | 95.65 ± 0.79 | 64.03 ± 1.42 | 0.00 | 245.6 | 0.367 |
| Adversarial, dense | 98.78 ± 0.09 | 78.48 ± 0.79 | 90.24 ± 0.32 | 245.6 | 0.357 |
| Adversarial, 3.75% | 96.60 ± 0.79 | 69.99 ± 1.56 | 83.55 ± 1.75 | 245.6 | 0.356 |
| **IMP + KD + INT8** | 98.65 ± 0.17 | 71.10 ± 1.47 | — | **75.0** | 0.396 |
| Global-pruned, 3.52% | 98.90 ± 0.07 | 73.59 ± 0.98 | 0.00 | 245.6 | — |
| Global-pruned, 1.15% | 98.20 ± 0.16 | 70.42 ± 1.75 | 0.00 | 245.6 | — |

**Findings:**
- **Lottery ticket reproduced on clean accuracy.** It is flat down to 13.8% of weights. MNIST-C falls faster at every level: −0.56 clean vs −3.89 MNIST-C at 3.75%. Rewind-to-init ≈ late rewinding. Fine-tune-without-reset is 6.73 pp worse at 1.30%, but that is **not significant** (p = 0.062 raw, 0.174 Holm) at n = 5.
- **Distillation** lifts the pruned student 98.52 → 98.64 clean and 69.83 → 71.49 MNIST-C, at the same size. Distillation alone is worth +0.17 pp clean (p = 0.011). The T/α grid spans only 0.16 pp, so the effect comes from distilling at all, not from the particular T.
- **INT8 QAT:** 245.6 → 75.0 KB (**3.27×**, measured on disk) for −0.17 pp (not significant). It is **1.07× slower** (0.359 → 0.383 ms, p = 0.0005), because on a model this small quantize/dequantize overhead exceeds the gain from integer arithmetic.
  - [CHECKED] 245.6 KB is the fp32 `state_dict` file (61,706 × 4 B = 241 KiB + container overhead).
  - 75.0 KB is about 60 KB of int8 weights + scales/zero points + fp32 biases + container.
  - Unstructured pruning doesn't shrink the file, because the zeros are still stored. Size gains need sparse formats or structured pruning.
- **Pruning increases adversarial robustness at small ε** [REPORT Table 3]:

  | Model | ε = 0.05 | ε = 0.1 | ε = 0.15 | ε = 0.3 |
  |---|---|---|---|---|
  | standard, dense | 51.98 ± 9.51 | 2.33 ± 0.90 | 0.01 | 0.00 |
  | standard, 13.8% | **82.18 ± 3.92** | 14.74 ± 5.02 | 0.13 | 0.00 |
  | adversarially trained, dense | 98.09 | 97.29 | 96.19 | 90.55 |

  - +30 pp at ε = 0.05 (p = 0.0010) and +12 pp at ε = 0.1 (p = 0.0046).
  - **Gradient masking was ruled out**: black-box transfer from the dense model is *weaker* than white-box PGD on the pruned model (68.15% vs 14.10% robust accuracy at ε = 0.1).
  - Both models are at 0% by ε = 0.2, so this is **not a defence**. It is probably the other face of the first-layer damage.
- **First-layer filters are destroyed** [REPORT Table 4]:

  | weights left | conv1 weights alive | filters entirely empty |
  |---|---|---|
  | 100% | 150 / 150 | 0 / 6 |
  | 13.8% | 21 / 150 | 0 / 6 |
  | 3.75% | **6 / 150** | **3 / 6** |
  | 1.30% | 2 / 150 | 4 / 6 |

  With a uniform per-layer rate, the 150-weight conv1 is starved long before the 48,000-weight conv5. So "compressed" and "first layer destroyed" become the **same condition**: a **confound**.

---

## 7. Phase 3: the schedule control (the key result) [REPORT §4]

- **Two explanations for the corruption penalty:**
  - (a) sparsity itself hurts robustness;
  - (b) the first layer was destroyed by the layer-wise schedule.
- **Control:** hold everything identical (protocol, rewinding, seeds, 20-epoch budget) and change only **layer-wise → global ranking**.
  - Global pruning keeps conv1 at **26.7%** of its weights at round 20, against 1.3% under layer-wise. Small layers whose weights have large magnitude survive.

| Weights | Schedule | Δ clean | Δ MNIST-C | Ratio |
|---|---|---|---|---|
| 3.75% | layer-wise | −0.56 | **−3.89** (p = 0.019) | 7.0× |
| 3.52% | global | −0.18 | **−0.13** (p = 0.83) | 0.7× |
| 1.30% | layer-wise | −3.43 | −9.70 | 2.8× |
| 1.15% | global | −0.88 | −3.30 | 3.8× |

- At about 3.5% of weights the penalty **disappears** under global pruning. Head to head, global is **+3.76 pp MNIST-C (p = 0.021)** and +0.38 pp clean (p = 0.002), so it is strictly better on both.
- Near 1%, global still loses 3.30 pp of MNIST-C. **Sparsity does eventually cost robustness, but much later and much less** than layer-wise pruning implied.
- **The effect depends on the training condition** (switching to global at about 13.5% of weights):

  | condition | effect on MNIST-C |
  |---|---|
  | standard | +2.15 pp (p = 0.010) |
  | augmented | +0.34 (p = 0.74) |
  | adversarial | +3.20 (p = 0.003) |

  - Augmentation already trains tolerance to spatial perturbations, so it has little to gain.
  - The adversarial arm becomes more stable under global pruning: robust accuracy 88.51 ± 1.17 vs 84.72 ± 5.86.
- **Conclusion:** "the standard layer-wise recipe costs corruption robustness that the sparsity itself does not require."

---

## 8. Two measurement corrections [REPORT §6]

1. **PGD-40 with 1 restart overstated robustness.** Adversarially trained models scored 93.40 ± 0.24 under it, but 90.63 ± 0.53 under PGD-100 with 5 restarts (p = 0.0001), with the same sign on all 5 seeds. Accuracy saturates by PGD-200 with 1 restart but keeps falling as restarts are added, so **restarts, not steps**, were the binding constraint. Four gradient-masking checks passed.
2. **Adversarial training of sparse networks needs an ε ramp.** At full ε = 0.3 from epoch 1, every pruned ticket collapsed to a constant prediction: loss pinned at ln 10, accuracy 11.35% (the most frequent class) on all seeds. A ramp fixes it (3.75% weights: 11.2 → 93.2 robust accuracy). Reason: when every example is adversarial to an untrained sparse net, there is no useful gradient. Reporting the collapse would have produced the false headline "pruned networks can't be adversarially trained".

---

## 9. Evaluation code on `main` [CODE]
- `src/models.py`: `InputNormalize` (pad + standardise as a layer); `LeNet5(activation, width)` (`width=2` is the teacher); `MLP`; `prunable_modules`.
- `src/attacks.py`:
  - `fgsm`, `pgd` (random start, sign steps, projection to the ε-ball then to [0, 1], model in eval mode);
  - `adversarial_accuracy` **measures** the max L∞ of the perturbations and the pixel range, and raises if the budget is exceeded, so projection bugs can't inflate robustness;
  - defaults ε = 0.3, α = 0.01, 40 steps.
- `src/data.py`: the fixed-seed 55k/5k split; `get_mnist_c_loaders` for the 15 corruptions.
- `src/evaluate.py`:
  - `clean_accuracy`, `mnist_c_accuracy` (per corruption + mean);
  - `latency` (single-thread, warm-up 100, median of 1,000);
  - `model_size` (real `torch.save` file size in KB, plus theoretical sizes).
- `scripts/check_attacks.py`, `check_mnistc.py`, `check_port.py`: sanity checks (attack budget, MNIST-C loader against the download, Part 1 → Part 2 port).
- `scripts/run_phase0.py`, `make_phase0_table.py`: Phase 0 runs and tables (CPU, about 3.5 h).

---

## 10. Limitations [REPORT §7]
- The pipeline and ε-sweep experiments still use **layer-wise** tickets, so the small-ε robustness gain may be a first-layer effect rather than a sparsity effect. This is the most useful follow-up.
- **Global pruning has its own pathology:** it empties a later conv layer to 0.3% while keeping conv1 at 26.7%. The two schedules bracket the question. A sensitivity-scaled per-layer rate would be the principled version.
- Tickets come from standard training only. One architecture, one dataset. n = 5 is underpowered for small effects.

---

## 11. Likely interview questions
1. *What is the lottery ticket hypothesis, and did you reproduce it?* → §4.2; flat clean accuracy to 13.8%; rewind ≈ late rewind; the fine-tune gap was not significant.
2. *Why is layer-wise pruning bad here?* → conv1 has 150 weights; a uniform rate starves it (6 left at 3.75%).
3. *How did you prove it was the schedule and not the sparsity?* → the global-pruning control at matched sparsity, everything else fixed (§7).
4. *Why does global pruning keep conv1?* → [RECON] PyTorch's default init scales weights as about 1/√fan_in. conv1 has fan-in 25 (bound ≈ 0.2), conv5 has 400 (≈ 0.05) and fc6 has 120. So conv1's weights are larger in magnitude, and one global threshold spares them.
5. *Why put normalisation inside the model?* → the ε-scale argument (§5).
6. *Explain PGD, and how you made sure the attack wasn't broken.* → §4.6, §8.1; perturbation-budget assertions.
7. *Why is INT8 slower?* → quantize/dequantize overhead on a tiny model; integer GEMM gains are negligible at this size.
8. *Why doesn't pruning reduce file size?* → unstructured zeros are stored; you need a sparse format or structured pruning.
9. *Explain 51.98 → 82.18.* → ε = 0.05, the 13.8% model, not a defence (0% by ε = 0.2), gradient masking ruled out by transfer.
10. *Why Welch and Holm-Bonferroni?* → unequal variances; 59 comparisons → family-wise error control.
11. *What did the test-set leak cost?* → +0.05–0.08 pp, and it manufactured significance for augmentation.
12. *Why did adversarial training of pruned nets collapse?* → no useful gradient at full ε from the start; an ε ramp fixes it.
13. *What exactly did you do in this project?* → **answer honestly** (§2).
14. *What is distillation's T, and why did the grid find nothing?* → softens targets; the 3 × 3 grid spans 0.16 pp, below seed noise.

---

## 12. Numbers cheat sheet
| Number | Meaning |
|---|---|
| 61,706 | LeNet-5 params (dense C3, tanh) |
| 99.08 ± 0.09 | honest dense clean accuracy (20 ep, val selection); legacy reported 99.01 (leaky, 10 ep) |
| 73.72 | dense MNIST-C mean accuracy |
| 3.75% → 98.52 / 69.83 | layer-wise pruned clean / MNIST-C |
| −0.56 vs −3.89 (7×) | clean vs MNIST-C cost at 3.75% (layer-wise) |
| −0.13 (p = 0.83) | MNIST-C cost under global pruning at 3.52% |
| +3.76 pp (p = 0.021) | global over layer-wise on MNIST-C, head to head |
| 6 / 150, 3 / 6 empty | conv1 survivors at 3.75% layer-wise |
| 26.7% vs 1.3% | conv1 kept at round 20: global vs layer-wise |
| 51.98 → 82.18 | PGD accuracy at ε = 0.05, dense → 13.8% weights |
| 245.6 → 75.0 KB (3.27×) | INT8 QAT on-disk size |
| 0.359 → 0.383 ms (1.07× slower) | INT8 CPU latency |
| 93.40 → 90.63 | adversarially trained model, PGD-40×1 → PGD-100×5 |
| 260 runs, 59 tests, 31 survive Holm | the experimental record |

---

## Appendix A: Key code, annotated

Real line numbers from `src/` in `harshith2302/cs725_Project` (`main` @ `3b2584b`). `⋮` = left out; `◀` = plain-English note.

**The pruning, distillation and quantization code is NOT on `main`.** It is on an unpushed branch. A.4 is the report's own Listing 1 (pseudocode) plus a reconstruction labelled [RECON]. Be upfront about this if asked to show the pruning code.

### Reading the code: constructs used below
| Construct | Meaning |
|---|---|
| `nn.Module.register_buffer(name, t)` | a tensor saved with the model and moved by `.to(device)`, but not trained |
| `F.pad(x, (2,2,2,2))` | pad 2 pixels left/right/top/bottom (28×28 → 32×32) |
| `nn.Conv2d(in, out, 5)`, `nn.AvgPool2d(2, 2)` | 5×5 convolution; 2×2 average pooling with stride 2 |
| `torch.autograd.grad(loss, x)` | gradient of the loss w.r.t. the INPUT image (not the weights) |
| `grad.sign()` | elementwise −1/0/+1, the FGSM/PGD step direction |
| `t.clamp(lo, hi)` | elementwise clip |
| `torch.randperm(n, generator=g)` | a seeded random permutation, giving a reproducible split |

### A.1 Normalisation lives inside the model: `models.py`
```python
22  class InputNormalize(nn.Module):
31      def __init__(self, mean=0.1307, std=0.3081, pad=2):
34          self.register_buffer("mean", torch.tensor(mean).view(1, 1, 1, 1))
35          self.register_buffer("std", torch.tensor(std).view(1, 1, 1, 1))
37      def forward(self, x):                        # ◀ x = RAW pixels in [0, 1], shape (B, 1, 28, 28)
39          x = F.pad(x, (self.pad,) * 4, mode="constant", value=0.0)   # ◀ 28 → 32, as LeCun's input
40          return (x - self.mean) / self.std
⋮
51  class LeNet5(nn.Module):
58      def __init__(self, activation="tanh", width=1, num_classes=10):
60          c1, c3, c5, f6 = 6 * width, 16 * width, 120 * width, 84 * width   # ◀ width=2 → the distillation teacher
62          self.normalize = InputNormalize()        # ◀ LAYER 0
63          self.conv1 = nn.Conv2d(1, c1, kernel_size=5)    # ◀ 150 weights: the layer pruning starves
64          self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
65          self.conv3 = nn.Conv2d(c1, c3, kernel_size=5)   # ◀ dense C3 (2,400 weights), not LeCun's sparse table
66          self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
67          self.conv5 = nn.Conv2d(c3, c5, kernel_size=5)   # ◀ 48,000 weights; output is 1×1
68          self.fc6 = nn.Linear(c5, f6)                    # ◀ 10,080
69          self.fc7 = nn.Linear(f6, num_classes)           # ◀ 840; pruned at HALF rate under layer-wise pruning
72      def forward(self, x):
73          x = self.normalize(x)
74          x = self.pool2(self.act(self.conv1(x)))   # ◀ 32→28→14
75          x = self.pool4(self.act(self.conv3(x)))   # ◀ 14→10→5
76          x = self.act(self.conv5(x))               # ◀ 5→1, 120 channels
77          x = torch.flatten(x, 1)
78          x = self.act(self.fc6(x))
79          return self.fc7(x)                         # ◀ logits; softmax + CE in the loss
⋮
82      def prunable_modules(self):                    # ◀ what IMP masks: conv1, conv3, conv5, fc6, fc7
```
Why normalisation is inside: PGD's ε = 0.3 must be in **pixel** units. If normalisation were a dataset transform, an ε-ball in the attacked tensor would be ε × 0.3081 ≈ 0.09 in pixels.

### A.2 PGD-L∞ with budget audits: `attacks.py`
```python
21  def assert_pixel_space_model(model) -> None:   # ◀ refuses to attack a model without InputNormalize
⋮
31  def _grad_wrt_input(model, x, y, loss_fn):
32      x = x.clone().detach().requires_grad_(True)
33      loss = loss_fn(model(x), y)
34      (grad,) = torch.autograd.grad(loss, x)     # ◀ ∂loss/∂pixels
⋮
49  def pgd(model, x, y, eps=0.3, alpha=0.01, steps=40, random_start=True, loss_fn=None):
65      if random_start:
66          adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)   # ◀ random point inside the ε-ball
67          adv = adv.clamp(0.0, 1.0)
71      for _ in range(steps):
72          grad = _grad_wrt_input(model, adv, y, loss_fn)
73          adv = adv.detach() + alpha * grad.sign()                 # ◀ step UP the loss
75          adv = x_orig + (adv - x_orig).clamp(-eps, eps)           # ◀ project back into the ε-ball
76          adv = adv.clamp(0.0, 1.0)                                # ◀ stay a valid image
⋮
114     max_linf = max(max_linf, (adv - x).abs().max().item())       # ◀ MEASURE the real perturbation
123     if max_linf > eps + 1e-5: raise AssertionError(...)          # ◀ a projection bug fails loudly, not as fake robustness
```
The report's final robustness numbers use PGD-100 with 5 restarts, because PGD-40 with 1 restart overstated robustness (§8).

### A.3 Leak-free split and honest measurement: `data.py`, `evaluate.py`
```python
65  def train_val_indices(n_total: int = 60_000):           # data.py
67      g = torch.Generator().manual_seed(SPLIT_SEED)         # ◀ SPLIT_SEED = 1234, fixed for every training seed
68      perm = torch.randperm(n_total, generator=g).tolist()
69      return perm[:N_TRAIN], perm[N_TRAIN : N_TRAIN + N_VAL]  # ◀ 55,000 train / 5,000 val; test set untouched
⋮
48  def latency(model, device=None, warmup=100, iters=1000, single_thread=True):   # evaluate.py
58      torch.set_num_threads(1)                              # ◀ batch-1, single-thread CPU
67          t0 = time.perf_counter()
69          samples.append((time.perf_counter() - t0) * 1000.0)
83      "latency_ms_median": statistics.median(samples),      # ◀ median of 1,000 runs
⋮
105 def model_size(model) -> dict:
115     fp32_kb = _file_size_kb(lambda p: torch.save(model.state_dict(), p))   # ◀ REAL on-disk size → 245.6 KB
```
`model_size` measures the saved file, which is why pruned FP32 checkpoints are still 245.6 KB: the zeros are stored.

### A.4 Iterative magnitude pruning (from the report's Listing 1; not in the public repo)
```python
def prune_step(model, scope, rate=0.20):
    if scope == "layerwise":
        for layer in prunable_layers(model):
            threshold = magnitude_quantile(layer.weight, rate)     # ◀ per-layer 20% cut (fc7 at 10%)
            layer.mask &= (layer.weight.abs() >= threshold)
    elif scope == "global":
        all_weights = concat_magnitudes(model)                   # ◀ ONE threshold across all layers
        threshold = magnitude_quantile(all_weights, rate)
        for layer in prunable_layers(model):
            layer.mask &= (layer.weight.abs() >= threshold)
    apply_masks(model)
```
[RECON] The surrounding lottery-ticket loop:
```python
init_state = copy.deepcopy(model.state_dict())          # θ₀
for round in range(20):
    train(model, epochs=20, select_on="val")             # mask kept applied after every optimizer step
    prune_step(model, scope, rate=0.20)                   # quantile over SURVIVING weights only
    model.load_state_dict(rewind(init_state), keep_masks=True)   # rewind survivors to θ₀ (or an early iterate)
```
