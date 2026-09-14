# UrbanSound8K Audio Classification (PyTorch)

End-to-end audio classification pipeline built on the **UrbanSound8K** dataset using PyTorch. The project moves from a simple baseline CNN to a carefully tuned pipeline through controlled, one-variable-at-a-time experiments — not by scaling up the model or training longer.

**Final result:** Test Accuracy **69.65%** · Test Macro-F1 **0.6960**
(baseline: 47.07% / 0.4597 → **+22.58 pts accuracy, +23.63 pts Macro-F1**)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Experiments](#experiments)
- [Results](#results)
- [Error Analysis](#error-analysis)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Overview

The goal of this project wasn't just a higher accuracy number — it was to understand and systematically improve every stage of an audio classification pipeline:

- Audio preprocessing and Mel-spectrogram representation
- CNN design for spectrogram inputs
- Class imbalance and loss weighting
- Precision/recall trade-offs and decision-boundary behavior
- Overfitting, regularization, and generalization
- Class-specific error analysis (especially acoustically confusable classes)
- Learning-rate scheduling, early stopping, and checkpointing

Every major change was introduced as a single controlled experiment and validated against **Macro-F1** before moving to the next one.

## Dataset

**UrbanSound8K**

| Property | Value |
|---|---|
| Total clips | 8,732 |
| Classes | 10 |
| Clip length | up to ~4 seconds |
| Format | WAV |
| Folds | 10 predefined folds |

**Classes:** `air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music`

### Split strategy (leakage-safe)

```
UrbanSound8K
     │
     ├── Folds 1–9  → Development
     │        │
     │        ├── Training     (~85%, stratified)
     │        └── Validation   (~15%, stratified)
     │
     └── Fold 10     → Test (untouched until final evaluation)
```

Fold 10 was **never** used during model development or hyperparameter tuning — only for the final, single evaluation pass.

## Pipeline

```
Raw WAV Audio
     ↓
Convert to Mono
     ↓
Resample to 22,050 Hz
     ↓
Fix Duration to 4 Seconds (pad/trim)
     ↓
Mel Spectrogram (128 Mel bins)
     ↓
Power → Decibel Conversion
     ↓
Fixed-Range dB Normalization
     ↓
SpecAugment (training only)
     ↓
CNN
```

**Input shape:** approximately `(1, 128, 173)` → (channel, Mel bins, time frames)

### Fixed-range dB normalization

The baseline used per-sample min-max normalization, which discards absolute loudness information across clips. The improved pipeline instead uses a fixed, physically meaningful dB range:

```python
mel_db = np.clip(mel_db, -80, 0)
normalized = (mel_db + 80) / 80
```

This keeps energy levels comparable across all samples rather than rescaling each clip independently.

### SpecAugment

Applied **only during training** (validation/test spectrograms are left untouched):

- **Frequency masking** — zeroes a random contiguous band of Mel bins
- **Time masking** — zeroes a random contiguous band of time frames

This exposes the model to partial views of the same sound and reduces overfitting.

## Model Architecture

A moderately sized CNN — intentionally not scaled up, to isolate the effect of training/loss changes from capacity changes.

```
Input Log-Mel Spectrogram
        ↓
[Conv2D → BatchNorm2D → ReLU → MaxPool2D → Dropout]  × 3 blocks
        ↓
Global Average Pooling
        ↓
Fully Connected Classifier
        ↓
10-way Softmax
```

| Component | Purpose |
|---|---|
| BatchNorm2D | Stabilizes activation distributions, allows higher effective LR, smoother training |
| Dropout | Prevents co-adaptation of neurons, improves robustness |
| Global Average Pooling | Reduces parameters before the classifier head, mitigates overfitting vs. a large flattened FC layer |

## Training Configuration

| Component | Setting |
|---|---|
| Loss | Weighted `CrossEntropyLoss` + label smoothing (`0.05`) |
| Class weighting | Square-root-damped inverse frequency |
| Optimizer | AdamW (`lr=1e-3`, `weight_decay=1e-4`) |
| Scheduler | `ReduceLROnPlateau` (`mode="max"`, monitors val Macro-F1, `factor=0.5`, `patience=2`) |
| Model selection | Best validation Macro-F1 (not final epoch) |
| Early stopping | Enabled, with best-checkpoint restoration |

### Why Macro-F1 over accuracy?

UrbanSound8K is not perfectly balanced, and accuracy can hide poor performance on individual classes (a model can look "good" overall while failing badly on one class). Macro-F1 scores every class equally and averages the result, so it was used as the primary model-selection metric throughout.

## Experiments

Each experiment changed **one thing** relative to the previous configuration.

| # | Configuration | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---|---|---|---|
| 1 | Baseline CNN + baseline (min-max) preprocessing | — | 47.07% | 0.4597 |
| 2 | + Fixed-range dB norm, BatchNorm, Dropout, GAP, SpecAugment, linear weighted CE, AdamW, scheduler | 0.6262 | 61.77% | 0.6048 |
| 3 | + Square-root-damped class weights + label smoothing (0.05) | **0.7213** | **69.65%** | **0.6960** |

### Experiment 2 → 3: why re-weight the loss?

Experiment 2's linear inverse-frequency weighting (`w_c = N / (K · n_c)`) gave the rarest class, `gun_shot`, a disproportionately large loss weight. This produced a specific, diagnosable failure:

```
gun_shot   → Precision 31.68%  |  Recall 100.00%  |  F1 0.4812
siren      → Precision 93.33%  |  Recall  16.87%  |  F1 0.2857
```

The model found *every* real gunshot but also mislabeled many other sounds as gunshots (`air_conditioner`, `siren`, `dog_bark`, `children_playing` were the main sources). Meanwhile `siren` was rarely predicted at all. This wasn't classic class imbalance — it was the loss weighting distorting the decision boundary in one direction.

**Fix:** dampen the weighting with a square root, which retains imbalance compensation while compressing the gap between rare and common classes:

```python
raw_weights  = N / (num_classes * class_counts)   # original (too aggressive)
sqrt_weights = np.sqrt(raw_weights)
sqrt_weights = sqrt_weights / sqrt_weights.mean() # renormalize to mean 1
```

Combined with label smoothing (`0.05`) to discourage overconfident predictions, this produced the largest single jump in the project: **Val Macro-F1 0.6262 → 0.7213**.

> **Note on ablation:** square-root weighting and label smoothing were introduced together in this experiment, so their individual contributions were not isolated. A follow-up ablation (linear → sqrt → sqrt + label smoothing) is listed under [Future Work](#future-work).

## Results

### Final classification report (test set, fold 10)

```
                  precision    recall  f1-score   support

 air_conditioner     0.4157    0.7400    0.5324       100
        car_horn     0.7812    0.7576    0.7692        33
children_playing     0.7867    0.5900    0.6743       100
        dog_bark     0.7500    0.6000    0.6667       100
        drilling     0.8218    0.8300    0.8259       100
   engine_idling     0.6789    0.7957    0.7327        93
        gun_shot     0.6667    1.0000    0.8000        32
      jackhammer     0.8288    0.9583    0.8889        96
           siren     0.9474    0.2169    0.3529        83
    street_music     0.7857    0.6600    0.7174       100

        accuracy                         0.6965       837
       macro avg     0.7463    0.7148    0.6960       837
    weighted avg     0.7460    0.6965    0.6875       837
```

### Class-wise F1, best to worst

| Rank | Class | F1 |
|---|---|---|
| 1 | jackhammer | 0.8889 |
| 2 | drilling | 0.8259 |
| 3 | gun_shot | 0.8000 |
| 4 | car_horn | 0.7692 |
| 5 | engine_idling | 0.7327 |
| 6 | street_music | 0.7174 |
| 7 | children_playing | 0.6743 |
| 8 | dog_bark | 0.6667 |
| 9 | air_conditioner | 0.5324 |
| 10 | siren | 0.3529 |

## Error Analysis

### `gun_shot`: from over-predicted to well-calibrated

| Metric | Before (Exp. 2) | After (Exp. 3) |
|---|---|---|
| Precision | 31.68% | **66.67%** |
| Recall | 100.00% | 100.00% |
| F1 | 0.4812 | **0.8000** |

False positives into `gun_shot`, before → after square-root weighting + label smoothing:

| Source class | Before | After |
|---|---|---|
| air_conditioner | 23 | **0** |
| siren | 22 | 5 |
| dog_bark | 13 | 9 |
| children_playing | 7 | 1 |

### `siren`: improved, but still the weakest class

| Metric | Before (Exp. 2) | After (Exp. 3) |
|---|---|---|
| Precision | 93.33% | 94.74% |
| Recall | 16.87% | 21.69% |
| F1 | 0.2857 | 0.3529 |

The dominant remaining confusion shifted: `siren → gun_shot` dropped from 22 to 5, but **`siren → air_conditioner` grew to 37 of 83 siren samples**. This points to genuine acoustic overlap between siren's tonal sweep and air_conditioner's low-frequency hum, rather than a loss-weighting problem — future work should target this directly (see below).

## Key Findings

1. **Fixed-range dB normalization** preserves cross-clip loudness information that per-sample min-max normalization discards.
2. **BatchNorm** stabilized optimization and enabled a higher effective learning rate.
3. **SpecAugment** provided audio-specific regularization by exposing the model to partial spectrogram views.
4. **Linear inverse-frequency weighting was too aggressive** — it pushed the rarest class (`gun_shot`) to be over-predicted at the expense of precision.
5. **Square-root-damped weighting** retained imbalance compensation while removing that distortion.
6. **Label smoothing** discouraged overconfident predictions and contributed to the best-performing configuration.
7. **Macro-F1-based checkpointing** mattered: validation Macro-F1 fluctuated significantly between epochs (e.g., 0.7213 → 0.3957 → 0.6867 across epochs 10–15), so the final training epoch was **not** the best model — best-checkpoint restoration was essential.
8. **More epochs is not automatically better.** The model peaked at epoch 10 and became unstable afterward; blindly training longer would have risked more overfitting, not less.

## Limitations

- **Siren remains weak** (F1 = 0.3529), dominated by confusion with `air_conditioner`.
- **Air conditioner remains challenging** (F1 = 0.5324) — high recall but lower precision, meaning several other classes still get misclassified as it.
- **No pretrained audio backbone** — the CNN learns representations from scratch rather than leveraging models like YAMNet, PANNs, or AST.
- **Square-root weighting and label smoothing were not ablated separately** — both were introduced in the same experiment, so their individual contributions are unverified.
- **Validation Macro-F1 was noisy** across epochs; early stopping and checkpointing mitigate but don't eliminate this.

## Future Work

- **Full ablation study:** linear weights → square-root weights → square-root weights + label smoothing, isolated one at a time.
- **Targeted siren improvement:** siren-specific augmentation, more diverse siren training samples, conservative siren-specific weighting, or hard-negative training focused on `siren ↔ air_conditioner`.
- **Focal Loss** as an alternative to weighted CE, to up-weight hard examples directly rather than by class frequency.
- **Stronger, more diverse augmentation:** time shifting, background noise injection, gain/pitch variation, time stretching, Mixup — introduced and evaluated individually.
- **Pretrained audio encoders** (YAMNet, PANNs, AST, wav2vec-style models, BEATs) as a stronger feature-extraction baseline to compare against the from-scratch CNN.

 
 
## Tech Stack

- Python
- PyTorch
- Librosa / torchaudio
- NumPy, Pandas
- scikit-learn
- Matplotlib, Seaborn
- Google Colab (NVIDIA T4 GPU)

## Author

**Aseem Lais**
PyTorch · Deep Learning · Audio Classification · Machine Learning
