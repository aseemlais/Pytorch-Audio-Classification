# UrbanSound8K Audio Classification with PyTorch

A PyTorch-based environmental sound classification project using the
UrbanSound8K dataset and a custom Convolutional Neural Network (CNN).

The project converts audio waveforms into log-Mel spectrograms and treats
them as 2D inputs to a CNN for multi-class classification.

## Dataset

UrbanSound8K contains 8,732 labeled audio clips (≤4 seconds) across 10
urban sound classes and provides predefined folds for evaluation.

Classes:

- air_conditioner
- car_horn
- children_playing
- dog_bark
- drilling
- engine_idling
- gun_shot
- jackhammer
- siren
- street_music

Dataset: https://urbansounddataset.weebly.com/urbansound8k.html

> The official dataset recommends using its predefined folds for evaluation.
> This project currently uses Fold 10 as a held-out test set for the baseline.

## Tech Stack

- Python
- PyTorch
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab
- NVIDIA T4 GPU

## Audio Preprocessing

Each audio file follows this pipeline:

WAV
→ Resample to 22,050 Hz
→ Pad/trim to 4 seconds
→ Mel Spectrogram
→ Convert to dB
→ Normalize
→ PyTorch Tensor

Configuration:

- Sampling rate: 22,050 Hz
- Duration: 4 seconds
- Mel bands: 128
- FFT size: 2,048
- Hop length: 512
- Input shape: `[1, 128, 173]`

## Data Split

For the current baseline:

- Folds 1–9 → development data
- 15% of development data → validation
- Fold 10 → final test set
- Stratified splitting based on class labels
- Class-weighted CrossEntropyLoss used to handle class imbalance

## CNN Architecture

```text
Input [1, 128, 173]
        ↓
Conv2D (1 → 16) + ReLU + MaxPool
        ↓
Conv2D (16 → 32) + ReLU + MaxPool
        ↓
Conv2D (32 → 64) + ReLU + MaxPool
        ↓
Adaptive Average Pooling
        ↓
Flatten
        ↓
Linear (64 → 128) + ReLU + Dropout
        ↓
Linear (128 → 10)
