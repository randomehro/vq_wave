# VQ-Wave: Spatio-Temporal Inception Network for Analysis of Functional Lung MRI Data

This repository contains the official PyTorch implementation, **pre-trained model**, and inference framework for **VQ-Wave**, as described in the paper:

> **VQ-Wave: A physics-driven spatio-temporal deep learning approach for non-contrast-enhanced lung ventilation and perfusion MRI**
> *Bauman G, Panos P, Bieri O.*
> (Submitted), 2026.

## Scope of Repository
This repository provides the tools necessary to replicate the inference results described in the manuscript using the provided pre-trained weights.
**Note:** The synthetic training data generator and the full training pipeline are **not** included in this repository to protect proprietary methodologies.

## Contents
* `vqwave_model.py`: Complete PyTorch implementation of the Spatio-Temporal Inception Network, including the custom Squeeze-and-Excitation (SE) blocks and hybrid pooling layers.
* `vqwave_inference.py`: Script for model inference which can be used to process time-resolved two-dimensional functional MRI data in DICOM format.
* `VQWaveNetwork_V13.pt`: **Pre-trained model weights** (trained on synthetic physics-based signals).
* `dataset/`: Contains a sample anonymized DICOM time-series for testing purposes.

## Usage Example
You can run the inference script directly on a folder of time-resolved DICOM images.

```bash
# Basic usage for a single slice time-series
python3 vqwave_inference.py -d ./data/volunteer_01/slice_05 -m VQWaveModel_V13.pt -o results.nrrd
```
The script performs the following steps:
1. Loads the DICOM time-series.
2. Normalizes the signal intensity (voxel-wise z-score).
3. Reshapes the data into spatio-temporal patches (3x3 neighbors).
4. Runs the pre-trained VQ-Wave model.
5. Exports the ventilation and perfusion amplitude and phase maps.
