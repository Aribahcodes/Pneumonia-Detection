# Pneumonia Detection from Chest X-Rays

Deep learning pipeline for detecting pneumonia from chest X-ray images, built on the
[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) dataset.

## Project Overview

- **Task**: 3-class classification (Normal / No Lung Opacity–Not Normal / Lung Opacity) from chest X-ray DICOM images
- **Model**: DenseNet121 (transfer learning, ImageNet-pretrained)
- **Explainability**: Grad-CAM visualizations for clinically meaningful predictions
- **Deployment**: Streamlit app for interactive demo

## Motivation

This project explores medical image classification end-to-end:
From raw DICOM handling and class imbalance, through model training, to explainability and deployment.
The RSNA dataset was chosen over more commonly used chest X-ray datasets (e.g. Kaggle's Paul Mooney set)
specifically for its DICOM format, bounding-box annotations, and clinical provenance.

## Project Structure
```
├── data/
│   ├── raw/            # untouched kagglehub download (gitignored)
│   ├── processed/       # patient-level resolved labels (labels.csv)
│   ├── splits/          # stratified train/val/test patientId splits
│   └── cache/           # preprocessed .npy images for fast loading (gitignored)
├── notebooks/
│   ├── 01_explore_data.ipynb      # EDA, label resolution, stratified split
│   ├── 02_dataset_pipeline.ipynb  # DICOM Dataset class, transforms, caching
│   ├── 03_model_training.ipynb    # DenseNet121, freeze/finetune/progressive comparison
│   ├── 04_evaluation_metrics.ipynb
│   ├── 05_gradcam_explainability.ipynb
│   └── 06_visualize_predictions.ipynb
├── src/
│   ├── preprocessing.py   # label mapping, patient-level label resolution, caching, transforms
│   ├── dataset.py         # DICOM-aware PyTorch Dataset (cache-enabled)
│   ├── utils.py            # kagglehub path resolution
│   ├── model.py            # DenseNet121 w/ freeze/fine-tune toggle
│   ├── train.py            # train/validate epoch loops, multi-epoch training wrapper
│   ├── evaluate.py         # (Phase 4)
│   └── gradcam.py          # (Phase 5)
├── configs/
│   └── config.yaml
├── app/
│   └── streamlit_app.py
├── models/     # saved checkpoints (gitignored)
└── outputs/    # figures and metrics for writeup
```
## Dataset

RSNA Pneumonia Detection Challenge dataset (~26,684 training DICOMs), downloaded via
`kagglehub` at runtime — not committed to this repo due to size. To reproduce:

```python
from src.utils import get_raw_dir
RAW_DIR = get_raw_dir()
```

You'll need to accept the competition rules on Kaggle and have your Kaggle API credentials
configured for `kagglehub` to authenticate.

## Key Technical Decisions

- **3-class over binary**: RSNA's detailed class CSV distinguishes `Normal` from
  `No Lung Opacity / Not Normal` — both map to `Target=0` in the binary scheme, but are
  clinically distinct. 3-class classification preserves that distinction.
- **`pd.concat` over `pd.merge` for label joining**: a naive merge on `patientId` inflated
  row count (30,227 → 37,629) due to duplicate patientIds across both CSVs (multi-box
  patients). Since both CSVs are row-aligned by construction, `pd.concat` on columns was
  the correct, leak-free approach.
- **Patient-level stratified split**: a manual 70/15/15 train/val/test split by `patientId`,
  stratified by class, since the dataset provides no official validation split.
- **Cache preprocessed images**: DICOM decoding is expensive; each image is decoded and
  resized once, then cached as `.npy`, avoiding repeated decode cost every epoch.
- **ImageNet-pretrained backbone**: grayscale X-rays are replicated to 3 channels and
  normalized with ImageNet mean/std to work with a pretrained DenseNet121.
- **Fine-tuning strategy comparison**: frozen backbone, full fine-tune, and progressive
  unfreezing were each trained and compared on validation accuracy. Full fine-tune
  (lr=1e-4) performed best, reaching ~72% val accuracy after 5 epochs — outperforming
  both frozen (~64%) and progressive unfreezing (~70% over 8 epochs).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run notebooks in order (`01` → `06`) for the full pipeline walkthrough, or import directly
from `src/` for reusable pipeline code.

## Status

- ✅ **Phase 1** — Dataset setup, EDA, label resolution, stratified split
- ✅ **Phase 2** — DICOM dataset pipeline, transforms, augmentation, caching, bbox validation
- ✅ **Phase 3** — Model architecture, freeze/fine-tune/progressive-unfreeze comparison
- ⬜ Phase 4 — Evaluation metrics
- ⬜ Phase 5 — Grad-CAM explainability
- ⬜ Phase 6 — Deployment (Streamlit app)

## Results

_(to be filled in after evaluation — accuracy-based model selection above is provisional;
final model choice pending per-class recall analysis in Phase 4)_

## Demo

_(Streamlit app link to be added after deployment)_