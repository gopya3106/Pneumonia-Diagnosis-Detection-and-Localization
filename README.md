# Pneumonia Diagnosis Detection and Localization

UE24CS352A Machine Learning mini-project (Problem 34).
Classifies a chest X-ray as NORMAL or PNEUMONIA with a fine-tuned DenseNet121 and
localizes the suspicious region with Grad-CAM heatmaps.

## Dataset
Kaggle "Chest X-Ray Images (Pneumonia)" (`paultimothymooney/chest-xray-pneumonia`).
It is downloaded automatically by `src/train.py` via `kagglehub`. The original
train/val/test folders are pooled and re-split 70/15/15 **by patient ID** to avoid leakage.

## Setup
```bash
git clone https://github.com/gopya3106/Pneumonia-Diagnosis-Detection-and-Localization.git
cd pneumonia-project
pip install -r requirements.txt
```
A GPU is strongly recommended (Google Colab T4 works).

## Train and evaluate
```bash
python src/train.py --epochs 6
```
Outputs: `models/best.pt`, and in `results/`: `metrics.json`, `confusion_matrix.png`,
`roc_curve.png`, and the split CSVs.

## Grad-CAM samples
```bash
python src/run_gradcam.py
```
Writes `results/gradcam_samples.png`.

## Run the demo
```bash
python app/app.py            # local, opens at http://127.0.0.1:7860
python app/app.py --share    # public link (use this in Colab)
```
Requires `models/best.pt` from the training step.

## Project structure
```
src/common.py        dataset, transforms, model, Grad-CAM
src/train.py         data download, patient-wise split, training, evaluation
src/run_gradcam.py   Grad-CAM sample grid
app/app.py           Gradio demo
results/             metrics and figures
```

## Results
Held-out test set (878 images, split by patient):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| NORMAL | 0.954 | 0.951 | 0.953 | 264 |
| PNEUMONIA | 0.979 | 0.980 | 0.980 | 614 |

Accuracy 0.972, ROC-AUC 0.996. Confusion matrix: 251 / 13 / 12 / 602.
Figures are in `results/`. The trained weights are `models/best.pt` (not tracked in git;
shared separately).

## Team
- Member 1: <name, SRN> - data pipeline, training, evaluation
- Member 2: Dharshini V [PES1UG24AM428] - Grad-CAM, demo app, write-up

## Disclaimer
Educational project only. Not for clinical use.
