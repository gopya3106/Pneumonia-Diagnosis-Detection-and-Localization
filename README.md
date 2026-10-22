# Pneumonia Diagnosis Detection and Localization

UE24CS352A Machine Learning mini-project (Problem 34).
Classifies a chest X-ray as NORMAL or PNEUMONIA with a fine-tuned DenseNet121 and
localizes the suspicious region with Grad-CAM heatmaps.

## Dataset
Kaggle "Chest X-Ray Images (Pneumonia)" (`paultimothymooney/chest-xray-pneumonia`).
It is downloaded automatically by `src/train.py` via `kagglehub`. The original
train/val/test folders are pooled and re-split 70/15/15 **by patient ID** to avoid leakage.

## Setup

**Linux / macOS**
```bash
git clone https://github.com/gopya3106/Pneumonia-Diagnosis-Detection-and-Localization.git
cd Pneumonia-Diagnosis-Detection-and-Localization
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/gopya3106/Pneumonia-Diagnosis-Detection-and-Localization.git
cd Pneumonia-Diagnosis-Detection-and-Localization
python -m venv venv
Set-ExecutionPolicy -Scope Process RemoteSigned
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run all commands below from the repository root.

## Run the demo
The trained weights are included at `models/best.pt`, so the demo works right after setup:

```bash
python app/app.py            # local, opens at http://127.0.0.1:7860
python app/app.py --share    # public link (use this in Colab)
```

Upload a chest X-ray and the app shows the NORMAL / PNEUMONIA prediction with a
Grad-CAM heatmap of the regions that drove it.

## Train and evaluate (optional)
Only needed if you want to retrain the model. A GPU is strongly recommended
(Google Colab T4 works).

```bash
python src/train.py --epochs 6
```

Useful flags: `--batch` (default 32, lower it if you run out of GPU memory),
`--lr`, `--workers` (use `--workers 0` on Windows if the dataloader crashes).

Outputs: `models/best.pt`, and in `results/`: `metrics.json`, `confusion_matrix.png`,
`roc_curve.png`, and the split CSVs.

If the dataset download asks for authentication, set your Kaggle credentials
(`KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables).

## Grad-CAM samples
```bash
python src/run_gradcam.py
```
Writes `results/gradcam_samples.png`.

## Project structure
```
src/common.py        dataset, transforms, model, Grad-CAM
src/train.py         data download, patient-wise split, training, evaluation
src/run_gradcam.py   Grad-CAM sample grid
app/app.py           Gradio demo
models/best.pt       trained weights
results/             metrics and figures
notebooks/           Colab training notebook
submission/          project slides and write-up
```

## Results
Held-out test set (878 images, split by patient):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| NORMAL | 0.954 | 0.951 | 0.953 | 264 |
| PNEUMONIA | 0.979 | 0.980 | 0.980 | 614 |

Accuracy 0.972, ROC-AUC 0.996. Confusion matrix: 251 / 13 / 12 / 602.
Figures are in `results/`.

## Team
- Member 1: Nakka Gopya [PES1UG24AM107] - data pipeline, training, evaluation
- Member 2: Dharshini V [PES1UG24AM428] - Grad-CAM, demo app, write-up

## Disclaimer
Educational project only. Not for clinical use.
