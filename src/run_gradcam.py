"""Generate a grid of Grad-CAM overlays from the test split.

Usage:  python src/run_gradcam.py
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from common import CLASSES, get_transforms, gradcam, load_model, overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--n", type=int, default=4, help="images per class")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(f"{args.out}/models/best.pt", device)
    test = pd.read_csv(f"{args.out}/results/test_split.csv")
    tfm = get_transforms(False)

    rows = []
    for label in (1, 0):
        rows.append(test[test.label == label].sample(args.n, random_state=1))
    sample = pd.concat(rows)

    fig, axes = plt.subplots(2, len(sample), figsize=(3 * len(sample), 6.4))
    for i, (_, r) in enumerate(sample.iterrows()):
        img = Image.open(r["path"]).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        cam, probs = gradcam(model, x)
        pred = int(probs.argmax())
        axes[0, i].imshow(img.resize((224, 224)), cmap="gray")
        axes[0, i].set_title(f"True: {CLASSES[r['label']]}", fontsize=9)
        axes[1, i].imshow(overlay(img, cam))
        axes[1, i].set_title(f"Pred: {CLASSES[pred]} ({probs[pred]:.2f})", fontsize=9)
        for a in axes[:, i]:
            a.axis("off")
    plt.tight_layout()
    plt.savefig(f"{args.out}/results/gradcam_samples.png", dpi=150)
    print("Saved results/gradcam_samples.png")


if __name__ == "__main__":
    main()
