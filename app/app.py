"""Gradio demo: upload a chest X-ray, get a prediction and a Grad-CAM heatmap.

Usage:  python app/app.py            (local)
        python app/app.py --share    (public link, useful in Colab)
"""
import argparse
import os
import sys

import gradio as gr
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import CLASSES, get_transforms, gradcam, load_model, overlay  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(os.path.join(ROOT, "models", "best.pt"), device)
tfm = get_transforms(False)


def predict(img):
    if img is None:
        return None, None
    img = img.convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    cam, probs = gradcam(model, x)
    scores = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    return scores, overlay(img, cam).resize(img.size)


import inspect  # noqa: E402
_params = inspect.signature(gr.Interface.__init__).parameters
no_flag = ({"flagging_mode": "never"} if "flagging_mode" in _params
           else {"allow_flagging": "never"})

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Chest X-ray"),
    outputs=[gr.Label(num_top_classes=2, label="Prediction"),
             gr.Image(label="Grad-CAM (red = region driving the prediction)")],
    title="Pneumonia Detection and Localization",
    description="Upload a chest X-ray. The model predicts NORMAL or PNEUMONIA and "
                "highlights the regions it used. Educational project, not a medical device.",
    **no_flag,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    demo.launch(share=args.share)
