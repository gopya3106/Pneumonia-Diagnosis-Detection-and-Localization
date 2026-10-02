"""Shared utilities: dataset, transforms, model, Grad-CAM."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms
import matplotlib

CLASSES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class XrayDataset(Dataset):
    def __init__(self, df, tfm):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(img), self.labels[i]


def build_model(pretrained: bool = True):
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
    return model


def load_model(path, device="cpu"):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def gradcam(model, x, target=None):
    """Grad-CAM on DenseNet121's final feature map.

    x: tensor (1, 3, H, W). Returns (cam in [0,1] as HxW numpy, probs numpy).
    """
    model.eval()
    x = x.requires_grad_(True) if not x.requires_grad else x
    feats = model.features(x)
    feats.retain_grad()
    pooled = F.adaptive_avg_pool2d(F.relu(feats), 1).flatten(1)
    logits = model.classifier(pooled)
    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    if target is None:
        target = int(logits.argmax(1))
    model.zero_grad()
    logits[0, target].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, probs


def overlay(pil_img, cam, alpha=0.4):
    """Blend a Grad-CAM heatmap over the (resized) X-ray."""
    img = np.asarray(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    out = (1 - alpha) * img + alpha * heat
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
