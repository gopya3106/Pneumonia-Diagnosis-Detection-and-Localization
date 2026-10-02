"""Download data, make a patient-wise split, train DenseNet121, evaluate on test set.

Usage:  python src/train.py --epochs 6
"""
import argparse
import json
import os
import re
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from common import CLASSES, XrayDataset, build_model, get_transforms

SEED = 42


def download_data():
    import kagglehub
    return kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")


def find_root(base):
    for dp, dn, _ in os.walk(base):
        if "train" in dn and "test" in dn:
            return dp
    raise FileNotFoundError("Could not find train/test folders under " + base)


def group_id(path, label):
    """Patient id so one patient's images never land in two splits."""
    name = os.path.basename(path)
    m = re.search(r"person(\d+)", name)
    if m:
        return "p" + m.group(1)
    m = re.search(r"IM-(\d+)", name)
    if m:
        return "n" + m.group(1)
    return name


def build_df(root):
    rows = []
    for split in ["train", "val", "test"]:
        for label, cls in enumerate(CLASSES):
            for p in glob(os.path.join(root, split, cls, "*")):
                if p.lower().endswith((".jpeg", ".jpg", ".png")):
                    rows.append({"path": p, "label": label, "group": group_id(p, label)})
    return pd.DataFrame(rows)


def make_splits(df):
    """Pool everything, re-split 70/15/15 grouped by patient."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    trval_idx, test_idx = next(gss.split(df, groups=df["group"]))
    trval, test = df.iloc[trval_idx], df.iloc[test_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=SEED)
    tr_idx, val_idx = next(gss.split(trval, groups=trval["group"]))
    return trval.iloc[tr_idx], trval.iloc[val_idx], test


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    os.makedirs(f"{args.out}/models", exist_ok=True)
    os.makedirs(f"{args.out}/results", exist_ok=True)

    root = find_root(download_data())
    df = build_df(root)
    train_df, val_df, test_df = make_splits(df)
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: {len(d)} images | NORMAL={int((d.label==0).sum())} "
              f"PNEUMONIA={int((d.label==1).sum())}")
        d.to_csv(f"{args.out}/results/{name}_split.csv", index=False)

    mk = lambda d, train: DataLoader(
        XrayDataset(d, get_transforms(train)), batch_size=args.batch,
        shuffle=train, num_workers=args.workers, pin_memory=(device == "cuda"))
    train_dl, val_dl, test_dl = mk(train_df, True), mk(val_df, False), mk(test_df, False)

    # class weights to handle imbalance
    counts = np.bincount(train_df["label"], minlength=2)
    weights = torch.tensor(len(train_df) / (2.0 * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_auc, history = 0.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
        sched.step()
        vp, vy = predict(model, val_dl, device)
        val_auc = roc_auc_score(vy, vp)
        val_acc = ((vp > 0.5) == vy).mean()
        tr_loss = total / len(train_df)
        history.append({"epoch": ep, "train_loss": tr_loss, "val_auc": val_auc, "val_acc": val_acc})
        print(f"Epoch {ep}/{args.epochs} | loss {tr_loss:.4f} | val AUC {val_auc:.4f} | val acc {val_acc:.4f}")
        if val_auc >= best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"{args.out}/models/best.pt")

    # ---- final evaluation on the held-out test set ----
    model.load_state_dict(torch.load(f"{args.out}/models/best.pt", map_location=device))
    tp, ty = predict(model, test_dl, device)
    pred = (tp > 0.5).astype(int)
    report = classification_report(ty, pred, target_names=CLASSES, output_dict=True)
    auc = roc_auc_score(ty, tp)
    print("\nTEST RESULTS")
    print(classification_report(ty, pred, target_names=CLASSES))
    print(f"ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(ty, pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues")
    plt.title("Confusion matrix (test)")
    plt.savefig(f"{args.out}/results/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(ty, tp)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve (test)"); plt.legend()
    plt.savefig(f"{args.out}/results/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(f"{args.out}/results/metrics.json", "w") as f:
        json.dump({"report": report, "roc_auc": auc, "confusion_matrix": cm.tolist(),
                   "history": history}, f, indent=2)
    print("Saved model to models/best.pt and results to results/")


if __name__ == "__main__":
    main()
