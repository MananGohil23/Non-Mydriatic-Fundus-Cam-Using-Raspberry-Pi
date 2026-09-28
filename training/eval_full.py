import glob
import os
import sys

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, r"D:\HTH\backend")
from inference import RetfoundInference

CKPT = r"D:\HTH\training\RETFound-main\output_dir\aptos_binary_lp\checkpoint-best.pth"
ROOT = r"D:\HTH\training\RETFound-main\data\test"

model = RetfoundInference(CKPT, "vit_large_patch16_224", 2, ("normal", "referable"), 224)
print("load_report:", model.load_report)

y_true, y_prob = [], []
for label, index in (("normal", 0), ("referable", 1)):
    files = sorted(glob.glob(os.path.join(ROOT, label, "*.png")))
    for path in files:
        image = cv2.imread(path)
        result = model.analyze(image, with_heatmap=False)
        y_true.append(index)
        y_prob.append(result["probabilities"]["referable"] / 100.0)

y_true = np.array(y_true)
y_prob = np.array(y_prob)

print("n =", len(y_true), "| positives =", int(y_true.sum()), "| negatives =", int((y_true == 0).sum()))
print("AUC:", round(float(roc_auc_score(y_true, y_prob)), 4))

for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / len(y_true)
    print(
        f"thr={threshold:.1f}  acc={acc:.4f}  sensitivity={sens:.4f}  specificity={spec:.4f}  "
        f"TP={tp} FP={fp} FN={fn} TN={tn}"
    )

print("calibration (predicted referable-probability vs actual referable-rate):")
bins = np.linspace(0.0, 1.0, 11)
for low, high in zip(bins[:-1], bins[1:]):
    mask = (y_prob >= low) & (y_prob < high)
    if mask.sum():
        print(
            f"  [{low:.1f},{high:.1f})  n={int(mask.sum()):3d}  mean_pred={y_prob[mask].mean():.3f}  "
            f"actual={y_true[mask].mean():.3f}"
        )
