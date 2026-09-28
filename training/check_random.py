import csv
import glob
import os
import random
import sys

import argparse

import cv2

sys.path.insert(0, r"D:\HTH\backend")
from inference import RetfoundInference

ROOT = r"D:\HTH\training\RETFound-main\data\test"
CSV = r"D:\HTH\training\aptos\train.csv"
OUT = r"D:\HTH\training\_check"
os.makedirs(OUT, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--image", default=None)
cli = parser.parse_args()

grade_of = {}
with open(CSV, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        grade_of[row["id_code"]] = int(row["diagnosis"])

candidates = []
for label in ("normal", "referable"):
    for path in glob.glob(os.path.join(ROOT, label, "*.png")):
        candidates.append((path, label))

if cli.image:
    path = cli.image
    true_label = os.path.basename(os.path.dirname(path))
else:
    path, true_label = random.choice(candidates)
code = os.path.splitext(os.path.basename(path))[0]
grade = grade_of.get(code, "?")

model = RetfoundInference(
    r"D:\HTH\training\RETFound-main\output_dir\aptos_binary_lp\checkpoint-best.pth",
    "vit_large_patch16_224",
    2,
    ("normal", "referable"),
    224,
)

image = cv2.imread(path)
result = model.analyze(image, with_heatmap=True)

correct = result["label"] == true_label
name = os.path.basename(path)
cv2.imwrite(os.path.join(OUT, "random_original.png"), image)
cv2.imwrite(os.path.join(OUT, "random_heatmap.png"), result["heatmap"])

print("random image:", name)
print("original APTOS grade (0-4):", grade)
print("dataset label:", true_label, "   (grade>=1 => referable)")
print("model prediction:", result["label"])
print("confidence:", result["confidence"], "%")
print("probabilities:", result["probabilities"])
print("CORRECT" if correct else "WRONG")
print("saved:", os.path.join(OUT, "random_original.png"), "and random_heatmap.png")
