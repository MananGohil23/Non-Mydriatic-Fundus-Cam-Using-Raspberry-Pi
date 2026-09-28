import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG")


def find_image(images_dir, code):
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{code}{ext}"
        if candidate.exists():
            return candidate
    matches = list(images_dir.rglob(f"{code}.*"))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="Folder containing APTOS images")
    parser.add_argument("--csv", required=True, help="APTOS train.csv")
    parser.add_argument("--out", default="data")
    parser.add_argument("--id-col", default="id_code")
    parser.add_argument("--label-col", default="diagnosis")
    parser.add_argument("--positive-threshold", type=int, default=1,
                        help="diagnosis >= threshold becomes 'referable'")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", help="Move instead of copy")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out = Path(args.out)

    label_of = {}
    with open(args.csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            code = row[args.id_col]
            label = "referable" if int(row[args.label_col]) >= args.positive_threshold else "normal"
            label_of[code] = label

    codes = list(label_of.keys())
    labels = [label_of[c] for c in codes]
    if len(set(labels)) < 2:
        raise SystemExit("only one class found - check --label-col / --positive-threshold")

    holdout = args.val_frac + args.test_frac
    train_ids, rest_ids, _, rest_y = train_test_split(
        codes, labels, test_size=holdout, stratify=labels, random_state=args.seed
    )
    rel_test = args.test_frac / holdout
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=rel_test, stratify=rest_y, random_state=args.seed
    )

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    missing = []
    for split, split_ids in splits.items():
        for code in split_ids:
            src = find_image(images_dir, code)
            if src is None:
                missing.append(code)
                continue
            dest_dir = out / split / label_of[code]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            if args.move:
                shutil.move(str(src), str(dest))
            else:
                shutil.copy2(src, dest)

    print(f"missing images: {len(missing)}")
    for split, ids in splits.items():
        counts = Counter(label_of[c] for c in ids)
        print(f"  {split}: {len(ids):5d}  {dict(counts)}")
    print(f"wrote ImageFolder tree to: {out.resolve()}")


if __name__ == "__main__":
    main()
