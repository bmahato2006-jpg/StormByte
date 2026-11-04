import csv
from collections import Counter
from pathlib import Path
from PIL import Image

manifest_dir = Path("data/processed/satellite/intensity/manifests")
train_p = manifest_dir / "intensity_train.csv"
val_p = manifest_dir / "intensity_validation.csv"
test_p = manifest_dir / "intensity_test.csv"
cv_p = manifest_dir / "intensity_cv_folds.csv"

manifests = {
    "train": train_p,
    "val": val_p,
    "test": test_p,
    "cv_folds": cv_p,
}

print("=" * 80)
print("VALIDATING GENERATED INTENSITY MANIFESTS")
print("=" * 80)

data = {}
for name, p in manifests.items():
    if not p.exists():
        print(f"ERROR: {p} does NOT exist!")
        continue
    with open(p, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data[name] = rows
    winds = [int(r["wind_speed_knots"]) for r in rows]
    cats = Counter(r["intensity_category"] for r in rows)
    print(f"\nManifest: {name} ({p.name})")
    print(f"  Row count: {len(rows)}")
    print(f"  Min wind speed: {min(winds)} kt")
    print(f"  Max wind speed: {max(winds)} kt")
    print(f"  Mean wind speed: {sum(winds)/len(winds):.2f} kt")
    print("  Category distribution:")
    for cat, cnt in sorted(cats.items()):
        print(f"    {cat:35s}: {cnt:3d} ({cnt/len(rows)*100:5.1f}%)")

# Check overlaps
train_ids = {r["sample_id"] for r in data["train"]}
val_ids = {r["sample_id"] for r in data["val"]}
test_ids = {r["sample_id"] for r in data["test"]}

print("\n" + "=" * 80)
print("OVERLAP & LEAKAGE CHECKS")
print("=" * 80)
print(f"Train & Val overlap: {len(train_ids.intersection(val_ids))}")
print(f"Train & Test overlap: {len(train_ids.intersection(test_ids))}")
print(f"Val & Test overlap: {len(val_ids.intersection(test_ids))}")
print(f"Total unique IDs across train/val/test: {len(train_ids | val_ids | test_ids)}")

# 84.jpg / 84(1).jpg placement
print("\nNear-Duplicate Placement Check (84.jpg & 84(1).jpg):")
for name, rows in data.items():
    found = [r for r in rows if r["img_name"] in ("84.jpg", "84(1).jpg")]
    if found:
        print(f"  In {name}: {len(found)} found")
        for r in found:
            print(f"    - img_name: {r['img_name']}, split: {r.get('split')}, fold: {r.get('fold')}")

# Dry-run validation of file existence on disk
print("\n" + "=" * 80)
print("DRY RUN: IMAGE FILE EXISTENCE & READABILITY ON DISK")
print("=" * 80)
unreadable = []
missing = []
for name, rows in data.items():
    for r in rows:
        ip = Path(r["image_path"])
        if not ip.exists():
            missing.append((name, r["sample_id"], str(ip)))
        else:
            try:
                with Image.open(ip) as im:
                    im.verify()
            except Exception as e:
                unreadable.append((name, r["sample_id"], str(e)))

print(f"Total missing image files across manifests: {len(missing)}")
print(f"Total unreadable image files: {len(unreadable)}")

# Check fold distribution in cv_folds
print("\n" + "=" * 80)
print("5-FOLD CV DISTRIBUTION")
print("=" * 80)
cv_rows = data["cv_folds"]
fold_counts = Counter(r["fold"] for r in cv_rows)
for fold_idx in sorted(fold_counts.keys()):
    f_rows = [r for r in cv_rows if r["fold"] == fold_idx]
    f_winds = [int(r["wind_speed_knots"]) for r in f_rows]
    print(f"Fold {fold_idx}: {len(f_rows)} samples, min={min(f_winds)} kt, max={max(f_winds)} kt, mean={sum(f_winds)/len(f_winds):.2f} kt")

