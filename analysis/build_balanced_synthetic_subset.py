import argparse
import csv
import shutil
from pathlib import Path


EMOTION_NAMES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def parse_classes(value: str):
    name_to_id = {name: idx for idx, name in EMOTION_NAMES.items()}
    labels = []
    for item in value.split(","):
        item = item.strip().lower()
        if item:
            labels.append(int(item) if item.isdigit() else name_to_id[item])
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--classes", default="disgust,fear,sad")
    parser.add_argument("--per-class", type=int, default=0, help="0 uses the smallest available class count")
    args = parser.parse_args()

    labels = parse_classes(args.classes)
    files_by_label = {}
    for label in labels:
        files_by_label[label] = sorted(args.accepted_dir.glob(f"{label}_*.png"))

    available = {label: len(files) for label, files in files_by_label.items()}
    if any(count == 0 for count in available.values()):
        raise ValueError(f"Cannot balance with empty classes: {available}")

    per_class = args.per_class if args.per_class > 0 else min(available.values())
    per_class = min(per_class, min(available.values()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in labels:
        class_name = EMOTION_NAMES[label]
        class_dir = args.out_dir / f"{label}_{class_name}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for src in files_by_label[label][:per_class]:
            dst = class_dir / src.name
            shutil.copy2(src, dst)
            rows.append(
                {
                    "path": str(dst),
                    "label": label,
                    "class_name": class_name,
                    "source": str(src),
                }
            )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "class_name", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"available: {available}")
    print(f"balanced per class: {per_class}")
    print(f"wrote {len(rows)} rows to {args.manifest}")


if __name__ == "__main__":
    main()
