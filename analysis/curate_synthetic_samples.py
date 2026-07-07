from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


EMOTION_NAMES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def parse_classes(value: str) -> list[int]:
    name_to_id = {name: idx for idx, name in EMOTION_NAMES.items()}
    labels = []
    for item in value.split(","):
        item = item.strip().lower()
        if item:
            labels.append(int(item) if item.isdigit() else name_to_id[item])
    return labels


def image_index(path: Path) -> int | None:
    match = re.search(r"_synthetic_(\d+)_", path.name)
    return int(match.group(1)) if match else None


def laplacian_variance(image: np.ndarray) -> float:
    center = image[1:-1, 1:-1]
    lap = (
        -4.0 * center
        + image[:-2, 1:-1]
        + image[2:, 1:-1]
        + image[1:-1, :-2]
        + image[1:-1, 2:]
    )
    return float(lap.var())


def center_score(image: np.ndarray) -> float:
    contrast = np.abs(image - image.mean())
    mask = contrast > max(float(image.std()) * 0.5, 0.03)
    if not mask.any():
        return 0.0
    ys, xs = np.nonzero(mask)
    cy = float(ys.mean()) / max(image.shape[0] - 1, 1)
    cx = float(xs.mean()) / max(image.shape[1] - 1, 1)
    distance = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    return max(0.0, 1.0 - distance / math.sqrt(0.5))


def symmetry_score(image: np.ndarray) -> float:
    left = image[:, : image.shape[1] // 2]
    right = image[:, -left.shape[1] :]
    diff = np.abs(left - np.fliplr(right)).mean()
    return max(0.0, 1.0 - float(diff) / 0.35)


def quality_metrics(path: Path) -> dict[str, float]:
    image = Image.open(path).convert("L").resize((48, 48))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    contrast = float(arr.std())
    sharpness = laplacian_variance(arr)
    clipped = float(((arr < 0.02) | (arr > 0.98)).mean())
    return {
        "contrast": contrast,
        "sharpness": sharpness,
        "clipped_fraction": clipped,
        "center_score": center_score(arr),
        "symmetry_score": symmetry_score(arr),
        "contrast_score": min(1.0, contrast / 0.22),
        "sharpness_score": min(1.0, sharpness / 0.018),
        "clipping_score": max(0.0, 1.0 - clipped / 0.25),
    }


def diversity_embedding(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((16, 16))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = arr - float(arr.mean())
    scale = float(arr.std())
    if scale > 1e-6:
        arr = arr / scale
    return arr.reshape(-1)


def composite_score(row: dict[str, float]) -> float:
    return (
        0.50 * row["confidence"]
        + 0.15 * row["contrast_score"]
        + 0.15 * row["sharpness_score"]
        + 0.10 * row["clipping_score"]
        + 0.05 * row["center_score"]
        + 0.05 * row["symmetry_score"]
    )


def accepted_summary(summary: Path) -> pd.DataFrame:
    df = pd.read_csv(summary)
    if "accepted" not in df.columns:
        raise ValueError(f"{summary} does not contain an accepted column")
    accepted = df[df["accepted"] == True].copy()  # noqa: E712
    accepted["index"] = accepted["index"].astype(int)
    accepted["requested_label"] = accepted["requested_label"].astype(int)
    return accepted.set_index("index")


def pairwise_distance(candidate: np.ndarray, selected: list[np.ndarray]) -> float:
    if not selected:
        return 1.0
    distances = [float(np.sqrt(np.mean((candidate - item) ** 2))) for item in selected]
    return min(1.0, min(distances) / 2.0)


def select_diverse(group: pd.DataFrame, per_class: int, diversity_weight: float) -> pd.DataFrame:
    if diversity_weight <= 0 or len(group) <= per_class:
        selected = group.head(per_class).copy()
        selected["diversity_score"] = 1.0
        selected["selection_score"] = selected["quality_score"]
        return selected

    remaining = group.copy()
    selected_rows = []
    selected_embeddings: list[np.ndarray] = []

    while len(selected_rows) < per_class and not remaining.empty:
        best_index = None
        best_score = -1.0
        best_diversity = 0.0
        for row in remaining.itertuples():
            diversity = pairwise_distance(row.embedding, selected_embeddings)
            score = (1.0 - diversity_weight) * float(row.quality_score) + diversity_weight * diversity
            if score > best_score:
                best_index = row.Index
                best_score = score
                best_diversity = diversity
        record = remaining.loc[best_index].copy()
        record["selection_score"] = best_score
        record["diversity_score"] = best_diversity
        selected_rows.append(record)
        selected_embeddings.append(record["embedding"])
        remaining = remaining.drop(index=best_index)

    return pd.DataFrame(selected_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores-out", type=Path, required=True)
    parser.add_argument("--classes", default="disgust,fear,sad")
    parser.add_argument("--per-class", type=int, default=35)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--min-contrast", type=float, default=0.0)
    parser.add_argument("--min-sharpness", type=float, default=0.0)
    parser.add_argument("--max-clipped-fraction", type=float, default=1.0)
    parser.add_argument("--min-center-score", type=float, default=0.0)
    parser.add_argument("--min-symmetry-score", type=float, default=0.0)
    parser.add_argument("--min-quality-score", type=float, default=0.0)
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.15,
        help="Weight for within-class visual diversity during greedy selection.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.diversity_weight <= 1.0:
        raise ValueError("--diversity-weight must be between 0 and 1")

    target_labels = parse_classes(args.classes)
    summary = accepted_summary(args.summary)

    scored_rows = []
    for path in sorted(args.accepted_dir.glob("*.png")):
        idx = image_index(path)
        if idx is None or idx not in summary.index:
            continue
        meta = summary.loc[idx]
        label = int(meta["requested_label"])
        confidence = float(meta["confidence"])
        if label not in target_labels or confidence < args.min_confidence:
            continue
        row = {
            "source": str(path),
            "index": idx,
            "label": label,
            "class_name": EMOTION_NAMES[label],
            "confidence": confidence,
        }
        row.update(quality_metrics(path))
        row["quality_score"] = composite_score(row)
        if row["contrast"] < args.min_contrast:
            continue
        if row["sharpness"] < args.min_sharpness:
            continue
        if row["clipped_fraction"] > args.max_clipped_fraction:
            continue
        if row["center_score"] < args.min_center_score:
            continue
        if row["symmetry_score"] < args.min_symmetry_score:
            continue
        if row["quality_score"] < args.min_quality_score:
            continue
        row["embedding"] = diversity_embedding(path)
        scored_rows.append(row)

    if not scored_rows:
        raise ValueError(f"No accepted images could be scored from {args.accepted_dir}")

    scored = pd.DataFrame(scored_rows).sort_values(
        ["label", "quality_score", "confidence"],
        ascending=[True, False, False],
    )
    scored_out = scored.drop(columns=["embedding"])
    args.scores_out.parent.mkdir(parents=True, exist_ok=True)
    scored_out.to_csv(args.scores_out, index=False)

    available = scored.groupby("label").size().to_dict()
    missing = [label for label in target_labels if available.get(label, 0) == 0]
    if missing:
        raise ValueError(f"Cannot curate balanced subset with empty classes: {available}")

    per_class = min(args.per_class, min(available[label] for label in target_labels))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in target_labels:
        selected = select_diverse(scored[scored["label"] == label], per_class, args.diversity_weight)
        selected = selected.sort_values(["selection_score", "quality_score"], ascending=[False, False])
        class_dir = args.out_dir / f"{label}_{EMOTION_NAMES[label]}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for old_png in class_dir.glob("*.png"):
            old_png.unlink()
        for record in selected.to_dict("records"):
            src = Path(record["source"])
            dst = class_dir / src.name
            shutil.copy2(src, dst)
            rows.append(
                {
                    "path": str(dst),
                    "label": label,
                    "class_name": EMOTION_NAMES[label],
                    "source": str(src),
                    "confidence": record["confidence"],
                    "quality_score": record["quality_score"],
                    "contrast": record["contrast"],
                    "sharpness": record["sharpness"],
                    "clipped_fraction": record["clipped_fraction"],
                    "center_score": record["center_score"],
                    "symmetry_score": record["symmetry_score"],
                    "diversity_score": record.get("diversity_score", 1.0),
                    "selection_score": record.get("selection_score", record["quality_score"]),
                }
            )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"available: {available}")
    print(f"curated per class: {per_class}")
    print(f"wrote scores to {args.scores_out}")
    print(f"wrote {len(rows)} rows to {args.manifest}")


if __name__ == "__main__":
    main()
