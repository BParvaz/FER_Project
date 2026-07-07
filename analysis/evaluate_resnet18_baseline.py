import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import resnet18


EMOTION_NAMES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


class FERDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        if "emotion" not in dataframe.columns:
            raise ValueError("Evaluation requires an emotion column; this CSV is unlabeled.")
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
        image = Image.fromarray(pixels.reshape(48, 48)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["emotion"])


def build_model() -> nn.Module:
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 7),
    )
    return model


def stratified_indices(df: pd.DataFrame, val_pct: float, seed: int):
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for emotion in sorted(df["emotion"].unique()):
        indices = df.index[df["emotion"] == emotion].to_numpy()
        rng.shuffle(indices)
        n_val = int(len(indices) * val_pct)
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())
    return train_indices, val_indices


def evaluate_checkpoint(checkpoint_path: Path, dataloader: DataLoader, device: torch.device):
    model = build_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    report = classification_report(
        labels_np,
        preds_np,
        labels=list(EMOTION_NAMES.keys()),
        target_names=[EMOTION_NAMES[i] for i in EMOTION_NAMES],
        output_dict=True,
        zero_division=0,
    )

    row = {
        "checkpoint": str(checkpoint_path),
        "n": len(labels_np),
        "loss": total_loss / len(labels_np),
        "accuracy": float((preds_np == labels_np).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels_np, preds_np)),
        "micro_f1": float(f1_score(labels_np, preds_np, average="micro")),
        "macro_f1": float(f1_score(labels_np, preds_np, average="macro")),
        "weighted_f1": float(f1_score(labels_np, preds_np, average="weighted")),
    }
    for class_id, class_name in EMOTION_NAMES.items():
        row[f"{class_name}_recall"] = report[class_name]["recall"]
        row[f"{class_name}_f1"] = report[class_name]["f1-score"]
    predictions = pd.DataFrame(
        {
            "checkpoint": str(checkpoint_path),
            "y_true": labels_np,
            "y_pred": preds_np,
        }
    )
    return row, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/FER2013/train.csv"))
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("models"),
        help="Directory containing */_tmp.pth checkpoints, a directory of .pth files, or one checkpoint file.",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/resnet18_baseline_eval.csv"))
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--val-pct", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    _, val_indices = stratified_indices(df, args.val_pct, args.seed)

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = FERDataset(df, transform=val_transform)
    val_dataset = Subset(dataset, val_indices)
    dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoints.is_file():
        checkpoints = [args.checkpoints]
    else:
        checkpoints = sorted(args.checkpoints.glob("*/_tmp.pth"))
        if not checkpoints:
            checkpoints = sorted(args.checkpoints.glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {args.checkpoints}")

    rows = []
    prediction_frames = []
    for checkpoint in checkpoints:
        print(f"evaluating {checkpoint}", flush=True)
        row, predictions = evaluate_checkpoint(checkpoint, dataloader, device)
        rows.append(row)
        prediction_frames.append(predictions)

    rows = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.out}", flush=True)
    print("best checkpoint:", rows[0]["checkpoint"], flush=True)
    print(
        "best metrics:",
        {k: rows[0][k] for k in ["accuracy", "balanced_accuracy", "micro_f1", "macro_f1", "weighted_f1"]},
        flush=True,
    )

    if args.predictions_out is not None:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(prediction_frames, ignore_index=True).to_csv(args.predictions_out, index=False)
        print(f"wrote {args.predictions_out}", flush=True)


if __name__ == "__main__":
    main()
