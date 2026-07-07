import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


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
            raise ValueError("Training requires an emotion column; this CSV is unlabeled.")
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labels = self.df["emotion"].to_numpy(dtype=np.int64)
        self.images = np.stack(
            [
                np.fromstring(pixels, sep=" ", dtype=np.uint8).reshape(48, 48)
                for pixels in self.df["pixels"]
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[idx])


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_model(pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
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


def make_transforms():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    return train_transform, val_transform


def class_weight_tensor(df: pd.DataFrame, train_indices, device: torch.device):
    labels = df.iloc[train_indices]["emotion"].to_numpy()
    counts = np.bincount(labels, minlength=7).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def class_weight_tensor_from_df(df: pd.DataFrame, device: torch.device):
    labels = df["emotion"].to_numpy()
    counts = np.bincount(labels, minlength=7).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def synthetic_manifest_to_dataframe(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows = []
    for row in manifest.itertuples(index=False):
        image = Image.open(row.path).convert("L").resize((48, 48))
        pixels = " ".join(str(int(value)) for value in np.asarray(image, dtype=np.uint8).reshape(-1))
        rows.append({"emotion": int(row.label), "pixels": pixels, "source": "synthetic_diffusion"})
    return pd.DataFrame(rows)


def run_epoch(model, dataloader, loss_fn, device, optimizer=None, scheduler=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    grad_context = torch.enable_grad() if is_train else torch.inference_mode()
    with grad_context:
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            preds = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (preds == labels).sum().item()
            total += batch_size
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "balanced_accuracy": float(balanced_accuracy_score(all_labels_np, all_preds_np)),
        "macro_f1": float(f1_score(all_labels_np, all_preds_np, average="macro")),
        "weighted_f1": float(f1_score(all_labels_np, all_preds_np, average="weighted")),
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: dict, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
            "emotion_names": EMOTION_NAMES,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/FER2013/train.csv"))
    parser.add_argument("--synthetic-manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("models/resnet18_baseline_retrain"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-pct", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--class-weights", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    train_indices, val_indices = stratified_indices(df, args.val_pct, args.seed)
    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()
    synthetic_count = 0
    if args.synthetic_manifest is not None:
        synthetic_df = synthetic_manifest_to_dataframe(args.synthetic_manifest)
        synthetic_count = len(synthetic_df)
        train_df = pd.concat([train_df, synthetic_df], ignore_index=True)
    print(
        f"loaded {len(df)} labelled rows; train={len(train_df)} val={len(val_df)} synthetic_train={synthetic_count}",
        flush=True,
    )
    train_transform, val_transform = make_transforms()
    train_dataset = FERDataset(train_df, transform=train_transform)
    val_dataset = FERDataset(val_df, transform=val_transform)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )
    weight = class_weight_tensor_from_df(train_df, device) if args.class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    metrics_path = args.out_dir / "metrics.csv"
    best_path = args.out_dir / "best.pth"
    last_path = args.out_dir / "last.pth"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_balanced_accuracy",
        "train_macro_f1",
        "train_weighted_f1",
        "val_loss",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
    ]

    best_metric = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer, scheduler)
            val_metrics = run_epoch(model, val_loader, loss_fn, device)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_weighted_f1": train_metrics["weighted_f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
            }
            writer.writerow(row)
            f.flush()
            print(
                "epoch {epoch:03d} "
                "train_loss {train_loss:.4f} train_acc {train_accuracy:.4f} "
                "val_loss {val_loss:.4f} val_acc {val_accuracy:.4f} "
                "val_macro_f1 {val_macro_f1:.4f}".format(**row),
                flush=True,
            )

            save_checkpoint(last_path, model, optimizer, scheduler, epoch, row, args)
            if row["val_macro_f1"] > best_metric:
                best_metric = row["val_macro_f1"]
                best_epoch = epoch
                stale_epochs = 0
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, row, args)
                print(f"saved new best checkpoint at epoch {epoch}", flush=True)
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(
                        f"early stopping after epoch {epoch}; best epoch was {best_epoch}",
                        flush=True,
                    )
                    break

    print(f"best checkpoint: {best_path}", flush=True)
    print(f"best val_macro_f1: {best_metric:.4f}", flush=True)


if __name__ == "__main__":
    main()
