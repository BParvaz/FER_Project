import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
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


def load_model(path: Path, device: torch.device) -> nn.Module:
    model = build_model().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def parse_classes(value: str):
    if value.lower() == "all":
        return set(EMOTION_NAMES)
    labels = set()
    name_to_id = {name: idx for idx, name in EMOTION_NAMES.items()}
    for item in value.split(","):
        item = item.strip().lower()
        if item:
            labels.add(int(item) if item.isdigit() else name_to_id[item])
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--target-classes", default="disgust,fear,sad")
    parser.add_argument("--max-per-class", type=int, default=0, help="0 means no cap")
    args = parser.parse_args()

    data = np.load(args.npz)
    images = data["arr_0"]
    labels = data["arr_1"]
    target_classes = parse_classes(args.target_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    accepted_counts = {label: 0 for label in EMOTION_NAMES}
    rows = []
    accepted_dir = args.out_dir / "accepted"
    rejected_dir = args.out_dir / "rejected_preview"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for idx, (array, requested_label) in enumerate(zip(images, labels)):
            requested_label = int(requested_label)
            pil_image = Image.fromarray(array).convert("RGB")
            tensor = transform(pil_image).unsqueeze(0).to(device)
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
            predicted_label = int(probs.argmax())
            confidence = float(probs[predicted_label])

            label_match = predicted_label == requested_label
            target_match = requested_label in target_classes
            under_cap = args.max_per_class <= 0 or accepted_counts[requested_label] < args.max_per_class
            accepted = label_match and target_match and confidence >= args.threshold and under_cap

            requested_name = EMOTION_NAMES[requested_label]
            filename = f"{requested_label}_{requested_name}_synthetic_{idx:05d}_conf_{confidence:.3f}.png"
            if accepted:
                pil_image.save(accepted_dir / filename)
                accepted_counts[requested_label] += 1
            elif idx < 50:
                pil_image.save(rejected_dir / filename)

            rows.append(
                {
                    "index": idx,
                    "requested_label": requested_label,
                    "requested_name": requested_name,
                    "predicted_label": predicted_label,
                    "predicted_name": EMOTION_NAMES[predicted_label],
                    "confidence": confidence,
                    "accepted": accepted,
                    "reason": "accepted"
                    if accepted
                    else f"label_match={label_match};target_match={target_match};under_cap={under_cap}",
                }
            )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.summary}")
    print("accepted counts:", {EMOTION_NAMES[k]: v for k, v in accepted_counts.items() if v})


if __name__ == "__main__":
    main()
