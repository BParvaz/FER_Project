import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw


EMOTION_NAMES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


class RefinementBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = 0.1 * out + identity
        return self.act(out)


class FlexibleGenerator(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, init_size: int, width: int = 512):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = init_size
        self.width = width
        self.fc = nn.Linear(latent_dim + num_classes, width * init_size * init_size)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(width, width // 2, 4, stride=2, padding=1),
            RefinementBlock(width // 2),
            nn.ConvTranspose2d(width // 2, width // 4, 4, stride=2, padding=1),
            RefinementBlock(width // 4),
            nn.ConvTranspose2d(width // 4, width // 8, 4, stride=2, padding=1),
            RefinementBlock(width // 8),
            RefinementBlock(width // 8),
            RefinementBlock(width // 8),
            nn.Conv2d(width // 8, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat((z, c), dim=1)
        out = self.fc(x)
        out = out.view(out.size(0), self.width, self.init_size, self.init_size)
        return self.net(out)


def parse_classes(value: str) -> list[int]:
    name_to_id = {name: idx for idx, name in EMOTION_NAMES.items()}
    labels = []
    for item in value.split(","):
        item = item.strip().lower()
        if item:
            labels.append(int(item) if item.isdigit() else name_to_id[item])
    return labels


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_init_size(state_dict: dict[str, torch.Tensor], width: int = 512) -> int:
    out_features = int(state_dict["fc.bias"].numel())
    spatial = out_features // width
    init_size = int(round(math.sqrt(spatial)))
    if width * init_size * init_size != out_features:
        raise ValueError(f"Cannot infer WGAN init_size from fc.bias length {out_features}")
    return init_size


def load_generator(checkpoint_path: Path, latent_dim: int, num_classes: int, device: torch.device) -> FlexibleGenerator:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["generator"] if isinstance(checkpoint, dict) and "generator" in checkpoint else checkpoint
    init_size = infer_init_size(state_dict)
    generator = FlexibleGenerator(latent_dim, num_classes, init_size=init_size).to(device)
    generator.load_state_dict(state_dict)
    generator.eval()
    print(f"loaded WGAN checkpoint {checkpoint_path} with init_size={init_size}", flush=True)
    return generator


def tensor_to_uint8_rgb(images: torch.Tensor) -> np.ndarray:
    images = images.detach().cpu().clamp(-1, 1)
    images = ((images + 1.0) * 127.5).round().to(torch.uint8)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    images = images.permute(0, 2, 3, 1).numpy()
    return images


def make_contact_sheet(images: np.ndarray, labels: np.ndarray, out: Path, limit: int = 120) -> None:
    images = images[:limit]
    labels = labels[:limit]
    columns = 10
    tile = 96
    label_height = 14
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * tile, rows * (tile + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, image in enumerate(images):
        x = (idx % columns) * tile
        y = (idx // columns) * (tile + label_height)
        tile_img = Image.fromarray(image).resize((tile, tile), Image.Resampling.NEAREST)
        sheet.paste(tile_img, (x, y))
        label = int(labels[idx])
        draw.text((x + 2, y + tile + 1), f"{label} {EMOTION_NAMES[label]}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--classes", default="disgust,fear,sad")
    parser.add_argument("--samples-per-class", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    labels_to_sample = parse_classes(args.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator(args.checkpoint, args.latent_dim, args.num_classes, device)

    all_images = []
    all_labels = []
    with torch.inference_mode():
        for label in labels_to_sample:
            remaining = args.samples_per_class
            while remaining > 0:
                batch_size = min(args.batch_size, remaining)
                z = torch.randn(batch_size, args.latent_dim, device=device)
                labels = torch.full((batch_size,), label, device=device, dtype=torch.long)
                images = generator(z, labels)
                all_images.append(tensor_to_uint8_rgb(images))
                all_labels.append(np.full((batch_size,), label, dtype=np.int64))
                remaining -= batch_size

    image_array = np.concatenate(all_images, axis=0)
    label_array = np.concatenate(all_labels, axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, image_array, label_array)

    if args.contact_sheet is not None:
        make_contact_sheet(image_array, label_array, args.contact_sheet)

    values, counts = np.unique(label_array, return_counts=True)
    print(f"wrote {args.out}")
    print({EMOTION_NAMES[int(label)]: int(count) for label, count in zip(values, counts)})


if __name__ == "__main__":
    main()
