import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
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


def make_grid(images: np.ndarray, labels: Optional[np.ndarray], out_path: Path, columns: int, tile: int) -> None:
    rows = math.ceil(len(images) / columns)
    label_height = 14 if labels is not None else 0
    sheet = Image.new("RGB", (columns * tile, rows * (tile + label_height)), "white")
    draw = ImageDraw.Draw(sheet)

    for i, image in enumerate(images):
        x = (i % columns) * tile
        y = (i // columns) * (tile + label_height)
        tile_img = Image.fromarray(image).resize((tile, tile), Image.Resampling.NEAREST)
        sheet.paste(tile_img, (x, y))
        if labels is not None:
            label = int(labels[i])
            draw.text((x + 2, y + tile + 1), f"{label} {EMOTION_NAMES.get(label, '')}", fill="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--tile", type=int, default=96)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    data = np.load(args.npz_path)
    images = data["arr_0"]
    labels = data["arr_1"] if "arr_1" in data.files else None
    if args.limit is not None:
        images = images[: args.limit]
        labels = labels[: args.limit] if labels is not None else None

    out_path = args.out or args.npz_path.with_name("contact_sheet.png")
    make_grid(images, labels, out_path, args.columns, args.tile)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
