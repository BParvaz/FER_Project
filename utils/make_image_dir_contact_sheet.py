import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--tile", type=int, default=96)
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()

    paths = sorted(args.image_dir.rglob("*.png"))[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No PNG files under {args.image_dir}")

    label_height = 14
    rows = math.ceil(len(paths) / args.columns)
    sheet = Image.new("RGB", (args.columns * args.tile, rows * (args.tile + label_height)), "white")
    draw = ImageDraw.Draw(sheet)

    for i, path in enumerate(paths):
        x = (i % args.columns) * args.tile
        y = (i // args.columns) * (args.tile + label_height)
        image = Image.open(path).convert("RGB").resize((args.tile, args.tile), Image.Resampling.NEAREST)
        sheet.paste(image, (x, y))
        draw.text((x + 2, y + args.tile + 1), path.stem[:18], fill="black")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
