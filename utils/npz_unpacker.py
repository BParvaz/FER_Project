import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def unpack_npz(npz_path: Path, out_dir: Path, limit: Optional[int] = None) -> None:
    data = np.load(npz_path)
    print(data.files)

    images = data["arr_0"]
    labels = data["arr_1"] if "arr_1" in data.files else None
    print(images.shape)

    if labels is not None:
        values, counts = np.unique(labels, return_counts=True)
        print(dict(zip(values.tolist(), counts.tolist())))

    out_dir.mkdir(parents=True, exist_ok=True)
    max_images = len(images) if limit is None else min(limit, len(images))

    for i, img in enumerate(images[:max_images]):
        prefix = f"{int(labels[i])}_" if labels is not None else ""
        Image.fromarray(img).save(out_dir / f"{prefix}sample_{i}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    out_dir = args.out_dir or args.npz_path.with_suffix("")
    unpack_npz(args.npz_path, out_dir, args.limit)


if __name__ == "__main__":
    main()
