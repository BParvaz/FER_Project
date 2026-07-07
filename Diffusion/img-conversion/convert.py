import argparse
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

# emotion labels
emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}


def export_split(csv_path: Path, out_dir: Path, clean: bool = False) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    has_labels = "emotion" in df.columns
    for i, row in df.iterrows():
        pixels = np.array(row["pixels"].split(), dtype=np.uint8)
        image = Image.fromarray(pixels.reshape(48, 48))

        if has_labels:
            emotion_id = int(row["emotion"])
            emotion_name = emotion_map[emotion_id]
            # Numeric prefix keeps guided-diffusion class ids aligned with FER2013.
            filename = f"{emotion_id}_{emotion_name}_{i}.png"
        else:
            filename = f"unlabeled_{i}.png"

        image.save(out_dir / filename)

    print(f"done with {out_dir.name}: {len(df)} images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=Path("../../data/FER2013/train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("../../data/FER2013/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("fer_images"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    export_split(args.train_csv, args.output_dir / "train", clean=args.clean)
    export_split(args.test_csv, args.output_dir / "test", clean=args.clean)


if __name__ == "__main__":
    main()
