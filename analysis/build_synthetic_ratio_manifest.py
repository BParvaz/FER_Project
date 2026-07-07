from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--min-per-class", type=int, default=1)
    parser.add_argument("--sort-column", default="quality_score")
    args = parser.parse_args()

    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")

    df = pd.read_csv(args.manifest)
    required = {"label", "class_name", "path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.manifest} missing columns: {sorted(missing)}")

    sort_column = args.sort_column if args.sort_column in df.columns else None
    rows = []
    counts = {}
    for label, group in df.groupby("label", sort=True):
        if sort_column is not None:
            group = group.sort_values(sort_column, ascending=False)
        else:
            group = group.sort_values("path")
        take = max(args.min_per_class, int(math.ceil(len(group) * args.fraction)))
        take = min(take, len(group))
        selected = group.head(take)
        rows.append(selected)
        counts[int(label)] = take

    out_df = pd.concat(rows, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"wrote {len(out_df)} rows to {args.out}")
    print(f"per-class counts: {counts}")


if __name__ == "__main__":
    main()
