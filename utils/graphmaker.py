#!/usr/bin/env python3
"""
plot_wgan_logs.py

Usage:
  python plot_wgan_logs.py /path/to/log.csv
  # or, from another script:
  from plot_wgan_logs import plot_wgan_csv
  plot_wgan_csv("log.csv", out_dir="plots")
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def plot_wgan_csv(
    csv_path: str | os.PathLike,
    out_dir: str | os.PathLike = "plots",
    smooth_window: int = 1,
    show: bool = False,
) -> Path:
    """
    Reads a WGAN-GP training log CSV and writes several PNG plots.

    Expected columns (case-sensitive):
      time, epoch, g_loss, d_loss, d_real, d_fake, gp, gap

    Args:
      csv_path: path to the CSV file
      out_dir: directory to write PNG files
      smooth_window: moving average window (>=1). 1 = no smoothing.
      show: if True, displays plots interactively

    Returns:
      Path to the output directory.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = ["epoch", "g_loss", "d_loss", "d_real", "d_fake", "gp", "gap"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Ensure numeric + sort by epoch
    for c in required + (["time"] if "time" in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)

    # Optional smoothing (moving average)
    def smooth(series: pd.Series) -> pd.Series:
        w = max(int(smooth_window), 1)
        if w <= 1:
            return series
        return series.rolling(window=w, min_periods=1, center=False).mean()

    epoch = df["epoch"]

    def save_plot(y_cols: list[str], title: str, filename: str) -> None:
        plt.figure()
        for c in y_cols:
            plt.plot(epoch, smooth(df[c]), label=c)
        plt.xlabel("epoch")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        if show:
            plt.show()
        plt.close()

    # 1) Losses
    save_plot(["g_loss", "d_loss"], "Generator / Critic Loss", "losses.png")

    # 2) Critic components
    save_plot(["d_real", "d_fake"], "Critic Scores: real vs fake", "critic_scores.png")

    # 3) Gradient penalty
    save_plot(["gp"], "Gradient Penalty (gp)", "gp.png")

    # 4) Gap (your logged metric)
    save_plot(["gap"], "Gap", "gap.png")

    # 5) One combined plot (optional convenience)
    save_plot(["g_loss", "d_loss", "gp", "gap"], "Overview", "overview.png")

    return out_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to the training log CSV")
    ap.add_argument("--out_dir", default="plots", help="Where to save PNGs")
    ap.add_argument("--smooth", type=int, default=1, help="Moving average window (1 = off)")
    ap.add_argument("--show", action="store_true", help="Show plots interactively")
    args = ap.parse_args()

    out = plot_wgan_csv(args.csv_path, out_dir=args.out_dir, smooth_window=args.smooth, show=args.show)
    print(f"Saved plots to: {out.resolve()}")