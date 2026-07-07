import argparse
import csv
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


METRICS = {
    "accuracy": lambda y, p: accuracy_score(y, p),
    "balanced_accuracy": lambda y, p: balanced_accuracy_score(y, p),
    "micro_f1": lambda y, p: f1_score(y, p, average="micro"),
    "macro_f1": lambda y, p: f1_score(y, p, average="macro"),
    "weighted_f1": lambda y, p: f1_score(y, p, average="weighted"),
}

EMOTION_NAMES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def class_recall(label: int):
    def metric(y, p):
        mask = y == label
        if not mask.any():
            return 0.0
        return float((p[mask] == label).mean())

    return metric


def class_f1(label: int):
    def metric(y, p):
        return f1_score(y, p, labels=[label], average="macro", zero_division=0)

    return metric


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    indices = np.arange(n)
    for _ in range(n_bootstrap):
        sample = rng.choice(indices, size=n, replace=True)
        scores.append(metric_fn(y_true[sample], y_pred[sample]))
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def exact_mcnemar_p(correct_a, correct_b):
    only_a = int(np.logical_and(correct_a, ~correct_b).sum())
    only_b = int(np.logical_and(~correct_a, correct_b).sum())
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return only_a, only_b, min(1.0, 2.0 * tail)


def load_predictions(path: Path):
    df = pd.read_csv(path)
    required = {"y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if "checkpoint" not in df.columns:
        df["checkpoint"] = str(path)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("reports/statistical_tests.csv"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.concat([load_predictions(path) for path in args.predictions], ignore_index=True)
    rows = []

    grouped = list(df.groupby("checkpoint", sort=False))
    for checkpoint, group in grouped:
        y_true = group["y_true"].to_numpy()
        y_pred = group["y_pred"].to_numpy()
        metric_items = list(METRICS.items())
        for label, name in EMOTION_NAMES.items():
            metric_items.append((f"{name}_recall", class_recall(label)))
            metric_items.append((f"{name}_f1", class_f1(label)))
        for metric_name, metric_fn in metric_items:
            score = float(metric_fn(y_true, y_pred))
            ci_low, ci_high = bootstrap_ci(y_true, y_pred, metric_fn, args.bootstrap, args.seed)
            rows.append(
                {
                    "test": "bootstrap_ci",
                    "checkpoint_a": checkpoint,
                    "checkpoint_b": "",
                    "metric": metric_name,
                    "score": score,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "statistic": "",
                    "p_value": "",
                    "n": len(group),
                }
            )

    if len(grouped) >= 2:
        for i, (checkpoint_a, group_a) in enumerate(grouped):
            for checkpoint_b, group_b in grouped[i + 1 :]:
                if len(group_a) != len(group_b) or not np.array_equal(
                    group_a["y_true"].to_numpy(), group_b["y_true"].to_numpy()
                ):
                    raise ValueError("Paired tests require prediction files with the same ordered y_true values.")
                y_true = group_a["y_true"].to_numpy()
                pred_a = group_a["y_pred"].to_numpy()
                pred_b = group_b["y_pred"].to_numpy()
                only_a, only_b, p_value = exact_mcnemar_p(pred_a == y_true, pred_b == y_true)
                rows.append(
                    {
                        "test": "mcnemar_exact",
                        "checkpoint_a": checkpoint_a,
                        "checkpoint_b": checkpoint_b,
                        "metric": "accuracy",
                        "score": "",
                        "ci_low": "",
                        "ci_high": "",
                        "statistic": f"only_a_correct={only_a};only_b_correct={only_b}",
                        "p_value": p_value,
                        "n": len(y_true),
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
