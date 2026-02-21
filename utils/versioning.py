import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunPaths:
    logs_root: Path
    version: int
    tag: str
    version_dir: Path
    plots_dir: Path
    gif_dir: Path
    samples_dir: Path
    checkpoints_dir: Path
    metrics_csv: Path


def _latest_version(logs_root: Path) -> int:
    if not logs_root.exists():
        return 0

    pat = re.compile(r"^version[_\-\s]?(\d+)$", re.IGNORECASE)
    latest = 0
    for p in logs_root.iterdir():
        if p.is_dir():
            m = pat.match(p.name)
            if m:
                latest = max(latest, int(m.group(1)))
    return latest


def make_run_dirs(logs_root: str | Path = "logs", version: Optional[int] = None) -> RunPaths:
    logs_root = Path(logs_root)
    logs_root.mkdir(parents=True, exist_ok=True)

    if version is None:
        version = _latest_version(logs_root) + 1

    tag = f"{version:03d}"  # 001, 002, ...
    version_dir = logs_root / f"version_{tag}"

    # Create structure requested:
    # logs/version_X/(plots X, gif X, samples X, checkpoints X, metrics X)
    plots_dir = version_dir / f"plots_{tag}"
    gif_dir = version_dir / f"gif_{tag}"
    samples_dir = version_dir / f"samples_{tag}"
    checkpoints_dir = version_dir / f"checkpoints_{tag}"
    metrics_csv = version_dir / f"metrics_{tag}.csv"

    # Fail fast if you accidentally reuse a version
    version_dir.mkdir(parents=True, exist_ok=False)
    plots_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        logs_root=logs_root,
        version=version,
        tag=tag,
        version_dir=version_dir,
        plots_dir=plots_dir,
        gif_dir=gif_dir,
        samples_dir=samples_dir,
        checkpoints_dir=checkpoints_dir,
        metrics_csv=metrics_csv,
    )
