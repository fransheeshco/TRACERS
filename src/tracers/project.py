from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir


def setup_project(project_root: str | Path | None = None) -> dict:
    """Load the training config and return the common runtime state."""
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path.cwd().parent.resolve()
    )

    for candidate in (root, root / "src"):
        if str(candidate) not in sys.path:
            sys.path.append(str(candidate))

    config_dir = (root / "conf").resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="train")

    warnings.filterwarnings("ignore", message=".*nested tensors.*")

    return {
        "root": root,
        "cfg": cfg,
        "device": device,
        "experiment_config": cfg.experiment,
        "hyperparams": cfg.hyperparams,
        "segment_frames": cfg.hyperparams["segment_frames"],
        "batch_size": cfg.hyperparams["batch_size"],
    }
