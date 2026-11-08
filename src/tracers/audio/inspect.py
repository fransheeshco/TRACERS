"""Inspect and validate a preprocessed TRACERS tensor file."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import Sequence

import torch


_TENSOR_DTYPES = {
    "specs": torch.float32,
    "beat_phase": torch.int32,
    "beat_num": torch.int32,
}


def inspect_tensor_file(path: str | Path) -> dict[str, object]:
    """Load a ``.pt`` file and validate the AutoOsu audio tensor contract."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tensor file not found: {path}")
    if path.suffix != ".pt":
        raise ValueError(f"Expected a .pt file, received: {path}")

    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError("Tensor file must contain a dictionary")

    missing = [name for name in _TENSOR_DTYPES if name not in data]
    if missing:
        raise ValueError(f"Tensor file is missing required entries: {', '.join(missing)}")

    for name, expected_dtype in _TENSOR_DTYPES.items():
        tensor = data[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a PyTorch tensor")
        if tensor.dtype != expected_dtype:
            raise ValueError(f"{name} must use {expected_dtype}, received {tensor.dtype}")
        if tensor.is_floating_point() and not torch.all(torch.isfinite(tensor)):
            raise ValueError(f"{name} contains non-finite values")

    specs = data["specs"]
    if specs.ndim != 3 or specs.shape[0] != 3 or specs.shape[2] != 80:
        raise ValueError(f"specs must have shape (3, T, 80), received {tuple(specs.shape)}")

    frame_count = specs.shape[1]
    for name in ("beat_phase", "beat_num"):
        tensor = data[name]
        if tensor.shape != (frame_count,):
            raise ValueError(
                f"{name} must have shape ({frame_count},), received {tuple(tensor.shape)}"
            )

    beat_phase = data["beat_phase"]
    if torch.any((beat_phase < 0) | (beat_phase > 48)):
        raise ValueError("beat_phase values must be in the range [0, 48]")
    beat_num = data["beat_num"]
    if torch.any((beat_num < 0) | (beat_num >= 4)):
        raise ValueError("beat_num values must be in the range [0, 4)")

    return data


def print_tensor_report(path: str | Path, data: dict[str, object]) -> None:
    """Print a concise human-readable report for validated tensors."""

    print(f"File: {path}")
    print("Status: valid TRACERS tensor file")
    for name in _TENSOR_DTYPES:
        tensor = data[name]
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and validate a TRACERS PyTorch tensor file."
    )
    parser.add_argument("file", type=Path, help="Preprocessed .pt file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        data = inspect_tensor_file(args.file)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    print_tensor_report(args.file, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
