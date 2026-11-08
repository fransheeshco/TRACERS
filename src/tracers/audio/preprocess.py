"""CLI and orchestration for model-ready audio preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from tracers.charts.bestdori import tempo_changes_from_bestdori

from .beat_features import TempoChange, build_beat_features
from .config import AudioTensorConfig
from .loading import load_audio
from .spectrogram import extract_log_mel_specs


def preprocess_audio(
    audio_path: str | Path,
    tempo_changes: Sequence[TempoChange],
    offset_seconds: float = 0.0,
    config: AudioTensorConfig = AudioTensorConfig(),
) -> dict[str, torch.Tensor]:
    """Convert audio and timing metadata into model-ready CPU tensors."""

    audio = load_audio(audio_path, config.sample_rate)
    specs = extract_log_mel_specs(audio, config)
    beat_phase, beat_num = build_beat_features(
        frame_count=specs.shape[1],
        frame_rate=config.frame_rate,
        tempo_changes=tempo_changes,
        offset_seconds=offset_seconds,
    )
    frame_times = np.arange(specs.shape[1], dtype=np.float32) / np.float32(config.frame_rate)
    return {
        "specs": torch.from_numpy(specs),
        "beat_phase": torch.from_numpy(beat_phase),
        "beat_num": torch.from_numpy(beat_num).long(),
        "frame_times": torch.from_numpy(frame_times),
    }


def save_tensors(
    tensors: dict[str, torch.Tensor],
    output: str | Path,
    *,
    config: AudioTensorConfig = AudioTensorConfig(),
    offset_seconds: float = 0.0,
) -> None:
    """Save tensors as native PyTorch ``.pt`` or compatibility ``.npz``."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sample_rate": config.sample_rate,
        "hop_length": config.hop_length,
        "n_ffts": config.n_ffts,
        "n_mels": config.n_mels,
        "offset_seconds": offset_seconds,
    }
    if output.suffix == ".pt":
        torch.save({**tensors, "metadata": metadata}, output)
        return
    if output.suffix == ".npz":
        np.savez_compressed(
            output,
            **{name: tensor.detach().cpu().numpy() for name, tensor in tensors.items()},
            sample_rate=np.int32(config.sample_rate),
            hop_length=np.int32(config.hop_length),
            n_ffts=np.asarray(config.n_ffts, dtype=np.int32),
            n_mels=np.int32(config.n_mels),
            offset_seconds=np.float32(offset_seconds),
        )
        return
    raise ValueError("Output filename must end in .pt (recommended) or .npz")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert audio to DDC-style log-Mel and beat-grid tensors."
    )
    parser.add_argument("audio", type=Path, help="Input WAV, MP3, FLAC, OGG, or M4A file")
    parser.add_argument("output", type=Path, help="Output .pt (recommended) or .npz file")
    timing = parser.add_mutually_exclusive_group(required=True)
    timing.add_argument("--bpm", type=float, help="Constant BPM")
    timing.add_argument("--chart-json", type=Path, help="Bestdori chart containing BPM events")
    parser.add_argument(
        "--offset-seconds",
        type=float,
        default=0.0,
        help="Audio time of beat 0 (default: 0)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.chart_json:
        with args.chart_json.open("r", encoding="utf-8") as file:
            tempo_changes = tempo_changes_from_bestdori(json.load(file))
    else:
        tempo_changes = [TempoChange(beat=0.0, bpm=args.bpm)]

    try:
        tensors = preprocess_audio(args.audio, tempo_changes, args.offset_seconds)
        save_tensors(tensors, args.output, offset_seconds=args.offset_seconds)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"specs: {tensors['specs'].shape} {tensors['specs'].dtype}")
    print(f"beat_phase: {tensors['beat_phase'].shape} {tensors['beat_phase'].dtype}")
    print(f"beat_num: {tensors['beat_num'].shape} {tensors['beat_num'].dtype}")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
