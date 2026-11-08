"""AutoOsu-compatible frame-aligned beat features."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True, order=True)
class TempoChange:
    beat: float
    bpm: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.beat):
            raise ValueError("Tempo-change beat must be finite")
        if not np.isfinite(self.bpm) or self.bpm <= 0:
            raise ValueError("BPM must be finite and greater than zero")


def build_beat_features(
    frame_count: int,
    frame_rate: float,
    tempo_changes: Sequence[TempoChange],
    offset_seconds: float = 0.0,
    beat_division: int = 48,
    meter: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Create AutoOsu's quantized beat phase and measure-relative beat number."""

    if frame_count < 0:
        raise ValueError("frame_count cannot be negative")
    if frame_rate <= 0 or not np.isfinite(frame_rate):
        raise ValueError("frame_rate must be finite and greater than zero")
    if len(tempo_changes) != 1:
        raise ValueError("Original AutoOsu conversion supports exactly one BPM")
    if beat_division <= 0:
        raise ValueError("beat_division must be greater than zero")
    if meter <= 0:
        raise ValueError("meter must be greater than zero")

    tempo = tempo_changes[0]
    frame_times_ms = torch.arange(frame_count) * (1000.0 / frame_rate)
    offset_ms = offset_seconds * 1000.0
    beat_length_ms = 60_000.0 / tempo.bpm

    phase = ((frame_times_ms - offset_ms) % beat_length_ms) / beat_length_ms
    beat_division_length = round(1 / beat_division, 5)
    beat_phase = (phase / beat_division_length).round().int()
    beat_num = (((frame_times_ms - offset_ms) / beat_length_ms) % meter).floor().int()
    return beat_phase.numpy(), beat_num.numpy()
