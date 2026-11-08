"""Frame-aligned beat phase and beat number features."""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class TempoChange:
    beat: float
    bpm: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.beat):
            raise ValueError("Tempo-change beat must be finite")
        if not np.isfinite(self.bpm) or self.bpm <= 0:
            raise ValueError("BPM must be finite and greater than zero")


def _prepare_tempo_map(tempo_changes: Iterable[TempoChange]) -> tuple[np.ndarray, np.ndarray]:
    changes = sorted(tempo_changes)
    if not changes:
        raise ValueError("At least one tempo change is required")
    beats = np.asarray([change.beat for change in changes], dtype=np.float64)
    bpms = np.asarray([change.bpm for change in changes], dtype=np.float64)
    if np.any(np.diff(beats) <= 0):
        raise ValueError("Tempo-change beats must be unique and strictly increasing")
    return beats, bpms


def build_beat_features(
    frame_count: int,
    frame_rate: float,
    tempo_changes: Sequence[TempoChange],
    offset_seconds: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create beat phase and absolute beat number for every audio frame."""

    if frame_count < 0:
        raise ValueError("frame_count cannot be negative")
    if frame_rate <= 0 or not np.isfinite(frame_rate):
        raise ValueError("frame_rate must be finite and greater than zero")
    if not np.isfinite(offset_seconds):
        raise ValueError("offset_seconds must be finite")

    beats, bpms = _prepare_tempo_map(tempo_changes)
    change_times = np.empty_like(beats)
    change_times[0] = offset_seconds + beats[0] * 60.0 / bpms[0]
    for index in range(1, beats.size):
        change_times[index] = (
            change_times[index - 1]
            + (beats[index] - beats[index - 1]) * 60.0 / bpms[index - 1]
        )

    frame_times = np.arange(frame_count, dtype=np.float64) / frame_rate
    segment = np.searchsorted(change_times, frame_times, side="right") - 1
    segment = np.clip(segment, 0, beats.size - 1)
    continuous_beat = beats[segment] + (
        frame_times - change_times[segment]
    ) * bpms[segment] / 60.0
    beat_num = np.floor(continuous_beat).astype(np.int32)
    beat_phase = (continuous_beat - beat_num).astype(np.float32)
    beat_phase %= np.float32(1.0)
    return beat_phase, beat_num
