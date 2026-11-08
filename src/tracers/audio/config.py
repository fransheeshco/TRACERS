"""Configuration for audio feature extraction."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioTensorConfig:
    """Settings used by the original AutoOsu audio converter."""

    sample_rate: int = 44_100
    hop_length: int = 441
    n_ffts: tuple[int, ...] = (1024, 2048, 4096)
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 11_000.0
    power: float = 2.0
    waveform_epsilon: float = 1e-9
    beat_division: int = 48

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length
