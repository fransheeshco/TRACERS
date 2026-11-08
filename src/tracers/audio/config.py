"""Configuration for audio feature extraction."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioTensorConfig:
    """Settings for the three-resolution DDC-style audio representation."""

    sample_rate: int = 44_100
    hop_length: int = 441
    n_ffts: tuple[int, ...] = (1024, 2048, 4096)
    n_mels: int = 80
    f_min: float = 27.5
    f_max: float = 16_000.0
    log_epsilon: float = 1e-16
    batch_frames: int = 512

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length
