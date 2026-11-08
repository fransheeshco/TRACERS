"""AutoOsu-compatible three-resolution log-Mel extraction."""

import numpy as np
import torch
import torchaudio

from .config import AudioTensorConfig


def extract_log_mel_specs(
    audio: np.ndarray,
    config: AudioTensorConfig = AudioTensorConfig(),
) -> np.ndarray:
    """Return the same ``(3, time, 80)`` representation as AutoOsu."""

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("audio must be a one-dimensional mono waveform")

    waveform = torch.from_numpy(audio)
    channels: list[torch.Tensor] = []
    for n_fft in config.n_ffts:
        converter = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=n_fft,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=config.power,
        )
        mel = converter(waveform + config.waveform_epsilon)
        channels.append(torch.log(mel.T))

    return torch.stack(channels, dim=0).numpy()
