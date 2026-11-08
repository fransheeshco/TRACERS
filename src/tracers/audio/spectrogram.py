"""Three-resolution log-Mel spectrogram extraction."""

import numpy as np
from scipy.fft import rfft

from .config import AudioTensorConfig


def _hz_to_mel(frequency: np.ndarray | float) -> np.ndarray:
    frequency = np.asarray(frequency, dtype=np.float64)
    f_sp = 200.0 / 3
    mel = frequency / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(
        frequency >= min_log_hz,
        min_log_mel + np.log(np.maximum(frequency, min_log_hz) / min_log_hz) / logstep,
        mel,
    )


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    f_sp = 200.0 / 3
    frequency = f_sp * mel
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(
        mel >= min_log_mel,
        min_log_hz * np.exp(logstep * (mel - min_log_mel)),
        frequency,
    )


def _mel_filterbank(config: AudioTensorConfig, n_fft: int) -> np.ndarray:
    if config.f_max > config.sample_rate / 2:
        raise ValueError("f_max cannot exceed the Nyquist frequency")

    mel_edges = np.linspace(
        _hz_to_mel(config.f_min),
        _hz_to_mel(config.f_max),
        config.n_mels + 2,
    )
    hz_edges = _mel_to_hz(mel_edges)
    frequencies = np.fft.rfftfreq(n_fft, d=1 / config.sample_rate)
    lower = hz_edges[:-2, None]
    center = hz_edges[1:-1, None]
    upper = hz_edges[2:, None]
    rising = (frequencies[None, :] - lower) / np.maximum(center - lower, 1e-12)
    falling = (upper - frequencies[None, :]) / np.maximum(upper - center, 1e-12)
    filters = np.maximum(0.0, np.minimum(rising, falling))
    filters *= (2.0 / np.maximum(hz_edges[2:] - hz_edges[:-2], 1e-12))[:, None]
    return filters.astype(np.float32)


def _blackman_harris_62(length: int) -> np.ndarray:
    if length < 2:
        return np.ones(length, dtype=np.float32)
    n = np.arange(length, dtype=np.float64)
    angle = 2 * np.pi * n / (length - 1)
    window = 0.44959 - 0.49364 * np.cos(angle) + 0.05677 * np.cos(2 * angle)
    return window.astype(np.float32)


def _centered_frames(
    audio: np.ndarray,
    frame_length: int,
    hop_length: int,
    start: int,
    stop: int,
) -> np.ndarray:
    centers = np.arange(start, stop, dtype=np.int64) * hop_length
    offsets = np.arange(frame_length, dtype=np.int64) - frame_length // 2
    indices = centers[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < audio.size)
    frames = np.zeros((stop - start, frame_length), dtype=np.float32)
    frames[valid] = audio[indices[valid]]
    return frames


def extract_log_mel_specs(
    audio: np.ndarray,
    config: AudioTensorConfig = AudioTensorConfig(),
) -> np.ndarray:
    """Extract a ``(3, time, 80)`` float32 log-Mel representation."""

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("audio must be a one-dimensional mono waveform")

    frame_count = int(np.ceil(audio.size / config.hop_length))
    channels: list[np.ndarray] = []
    for n_fft in config.n_ffts:
        filterbank = _mel_filterbank(config, n_fft)
        window = _blackman_harris_62(n_fft)
        output = np.empty((frame_count, config.n_mels), dtype=np.float32)
        for start in range(0, frame_count, config.batch_frames):
            stop = min(start + config.batch_frames, frame_count)
            frames = _centered_frames(audio, n_fft, config.hop_length, start, stop)
            magnitude = np.abs(rfft(frames * window[None, :], axis=1))
            mel = magnitude @ filterbank.T
            output[start:stop] = np.log(mel + config.log_epsilon).astype(np.float32)
        channels.append(output)
    return np.stack(channels, axis=0)
