from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

from tracers.audio import (
    AudioTensorConfig,
    TempoChange,
    build_beat_features,
    extract_log_mel_specs,
)
from tracers.audio.preprocess import default_output_path, preprocess_audio, save_tensors
from tracers.charts import tempo_changes_from_bestdori


def _write_sine_wave(path: Path, seconds: float = 1.0, sample_rate: int = 44_100) -> None:
    time = np.arange(round(seconds * sample_rate)) / sample_rate
    audio = (0.2 * np.sin(2 * np.pi * 440 * time) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(audio.tobytes())


def test_constant_tempo_features() -> None:
    phase, number = build_beat_features(
        frame_count=101,
        frame_rate=100,
        tempo_changes=[TempoChange(beat=0, bpm=120)],
    )

    assert phase.shape == (101,)
    assert number.shape == (101,)
    assert phase[0] == 0
    assert phase[25] == 24
    assert phase[50] == 0
    assert number[49] == 0
    assert number[50] == 1


def test_spectrogram_matches_original_autoosu_formula() -> None:
    config = AudioTensorConfig()
    time = np.arange(4_410, dtype=np.float32) / config.sample_rate
    audio = np.sin(2 * np.pi * 440 * time).astype(np.float32)

    actual = extract_log_mel_specs(audio, config)
    waveform = torch.from_numpy(audio)
    expected = torch.stack(
        [
            torch.log(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=44_100,
                    n_fft=n_fft,
                    hop_length=441,
                    f_max=11_000,
                    n_mels=80,
                    power=2,
                )(waveform + 1e-9).T
            )
            for n_fft in (1024, 2048, 4096)
        ]
    )

    assert np.array_equal(actual, expected.numpy())


def test_tempo_change_is_rejected_like_original_converter() -> None:
    with pytest.raises(ValueError, match="exactly one BPM"):
        build_beat_features(
            frame_count=301,
            frame_rate=100,
            tempo_changes=[TempoChange(beat=0, bpm=120), TempoChange(beat=4, bpm=60)],
        )


def test_bestdori_tempo_events() -> None:
    changes = tempo_changes_from_bestdori(
        [
            {"type": "Single", "lane": 3, "beat": 1},
            {"type": "BPM", "beat": 0, "bpm": 120},
            {"type": "BPM", "beat": 8, "bpm": 160},
        ]
    )

    assert changes == [TempoChange(beat=0, bpm=120), TempoChange(beat=8, bpm=160)]


def test_end_to_end_shapes(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_sine_wave(audio_path)

    tensors = preprocess_audio(
        audio_path,
        tempo_changes=[TempoChange(beat=0, bpm=120)],
        config=AudioTensorConfig(),
    )

    assert tensors["specs"].shape == (3, 101, 80)
    assert tensors["specs"].dtype == torch.float32
    assert tensors["beat_phase"].shape == (101,)
    assert tensors["beat_phase"].dtype == torch.int32
    assert tensors["beat_num"].shape == (101,)
    assert tensors["beat_num"].dtype == torch.int32
    assert torch.all(torch.isfinite(tensors["specs"]))


def test_save_and_load_pytorch_tensors(tmp_path: Path) -> None:
    tensors = {
        "specs": torch.zeros((3, 10, 80), dtype=torch.float32),
        "beat_phase": torch.zeros(10, dtype=torch.int32),
        "beat_num": torch.arange(10, dtype=torch.int32) % 4,
    }
    output = tmp_path / "song.pt"

    save_tensors(tensors, output)
    loaded = torch.load(output, map_location="cpu", weights_only=True)

    assert torch.equal(loaded["specs"], tensors["specs"])
    assert loaded["beat_num"].dtype == torch.int32
    assert list(loaded) == ["specs", "beat_phase", "beat_num"]


def test_default_output_path_preserves_audio_filename() -> None:
    output = default_output_path("/uploads/My Original Song.m4a")

    assert output == Path("data/processed/My Original Song.pt")


def test_missing_audio_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        preprocess_audio(
            tmp_path / "missing.wav",
            tempo_changes=[TempoChange(beat=0, bpm=120)],
        )
