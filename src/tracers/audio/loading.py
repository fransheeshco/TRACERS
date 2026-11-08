"""Audio loading, mono conversion, and AutoOsu-compatible resampling."""

from pathlib import Path
import subprocess

from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio(path: str | Path, target_sample_rate: int = 44_100) -> np.ndarray:
    """Decode normalized float audio, mix to mono, and resample with Torchaudio."""

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio, source_sample_rate = sf.read(path, always_2d=True, dtype="float32")
        mono = torch.from_numpy(audio).mean(dim=1)
    except sf.LibsndfileError:
        command = [
            get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(target_sample_rate),
            "pipe:1",
        ]
        try:
            decoded = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            detail = (
                error.stderr.decode("utf-8", errors="replace").strip()
                if isinstance(error, subprocess.CalledProcessError)
                else str(error)
            )
            raise ValueError(f"Could not decode audio file {path}: {detail}") from error
        mono = torch.from_numpy(np.frombuffer(decoded.stdout, dtype="<f4").copy())
        source_sample_rate = target_sample_rate

    if source_sample_rate != target_sample_rate:
        mono = torchaudio.functional.resample(mono, source_sample_rate, target_sample_rate)
    if mono.numel() == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if not mono.isfinite().all():
        raise ValueError(f"Audio contains non-finite samples: {path}")
    return mono.numpy()
