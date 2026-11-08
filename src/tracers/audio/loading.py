"""Audio decoding, mono conversion, and resampling."""

from pathlib import Path
import subprocess

from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


def load_audio(path: str | Path, target_sample_rate: int = 44_100) -> np.ndarray:
    """Load an audio file, mix it to mono, and resample it."""

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {path}. Replace the example path with the "
            "actual path to your WAV, MP3, FLAC, OGG, or M4A file."
        )

    try:
        audio, source_sample_rate = sf.read(path, always_2d=True, dtype="float32")
    except sf.LibsndfileError:
        # libsndfile does not decode common AAC/M4A files. The bundled FFmpeg
        # fallback emits mono float32 PCM at the target sample rate.
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

        mono = np.frombuffer(decoded.stdout, dtype="<f4").copy()
        source_sample_rate = target_sample_rate
        audio = mono[:, None]

    mono = audio.mean(axis=1, dtype=np.float32)
    if source_sample_rate != target_sample_rate:
        divisor = int(np.gcd(source_sample_rate, target_sample_rate))
        mono = resample_poly(
            mono,
            target_sample_rate // divisor,
            source_sample_rate // divisor,
        ).astype(np.float32, copy=False)

    if mono.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if not np.all(np.isfinite(mono)):
        raise ValueError(f"Audio contains non-finite samples: {path}")
    return mono
