# TRACERS

**TRACERS (Transformer-based Rhythm-Aware Chart Encoding and Rendering System)**
is a transformer-based rhythm game chart generation system developed as a
midterm project for CSC 173 – Intelligent Systems. The project explores the
application of transformer architectures in analyzing musical patterns and
automatically generating playable rhythm charts for Sonolus. It aims to reduce
manual chart creation while producing note sequences that align with a song's
rhythm, tempo, and musical structure.

This project is submitted in partial fulfillment of the requirements for
CSC 173 – Intelligent Systems.

## Project Members (Group 6)

- Cejas, Francis Nelson M.
- Haganas, Joshua Ray D.
- Saladaga Carlo James.

## Current pipeline

```text
Audio + BPM/offset
        |
        v
Audio preprocessing
        |
        +-- specs       (3, T, 80) torch.float32
        +-- beat_phase  (T,)       torch.int32
        +-- beat_num    (T,)       torch.int32
        |
        v
PyTorch Transformer (next stage)
        |
        v
Bestdori/Bandori chart JSON
        |
        v
Play through a compatible Sonolus server on a phone
```

`T` is the number of 10 ms frames. For example, `T = 9102` represents about
91.02 seconds of audio.

## Repository layout

```text
src/tracers/audio/   Audio loading and tensor preprocessing
src/tracers/charts/  Bestdori chart parsing utilities
tests/               Automated tests
examples/            Small, commit-safe example charts
data/                Local datasets and generated files (ignored by Git)
```

## Requirements

- Python 3.11 or newer
- macOS, Linux, or Windows

The project uses PyTorch, Torchaudio, and NumPy. Spectrogram extraction and
resampling follow the original AutoOsu converter's Torchaudio implementation.

## Setup

Clone the repository and enter it:

```bash
git clone https://github.com/fransheeshco/TRACERS.git
cd TRACERS
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install TRACERS and its test dependencies:

```bash
python -m pip install -e ".[test]"
```

## Preprocess audio

The preprocessor needs an audio file and timing information. Timing can come
from a Bestdori chart containing exactly one BPM event:

```bash
preprocess-audio \
  "path/to/song.m4a" \
  --chart-json "examples/first-chart.json" \
  --offset-seconds 0
```

If a song has a constant known BPM, provide it directly:

```bash
preprocess-audio \
  "path/to/song.m4a" \
  --bpm 120 \
  --offset-seconds 0
```

`offset-seconds` is the position in the audio at which beat 0 occurs. Like the
original AutoOsu converter, this conversion accepts one constant BPM and does
not estimate BPM or offset automatically.

The output filename is created automatically from the original audio filename.
For example, processing `My Original Song.m4a` produces:

```text
data/processed/My Original Song.pt
```

The preprocessor creates `data/processed/` when necessary. To use a custom
filename or location, provide it as the second positional argument:

```bash
preprocess-audio \
  "path/to/My Original Song.m4a" \
  "data/processed/custom-name.pt" \
  --bpm 120
```

### Output tensors

Load a generated file directly with PyTorch:

```python
import torch

data = torch.load(
    "data/processed/song.pt",
    map_location="cpu",
    weights_only=True,
)

specs = data["specs"]
beat_phase = data["beat_phase"]
beat_num = data["beat_num"]

print(specs.shape, specs.dtype)
print(beat_phase.shape, beat_phase.dtype)
print(beat_num.shape, beat_num.dtype)
```

For one-song model input, add a batch dimension:

```python
specs = specs.unsqueeze(0)            # (1, 3, T, 80)
beat_phase = beat_phase.unsqueeze(0)  # (1, T)
beat_num = beat_num.unsqueeze(0)      # (1, T)
```

The `.pt` dictionary contains exactly `specs`, `beat_phase`, and `beat_num`,
matching the original AutoOsu audio tensor format. A `.npz` output filename
can still be used when NumPy compatibility is required.

### Inspect a tensor file

Use the inspector to view tensor shapes and dtypes:

```bash
inspect-tensors "data/processed/My Original Song.pt"
```

The command also validates that all required tensors are present, their shapes
and dtypes match the TRACERS format, their frame counts agree, and their values
are finite. It reports an actionable error when the file is missing or invalid.

## Preprocessing representation

The audio representation follows the original AutoOsu converter:

| Setting | Value |
|---|---:|
| Sample rate | 44,100 Hz |
| Hop length | 441 samples |
| Frame rate | 100 frames/second |
| FFT sizes | 1024, 2048, 4096 |
| Mel bands | 80 |
| Frequency range | 0 Hz to 11,000 Hz |
| Spectral value | Natural-log Mel power |
| Window/padding | Torchaudio defaults (Hann/reflect) |

The three FFT sizes show the model short transients and longer musical context
on the same 100 Hz timeline. `beat_phase` is the original quantized 1/48-beat
index, and `beat_num` is the original measure-relative beat index in `[0, 4)`.

## Run tests

With the virtual environment active:

```bash
pytest
```

The tests cover AutoOsu beat quantization, rejection of tempo changes, Bestdori
BPM parsing, tensor shapes and dtypes, PyTorch saving/loading, and missing-file
errors.

## Data and Git

Do not commit songs, generated tensors, or model checkpoints. Place them under:

```text
data/raw/
data/processed/
data/checkpoints/
```

These paths and common audio/model file extensions are excluded by
`.gitignore`. This keeps copyrighted media and large generated artifacts out of
the repository.

## Project status

Implemented:

- Audio decoding with libsndfile and FFmpeg fallback
- Mono conversion and 44.1 kHz resampling
- Three-resolution log-Mel extraction
- Frame-aligned beat features
- Bestdori BPM parsing
- Native PyTorch `.pt` output

Not implemented yet:

- Bestdori chart-target tensors
- PyTorch Dataset and batching
- Transformer model and training loop
- Predicted tensor to Bestdori JSON conversion
