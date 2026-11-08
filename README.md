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
        +-- beat_phase  (T,)       torch.float32
        +-- beat_num    (T,)       torch.int64
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

The project uses PyTorch, NumPy, SciPy, SoundFile, and a bundled FFmpeg binary
for formats such as M4A/AAC.

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
from the BPM events in a Bestdori chart:

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

`offset-seconds` is the position in the audio at which beat 0 occurs. The
current preprocessor does not estimate BPM or offset automatically; use chart
timing or verified song metadata.

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

The `.pt` file also contains `frame_times` and extraction metadata. A `.npz`
output filename can be used when NumPy compatibility is required.

## Preprocessing representation

The audio representation follows the three-resolution front end used by DDC:

| Setting | Value |
|---|---:|
| Sample rate | 44,100 Hz |
| Hop length | 441 samples |
| Frame rate | 100 frames/second |
| FFT sizes | 1024, 2048, 4096 |
| Mel bands | 80 |
| Frequency range | 27.5 Hz to 16,000 Hz |
| Spectral value | Natural-log Mel magnitude |

The three FFT sizes show the model short transients and longer musical context
on the same 100 Hz timeline. `beat_phase` describes progress through the
current beat in `[0, 1)`. `beat_num` is the absolute integer beat index.

## Run tests

With the virtual environment active:

```bash
pytest
```

The tests cover constant tempo, tempo changes, Bestdori BPM parsing, tensor
shapes and dtypes, PyTorch saving/loading, and missing-file errors.

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

- Audio decoding, including M4A/AAC fallback
- Mono conversion and 44.1 kHz resampling
- Three-resolution log-Mel extraction
- Frame-aligned beat features
- Bestdori BPM parsing
- Native PyTorch `.pt` output

Not implemented yet:

- Chart target tensors (`actions`, `onsets`, and encoded `beatmap`)
- PyTorch Dataset and batching
- Transformer model and training loop
- Predicted tensor to Bestdori JSON conversion
