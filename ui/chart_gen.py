from __future__ import annotations

import inspect
import queue
import sys
import threading
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - handled by the app entry point
    gr = None
    GRADIO_IMPORT_ERROR = exc
else:
    GRADIO_IMPORT_ERROR = None

import torch

from tracers.animations.chart_preview import NoteAssets, render_progress
from tracers.audio.beat_features import TempoChange
from tracers.audio.config import AudioTensorConfig
from tracers.audio.preprocess import preprocess_audio, save_tensors
from tracers.export.chart import export_generated_chart
from tracers.generate.converter import export_generation_debug
from tracers.pipelines.generate import run_custom_generation
from tracers.pipelines.trainer_factory import build_trainer
from tracers.project import setup_project

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "notebooks" / "checkpoints" / "tracer_np" / "best_hope.pt"
)
DEFAULT_SEGMENT_FRAMES = 2000

MODEL_STATE: dict | None = None
MODEL_LOCK = threading.Lock()


APP_CSS = r"""
:root {
    --tr-canvas: #14131A;
    --tr-surface: #1D1B26;
    --tr-skeleton: #232030;
    --tr-button: #262333;
    --tr-active: #2B273A;
    --tr-button-hover: #312D42;
    --tr-header-divider: #262233;
    --tr-divider: #2B2738;
    --tr-border: #2E2A3D;
    --tr-button-border: #353047;
    --tr-switch-border: #38334A;
    --tr-input-hover: #423D57;
    --tr-pink: #FF66AA;
    --tr-pink-hover: #FF7DB6;
    --tr-pink-pressed: #EB5297;
    --tr-pink-text: #FF7EB8;
    --tr-download: #FF8DC0;
    --tr-cyan: #38BDF8;
    --tr-text: #F4F3F8;
    --tr-label: #E5E3F0;
    --tr-option: #C7C3D9;
    --tr-status: #B8B4CC;
    --tr-secondary: #A6A2BA;
    --tr-secondary-alt: #A29DB8;
    --tr-subtitle: #9E99B3;
    --tr-helper: #8E89A6;
    --tr-helper-alt: #8B86A3;
    --tr-file-meta: #837E99;
    --tr-muted: #6E6A85;
    --tr-placeholder: #65607A;
    --tr-line-number: #58536D;
    --tr-success: #34D399;
    --tr-error: #F87171;
    --tr-error-text: #FCA5A5;
    --tr-json-key: #7DD3FC;
    --tr-json-string: #F9A8D4;
    --tr-json-boolean: #6EE7B7;
    --tr-json-number: #FCD34D;
    --tr-json-null: #A78BFA;
    --tr-json-punctuation: #7A7591;
    color-scheme: dark;
}

body,
.gradio-container {
    background: var(--tr-canvas) !important;
    color: var(--tr-text) !important;
}

.gradio-container {
    max-width: none !important;
    min-height: 100vh;
    padding: 0 !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif !important;
}

#tracer-shell {
    width: min(1480px, calc(100% - 48px));
    margin: 0 auto;
    padding: 32px 0 56px;
}

#tracer-header {
    margin: 0 !important;
    padding: 0 0 26px !important;
    border: 0 !important;
    border-bottom: 1px solid var(--tr-header-divider) !important;
    background: transparent !important;
}

.tracer-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.tracer-mark {
    position: relative;
    width: 30px;
    height: 30px;
    flex: 0 0 30px;
    border: 3px solid var(--tr-pink);
    border-radius: 999px;
}

.tracer-mark::before {
    content: "";
    position: absolute;
    inset: 5px;
    border: 3px solid var(--tr-cyan);
    border-radius: inherit;
}

.tracer-mark::after {
    content: "";
    position: absolute;
    inset: 11px;
    background: var(--tr-text);
    border-radius: inherit;
}

.tracer-title {
    margin: 0;
    color: var(--tr-text);
    font-size: clamp(1.45rem, 2.4vw, 2rem);
    font-weight: 720;
    letter-spacing: -0.035em;
}

.tracer-subtitle {
    max-width: 700px;
    margin: 6px 0 0 44px;
    color: var(--tr-subtitle);
    font-size: 0.94rem;
    line-height: 1.55;
}

#config-grid,
#output-grid {
    align-items: stretch;
    gap: 20px;
    margin-top: 24px;
}

.tracer-card {
    min-width: 0 !important;
    padding: 24px !important;
    border: 1px solid var(--tr-border) !important;
    border-radius: 16px !important;
    background: var(--tr-surface) !important;
    box-shadow: none !important;
}

.section-heading {
    margin: 0 0 20px !important;
    padding: 0 0 18px !important;
    border-bottom: 1px solid var(--tr-divider) !important;
    background: transparent !important;
}

.section-heading h2 {
    margin: 0 0 5px !important;
    color: var(--tr-text) !important;
    font-size: 1.08rem !important;
    font-weight: 680 !important;
    letter-spacing: -0.015em;
}

.section-heading p {
    margin: 0 !important;
    color: var(--tr-helper) !important;
    font-size: 0.86rem !important;
    line-height: 1.5 !important;
}

#preview-card .section-heading,
#json-card .section-heading {
    border-bottom: 0 !important;
}

.tracer-card .form,
.tracer-card .block,
.tracer-card .wrap,
.tracer-card label,
.tracer-card fieldset {
    background: transparent;
}

.tracer-card label > span,
.tracer-card .label-wrap span,
.tracer-card legend,
.tracer-card .label {
    color: var(--tr-label) !important;
    font-weight: 560 !important;
}

.tracer-card input,
.tracer-card textarea,
.tracer-card select,
#generated_chart_json_text textarea,
#status-output textarea {
    color: var(--tr-text) !important;
    border-color: var(--tr-border) !important;
    background: var(--tr-canvas) !important;
    box-shadow: none !important;
}

.tracer-card input:hover,
.tracer-card textarea:hover,
.tracer-card select:hover,
.tracer-card input:focus,
.tracer-card textarea:focus,
.tracer-card select:focus {
    border-color: var(--tr-input-hover) !important;
}

.tracer-card input:focus-visible,
.tracer-card textarea:focus-visible,
.tracer-card button:focus-visible,
#generate-chart-button:focus-visible {
    outline: 2px solid var(--tr-cyan) !important;
    outline-offset: 2px !important;
}

.tracer-card input::placeholder,
.tracer-card textarea::placeholder {
    color: var(--tr-placeholder) !important;
}

.tracer-card input[type="range"],
.tracer-card input[type="checkbox"] {
    accent-color: var(--tr-pink) !important;
}

#generate-chart-button {
    min-height: 52px;
    margin-top: 20px !important;
    color: #14131A !important;
    border: 1px solid var(--tr-pink) !important;
    border-radius: 12px !important;
    background: var(--tr-pink) !important;
    font-size: 1rem !important;
    font-weight: 720 !important;
    box-shadow: none !important;
    transition: background-color 140ms ease, border-color 140ms ease,
        transform 140ms ease;
}

#generate-chart-button:hover {
    border-color: var(--tr-pink-hover) !important;
    background: var(--tr-pink-hover) !important;
}

#generate-chart-button:active {
    border-color: var(--tr-pink-pressed) !important;
    background: var(--tr-pink-pressed) !important;
    transform: translateY(1px);
}

#generate-chart-button:disabled {
    color: var(--tr-status) !important;
    border-color: var(--tr-button-border) !important;
    background: var(--tr-button) !important;
}

#preview-card {
    overflow: hidden;
}

#preview-card .image-container,
#preview-card [data-testid="image"] {
    min-height: 420px;
    border: 1px solid var(--tr-border) !important;
    border-radius: 12px !important;
    background: var(--tr-canvas) !important;
}

#preview-card img {
    width: 100%;
    object-fit: contain;
}

#result-rail {
    gap: 20px;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

#status-card,
#json-card {
    width: 100%;
}

#status-output,
#download-json {
    margin-top: 10px;
}

#status-output textarea {
    min-height: 74px !important;
    color: var(--tr-status) !important;
    resize: none;
}

#download-json button,
#generated_chart_json_text button {
    color: var(--tr-download) !important;
    border: 1px solid rgba(255, 102, 170, 0.32) !important;
    background: rgba(255, 102, 170, 0.16) !important;
}

#download-json button:hover,
#generated_chart_json_text button:hover {
    background: rgba(255, 102, 170, 0.24) !important;
}

#generated_chart_json_text textarea {
    min-height: 360px !important;
    color: var(--tr-status) !important;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.65 !important;
}

.tracer-card .file-preview,
.tracer-card [data-testid="file"] {
    border-color: var(--tr-border) !important;
    background: var(--tr-canvas) !important;
}

.tracer-card button:not(#generate-chart-button) {
    color: var(--tr-label);
    border-color: var(--tr-button-border);
    background: var(--tr-button);
}

.tracer-card button:not(#generate-chart-button):hover {
    background: var(--tr-button-hover);
}

footer {
    display: none !important;
}

@media (min-width: 901px) and (min-height: 620px) {
    body,
    .gradio-container {
        height: 100dvh;
        overflow: hidden !important;
    }

    #tracer-shell {
        display: grid !important;
        grid-template-columns: minmax(410px, 0.9fr) minmax(0, 2.1fr);
        grid-template-rows: 56px minmax(0, 1fr) 46px;
        grid-template-areas:
            "header header"
            "config output"
            "action output";
        gap: 12px 16px;
        width: min(1600px, calc(100% - 24px));
        height: calc(100dvh - 32px);
        padding: 10px 0;
        overflow: hidden;
    }

    #tracer-header {
        grid-area: header;
        display: grid;
        grid-template-columns: auto 1fr;
        align-items: center;
        column-gap: 20px;
        min-height: 0;
        padding: 0 0 8px !important;
    }

    .tracer-mark {
        width: 26px;
        height: 26px;
        flex-basis: 26px;
    }

    .tracer-title {
        font-size: 1.42rem;
    }

    .tracer-subtitle {
        align-self: center;
        max-width: none;
        margin: 0;
        font-size: 0.8rem;
        line-height: 1.35;
        text-align: right;
    }

    #config-grid {
        grid-area: config;
        display: grid !important;
        grid-template-columns: 1fr;
        grid-template-rows: 175px minmax(0, 1fr);
        gap: 10px;
        min-height: 0;
        margin: 0;
        overflow: hidden;
    }

    #audio-card,
    #parameters-card {
        width: 100%;
        min-height: 0;
        overflow: hidden;
    }

    #generate-chart-button {
        grid-area: action;
        min-height: 46px;
        margin: 0 !important;
        border-radius: 10px !important;
    }

    #output-grid {
        grid-area: output;
        display: grid !important;
        grid-template-columns: minmax(0, 1.8fr) minmax(300px, 0.75fr);
        gap: 12px;
        min-height: 0;
        margin: 0;
        overflow: hidden;
    }

    .tracer-card {
        padding: 10px !important;
        border-radius: 13px !important;
    }

    .section-heading {
        margin: 0 0 5px !important;
        padding: 0 0 5px !important;
    }

    .section-heading h2 {
        margin: 0 !important;
        font-size: 0.92rem !important;
    }

    .section-heading p {
        display: none;
    }

    #audio-card,
    #parameters-card,
    #preview-card,
    #json-card {
        flex-wrap: nowrap !important;
    }

    #audio-card {
        gap: 6px;
        padding: 12px !important;
    }

    #parameters-card {
        gap: 12px;
        padding: 12px !important;
    }

    #preview-card,
    #json-card {
        gap: 4px;
    }

    #audio-card .block,
    #parameters-card .block {
        min-height: 0 !important;
    }

    #audio-card .form,
    #parameters-card .form,
    #status-actions .form {
        min-width: 0 !important;
    }

    #audio-card > .block:not(.section-heading) {
        flex: 0 0 68px !important;
        height: 68px !important;
        min-height: 68px !important;
    }

    #audio-card > .block:not(.section-heading) .upload-container {
        height: 100% !important;
        min-height: 0 !important;
    }

    #audio-card [data-testid="upload-text"] {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: center;
        gap: 6px;
        padding-top: 17px;
        font-size: 0;
    }

    #audio-card [data-testid="upload-text"]::after {
        content: "Drop or choose audio / .pt";
        color: var(--tr-secondary-alt);
        font-size: 0.76rem;
    }

    #audio-card [data-testid="upload-icon"] {
        width: 18px;
        height: 18px;
    }

    #audio-card [data-testid="upload-text"] .or {
        display: none;
    }

    #timing-fields {
        flex: 0 0 43px !important;
        min-height: 43px !important;
    }

    #parameters-card > .form {
        flex: 0 0 67px !important;
        min-height: 67px !important;
    }

    #parameters-card > .form > .block {
        overflow: visible !important;
    }

    #parameters-card > .row {
        flex: 0 0 38px !important;
        min-height: 38px !important;
    }

    #audio-card .block.padded:not(.section-heading),
    #parameters-card .block.padded:not(.section-heading) {
        padding: 4px 8px !important;
    }

    #audio-card label.container,
    #parameters-card label.container {
        display: grid !important;
        grid-template-columns: auto minmax(0, 1fr);
        align-items: center;
        gap: 7px;
        height: 100%;
    }

    #audio-card [data-testid="block-info"],
    #parameters-card [data-testid="block-info"] {
        margin: 0 !important;
        font-size: 0.7rem !important;
        white-space: nowrap;
    }

    #audio-card input,
    #parameters-card input {
        min-height: 28px !important;
        padding-top: 3px !important;
        padding-bottom: 3px !important;
    }

    #audio-card .upload-container,
    #audio-card [data-testid="file-upload"] {
        min-height: 0 !important;
    }

    #parameters-card input[type="range"] {
        --slider-color: var(--tr-pink);
        accent-color: var(--tr-pink) !important;
        background: linear-gradient(
            to right,
            var(--tr-pink) 0 var(--range_progress),
            var(--tr-button-border) var(--range_progress) 100%
        ) !important;
    }

    #parameters-card input[type="range"]::-webkit-slider-thumb {
        width: 22px !important;
        height: 22px !important;
        border: 2px solid #C7C3D9 !important;
        border-radius: 50% !important;
        background: #F4F3F8 !important;
        box-shadow: none !important;
    }

    #parameters-card input[type="range"]::-webkit-slider-runnable-track {
        height: 8px !important;
        border-radius: 999px !important;
    }

    #parameters-card input[type="range"]::-moz-range-progress {
        background: var(--tr-pink) !important;
    }

    #parameters-card input[type="range"]::-moz-range-thumb {
        width: 22px !important;
        height: 22px !important;
        border: 2px solid #C7C3D9 !important;
        border-radius: 50% !important;
        background: #F4F3F8 !important;
    }

    #timing-fields,
    #seed-options,
    #status-actions {
        gap: 8px;
    }

    #seed-options > :last-child {
        align-self: end;
        min-height: 38px;
    }

    #preview-card {
        display: flex !important;
        flex-direction: column;
        height: 100%;
        min-height: 0;
        overflow: hidden;
    }

    #preview-card .image-container,
    #preview-card [data-testid="image"] {
        min-height: 0 !important;
        height: 100% !important;
    }

    #preview-card > .block:not(.section-heading) {
        flex: 1 1 auto !important;
        min-height: 0 !important;
        height: auto !important;
    }

    #result-rail {
        display: block !important;
        height: 100%;
        min-height: 0;
        overflow: hidden;
    }

    #json-card {
        display: flex !important;
        flex-direction: column;
        width: 100%;
        height: 100%;
        min-height: 0;
        overflow: hidden;
    }

    #json-toolbar {
        flex-wrap: nowrap !important;
        flex: 0 0 42px !important;
        min-height: 42px !important;
        align-items: center;
        gap: 8px;
    }

    #download-json {
        flex: 0 0 132px !important;
        width: 132px;
        min-width: 132px;
        height: 38px;
        min-height: 38px !important;
        margin: 0;
        overflow: hidden;
    }

    #download-json .empty,
    #download-json [data-testid="file"] {
        min-height: 36px !important;
        height: 36px !important;
        padding: 3px 7px !important;
    }

    #generated_chart_json_text,
    #generated_chart_json_text > div {
        flex: 1 1 auto !important;
        min-height: 0 !important;
        height: 100% !important;
    }

    #generated_chart_json_text textarea {
        min-height: 0 !important;
        height: 100% !important;
        font-size: 0.72rem !important;
        line-height: 1.45 !important;
    }
}

@media (max-width: 900px) {
    #tracer-shell {
        width: min(100% - 28px, 720px);
        padding-top: 22px;
    }

    #config-grid,
    #output-grid {
        flex-direction: column;
    }

    .tracer-card {
        padding: 20px !important;
    }

    #preview-card .image-container,
    #preview-card [data-testid="image"] {
        min-height: 300px;
    }
}

@media (max-width: 520px) {
    #tracer-shell {
        width: min(100% - 20px, 480px);
    }

    .tracer-subtitle {
        margin-left: 0;
    }

    .tracer-card {
        padding: 16px !important;
        border-radius: 13px !important;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
    }
}
"""


def _format_tag(value):
    if value is None:
        return "none"
    return str(value).replace(".", "p")


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def build_chart_output_path(
    output_root: str | Path,
    audio_name: str,
    difficulty,
    context_frames,
    temperature,
    seed,
) -> Path:
    output_dir = Path(output_root) / audio_name
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"chart_diff{difficulty}_ctx{context_frames}"
        f"_temp{_format_tag(temperature)}_seed{seed}.json"
    )
    chart_path = output_dir / filename
    reserved = _unique_path(chart_path)
    reserved.touch(exist_ok=True)
    return reserved


def validate_preprocessed_tensor(data) -> dict[str, torch.Tensor]:
    if not isinstance(data, dict):
        raise ValueError("Uploaded .pt file must contain a dictionary of tensors.")

    required_keys = ("specs", "beat_phase", "beat_num")
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"Missing required tensor keys: {', '.join(missing)}")

    validated = {}
    for key in required_keys:
        value = data[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected '{key}' to be a torch.Tensor.")
        validated[key] = value

    if validated["specs"].ndim != 3:
        raise ValueError("'specs' must have shape [channels, frames, features].")
    if validated["beat_phase"].ndim != 1:
        raise ValueError("'beat_phase' must be a 1D tensor.")
    if validated["beat_num"].ndim != 1:
        raise ValueError("'beat_num' must be a 1D tensor.")
    if validated["specs"].shape[1] != validated["beat_phase"].shape[0]:
        raise ValueError("'specs' frame count must match 'beat_phase' length.")
    if validated["specs"].shape[1] != validated["beat_num"].shape[0]:
        raise ValueError("'specs' frame count must match 'beat_num' length.")

    return validated


def _ensure_model_loaded() -> dict:
    global MODEL_STATE
    with MODEL_LOCK:
        if MODEL_STATE is not None:
            return MODEL_STATE

        if not DEFAULT_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {DEFAULT_CHECKPOINT}. "
                "Please ensure the trained model exists before launching the UI."
            )

        setup = setup_project(PROJECT_ROOT)
        trainer, model = build_trainer(
            model_key="tracer_np",
            hyperparams=setup["hyperparams"],
            experiment_config=setup["experiment_config"],
            device=setup["device"],
            checkpoint_path=str(DEFAULT_CHECKPOINT),
            train_loader=None,
            valid_loader=None,
        )

        MODEL_STATE = {
            "setup": setup,
            "trainer": trainer,
            "model": model,
            "segment_frames": setup.get("segment_frames", DEFAULT_SEGMENT_FRAMES),
        }
        return MODEL_STATE


def _sanitize_audio_name(name: str) -> str:
    cleaned = Path(name).stem
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in cleaned) or "song"


def _coerce_optional_int(value, *, allow_zero: bool = False):
    if value is None or value == "" or str(value).strip().lower() == "none":
        return None
    value = int(float(value))
    if not allow_zero and value <= 0:
        return None
    return value


def _coerce_optional_float(value, *, allow_zero: bool = False):
    if value is None or value == "" or str(value).strip().lower() == "none":
        return None
    value = float(value)
    if not allow_zero and value <= 0:
        return None
    return value


def _prepare_preprocessed_pt(
    upload_path: str | None,
    bpm: float,
    offset_ms: float,
) -> tuple[Path, str]:
    if upload_path is None:
        raise ValueError("Please upload an audio file or a preprocessed .pt tensor.")

    source_path = Path(upload_path)
    if not source_path.exists():
        raise ValueError(f"Input file not found: {source_path}")

    if source_path.suffix.lower() == ".pt":
        payload = torch.load(source_path, map_location="cpu", weights_only=True)
        validate_preprocessed_tensor(payload)
        return source_path, _sanitize_audio_name(source_path.name)

    output_dir = PROJECT_ROOT / "data" / "processed" / "ui_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_path.stem}_bpm{float(bpm)}_offset{int(offset_ms)}.pt"
    output_path = _unique_path(output_path)

    tempo_changes = [TempoChange(beat=0.0, bpm=float(bpm))]
    audio_tensors = preprocess_audio(
        source_path,
        tempo_changes,
        offset_seconds=float(offset_ms) / 1000.0,
        config=AudioTensorConfig(),
    )
    save_tensors(audio_tensors, output_path)
    return output_path, _sanitize_audio_name(source_path.name)


def _configure_queue(app):
    queue_kwargs = {}
    try:
        signature = inspect.signature(app.queue)
    except (TypeError, ValueError):
        signature = None

    if signature is not None and "concurrency_count" in signature.parameters:
        queue_kwargs["concurrency_count"] = 1
    elif signature is not None and "default_concurrency_limit" in signature.parameters:
        queue_kwargs["default_concurrency_limit"] = 1
    else:
        queue_kwargs["concurrency_count"] = 1

    try:
        return app.queue(**queue_kwargs)
    except TypeError:
        return app.queue(default_concurrency_limit=1)


def _make_debug_path(output_dir: Path, audio_name: str, params: dict) -> Path:
    debug_name = (
        f"debug_diff{params['difficulty']}_ctx{params['context_frames']}"
        f"_temp{_format_tag(params['temperature'])}_seed{params['seed']}.csv"
    )
    return _unique_path(output_dir / debug_name)


def _render_live_preview(all_actions, total_frames: int, assets: NoteAssets):
    return render_progress(all_actions, assets, total_frames=total_frames)


def _read_json_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def _button_state(text: str, *, disabled: bool = False):
    if gr is None:
        return text
    return gr.update(value=text, interactive=not disabled)


def _copy_button_kwargs() -> dict:
    """Built-in Textbox copy button: `buttons=["copy"]` on Gradio 6, `show_copy_button` before."""
    if gr is None:
        return {}
    params = inspect.signature(gr.Textbox.__init__).parameters
    if "buttons" in params:
        return {"buttons": ["copy"]}
    return {"show_copy_button": True}


def _notify_copied():
    if gr is None:
        return
    gr.Info("Chart JSON copied to clipboard", duration=2)


def _generate_chart_live(
    *,
    trainer,
    pt_path: str | Path,
    audio_name: str,
    difficulty,
    context_frames,
    temperature,
    top_k,
    top_p,
    seed,
    save_debug_csv: bool,
    bpm: float,
    offset_ms: float,
    segment_frames: int,
    output_root: str | Path,
):
    action_queue: queue.Queue = queue.Queue()
    all_actions: list[int] = []
    output_dir = Path(output_root) / audio_name
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_path = build_chart_output_path(
        output_dir.parent,
        audio_name,
        difficulty,
        context_frames,
        temperature,
        seed,
    )
    debug_csv_path = _make_debug_path(output_dir, audio_name, {
        "difficulty": difficulty,
        "context_frames": context_frames,
        "temperature": temperature,
        "seed": seed,
    }) if save_debug_csv else None

    note_assets = NoteAssets(
        str(PROJECT_ROOT / "src/tracers/animations" / "notes_sheet.png"),
        str(PROJECT_ROOT / "src/tracers/animations" / "hold_body.png"),
    )

    tensor = torch.load(Path(pt_path), map_location="cpu", weights_only=True)
    total_frames = int(tensor["specs"].shape[1])

    def worker():
        try:
            generated, debug_segments = run_custom_generation(
                trainer,
                str(pt_path),
                difficulty=float(difficulty),
                segment_frames=segment_frames,
                context_frames=int(context_frames),
                seed_action=0,
                seed=int(seed),
                temperature=float(temperature) if temperature not in (None, "", "None") else None,
                top_k=int(top_k) if top_k not in (None, "", "None") else None,
                top_p=float(top_p) if top_p not in (None, "", "None") else None,
                collect_debug=bool(save_debug_csv),
                on_window=lambda start_frame, new_actions: action_queue.put(
                    ("actions", int(start_frame), new_actions.detach().cpu())
                ),
            )

            export_generated_chart(
                generated,
                chart_path,
                bpm=int(float(bpm)),
                start_frame=0,
                offset_ms=int(round(float(offset_ms))),
                lane_map=(0, 2, 4, 6),
            )

            if save_debug_csv and debug_segments:
                export_generation_debug(
                    debug_segments,
                    str(debug_csv_path),
                    top_k=5,
                )

            action_queue.put(("done", chart_path))
        except Exception as exc:  # pragma: no cover - exercised in runtime UI
            action_queue.put(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        item = action_queue.get()
        if item[0] == "actions":
            _, _, actions = item
            all_actions.extend(int(a) for a in actions.reshape(-1).tolist())
            preview = _render_live_preview(all_actions, total_frames, note_assets)
            yield preview, "", None, f"Generating... {len(all_actions)} frames", _button_state("Generating...", disabled=True)
        elif item[0] == "done":
            _, final_path = item
            final_preview = _render_live_preview(all_actions, total_frames, note_assets)
            chart_text = _read_json_text(final_path)
            yield final_preview, chart_text, str(final_path), "Generation complete.", _button_state("Generate Chart", disabled=False)
            return
        elif item[0] == "error":
            _, error_message = item
            yield None, "", None, f"ERROR: {error_message}", _button_state("Generate Chart", disabled=False)
            return


def generate_chart(
    input_file,
    bpm,
    offset_ms,
    difficulty,
    context_frames,
    temperature,
    top_k,
    top_p,
    seed,
    save_debug_csv,
):
    if gr is None:
        raise RuntimeError(f"Gradio is not installed: {GRADIO_IMPORT_ERROR}")

    file_path = input_file[0] if isinstance(input_file, list) else input_file
    if file_path is None:
        yield None, "", None, "ERROR: Please upload an audio file or a preprocessed .pt tensor.", _button_state("Generate Chart", disabled=False)
        return

    try:
        trainer_state = _ensure_model_loaded()
        pt_path, audio_name = _prepare_preprocessed_pt(
            upload_path=file_path,
            bpm=float(bpm),
            offset_ms=float(offset_ms),
        )

        context_frames = int(context_frames)
        if not 0 <= context_frames < DEFAULT_SEGMENT_FRAMES:
            raise ValueError(
                f"Context Frames must be in [0, {DEFAULT_SEGMENT_FRAMES - 1}]."
            )

        top_k = _coerce_optional_int(top_k)
        if top_k is not None and top_k <= 0:
            top_k = None

        top_p = _coerce_optional_float(top_p, allow_zero=True)
        if top_p is not None and top_p <= 0:
            top_p = None
        elif top_p is not None and not 0 < top_p <= 1:
            raise ValueError("Top-P must be greater than 0 and at most 1 when specified.")

        temperature = _coerce_optional_float(temperature, allow_zero=True)
        if temperature is not None and temperature < 0:
            raise ValueError("Temperature must be non-negative.")

        seed = int(seed)

        # First output is the preview image, so it must not receive a button update.
        yield None, "", None, "Generating...", _button_state("Generating...", disabled=True)
        yield from _generate_chart_live(
            trainer=trainer_state["trainer"],
            pt_path=pt_path,
            audio_name=audio_name,
            difficulty=difficulty,
            context_frames=context_frames,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            save_debug_csv=bool(save_debug_csv),
            bpm=float(bpm),
            offset_ms=float(offset_ms),
            segment_frames=trainer_state["segment_frames"],
            output_root=PROJECT_ROOT / "charts",
        )
    except Exception as exc:
        yield None, "", None, f"ERROR: {exc}", _button_state("Generate Chart", disabled=False)


def build_app():
    if gr is None:
        raise RuntimeError(f"Gradio is not installed: {GRADIO_IMPORT_ERROR}")

    with gr.Blocks(title="Tracer Chart Generator", css=APP_CSS) as app:
        with gr.Column(elem_id="tracer-shell"):
            gr.Markdown(
                """
                <div class="tracer-brand">
                    <span class="tracer-mark" aria-hidden="true"></span>
                    <h1 class="tracer-title">Tracer Chart Generator</h1>
                </div>
                <p class="tracer-subtitle">
                    Transform audio or preprocessed tensors into a playable rhythm chart,
                    then inspect the live render and export the generated JSON.
                </p>
                """,
                elem_id="tracer-header",
            )

            with gr.Row(elem_id="config-grid"):
                with gr.Column(
                    scale=5,
                    elem_id="audio-card",
                    elem_classes=["tracer-card"],
                ):
                    gr.Markdown(
                        "## Audio & timing\nChoose a source and align its tempo before generation.",
                        elem_classes=["section-heading"],
                    )
                    input_file = gr.File(
                        label="Audio / Preprocessed .pt",
                        file_types=[".mp3", ".wav", ".flac", ".ogg", ".pt"],
                        type="filepath",
                    )
                    with gr.Row(elem_id="timing-fields"):
                        bpm = gr.Number(value=120, label="BPM", minimum=1)
                        offset_ms = gr.Number(value=0, label="Offset (ms)")

                with gr.Column(
                    scale=7,
                    elem_id="parameters-card",
                    elem_classes=["tracer-card"],
                ):
                    gr.Markdown(
                        "## Generation parameters\nTune chart density, sampling behavior, and reproducibility.",
                        elem_classes=["section-heading"],
                    )
                    difficulty = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=5,
                        step=0.1,
                        label="Difficulty",
                    )
                    with gr.Row():
                        context_frames = gr.Number(
                            value=0,
                            label="Context Frames",
                            precision=0,
                        )
                        temperature = gr.Number(
                            value=None,
                            label="Temperature",
                            minimum=0,
                            maximum=2,
                            precision=3,
                        )
                    with gr.Row():
                        top_k = gr.Number(
                            value=None,
                            label="Top-K",
                            precision=0,
                            minimum=0,
                        )
                        top_p = gr.Number(
                            value=None,
                            label="Top-P",
                            minimum=0,
                            maximum=1,
                            step=0.01,
                        )
                    with gr.Row(elem_id="seed-options"):
                        seed = gr.Number(value=0, label="Seed", precision=0)
                        save_debug_csv = gr.Checkbox(
                            value=True,
                            label="Save Debug CSV",
                        )

            generate_button = gr.Button(
                "Generate Chart",
                elem_id="generate-chart-button",
            )

            with gr.Row(elem_id="output-grid"):
                with gr.Column(
                    scale=8,
                    elem_id="preview-card",
                    elem_classes=["tracer-card"],
                ):
                    gr.Markdown(
                        "## Live chart preview\nThe chart render updates as generation progresses.",
                        elem_classes=["section-heading"],
                    )
                    preview_output = gr.Image(
                        label="Live Chart Preview",
                        type="pil",
                        show_label=False,
                        buttons=["download", "fullscreen", "share"],
                    )

                with gr.Column(scale=4, elem_id="result-rail"):
                    with gr.Column(
                        elem_id="json-card",
                        elem_classes=["tracer-card"],
                    ):
                        with gr.Row(elem_id="json-toolbar"):
                            gr.Markdown(
                                "## Generated chart JSON\nReview or copy the final chart payload.",
                                elem_classes=["section-heading"],
                            )
                            chart_output = gr.File(
                                label="Download JSON",
                                elem_id="download-json",
                                visible=False,
                            )
                        chart_json_output = gr.Textbox(
                            label="Generated Chart JSON",
                            value="",
                            lines=20,
                            max_lines=40,
                            show_label=False,
                            elem_id="generated_chart_json_text",
                            **_copy_button_kwargs(),
                        )
                        status_output = gr.Textbox(
                            label="Status",
                            interactive=False,
                            visible=False,
                            elem_id="status-output",
                        )

        # Fires when the user clicks the Textbox's built-in copy button.
        chart_json_output.copy(
            fn=_notify_copied,
            inputs=None,
            outputs=None,
            show_progress="hidden",
        )

        generate_button.click(
            fn=generate_chart,
            inputs=[
                input_file,
                bpm,
                offset_ms,
                difficulty,
                context_frames,
                temperature,
                top_k,
                top_p,
                seed,
                save_debug_csv,
            ],
            outputs=[preview_output, chart_json_output, chart_output, status_output, generate_button],
        )

    _ensure_model_loaded()
    return app


def main():
    if gr is None:
        raise RuntimeError(f"Gradio is not installed: {GRADIO_IMPORT_ERROR}")

    app = build_app()
    _configure_queue(app)
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
