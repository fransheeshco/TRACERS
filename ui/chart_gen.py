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

    with gr.Blocks(title="Tracer Chart Generator") as app:
        gr.Markdown("# Tracer Chart Generator")

        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="Audio / Preprocessed .pt",
                    file_types=[".mp3", ".wav", ".flac", ".ogg", ".pt"],
                    type="filepath",
                )

                bpm = gr.Number(value=120, label="BPM", minimum=1)
                offset_ms = gr.Number(value=0, label="Offset (ms)")

            with gr.Column(scale=1):
                difficulty = gr.Slider(minimum=0, maximum=10, value=5, step=0.1, label="Difficulty")
                context_frames = gr.Number(value=0, label="Context Frames", precision=0)
                temperature = gr.Number(value=None, label="Temperature", minimum=0, maximum=2, precision=3)
                top_k = gr.Number(value=None, label="Top-K", precision=0, minimum=0)
                top_p = gr.Number(value=None, label="Top-P", minimum=0, maximum=1, step=0.01)
                seed = gr.Number(value=0, label="Seed", precision=0)
                save_debug_csv = gr.Checkbox(value=True, label="Save Debug CSV")

        generate_button = gr.Button("Generate Chart")
        preview_output = gr.Image(label="Live Chart Preview", type="pil")
        chart_json_output = gr.Textbox(
            label="Generated Chart JSON",
            value="",
            lines=20,
            max_lines=40,
            elem_id="generated_chart_json_text",
            **_copy_button_kwargs(),
        )
        chart_output = gr.File(label="Download JSON")
        status_output = gr.Textbox(label="Status", interactive=False)

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
