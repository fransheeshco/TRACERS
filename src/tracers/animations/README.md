The renderer is implemented as `chart_preview.py`, and an example, `preview_example.png`, was rendered using fake notes. It has not yet been tested on real model output or inside Gradio.

## Visual Layout

- **Only generated notes are drawn.** The part of the chart that has not been generated yet is dimmed, with a green line at the generation front, so the chart appears to grow with each redraw.
- **Three columns, left to right.** Each column holds about 7 seconds with older notes at the bottom and newer on top, so a full 20-second window fits on screen without sprites overlapping badly. A header shows the progress bar and a note count.
- **Taps** use the pink diamond. **Hold starts and ends** use the same green oval, and the hold body is stretched from the green skin texture between them.
- **Orphans** (a hold start with no end, or an end with no start) are drawn as a green note with no body, allowing grammar problems in logs to be identified visually. A hold that is still open at the front keeps growing until its end arrives.

## Asset Assumptions

The sheet was cropped using a measured grid of 310 × 122 px cells. Cell `(row, col)` counts from zero, so `(0,0)` is the pink diamond, `(0,1)` the green oval and `(0,2)` the cyan oval. To use a different tap sprite, one argument can be changed:

```python
assets = NoteAssets("assets/notes_sheet.png", "assets/hold_body.png", tap=(0, 2), hold=(0, 1))

```

The body texture is inherently about 38% opaque, so it was boosted (`body_opacity_boost=2.3`) to make it readable on a dark background. This can be set to 1.0 to restore the original look.

## Gradio Integration

A per-window callback is required in `run_custom_generation` (`on_window(start_frame, new_actions)`, called with only the new frames). Then:

```python
import queue, threading, torch
from chart_preview import NoteAssets, render_progress

assets = NoteAssets("assets/notes_sheet.png", "assets/hold_body.png")

def generate_live(audio, bpm, offset_ms, difficulty, temperature, context, seed):
    pt_path = make_song_pt(audio, bpm, offset_ms)                 # preprocessing step
    total_frames = torch.load(pt_path, weights_only=True)["specs"].shape[1] - 1
    q, all_actions = queue.Queue(), []

    def work():
        try:
            result = run_custom_generation(
                trainer, pt_path, difficulty=difficulty, segment_frames=segment_frames,
                context_frames=int(context), seed_action=0, seed=int(seed),
                temperature=temperature or None, top_k=None, top_p=None,
                collect_debug=False, on_window=lambda start, acts: q.put(acts),
            )
            q.put(("done", result))
        except Exception as e:
            q.put(("error", e))

    threading.Thread(target=work, daemon=True).start()

    while True:
        item = q.get()
        if isinstance(item, tuple):
            kind, payload = item
            if kind == "error":
                raise payload
            generated, _ = payload
            chart_path = export_chart(generated, bpm, offset_ms)   # exporter step
            yield render_progress(all_actions, assets, total_frames=total_frames), chart_path
            return
        all_actions.extend(int(a) for a in item)
        yield render_progress(all_actions, assets, total_frames=total_frames), None

```

This generator function should be wired to outputs of `[gr.Image(label="Live preview"), gr.File(label="Chart JSON")]`, and `demo.queue()` must be called before `launch()`.

## Implementation Details

- Each redraw replaces the view with the latest 20 seconds, so with 2,000-frame windows, one block of the chart is visible at a time. For a rolling view, the windows must be shortened or a callback added every few hundred frames inside `Trainer.generate`.
- Dense sections still overlap slightly, as 16th notes are about 11 px apart at `px_per_s=90`. This can be adjusted by raising `px_per_s` or lowering `view_seconds`.
- If a note reaches into a column's edge, half of it appears in each neighboring column. That is normal for printed chart viewers.
- Frame numbers use `first_frame=1`, matching standard window printouts ("1–2000").
- A smooth scrolling player synced to the audio after generation finishes requires Gradio 6's `js_on_load`.
