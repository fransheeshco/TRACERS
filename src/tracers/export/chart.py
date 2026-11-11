from __future__ import annotations

import json
from pathlib import Path

import torch


def export_generated_chart(generated, output_path, *, bpm=120, start_frame=0, offset_ms=0, lane_map=(0, 2, 4, 6)):
    from tracers.generate.converter import predictions_to_json

    events = predictions_to_json(
        action_predictions=generated,
        bpm=bpm,
        start_frame=start_frame,
        offset_ms=offset_ms,
        lane_map=lane_map,
        valid_frames=torch.ones(len(generated), dtype=torch.bool),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=4)

    return events
