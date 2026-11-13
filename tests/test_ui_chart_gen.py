from pathlib import Path

import torch

from ui.chart_gen import build_chart_output_path, validate_preprocessed_tensor


def test_validate_preprocessed_tensor_accepts_expected_keys():
    payload = {
        "specs": torch.randn(1, 10, 128),
        "beat_phase": torch.zeros(10, dtype=torch.long),
        "beat_num": torch.zeros(10, dtype=torch.long),
    }

    validated = validate_preprocessed_tensor(payload)

    assert validated["specs"].shape == (1, 10, 128)
    assert validated["beat_phase"].shape == (10,)
    assert validated["beat_num"].shape == (10,)


def test_build_chart_output_path_avoids_overwrite(tmp_path):
    stem = "song"
    path = build_chart_output_path(tmp_path, stem, 5, 0, 0.6, 42)
    assert path.name == "chart_diff5_ctx0_temp0p6_seed42.json"

    second = build_chart_output_path(tmp_path, stem, 5, 0, 0.6, 42)
    assert second.name.startswith("chart_diff5_ctx0_temp0p6_seed42_")
    assert second.suffix == ".json"
