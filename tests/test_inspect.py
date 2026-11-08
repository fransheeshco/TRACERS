from pathlib import Path

import pytest
import torch

from tracers.audio.inspect import inspect_tensor_file, main
from tracers.audio.preprocess import save_tensors


def _save_valid_file(path: Path) -> None:
    tensors = {
        "specs": torch.zeros((3, 10, 80), dtype=torch.float32),
        "beat_phase": torch.zeros(10, dtype=torch.int32),
        "beat_num": torch.arange(10, dtype=torch.int32) % 4,
    }
    save_tensors(tensors, path)


def test_inspect_command_reports_valid_file(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    output = tmp_path / "My Song.pt"
    _save_valid_file(output)

    assert main([str(output)]) == 0

    report = capsys.readouterr().out
    assert "Status: valid TRACERS tensor file" in report
    assert "specs: shape=(3, 10, 80), dtype=torch.float32" in report
    assert "beat_num: shape=(10,), dtype=torch.int32" in report


def test_inspector_rejects_misaligned_frames(tmp_path: Path) -> None:
    output = tmp_path / "invalid.pt"
    _save_valid_file(output)
    data = torch.load(output, map_location="cpu", weights_only=True)
    data["beat_num"] = torch.zeros(9, dtype=torch.int32)
    torch.save(data, output)

    with pytest.raises(ValueError, match="beat_num must have shape"):
        inspect_tensor_file(output)
