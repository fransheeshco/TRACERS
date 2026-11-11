"""PyTorch data loading utilities for the 4-key AutoOsu dataset.

Expected layout (adjust the paths when constructing the Dataset)::

    dataset/
      audio/<song_id>.pt
      beatmap/4keys/<chart_id>.pt

Each chart is paired with audio by taking the part of its filename before
its first hyphen, e.g. 1003093-108-4.pt -> 1003093.pt.

Training samples reproduce AutoOsu's one-frame teacher-forcing alignment:
    features/onsets: frames [start + 1, ..., start + L]
    decoder input:    actions  [start,     ..., start + L - 1]
    target actions:   actions  [start + 1, ..., start + L]

A loss_mask is returned because an action token combines all four lanes. If a
crop cuts through a long note, the whole action frame at the dangling boundary
is masked (for both onset and action losses) rather than training on a broken
start/end pair. Other notes in that same frame are consequently masked too.
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import torch
from torch.utils.data import Dataset, DataLoader, default_collate


CropMode = Literal["random", "all", "center"]


@dataclass(frozen=True)
class ChartRecord:
    chart_path: Path
    audio_path: Path
    chart_id: str
    song_id: str


def _torch_load(path: Path) -> dict:
    """Load a tensor-only .pt dictionary on CPU, preferring safe loading."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(obj).__name__}")
    return obj


def song_id_from_chart_path(path: Path) -> str:
    """AutoOsu chart IDs look like '<song_id>-<difficulty>-<keys>'."""
    return path.stem.split("-", maxsplit=1)[0]


def _scalar_float(value, field_name: str, path: Path) -> float:
    tensor = torch.as_tensor(value).detach().cpu().float().reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"{path}: '{field_name}' must be scalar; got {tuple(tensor.shape)}")
    result = float(tensor.item())
    if not math.isfinite(result):
        raise ValueError(f"{path}: '{field_name}' is not finite")
    return result


def decode_4key_actions(actions: torch.Tensor) -> torch.Tensor:
    """Decode [T] base-4 action tokens into [T, 4] lane states (0..3)."""
    actions = torch.as_tensor(actions).long()
    return torch.stack([(actions >> (2 * lane)) & 3 for lane in range(4)], dim=-1)


def find_long_note_pairs(actions: torch.Tensor) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    """Return (lane, start_frame, end_frame) pairs and orphan (lane, frame) events.

    AutoOsu represents a long note with state 2 at its start and state 3 at
    its end. Pairs are matched in time order independently per lane.
    """
    states = decode_4key_actions(actions)
    pairs: list[tuple[int, int, int]] = []
    orphan_events: list[tuple[int, int]] = []

    for lane in range(4):
        pending_start: Optional[int] = None
        for frame, state in enumerate(states[:, lane].tolist()):
            if state == 2:
                # A second start before an end is malformed for a single lane.
                if pending_start is not None:
                    orphan_events.append((lane, pending_start))
                pending_start = frame
            elif state == 3:
                if pending_start is None:
                    orphan_events.append((lane, frame))
                else:
                    pairs.append((lane, pending_start, frame))
                    pending_start = None
        if pending_start is not None:
            orphan_events.append((lane, pending_start))

    return pairs, orphan_events


def _safe_crop_starts(
    total_frames: int,
    segment_frames: int,
    pairs: Sequence[tuple[int, int, int]],
) -> torch.Tensor:
    """Find starts that do not split a paired long note across target frames.

    A sample starts at action frame s. The decoder receives action[s], then
    predicts target frames s+1 through s+L. Therefore s itself is available as
    decoder context. We reject starts strictly inside a hold, and reject a
    crop whose right edge falls after a target long-note start but before its
    matching end.
    """
    max_start = total_frames - segment_frames - 1
    if max_start < 0:
        return torch.empty(0, dtype=torch.long)

    valid = torch.ones(max_start + 1, dtype=torch.bool)
    for _lane, note_start, note_end in pairs:
        # Invalid if the sample begins while this long note is already active.
        lo = max(0, note_start + 1)
        hi = min(max_start, note_end - 1)
        if lo <= hi:
            valid[lo : hi + 1] = False

        # Invalid if the target contains the start but not its matching end.
        # s < note_start <= s + L < note_end
        lo = max(0, note_start - segment_frames)
        hi = min(max_start, note_start - 1, note_end - segment_frames - 1)
        if lo <= hi:
            valid[lo : hi + 1] = False

    return torch.nonzero(valid, as_tuple=False).flatten()


def _validate_pair(
    chart: dict,
    audio: dict,
    chart_path: Path,
    audio_path: Path,
    check_beatmap_alignment: bool = True,
) -> dict:
    """Validate shapes, ranges, alignment, and optionally event reconstruction."""
    required_chart = {"actions", "onsets", "beatmap", "difficulty"}
    required_audio = {"specs", "beat_phase", "beat_num"}
    missing_chart = required_chart - chart.keys()
    missing_audio = required_audio - audio.keys()
    if missing_chart:
        raise KeyError(f"{chart_path}: missing chart keys {sorted(missing_chart)}")
    if missing_audio:
        raise KeyError(f"{audio_path}: missing audio keys {sorted(missing_audio)}")

    specs = torch.as_tensor(audio["specs"]).cpu()
    beat_phase = torch.as_tensor(audio["beat_phase"]).cpu()
    beat_num = torch.as_tensor(audio["beat_num"]).cpu()
    actions = torch.as_tensor(chart["actions"]).cpu().long().reshape(-1)
    onsets = torch.as_tensor(chart["onsets"]).cpu().reshape(-1)
    beatmap = torch.as_tensor(chart["beatmap"]).cpu()

    if specs.ndim != 3 or specs.shape[0] != 3 or specs.shape[2] != 80:
        raise ValueError(f"{audio_path}: expected specs [3,T,80], got {tuple(specs.shape)}")
    if beat_phase.ndim != 1 or beat_num.ndim != 1:
        raise ValueError(f"{audio_path}: beat_phase and beat_num must be 1-D")
    if beatmap.ndim != 2 or beatmap.shape[1] != 3:
        raise ValueError(f"{chart_path}: expected beatmap [N,3], got {tuple(beatmap.shape)}")

    lengths = {
        "specs": int(specs.shape[1]),
        "beat_phase": int(beat_phase.numel()),
        "beat_num": int(beat_num.numel()),
        "actions": int(actions.numel()),
        "onsets": int(onsets.numel()),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Length mismatch for {chart_path.name} / {audio_path.name}: {lengths}"
        )
    total_frames = lengths["actions"]
    if total_frames == 0:
        raise ValueError(f"{chart_path}: actions cannot be empty")
    if total_frames < 2:
        raise ValueError(f"{chart_path}: need at least two frames for shifted targets")

    if actions.min().item() < 0 or actions.max().item() > 255:
        raise ValueError(f"{chart_path}: 4-key actions must be in [0,255]")
    if not torch.all((onsets == 0) | (onsets == 1)):
        raise ValueError(f"{chart_path}: onsets must be binary")
    expected_onsets = actions.ne(0).to(onsets.dtype)
    if not torch.equal(onsets, expected_onsets):
        mismatch = int((onsets != expected_onsets).sum().item())
        raise ValueError(f"{chart_path}: {mismatch} onsets do not match actions != 0")

    if beat_phase.numel() and (beat_phase.min().item() < 0 or beat_phase.max().item() > 48):
        raise ValueError(f"{audio_path}: beat_phase must be in [0,48]")
    if beat_num.numel() and (beat_num.min().item() < 0 or beat_num.max().item() > 3):
        raise ValueError(f"{audio_path}: beat_num must be in [0,3]")

    difficulty = _scalar_float(chart["difficulty"], "difficulty", chart_path)
    if beatmap.numel():
        if not torch.isfinite(beatmap).all():
            raise ValueError(f"{chart_path}: beatmap contains non-finite values")
        lanes = beatmap[:, 1]
        note_types = beatmap[:, 2]
        if not torch.all(lanes == lanes.round()) or lanes.min().item() < 0 or lanes.max().item() > 3:
            raise ValueError(f"{chart_path}: 4-key beatmap lane indices must be integers 0..3")
        if not torch.all(note_types == note_types.round()) or not torch.all(
            (note_types >= 1) & (note_types <= 3)
        ):
            raise ValueError(f"{chart_path}: note types must be 1, 2, or 3")

    alignment_checked = False
    if check_beatmap_alignment:
        # The converter rounds event times to the nearest 10ms frame. Rebuild
        # the lane matrix and compare it with the stored action tokens.
        expected_lanes = torch.zeros((total_frames, 4), dtype=torch.long)
        valid_event_times = True
        for event in beatmap.tolist():
            time_ms, lane, note_type = event
            frame = int(torch.round(torch.tensor(time_ms / 10.0)).item())
            # The converter uses Python/Torch indexing; negative indices within
            # range wrap from the end. Preserve that behavior for validation.
            if frame < 0:
                frame += total_frames
            if frame < 0 or frame >= total_frames:
                valid_event_times = False
                break
            expected_lanes[frame, int(lane)] = int(note_type)

        if valid_event_times:
            weights = torch.tensor([1, 4, 16, 64], dtype=torch.long)
            expected_actions = (expected_lanes * weights).sum(dim=1)
            if not torch.equal(actions, expected_actions):
                mismatch = int((actions != expected_actions).sum().item())
                raise ValueError(
                    f"{chart_path}: beatmap reconstruction differs from actions at "
                    f"{mismatch} frames"
                )
            alignment_checked = True

    pairs, orphan_events = find_long_note_pairs(actions)
    return {
        "frames": total_frames,
        "active_frames": int(actions.ne(0).sum().item()),
        "difficulty_index": difficulty,
        "long_note_pairs": len(pairs),
        "orphan_long_note_events": len(orphan_events),
        "beatmap_alignment_checked": alignment_checked,
    }


class AutoOsu4KDataset(Dataset):
    """Lazy-loading Dataset for AutoOsu four-key charts.

    Args:
        beatmap_dir: Folder containing 4-key chart .pt files.
        audio_dir: Folder containing audio feature .pt files.
        segment_frames: Number of prediction frames per sample. At 10ms/frame,
            1000 means 10 seconds. Set to None to return full charts (not
            recommended for batching variable-length examples).
        crop_mode:
            'random' -> random crop each access; repeat each chart
                `crops_per_chart` times per epoch.
            'all' -> deterministic windows covering every target frame.
            'center' -> one deterministic center crop per chart.
        stride_frames: Window stride for crop_mode='all'. Defaults to
            segment_frames (non-overlapping windows).
        crops_per_chart: Number of random samples per chart per epoch.
        song_ids: Optional allow-list for grouped train/validation/test splits.
        audio_cache_size: Number of audio dictionaries cached in RAM per
            Dataset worker. 0 disables caching. Start with 4-8 on an 8GB GPU;
            this is CPU RAM caching, not GPU VRAM.
        check_on_load: Run inexpensive per-sample tensor checks when loading.
    """

    def __init__(
        self,
        beatmap_dir: str | Path,
        audio_dir: str | Path,
        segment_frames: Optional[int] = 1000,
        crop_mode: CropMode = "random",
        stride_frames: Optional[int] = None,
        crops_per_chart: int = 1,
        song_ids: Optional[Iterable[str]] = None,
        audio_cache_size: int = 8,
        check_on_load: bool = True,
    ):
        self.beatmap_dir = Path(beatmap_dir)
        self.audio_dir = Path(audio_dir)
        self.segment_frames = segment_frames
        self.crop_mode = crop_mode
        self.stride_frames = stride_frames if stride_frames is not None else segment_frames
        self.crops_per_chart = int(crops_per_chart)
        self.audio_cache_size = max(0, int(audio_cache_size))
        self.check_on_load = check_on_load
        self._audio_cache: OrderedDict[str, dict] = OrderedDict()

        if not self.beatmap_dir.is_dir():
            raise FileNotFoundError(f"Beatmap directory not found: {self.beatmap_dir}")
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")
        if crop_mode not in {"random", "all", "center"}:
            raise ValueError("crop_mode must be 'random', 'all', or 'center'")
        if segment_frames is not None and segment_frames < 1:
            raise ValueError("segment_frames must be positive or None")
        if crop_mode != "random" and segment_frames is None:
            raise ValueError("segment_frames is required for 'all' and 'center' modes")
        if self.crops_per_chart < 1:
            raise ValueError("crops_per_chart must be >= 1")
        if crop_mode == "all" and (self.stride_frames is None or self.stride_frames < 1):
            raise ValueError("stride_frames must be positive")

        audio_paths = sorted(self.audio_dir.glob("*.pt"))
        audio_by_song: dict[str, Path] = {}
        duplicates = []
        for path in audio_paths:
            if path.stem in audio_by_song:
                duplicates.append(path.stem)
            audio_by_song[path.stem] = path
        if duplicates:
            raise ValueError(f"Duplicate audio song IDs: {sorted(set(duplicates))[:10]}")

        allowed_song_ids = set(str(x) for x in song_ids) if song_ids is not None else None
        records: list[ChartRecord] = []
        missing_audio: list[str] = []
        for chart_path in sorted(self.beatmap_dir.glob("*.pt")):
            song_id = song_id_from_chart_path(chart_path)
            if allowed_song_ids is not None and song_id not in allowed_song_ids:
                continue
            audio_path = audio_by_song.get(song_id)
            if audio_path is None:
                missing_audio.append(f"{chart_path.name} -> {song_id}.pt")
                continue
            records.append(
                ChartRecord(chart_path, audio_path, chart_path.stem, song_id)
            )

        if missing_audio:
            preview = "\n".join(missing_audio[:10])
            raise FileNotFoundError(
                f"{len(missing_audio)} charts have no matching audio file. Examples:\n{preview}"
            )
        if not records:
            raise RuntimeError("No matching 4-key chart/audio pairs found")
        self.records = records
        self.song_ids = sorted({record.song_id for record in records})

        # For deterministic evaluation modes, create a virtual index of windows.
        self._window_index: list[tuple[int, int]] = []
        if crop_mode in {"all", "center"}:
            assert segment_frames is not None
            length = segment_frames

            for record_idx, record in enumerate(self.records):
                chart = _torch_load(record.chart_path)
                if "actions" not in chart:
                    raise KeyError(f"{record.chart_path}: missing 'actions'")

                total_frames = int(torch.as_tensor(chart["actions"]).numel())
                if total_frames < 2:
                    raise ValueError(f"{record.chart_path}: fewer than 2 frames")

                target_frames = total_frames - 1

                if crop_mode == "center":
                    start = max(0, (target_frames - length) // 2)
                    self._window_index.append((record_idx, start))
                else:
                    stride = self.stride_frames
                    if stride is None or stride < 1:
                        raise ValueError("stride_frames must be positive")

                    for start in range(0, target_frames, stride):
                        self._window_index.append((record_idx, start))

    def __len__(self) -> int:
        if self.crop_mode == "random":
            return len(self.records) * self.crops_per_chart
        return len(self._window_index)

    def _load_audio(self, record: ChartRecord) -> dict:
        song_id = record.song_id
        if self.audio_cache_size > 0 and song_id in self._audio_cache:
            self._audio_cache.move_to_end(song_id)
            return self._audio_cache[song_id]

        audio = _torch_load(record.audio_path)
        if self.audio_cache_size > 0:
            self._audio_cache[song_id] = audio
            self._audio_cache.move_to_end(song_id)
            while len(self._audio_cache) > self.audio_cache_size:
                self._audio_cache.popitem(last=False)
        return audio

    def _choose_random_start(self, total_frames: int, actions: torch.Tensor, pairs):
        assert self.segment_frames is not None
        length = self.segment_frames
        max_start = max(0, total_frames - length - 1)
        if total_frames <= length + 1:
            return 0

        safe_starts = _safe_crop_starts(total_frames, length, pairs)
        if safe_starts.numel() > 0:
            pick = int(torch.randint(safe_starts.numel(), (1,)).item())
            return int(safe_starts[pick].item())

        # No crop can preserve all long-note pairs. A random crop is still
        # usable because the corresponding dangling boundary target is masked.
        return int(torch.randint(max_start + 1, (1,)).item())

    def _make_loss_mask(
        self,
        start: int,
        valid_length: int,
        segment_frames: int,
        pairs: Sequence[tuple[int, int, int]],
        orphan_events: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        """Mask targets at incomplete long-note boundaries and right padding."""
        mask = torch.zeros(segment_frames, dtype=torch.bool)
        mask[:valid_length] = True
        last_target_frame = start + valid_length

        def mask_event(frame: int):
            if start < frame <= last_target_frame:
                mask[frame - start - 1] = False

        for _lane, note_start, note_end in pairs:
            # Crop begins inside a hold: its end is orphaned in this sample.
            if note_start < start < note_end:
                mask_event(note_end)
            # Crop ends inside a hold after its start was a prediction target.
            if start < note_start <= last_target_frame < note_end:
                mask_event(note_start)

        for _lane, frame in orphan_events:
            mask_event(frame)

        return mask

    def __getitem__(self, index: int) -> dict:
        if self.crop_mode == "random":
            record_idx = index // self.crops_per_chart
            record = self.records[record_idx]
            fixed_start = None
        else:
            record_idx, fixed_start = self._window_index[index]
            record = self.records[record_idx]

        chart = _torch_load(record.chart_path)
        audio = self._load_audio(record)

        if self.check_on_load:
            _validate_pair(
                chart, audio, record.chart_path, record.audio_path,
                check_beatmap_alignment=False,
            )

        specs = torch.as_tensor(audio["specs"]).cpu().float()
        beat_phase = torch.as_tensor(audio["beat_phase"]).cpu().long().reshape(-1)
        beat_num = torch.as_tensor(audio["beat_num"]).cpu().long().reshape(-1)
        actions = torch.as_tensor(chart["actions"]).cpu().long().reshape(-1)
        onsets = torch.as_tensor(chart["onsets"]).cpu().float().reshape(-1)
        difficulty_index = _scalar_float(chart["difficulty"], "difficulty", record.chart_path)
        difficulty_value = difficulty_index * 0.2

        total_frames = actions.numel()
        pairs, orphan_events = find_long_note_pairs(actions)

        if self.segment_frames is None:
            start = 0
            requested_length = total_frames - 1
        else:
            requested_length = int(self.segment_frames)
            if fixed_start is None:
                start = self._choose_random_start(total_frames, actions, pairs)
            else:
                start = fixed_start

        # Each prediction target is one frame after its decoder input.
        available_length = max(0, total_frames - start - 1)
        valid_length = min(requested_length, available_length)
        if valid_length <= 0:
            raise ValueError(f"{record.chart_path}: crop has no target frames")

        # Inputs are aligned to target frames [start+1, ..., start+L].
        specs_segment = specs[:, start + 1 : start + valid_length + 1, :]
        beat_phase_segment = beat_phase[start + 1 : start + valid_length + 1]
        beat_num_segment = beat_num[start + 1 : start + valid_length + 1]
        onsets_segment = onsets[start + 1 : start + valid_length + 1]
        actions_input = actions[start : start + valid_length]
        actions_target = actions[start + 1 : start + valid_length + 1]

        loss_mask = self._make_loss_mask(
            start, valid_length, requested_length, pairs, orphan_events
        )

        # Pad short final windows on the right so DataLoader can stack them.
        pad = requested_length - valid_length
        if pad > 0:
            specs_segment = torch.nn.functional.pad(specs_segment, (0, 0, 0, pad))
            beat_phase_segment = torch.nn.functional.pad(beat_phase_segment, (0, pad))
            beat_num_segment = torch.nn.functional.pad(beat_num_segment, (0, pad))
            onsets_segment = torch.nn.functional.pad(onsets_segment, (0, pad))
            actions_input = torch.nn.functional.pad(actions_input, (0, pad))
            actions_target = torch.nn.functional.pad(actions_target, (0, pad))

        difficulty = torch.full((requested_length, 1), difficulty_value, dtype=torch.float32)

        return {
            "specs": specs_segment.contiguous(),                 # [3, L, 80]
            "beat_phase": beat_phase_segment.contiguous(),       # [L]
            "beat_num": beat_num_segment.contiguous(),           # [L]
            "difficulty": difficulty,                            # [L, 1]
            "onsets": onsets_segment.contiguous(),               # [L]
            "actions_input": actions_input.contiguous(),         # [L]
            "actions_target": actions_target.contiguous(),       # [L]
            "valid_frames": torch.arange(requested_length) < valid_length,  # [L], bool; masks padding for attention
            "loss_mask": loss_mask,                               # [L], bool; also masks dangling LN boundaries
            "song_id": record.song_id,
            "chart_id": record.chart_id,
            "start_frame": int(start),
            "valid_length": int(valid_length),
        }

    def validate_dataset(
        self,
        max_charts: Optional[int] = None,
        check_beatmap_alignment: bool = True,
    ) -> dict:
        """Validate paired files and return a compact dataset summary.

        Raises on the first malformed pair. Set max_charts for a quick smoke
        test; use None to scan every chart.
        """
        records = self.records if max_charts is None else self.records[:max_charts]
        total_frames = active_frames = long_note_pairs = orphan_events = 0
        alignment_checked = 0
        chart_lengths = []

        for record in records:
            chart = _torch_load(record.chart_path)
            audio = self._load_audio(record)
            stats = _validate_pair(
                chart, audio, record.chart_path, record.audio_path,
                check_beatmap_alignment=check_beatmap_alignment,
            )
            total_frames += stats["frames"]
            active_frames += stats["active_frames"]
            long_note_pairs += stats["long_note_pairs"]
            orphan_events += stats["orphan_long_note_events"]
            alignment_checked += int(stats["beatmap_alignment_checked"])
            chart_lengths.append(stats["frames"])

        return {
            "charts_checked": len(records),
            "unique_songs_in_checked_charts": len({r.song_id for r in records}),
            "total_frames": total_frames,
            "active_frames": active_frames,
            "empty_frame_percent": 100.0 * (1.0 - active_frames / total_frames),
            "average_chart_frames": sum(chart_lengths) / len(chart_lengths),
            "long_note_pairs": long_note_pairs,
            "orphan_long_note_events": orphan_events,
            "beatmap_alignment_checks_passed": alignment_checked,
        }


def split_song_ids(
    song_ids: Iterable[str],
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministically split song IDs into train, validation, and test IDs.

    All charts belonging to a song must use the same partition to avoid audio
    leakage across difficulties.
    """
    ids = sorted(set(str(song_id) for song_id in song_ids))
    if len(ids) < 3:
        raise ValueError("Need at least 3 unique songs for train/val/test splitting")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be >= 0 and sum to < 1")

    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_fraction))
    n_test = int(round(len(ids) * test_fraction))
    if val_fraction > 0:
        n_val = max(1, n_val)
    if test_fraction > 0:
        n_test = max(1, n_test)
    if n_val + n_test >= len(ids):
        raise ValueError("Not enough songs left for the training split")

    test_ids = sorted(ids[:n_test])
    val_ids = sorted(ids[n_test : n_test + n_val])
    train_ids = sorted(ids[n_test + n_val :])
    return train_ids, val_ids, test_ids


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    collate_fn=None,
) -> DataLoader:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )


class FullSongAudioStore:
    """Lazy CPU store for full-song spectrograms."""

    def __init__(
        self,
        audio_dir,
        song_ids=None,
        cache_size=2
    ):
        self.audio_dir = Path(audio_dir)
        self.allowed_song_ids = (
            None if song_ids is None
            else set(str(x) for x in song_ids)
        )

        self.paths = {
            path.stem: path
            for path in self.audio_dir.glob("*.pt")
        }

        self.cache_size = max(0, int(cache_size))
        self.cache = OrderedDict()

    def __getitem__(self, song_id):
        song_id = str(song_id)

        if (
            self.allowed_song_ids is not None
            and song_id not in self.allowed_song_ids
        ):
            raise KeyError(f"Song {song_id} is outside this dataset split.")

        if song_id in self.cache:
            self.cache.move_to_end(song_id)
            return self.cache[song_id]

        path = self.paths.get(song_id)

        if path is None:
            raise FileNotFoundError(
                f"Missing full-song audio: {song_id}.pt"
            )

        audio = _torch_load(path)

        specs = torch.as_tensor(
            audio["specs"]
        ).cpu().float()

        if specs.ndim != 3 or specs.shape[0] != 3 or specs.shape[2] != 80:
            raise ValueError(
                f"{path}: expected specs [3,T,80], got {tuple(specs.shape)}"
            )

        if specs.size(1) < 2:
            raise ValueError(f"{path}: not enough audio frames.")

        # Match the dataset's teacher-forcing alignment:
        # local crop starts at original audio frame start + 1.
        #
        # Therefore, global index 0 corresponds to original frame 1.
        specs = specs[:, 1:, :].contiguous()

        if self.cache_size > 0:
            self.cache[song_id] = specs
            self.cache.move_to_end(song_id)

            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

        return specs


class GlobalAudioCollator:
    """
    Adds unique full-song spectrograms to a normal chart batch.

    Multiple charts/crops from the same song share one entry.
    """

    def __init__(
        self,
        audio_dir,
        song_ids=None,
        cache_size=2
    ):
        self.audio_store = FullSongAudioStore(
            audio_dir=audio_dir,
            song_ids=song_ids,
            cache_size=cache_size
        )

    def __call__(self, samples):
        batch = default_collate(samples)

        song_to_index = {}
        unique_specs = []
        global_indices = []

        for sample in samples:
            song_id = sample["song_id"]

            if song_id not in song_to_index:
                song_to_index[song_id] = len(unique_specs)
                unique_specs.append(
                    self.audio_store[song_id]
                )

            global_indices.append(song_to_index[song_id])

        # A list because full songs have different lengths.
        batch["global_specs"] = unique_specs

        # Maps each chart sample to its full-song tensor.
        batch["global_indices"] = torch.tensor(
            global_indices,
            dtype=torch.long
        )

        return batch
