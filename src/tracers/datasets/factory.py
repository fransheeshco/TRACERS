from __future__ import annotations

from pathlib import Path

from tracers.datasets import AutoOsu4KDataset, make_dataloader, split_song_ids
from tracers.datasets.dataset import GlobalAudioCollator


def build_song_split(root: str | Path, dataset_folder_name: str = "dataset", *, segment_frames: int, seed: int = 42):
    root = Path(root)
    audio_dir = root / dataset_folder_name / "audio"
    beatmap_dir = root / dataset_folder_name / "beatmap" / "4keys"

    base_dataset = AutoOsu4KDataset(
        beatmap_dir=beatmap_dir,
        audio_dir=audio_dir,
        segment_frames=segment_frames,
        crop_mode="random",
        audio_cache_size=8,
    )

    train_ids, val_ids, test_ids = split_song_ids(
        base_dataset.song_ids,
        val_fraction=0.10,
        test_fraction=0.10,
        seed=seed,
    )
    return audio_dir, beatmap_dir, train_ids, val_ids, test_ids


def build_datasets(root: str | Path, dataset_folder_name: str = "dataset", *, segment_frames: int, seed: int = 42):
    audio_dir, beatmap_dir, train_ids, val_ids, test_ids = build_song_split(
        root,
        dataset_folder_name,
        segment_frames=segment_frames,
        seed=seed,
    )

    train_dataset = AutoOsu4KDataset(
        beatmap_dir,
        audio_dir,
        segment_frames=segment_frames,
        crop_mode="random",
        crops_per_chart=2,
        song_ids=train_ids,
    )

    val_dataset = AutoOsu4KDataset(
        beatmap_dir,
        audio_dir,
        segment_frames=segment_frames,
        crop_mode="all",
        song_ids=val_ids,
        original_only=True
    )

    test_dataset = AutoOsu4KDataset(
        beatmap_dir,
        audio_dir,
        segment_frames=segment_frames,
        crop_mode="all",
        song_ids=test_ids,
        original_only=True
    )

    return audio_dir, beatmap_dir, train_dataset, val_dataset, test_dataset, train_ids, val_ids, test_ids


def build_loader_for_model(dataset, *, batch_size: int, shuffle: bool, num_workers: int, model_key: str):
    collate_fn = None
    model_key = model_key.lower()
    if model_key in {"tracer_global_np", "global", "global_np", "tracer_global"}:
        collate_fn = GlobalAudioCollator(
            dataset.audio_dir,
            song_ids=dataset.song_ids,
            cache_size=2,
        )

    return make_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
