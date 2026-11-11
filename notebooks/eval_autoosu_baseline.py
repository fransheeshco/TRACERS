from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    entry_str = str(entry)
    if entry_str not in sys.path:
        sys.path.insert(0, entry_str)

from autoosu_reference.models import OsuModel
from tracers.datasets import AutoOsu4KDataset, make_dataloader, split_song_ids
from tracers.models.loss import get_loss
from tracers.training.metrics import finalize_metrics, init_metric_counts, update_metric_counts


def build_autoosu_val_loader(
    root: Path,
    *,
    segment_frames: int = 1000,
    seed: int = 42,
    batch_size: int = 2,
):
    audio_dir = root / "dataset" / "audio"
    beatmap_dir = root / "dataset" / "beatmap" / "4keys"

    base_dataset = AutoOsu4KDataset(
        beatmap_dir=beatmap_dir,
        audio_dir=audio_dir,
        segment_frames=segment_frames,
        crop_mode="random",
        audio_cache_size=8,
    )

    _, val_ids, _ = split_song_ids(
        base_dataset.song_ids,
        val_fraction=0.10,
        test_fraction=0.10,
        seed=seed,
    )

    val_dataset = AutoOsu4KDataset(
        beatmap_dir,
        audio_dir,
        segment_frames=segment_frames,
        crop_mode="all",
        song_ids=val_ids,
        original_only=True,
    )

    return make_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def evaluate_autoosu_checkpoint(
    *,
    root: Path,
    checkpoint_name: str = "osu_model_v2.pt",
    batch_size: int = 2,
    segment_frames: int = 1000,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = root / "autoosu_reference" / "checkpoints" / checkpoint_name

    hyperparams = {
        "bp_emb_dim": 32,
        "bn_emb_dim": 16,
        "diff_emb_dim": 16,
        "np_hidden_size": 256,
        "np_num_layers": 2,
        "ns_pre_proj_size": 32,
        "ns_hidden_size": 256,
        "ns_num_layers": 2,
        "action_emb_dim": 32,
    }

    model = OsuModel(hyperparams).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = build_autoosu_val_loader(
        root,
        segment_frames=segment_frames,
        seed=seed,
        batch_size=batch_size,
    )

    loss_sums = {"total": 0.0, "np": 0.0, "ns": 0.0}
    counts = init_metric_counts()

    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            np_pred, ns_logits = model(
                batch["specs"],
                batch["beat_phase"],
                batch["beat_num"],
                batch["difficulty"],
                batch["actions_input"],
            )

            total_loss, np_loss, ns_loss = get_loss(
                np_pred,
                ns_logits,
                batch,
                np_gamma=2.0,
                np_pos_weight=0.8,
                ns_gamma=3.0,
                ns_pos_weight=0.5,
                np_loss_multiplier=7.0,
            )

            valid_count = (
                batch["loss_mask"].bool() & batch["valid_frames"].bool()
            ).sum().item()

            loss_sums["total"] += total_loss.item() * valid_count
            loss_sums["np"] += np_loss.item() * valid_count
            loss_sums["ns"] += ns_loss.item() * valid_count

            update_metric_counts(counts, np_pred, ns_logits, batch)

    denominator = max(counts["valid_count"], 1)
    losses = {key: value / denominator for key, value in loss_sums.items()}
    metrics = finalize_metrics(counts)

    print(f"\n{'=' * 60}\nAutoOsu baseline ({checkpoint_name})\n{'=' * 60}")
    print(
        "Loss   | Total: {:.4f} | NP: {:.4f} | NS: {:.4f}".format(
            losses["total"], losses["np"], losses["ns"]
        )
    )
    print(
        "Onset  | P: {:.4f} | R: {:.4f} | F1: {:.4f}".format(
            metrics["onset_precision"],
            metrics["onset_recall"],
            metrics["onset_f1"],
        )
    )
    print(
        "Action | P: {:.4f} | R: {:.4f} | F1: {:.4f}".format(
            metrics["action_precision"],
            metrics["action_recall"],
            metrics["action_f1"],
        )
    )
    print("Active Exact Accuracy: {:.4f}".format(metrics["active_exact_accuracy"]))
    print(
        "GT Active Rate: {:.2%} | Predicted Active Rate: {:.2%}".format(
            metrics["gt_active_rate"],
            metrics["predicted_active_rate"],
        )
    )

    return {"losses": losses, "metrics": metrics}


if __name__ == "__main__":
    evaluate_autoosu_checkpoint(root=ROOT)
