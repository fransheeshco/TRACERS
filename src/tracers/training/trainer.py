from pathlib import Path

import time
import torch
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

from .metrics import (
    init_metric_counts,
    finalize_metrics,
    update_metric_counts,
)
from ..models.loss import get_loss


def select_action(logits, temperature=None, top_k=None, top_p=None):
    """logits: [B, 256]. temperature=None (or <=0) means greedy."""
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k is not None and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        prev_cum = sorted_probs.cumsum(dim=-1) - sorted_probs
        sorted_logits = sorted_logits.masked_fill(prev_cum >= top_p, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        valid_loader,
        device,
        experiment_config,
        hyperparams,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device

        self.experiment_config = experiment_config
        self.hyperparams = hyperparams

        self.checkpoint_path = Path(
            experiment_config.checkpoint_path
        )
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self.num_epochs = experiment_config.num_epochs

        self.np_fl_gamma = hyperparams.np_fl_gamma
        self.np_fl_weight = hyperparams.np_fl_weight
        self.ns_fl_gamma = hyperparams.ns_fl_gamma
        self.ns_fl_weight = hyperparams.ns_fl_weight
        self.np_loss_multiplier = hyperparams.np_loss_multiplier

        self.start_epoch = 0
        self.best_val_loss = float("inf")

    # =========================================
    # CHECKPOINTING
    # =========================================

    def save_checkpoint(self, epoch, filename):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        path = self.checkpoint_path / filename
        torch.save(checkpoint, path)

        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, filename, load_optimizer=True):
        checkpoint = torch.load(filename, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.start_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Loaded (epoch {self.start_epoch}, "
            f"best val loss {self.best_val_loss:.4f})")

    def _get_global_model_kwargs(self, batch):
        # Original models don't need global audio.
        if not hasattr(self.model, "prepare_global_context"):
            return {}

        # Global models require these additional batch fields.
        required = ["global_specs", "global_indices", "start_frame"]
        missing = [key for key in required if key not in batch]

        if missing:
            raise ValueError(
                f"Global-context model requires batch fields: {missing}. "
                "Use GlobalAudioCollator with this DataLoader."
            )

        return {
            "global_specs": batch["global_specs"],
            "global_indices": batch["global_indices"],
            "start_frames": batch["start_frame"],
        }

    # =========================================
    # ONE EPOCH
    # =========================================

    def _run_epoch(self, loader, training):
        if training:
            self.model.train()
            description = "Training"
        else:
            self.model.eval()
            description = "Validation"

        loss_sums = {
            "total": 0.0,
            "np": 0.0,
            "ns": 0.0,
        }

        metric_counts = init_metric_counts()

        # Enable gradients for training, disable for validation.
        with torch.set_grad_enabled(training):

            for batch in tqdm(loader, desc=description):

                # Move tensor values in the batch to the device.
                batch = {
                    key: value.to(self.device)
                    if torch.is_tensor(value)
                    else value
                    for key, value in batch.items()
                }

                specs = batch["specs"]
                beat_phases = batch["beat_phase"]
                beat_nums = batch["beat_num"]
                difficulties = batch["difficulty"]
                actions_input = batch["actions_input"]
                valid_frames = batch["valid_frames"]

                model_kwargs = self._get_global_model_kwargs(batch)


                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                # Forward pass
                np_pred, ns_logits = self.model(
                    specs,
                    beat_phases,
                    beat_nums,
                    difficulties,
                    actions_input,
                    valid_frames,
                    **model_kwargs
                )

                # Calculate losses
                total_loss, np_loss, ns_loss = get_loss(
                    np_pred,
                    ns_logits,
                    batch,
                    np_gamma=self.np_fl_gamma,
                    np_pos_weight=self.np_fl_weight,
                    ns_gamma=self.ns_fl_gamma,
                    ns_pos_weight=self.ns_fl_weight,
                    np_loss_multiplier=self.np_loss_multiplier,
                )

                # Backpropagation and optimizer update (training only)
                if training:
                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0,
                    )

                    self.optimizer.step()

                # Count frames contributing to the loss.
                valid_count = (
                    batch["loss_mask"].bool()
                    & valid_frames.bool()
                ).sum().item()

                # Accumulate weighted losses.
                loss_sums["total"] += total_loss.item() * valid_count
                loss_sums["np"] += np_loss.item() * valid_count
                loss_sums["ns"] += ns_loss.item() * valid_count

                # Accumulate metric counts.
                update_metric_counts(
                    metric_counts,
                    np_pred,
                    ns_logits,
                    batch,
                )

        # =====================================
        # EPOCH RESULTS
        # =====================================

        denominator = max(metric_counts["valid_count"], 1)

        losses = {
            key: value / denominator
            for key, value in loss_sums.items()
        }

        metrics = finalize_metrics(metric_counts)

        return losses, metrics

    # =========================================
    # FULL TRAINING
    # =========================================

    def fit(self):

        self.model.to(self.device)

        # Initialize W&B
        use_wandb = self.experiment_config.log_to_wandb

        if use_wandb:
            wandb.init(
                project=self.experiment_config.wandb_project,
                name=self.experiment_config.run_name,
                config={
                    "experiment": OmegaConf.to_container(
                        self.experiment_config, resolve=True,
                    ),
                    "hyperparams": OmegaConf.to_container(
                        self.hyperparams, resolve=True,
                    ),
                },
                settings=wandb.Settings(console="wrap"),
            )

        try:
            for epoch in tqdm(
                range(self.start_epoch, self.num_epochs),
                desc="Epochs",
            ):
                epoch_start = time.perf_counter()

                # TRAINING
                train_start = time.perf_counter()
                train_losses, train_metrics = self._run_epoch(
                    self.train_loader, training=True,
                )
                train_seconds = time.perf_counter() - train_start

                # VALIDATION
                val_start = time.perf_counter()
                val_losses, val_metrics = self._run_epoch(
                    self.valid_loader, training=False,
                )
                val_seconds = time.perf_counter() - val_start
                epoch_seconds = time.perf_counter() - epoch_start

                # PRINT RESULTS
                print(f"\n{'=' * 60}")
                print(f"Epoch {epoch + 1}/{self.num_epochs}")
                print(f"{'=' * 60}")

                self._print_results("TRAIN", train_losses, train_metrics)
                self._print_results("VAL", val_losses, val_metrics)

                # W&B LOGGING
                if use_wandb:
                    log_data = {
                        "epoch": epoch + 1,

                        "train/total_loss": train_losses["total"],
                        "train/np_loss": train_losses["np"],
                        "train/ns_loss": train_losses["ns"],

                        "val/total_loss": val_losses["total"],
                        "val/np_loss": val_losses["np"],
                        "val/ns_loss": val_losses["ns"],
                        "time/train_seconds": train_seconds,
                        "time/val_seconds": val_seconds,
                        "time/epoch_seconds": epoch_seconds,
                    }

                    for key, value in train_metrics.items():
                        log_data[f"train/{key}"] = value

                    for key, value in val_metrics.items():
                        log_data[f"val/{key}"] = value

                    wandb.log(log_data)



                if val_losses["total"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total"]
                    self.save_checkpoint(epoch=epoch + 1, filename="best.pt")

                # CHECKPOINTING
                self.save_checkpoint(epoch=epoch + 1, filename="latest.pt")

        finally:
            if use_wandb:
                wandb.finish()

    @torch.inference_mode()
    def predict(self, batch):

        self.model.eval()

        # Move batch tensors to device
        batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

        model_kwargs = self._get_global_model_kwargs(batch)

        # Forward pass
        np_pred, ns_logits = self.model(
            batch["specs"],
            batch["beat_phase"],
            batch["beat_num"],
            batch["difficulty"],
            batch["actions_input"],
            batch["valid_frames"],
            **model_kwargs
        )

        # Select the most likely action token
        action_predictions = ns_logits.argmax(dim=-1)

        return {
            "np_pred": np_pred.cpu(),
            "ns_logits": ns_logits.cpu(),
            "action_predictions": action_predictions.cpu(),
            "actions_target": batch["actions_target"].cpu(),
            "valid_frames": batch["valid_frames"].bool().cpu(),
            "loss_mask": batch["loss_mask"].bool().cpu(),
        }



    @torch.inference_mode()
    def generate(
        self,
        batch,
        seed_action=0,
        temperature=None,
        top_k=None,
        top_p=None,
        seed=None,
        prefix_actions=None,
        global_context=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        self.model.eval()

        # =====================================
        # INPUTS
        # =====================================

        specs = batch["specs"].to(self.device)
        beat_phases = batch["beat_phase"].to(self.device)
        beat_nums = batch["beat_num"].to(self.device)
        difficulties = batch["difficulty"].to(self.device)

        batch_size = specs.size(0)
        seq_len = specs.size(2)

        if seq_len == 0:
            raise ValueError("Cannot generate an empty sequence.")

        if not 0 <= seed_action < 256:
            raise ValueError("seed_action must be between 0 and 255.")

        valid_frames = batch.get("valid_frames")

        if valid_frames is None:
            valid_frames = torch.ones(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=self.device,
            )
        else:
            valid_frames = valid_frames.to(self.device).bool()

        # =====================================
        # INITIALIZE ACTION INPUTS
        # =====================================

        actions_input = torch.zeros(
            batch_size,
            seq_len,
            dtype=torch.long,
            device=self.device,
        )

        generated_actions = torch.zeros_like(actions_input)

        # Determine where autoregressive generation begins.
        if prefix_actions is None:
            # First window: use the initial seed.
            actions_input[:, 0] = seed_action
            generate_from = 0

        else:
            # Later windows: load the previous generated history.
            prefix_actions = torch.as_tensor(
                prefix_actions,
                dtype=torch.long,
                device=self.device,
            )

            # Allow a single-song prefix with shape [P].
            if prefix_actions.ndim == 1:
                prefix_actions = prefix_actions.unsqueeze(0)

            if prefix_actions.ndim != 2:
                raise ValueError("prefix_actions must have shape [B, P].")

            if prefix_actions.size(0) != batch_size:
                raise ValueError("prefix_actions batch size does not match.")

            prefix_len = prefix_actions.size(1)

            if prefix_len == 0 or prefix_len > seq_len:
                raise ValueError(
                    "prefix_actions length must be between 1 and seq_len."
                )

            # Load previous action history.
            actions_input[:, :prefix_len] = prefix_actions

            # The first P-1 output actions are already known.
            generate_from = prefix_len - 1

            generated_actions[:, :generate_from] = prefix_actions[:, 1:]

        # =====================================
        # AUTOREGRESSIVE GENERATION
        # =====================================

        ns_logits_steps = None
        np_pred = None

        for t in range(generate_from, seq_len):
            model_kwargs = {}

            if global_context is not None:
                model_kwargs["global_context"] = global_context

            np_pred, ns_logits = self.model(
                specs,
                beat_phases,
                beat_nums,
                difficulties,
                actions_input,
                valid_frames,
                **model_kwargs,
            )

            if ns_logits_steps is None:
                # Zero initialization keeps prefix logits defined.
                ns_logits_steps = torch.zeros_like(ns_logits)

            current_logits = ns_logits[:, t, :]
            ns_logits_steps[:, t, :] = current_logits

            # Greedy or sampled action.
            next_action = select_action(
                current_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Ignore padded positions.
            next_action = torch.where(
                valid_frames[:, t],
                next_action,
                torch.zeros_like(next_action),
            )

            generated_actions[:, t] = next_action

            # Feed prediction into the next timestep.
            if t + 1 < seq_len:
                actions_input[:, t + 1] = next_action

        return {
            "action_predictions": generated_actions.cpu(),
            "action_inputs": actions_input.cpu(),
            "np_pred": np_pred.cpu() if np_pred is not None else None,
            "ns_logits": ns_logits_steps.cpu() if ns_logits_steps is not None else None,
            "valid_frames": valid_frames.cpu(),
        }

    def _print_results(self, name, losses, metrics):
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        print(f"Loss   | Total: {losses['total']:.4f} | "
            f"NP: {losses['np']:.4f} | NS: {losses['ns']:.4f}")
        print(f"Onset  | P: {metrics['onset_precision']:.4f} | "
            f"R: {metrics['onset_recall']:.4f} | F1: {metrics['onset_f1']:.4f}")
        print(f"Action | P: {metrics['action_precision']:.4f} | "
            f"R: {metrics['action_recall']:.4f} | F1: {metrics['action_f1']:.4f}")
        print(f"Active Exact Accuracy: {metrics['active_exact_accuracy']:.4f}")
        print(f"GT Active Rate: {metrics['gt_active_rate']:.2%} | "
            f"Predicted Active Rate: {metrics['predicted_active_rate']:.2%}")

    def evaluate(self, loader=None, checkpoint=None, name="VAL"):
        """Evaluate one checkpoint on one loader. No training, no W&B."""
        loader = self.valid_loader if loader is None else loader
        if loader is None:
            raise ValueError("Pass a loader, or construct the Trainer with valid_loader.")

        if checkpoint is not None:
            self.load_checkpoint(checkpoint, load_optimizer=False)

        self.model.to(self.device)

        start = time.perf_counter()
        losses, metrics = self._run_epoch(loader, training=False)
        seconds = time.perf_counter() - start

        self._print_results(f"{name} ({seconds:.0f}s)", losses, metrics)
        return {"losses": losses, "metrics": metrics}