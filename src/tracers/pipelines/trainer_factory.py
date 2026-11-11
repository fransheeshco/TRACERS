from __future__ import annotations

import torch

from tracers.models.registry import get_model_class
from tracers.training.trainer import Trainer


def build_trainer(
    *,
    model_key: str,
    hyperparams,
    experiment_config,
    device,
    checkpoint_path: str | None = None,
    train_loader=None,
    valid_loader=None,
):
    model_cls = get_model_class(model_key)
    model = model_cls(hyperparams).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        experiment_config=experiment_config,
        hyperparams=hyperparams,
    )

    if checkpoint_path is not None:
        trainer.load_checkpoint(checkpoint_path)

    return trainer, model
