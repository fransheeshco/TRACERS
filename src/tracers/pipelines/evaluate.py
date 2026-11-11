from __future__ import annotations


def run_evaluation(trainer, loader, *, checkpoint=None, name="eval"):
    trainer.evaluate(loader, checkpoint=checkpoint, name=name)
