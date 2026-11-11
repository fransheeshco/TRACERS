from __future__ import annotations

from tracers.models.transformer import TracerGlobalNpModel, TracerNpModel


MODEL_REGISTRY = {
    "tracer_np": TracerNpModel,
    "tracer_global_np": TracerGlobalNpModel,
    "local": TracerNpModel,
    "global": TracerGlobalNpModel,
    "np": TracerNpModel,
    "global_np": TracerGlobalNpModel,
}


def get_model_class(model_key: str):
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as exc:
        raise ValueError(f"Unknown model key: {model_key!r}") from exc
