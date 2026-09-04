"""Model / trainer registry.

``get_model_cls(model_name, training_type)`` returns the trainer class that
implements ``load_components`` / ``compute_loss`` / ``validation_step`` for a
given backbone.  Currently only the StM Composer built on CogVideoX-I2V is
registered (both ``sft`` and ``lora`` training types share the class).
"""

from __future__ import annotations

from typing import Dict, Literal, Type

SUPPORTED_MODELS: Dict[str, Dict[str, Type]] = {}


def register(model_name: str, training_type: Literal["lora", "sft"], trainer_cls: Type) -> None:
    SUPPORTED_MODELS.setdefault(model_name, {})[training_type] = trainer_cls


def get_model_cls(model_name: str, training_type: Literal["lora", "sft"]) -> Type:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' is not registered. Available: {list(SUPPORTED_MODELS)}")
    if training_type not in SUPPORTED_MODELS[model_name]:
        raise ValueError(f"Training type '{training_type}' is not supported for '{model_name}'")
    return SUPPORTED_MODELS[model_name][training_type]


def show_supported_models() -> None:
    for name, types in SUPPORTED_MODELS.items():
        print(f"{name}: {', '.join(types)}")


# register built-ins (import at the end to avoid circular imports)
from .stm_composer import StMComposerTrainer  # noqa: E402,F401

register("cogvideox-i2v", "sft", StMComposerTrainer)
register("cogvideox-i2v", "lora", StMComposerTrainer)

__all__ = ["SUPPORTED_MODELS", "register", "get_model_cls", "show_supported_models", "StMComposerTrainer"]
