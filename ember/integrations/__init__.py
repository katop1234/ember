"""Framework carve-out recipes (raw). FSDP2 needs none — Ember auto-detects DTensor."""
from .deepspeed import build_deepspeed_carveout, EmberDeepSpeedStepper
from .megatron import megatron_embedding_params

__all__ = ["build_deepspeed_carveout", "EmberDeepSpeedStepper", "megatron_embedding_params"]
