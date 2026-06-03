"""Megatron-LM (tensor / vocab parallel) for Ember.  [RAW — adapt to your Megatron.]

Megatron's `VocabParallelEmbedding` shards the table along rows (vocab) across the
tensor-parallel group — each TP rank holds V/TP whole rows. That's the clean case for
Ember: `r` (per-row) is local to each rank's rows, and `c` (per-col) + `mean` aggregate
over all rows, so they need one all-reduce over the **tensor-parallel group**.

Megatron params aren't DTensors, so Ember can't auto-detect the sharding — you hand it the
group explicitly via `row_shard_group`:

    from megatron.core import mpu                      # or megatron.mpu, version-dependent
    from ember import Ember
    from ember.integrations.megatron import megatron_embedding_params

    emb_params, other_params = megatron_embedding_params(model)
    ember = Ember(emb_params, lr=1e-3,
                  row_shard_group=mpu.get_tensor_model_parallel_group())
    base_opt = <your Megatron distributed optimizer on other_params>

Keep the embedding out of Megatron's distributed-optimizer param sharding (carve-out, same
spirit as the DeepSpeed recipe) — Ember's state is tiny and replicated per TP rank.
"""
import torch.nn as nn


def megatron_embedding_params(model, class_names=("VocabParallelEmbedding",)):
    """Return (embedding_params, other_params), matching Megatron's vocab-parallel
    embedding by module class name (avoids importing megatron here)."""
    emb_ids = set()
    for m in model.modules():
        if type(m).__name__ in class_names and getattr(m, "weight", None) is not None:
            emb_ids.add(id(m.weight))
    emb, other, seen = [], [], set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        (emb if id(p) in emb_ids else other).append(p)
    return emb, other
