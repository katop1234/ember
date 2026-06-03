"""Routing helper: send the embedding table(s) to Ember, everything else to your optimizer.

Usage:
    from ember import Ember, split_embedding_params

    emb_params, other_params = split_embedding_params(model)
    opt_emb   = Ember(emb_params, lr=1e-3)
    opt_other = torch.optim.AdamW(other_params, lr=...)   # or your existing ZeRO optimizer

Then step both. Put `other_params` in your normal ZeRO/FSDP partition; keep `emb_params`
OUT of it (replicate the tiny Ember state). See README for the per-framework two-liner.
"""
import torch.nn as nn


def split_embedding_params(model, extra_names=()):
    """Return (embedding_params, other_params).

    An embedding parameter is the .weight of an nn.Embedding, or any 2-D parameter whose
    name contains one of: embed, wte, tok_emb, word_embeddings (+ any `extra_names`).
    Tied weights are de-duplicated by identity.
    """
    keys = ("embed", "wte", "tok_emb", "word_embeddings") + tuple(extra_names)
    emb_ids = set()
    for m in model.modules():
        if isinstance(m, nn.Embedding) and m.weight is not None:
            emb_ids.add(id(m.weight))
    emb, other, seen = [], [], set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        is_emb = id(p) in emb_ids or (p.dim() == 2 and any(k in name.lower() for k in keys))
        (emb if is_emb else other).append(p)
    return emb, other
