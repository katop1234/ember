"""Small helpers: route the embedding to Ember, and a tiny model for tests/examples."""
import torch
import torch.nn as nn


def split_embedding_params(model, extra_names=()):
    """Return (embedding_params, other_params).

    Embedding = the .weight of any nn.Embedding, or any 2-D parameter whose name contains
    one of: embed, wte, tok_emb, word_embeddings (+ `extra_names`). Tied weights de-duped.

        emb, other = split_embedding_params(model)
        opt_emb   = Ember(emb, lr=1e-3)
        opt_other = torch.optim.AdamW(other, lr=3e-4)
    """
    keys = ("embed", "wte", "tok_emb", "word_embeddings") + tuple(extra_names)
    emb_ids = {id(m.weight) for m in model.modules()
               if isinstance(m, nn.Embedding) and m.weight is not None}
    emb, other, seen = [], [], set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        is_emb = id(p) in emb_ids or (p.dim() == 2 and any(k in name.lower() for k in keys))
        (emb if is_emb else other).append(p)
    return emb, other


class TinyLM(nn.Module):
    """Minimal embedding + linear head, for the tests and the FSDP2 demo."""
    def __init__(self, vocab=4096, dim=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        return self.head(self.emb(x))
