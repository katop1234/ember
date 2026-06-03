# Ember

A drop-in optimizer for the **token-embedding table**. Matches AdamW quality at **O(V + D)**
optimizer state instead of Adam's **O(2·V·D)** — about **1500× less** optimizer memory for
the embedding at a 50K vocab, and the gap grows with vocabulary.

Ember stores a row × column factored second moment and no first moment. The embedding's
per-coordinate update scale factorizes (token participation × feature), so the full `V×D`
second moment is redundant — `V + D` captures it.

```python
import torch
from ember import Ember, split_embedding_params

emb, other = split_embedding_params(model)
opt_emb   = Ember(emb, lr=1e-3)
opt_other = torch.optim.AdamW(other, lr=3e-4)
```

## Install
```bash
pip install git+https://github.com/katop1234/ember.git
```
Or just copy [`ember.py`](ember.py) — it's the whole optimizer in one file (single-device,
and sharding-aware when DTensor is available). PyTorch ≥ 2.4.

## Distributed
Ember runs under FSDP2, tensor parallelism, and ZeRO. Its state is ~1 MB, so **replicate it
— don't shard it.** Flat-sharding a tiny, structured state (`v_row`/`v_col`) just adds
communication for no memory; keep it replicated and out of your ZeRO partition.

When the embedding gradient is sharded, only the column factor needs syncing — one small
all-reduce (`D` floats + a scalar) over the row-shard group. `Ember` does this automatically
for DTensor (FSDP2), or pass `row_shard_group=<group>` for non-DTensor frameworks.

| setup | what you do |
|---|---|
| single-device / DDP / ZeRO-1 | nothing — full gradient, replicated state |
| FSDP2 (`fully_shard`) | nothing — auto-detected from the DTensor placement |
| Megatron / custom TP | `Ember(emb, row_shard_group=<group>)` |
| DeepSpeed ZeRO-2/3 | keep the embedding's state replicated (out of the ZeRO partition) |

Bit-identical to single device whether the table is replicated or row-sharded. State is
**fp32** (mixed-precision safe), checkpoints with the standard optimizer `state_dict`, and
the only communication is the `O(D)` column all-reduce — small enough that no compute/comms
overlap is needed.

## When it helps
Free the embedding optimizer state and spend that memory on batch size, context length, or
model capacity. The win grows with vocabulary, so it's especially handy for large or
multilingual vocabularies, per-layer embeddings, and memory-bound fine-tuning — but it's a
clean drop-in for any embedding table.

## Files
```
ember.py     # the optimizer + split_embedding_params — one file
```
