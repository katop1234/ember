# Ember

A drop-in optimizer for the **token-embedding table**. Matches AdamW quality at **O(V + D)**
optimizer state instead of Adam's **O(2·V·D)** — about **1500× less** optimizer memory for
the embedding at a 50K vocab, and the gap grows with vocabulary.

Ember stores a row × column factored second moment and no first moment. The embedding's
per-coordinate update scale factorizes (token participation × feature), so the full `V×D`
second moment is redundant — `V + D` captures it.

```python
from ember import Ember, split_embedding_params

emb, other = split_embedding_params(model)
opt_emb   = Ember(emb, lr=1e-3)
opt_other = torch.optim.AdamW(other, lr=3e-4)
```

## Install
```bash
pip install git+https://github.com/katop1234/ember.git
```
PyTorch ≥ 2.4.

## Distributed
Ember's state is ~1 MB, so **replicate it like a LayerNorm parameter — don't shard it**
(exclude it from your ZeRO optimizer partition). The framework still shards the embedding
*parameter* (FSDP2/TP); Ember rides along — `r` (per-row) stays local, and `c` (per-col)
plus `mean` need one all-reduce (`D` floats + a scalar) over the row-shard group.

| setup | what you do |
|---|---|
| single-device / DDP | nothing |
| FSDP2 (`fully_shard`) | nothing — auto-detected from the DTensor placement |
| Megatron / custom | `Ember(emb, row_shard_group=<group>)` |
| DeepSpeed ZeRO | keep the embedding out of the ZeRO optimizer, step Ember around `engine.step()` |

`Ember` is bit-identical to the single-device `EmberReference` whether the table is
replicated or row-sharded (`tests/`).

```bash
python tests/test_reference.py                          # single-device
torchrun --nproc_per_node=2 tests/test_distributed.py   # sharded == reference
torchrun --nproc_per_node=2 examples/fsdp2_minimal.py
```

## When it helps
Free the embedding optimizer state and spend that memory on batch size, context length, or
model capacity. The win grows with vocabulary, so it's especially handy for large or
multilingual vocabularies, per-layer embeddings, and memory-bound fine-tuning — but it's a
clean drop-in for any embedding table.

## Files
```
ember/
  ember_reference.py   # the spec — read first
  ember.py             # distributed (DTensor-aware)
  param_utils.py       # split_embedding_params(model)
```
