# Ember

Ember is a drop-in optimizer for embedding / LM-head matrices that uses **O(V + D)** optimizer
state instead of Adam's **O(2·V·D)**, at matched quality. About **1500× less** optimizer memory
for the embedding at a 50K vocab × 768 dim — and the gap grows with vocabulary.

Ember stores a row × column factored second moment and **no first moment**. The embedding's
per-coordinate update scale factorizes (token participation × feature), so the full `V×D` second
moment is redundant — `V + D` captures it. The only knob is `beta2` (default `0.999`).

## Install
```bash
pip install -e .
```
Or just copy [`ember.py`](ember.py) into your project — it's the whole optimizer in one file
(single-device, and sharding-aware when DTensor is available). PyTorch ≥ 2.1.

## Quickstart — a one-line diff from Adam

`Ember` takes the **exact** `torch.optim.Adam` constructor signature, so swapping is literally
one line and nothing else changes:

```python
from ember import Ember
# was: optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = Ember(model.parameters(), lr=1e-3)   # same call, ~D× less state on token tables
```

`Ember(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)`. Ember has no first
moment, so `betas[0]` is ignored and `betas[1]` is used as `beta2` (you can also pass `beta2=`
directly; it wins if both are given). Handed a whole model, 2-D matrices get the factored Ember
update and any non-2-D parameter (biases, norms, 1-D) falls back to a plain scaled-SGD step, so
the optimizer never errors. `.step()`, `.zero_grad()`, and `state_dict`/`load_state_dict` are the
standard `torch.optim.Optimizer` methods.

## Recommended usage — Ember on the token tables, your optimizer on the rest

The win is on the embedding / LM-head. Route those to Ember and keep AdamW (or Muon) on the body:

```python
import torch
from ember import Ember, split_embedding_params

emb, other = split_embedding_params(model)         # embed/wte/lm_head -> emb; tied weights de-duped
opt_emb   = Ember(emb, lr=1e-3)
opt_other = torch.optim.AdamW(other, lr=3e-4)

for batch in loader:
    loss = model(batch).loss
    opt_emb.zero_grad(); opt_other.zero_grad()
    loss.backward()
    opt_emb.step();      opt_other.step()
```

## What state it stores

For a `V × D` token table, Adam stores **two** dense `V × D` buffers (first and second moment) —
`2·V·D` numbers. Ember stores a length-`V` row factor and a length-`D` column factor — `V + D`
numbers, no first moment — and reconstructs the per-entry denominator on the fly via geometric-mean
normalization of those two vectors. At a 50K × 768 table that's ~77M Adam buffers vs ~51K Ember
buffers (~1500× less); the ratio grows with vocabulary.

## Distributed
Ember runs under FSDP2, tensor parallelism, and ZeRO. Its state is ~1 MB, so **replicate it — don't
shard it.** Flat-sharding a tiny, structured state just adds communication for no memory.

When the embedding gradient is sharded along rows, only the column factor needs syncing — one small
all-reduce (`D` floats + a scalar) over the row-shard group. `Ember` does this automatically for
DTensor (FSDP2), or pass `row_shard_group=<group>` for non-DTensor frameworks (e.g. Megatron).

| setup | what you do |
|---|---|
| single-device / DDP / ZeRO-1 | nothing — full gradient, replicated state |
| FSDP2 (`fully_shard`) | nothing — auto-detected from the DTensor placement |
| Megatron / custom TP | `Ember(emb, row_shard_group=<group>)` |
| DeepSpeed ZeRO-2/3 | keep the embedding's state replicated (out of the ZeRO partition) |

Bit-identical to single device whether the table is replicated or row-sharded. State is **fp32**
(mixed-precision safe) and checkpoints with the standard optimizer `state_dict`.

## Files
```
ember.py        # the optimizer + split_embedding_params — one file
test_ember.py   # CPU test: Adam-signature construct, train, whole-model step, state_dict round-trip
```

Run the test with `python test_ember.py`.
