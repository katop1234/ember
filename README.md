<p align="center">
  <img src="assets/ember-logo.png" alt="Ember" width="440">
</p>

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

Hand `Ember` the **model** and it figures out the routing itself — one line, nothing else changes:

```python
from ember import Ember
# was: optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = Ember(model, lr=1e-3)
```

Given the model, Ember puts its factored update on the **token tables only** — every
`nn.Embedding` weight plus the LM head — and runs a standard **AdamW** on everything else
(attention/MLP linears, norms, biases). **Hidden linear layers are never Embered.** It's a single
optimizer with the usual `step()` / `zero_grad()` / `state_dict()` API.

`Ember(model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, body_lr=None)`. `betas` /
`eps` / `weight_decay` apply to the AdamW side; `betas[1]` (or an explicit `beta2=`) sets Ember's
second moment. `body_lr` overrides the learning rate on the non-table params (defaults to `lr`).
Tied embeddings are de-duped.

## Recommended usage — a different optimizer on the body (e.g. Muon)

`Ember(model, ...)` already keeps AdamW on the body. If you want **Muon** (or any other optimizer)
on the body instead, split the params yourself and run two optimizers:

```python
import torch
from ember import Ember, split_embedding_params

emb, other = split_embedding_params(model)         # nn.Embedding + lm_head -> emb; tied de-duped
opt_emb   = Ember(emb, lr=1e-3)
opt_other = torch.optim.AdamW(other, lr=3e-4)       # or Muon(other, ...)

for batch in loader:
    loss = model(batch).loss
    opt_emb.zero_grad(); opt_other.zero_grad()
    loss.backward()
    opt_emb.step();      opt_other.step()
```

(Handed an explicit param iterable like `Ember(emb, ...)`, Ember applies its factored update to
every 2-D tensor you give it — so only pass it the token-table params.)

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
