# Ember

A drop-in optimizer for the **token-embedding table** that matches AdamW quality at
**O(V + D)** optimizer state instead of Adam's **O(2·V·D)** — roughly **1500× less**
optimizer memory for the embedding at a 50K vocab, and it scales with vocabulary.

Ember keeps a row × column *factored* second moment and no first moment. It's a
diagonal-Fisher preconditioner specialized to the embedding's structure: the per-coordinate
update scale factorizes (token participation × feature), so storing the full `V×D` second
moment is unnecessary.

```python
from ember import Ember, split_embedding_params

emb_params, other_params = split_embedding_params(model)
opt_emb   = Ember(emb_params, lr=1e-3)                  # ~1 MB state
opt_other = torch.optim.AdamW(other_params, lr=3e-4)    # or your existing optimizer
```

## Install
```bash
pip install git+https://github.com/<you>/ember.git
```
Requires PyTorch ≥ 2.4 (DTensor / FSDP2). Single-device and DDP need nothing extra.

## Distributed — one rule
> **Place Ember's state where your framework places a LayerNorm parameter: replicated,
> kept consistent by the gradient all-reduce you already do.** Exclude it from your ZeRO
> optimizer partition — it's ~1 MB, there is nothing to save by sharding it (and sharding
> *breaks* the row/col factoring, since ZeRO flat-shards across row boundaries).

The framework can still shard the embedding **parameter** (FSDP2/TP) for activation and
param memory. Ember rides along: `r` (per-row) shards with the rows for free, and `c`
(per-col) + `mean(r)` need **one small all-reduce over the row-shard group** — `D` floats
plus a scalar. That's the entire communication cost. Compared to Adam (needs ZeRO to shard
its bulky embedding state) or Muon (needs all-to-all to orthonormalize sharded matrices),
Ember's distributed footprint is negligible.

| setup | what Ember does | the change you make |
|---|---|---|
| single-device / DDP | identical to reference, **zero extra comm** | nothing |
| **FSDP2** (`fully_shard`, dim-0) | `c`+`mean` all-reduce over the row-shard sub-mesh | `fully_shard` the embedding; give its param to Ember |
| **Megatron / TP** (vocab-parallel) | `c`+`mean` all-reduce over the tensor-parallel group | route the embedding param to Ember |
| **DeepSpeed ZeRO-1/2/3** | keep `r`/`c` replicated, step locally | put the embedding in a **separate param group excluded from ZeRO**; rest stays in your ZeRO+Adam |

`Ember` reads a param's DTensor mesh/placement automatically — if rows aren't sharded it's
exactly the single-device reference with no communication. For non-DTensor frameworks pass
the group explicitly: `Ember(emb_params, row_shard_group=<group>)`.

Raw carve-out helpers for the invasive frameworks live in `ember/integrations/`
(`deepspeed.py`, `megatron.py`) — recipes + thin wrappers, adapt to your version.

## Correctness
`Ember` produces **bit-identical** updates to the single-device `EmberReference`, whether
the table is replicated or row-sharded — verified in `tests/`. You're getting the same
optimizer, placed correctly, not an approximation.

```bash
python tests/test_reference.py                          # single-device
torchrun --nproc_per_node=2 tests/test_distributed.py   # sharded == reference, bit-for-bit
torchrun --nproc_per_node=2 examples/fsdp2_minimal.py   # runnable FSDP2 example
```

## Files
```
ember/
  ember_reference.py   # the spec — single-device, read this first
  ember.py             # distributed: DTensor-aware, the v_col all-reduce
  param_utils.py       # split_embedding_params(model)
```

## Scope (stated honestly)
Optimizer-state memory only matters when you're **memory-bound** — large vocab, large
model, or per-layer embeddings (the embedding fraction grows with vocab). Ember's win is
biggest for **vocab-heavy, lightly-sharded** training (multilingual / on-device /
fine-tuning); at hyperscale MoE with thousands of GPUs the embedding is a rounding error
and sharding already amortizes it. Use it where the embedding optimizer state is a real
slice of your budget.
