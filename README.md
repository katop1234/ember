<p align="center">
  <img src="ember-logo.png" alt="Ember" width="440">
</p>

<p align="center">
  📄 <b>Paper:</b> <a href="https://arxiv.org/abs/2607.01455">Token Geometry (arXiv:2607.01455)</a>
</p>

The embedding table and LM-head are a language model's read/write interface between discrete
tokens and continuous computation. Their gradient geometry is different from dense hidden
weights — Ember exploits that: a drop-in optimizer for token tables using **O(V + D)** state
instead of Adam's **O(2VD)**, at matched quality. ~**1500× less** optimizer memory at 50K
vocab × 768 dim, growing with vocabulary.

Row × column factored second moment, no first moment. One knob: `beta2` (default `0.999`).

**Why it helps:**
- **Memory:** state is ~1 MB → replicate it, never shard it. Token tables drop out of
  ZeRO/FSDP optimizer-state sharding.
- **Distributed:** row-sharded tables sync with one ~D-float all-reduce per step
  (put it on a dedicated communicator — see `ember.py`).
- **Deterministic:** contiguous reductions only, no atomics → bitwise reproducible at
  fixed world size.
- **Cheap step:** touches only gradient + weights (no m/v buffers) — ~3× less memory
  traffic than Adam's step.

## Install

```bash
pip install -e .
```
Or copy [`ember.py`](ember.py) — the whole repo is one file. PyTorch ≥ 2.1.

## Quickstart

```python
from ember import Ember
# was: optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = Ember(model, lr=1e-3)
```

Given the model, Ember routes automatically: factored update on token tables (every
`nn.Embedding` + the LM head, tied weights de-duped), internal **AdamW** on everything else.
Standard `step()` / `zero_grad()` / `state_dict()`. `betas`/`eps`/`weight_decay` apply to the
AdamW side; `beta2=` overrides Ember's; `body_lr=` overrides the non-table lr.

## Different body optimizer (e.g. Muon)

```python
from ember import Ember, split_embedding_params

emb, other = split_embedding_params(model)
opt_emb   = Ember(emb, lr=1e-3)
opt_other = Muon(other, lr=2e-2)   # or torch.optim.AdamW(other, ...)
```

## Measured (July 2026)

- **Parity with tuned Adam on Adam's home turf**: on the
  [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt), swapping
  Adam→Ember on the token tables at each optimizer's own optimum differs by ~0.001 val
  loss — within seed noise (n=10 vs n=15 at 8×H100).
- **Memory:** 2 GB → ~400 KB optimizer state at Pythia-2.8B; −3 GB peak VRAM at speedrun scale.
- **Stability:** Adam's dense second moment goes stale on rare rows and throws 10²–10⁴×
  oversized steps late in training; Ember's row statistic stays 1–3× throughout.
- **One config for the whole token interface** — input table + LM-head, no per-table tuning.

## Distributed reference

| setup | what you do |
|---|---|
| single-device / DDP / ZeRO-1 | nothing |
| FSDP2 (`fully_shard`) | nothing — DTensor row-sharding auto-detected |
| Megatron / custom TP | `Ember(emb, row_shard_group=<group>)` |
| DeepSpeed ZeRO-2/3 | keep the table's state replicated (out of the ZeRO partition) |

State is fp32 (mixed-precision safe); checkpoints via the standard optimizer `state_dict`.
