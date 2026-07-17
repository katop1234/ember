<p align="center">
  <img src="ember-logo.png" alt="Ember" width="440">
</p>

<p align="center">
  📄 <b>Paper:</b> <a href="https://arxiv.org/abs/2607.01455">Token Geometry (arXiv:2607.01455)</a>
</p>

Ember is a drop-in optimizer for embedding / LM-head matrices that uses **O(V + D)** optimizer
state instead of Adam's **O(2·V·D)**, at matched quality — about **1500× less** optimizer memory
at a 50K vocab × 768 dim, growing with vocabulary.

It stores a row × column factored second moment and **no first moment**. The only knob is
`beta2` (default `0.999`).

**Distributed-training aware, by design:**
- State is ~1 MB → **replicated, never sharded**. Token tables drop out of ZeRO/FSDP
  optimizer-state sharding entirely.
- Row-sharded tables sync with **one ~D-float all-reduce** per step — below NCCL's latency
  floor, effectively free.
- Stats are built with contiguous reductions only (no atomics) → updates are **bitwise
  identical at any world size**. Debugging distributed training stops being archaeology.

## Install
```bash
pip install -e .
```
Or copy [`ember.py`](ember.py) into your project — the whole optimizer is one file. PyTorch ≥ 2.1.

## Quickstart — a one-line diff from Adam

```python
from ember import Ember
# was: optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = Ember(model, lr=1e-3)
```

Handed the model, Ember routes automatically: the factored update on token tables only (every
`nn.Embedding` + the LM head, tied weights de-duped), a standard internal **AdamW** on everything
else — hidden linears are never Embered. Usual `step()` / `zero_grad()` / `state_dict()` API.

`betas` / `eps` / `weight_decay` apply to the AdamW side; `beta2=` overrides Ember's second
moment (default `betas[1]`); `body_lr=` overrides the non-table lr (default `lr`).

## With a different body optimizer (e.g. Muon)

Split the params and run two optimizers:

```python
from ember import Ember, split_embedding_params

emb, other = split_embedding_params(model)
opt_emb   = Ember(emb, lr=1e-3)          # explicit params: factored update on 2-D tensors
opt_other = Muon(other, lr=2e-2)         # or torch.optim.AdamW(other, ...)
```

## What we've measured (July 2026)

- **Parity with tuned Adam on Adam's home turf.** In the [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt) — the most heavily Adam-tuned public benchmark — swapping Adam→Ember on the token tables at each optimizer's own optimum differs by ~0.001 val loss, within seed noise.
- **Inside a faster-than-record recipe.** An Ember-carrying recipe reached the speedrun target under the leaderboard's statistical rules in fewer steps than the standing record (PR in preparation).
- **Memory:** 2 GB → ~400 KB optimizer state at Pythia-2.8B; ~3 GB lower peak VRAM at speedrun scale.
- **Late-training stability:** Adam's dense second moment goes stale on rare rows and throws transiently oversized steps (measured 10²–10⁴× calibrated); Ember's row statistic is refreshed by whole-row traffic and stays ~1–3× throughout.
- **One config, whole token interface.** The same recipe on input embedding + LM-head tables was parity-or-better in every paired trial — no per-table tuning.

## Distributed reference

| setup | what you do |
|---|---|
| single-device / DDP / ZeRO-1 | nothing |
| FSDP2 (`fully_shard`) | nothing — DTensor row-sharding auto-detected |
| Megatron / custom TP | `Ember(emb, row_shard_group=<group>)` |
| DeepSpeed ZeRO-2/3 | keep the table's state replicated (out of the ZeRO partition) |

State is fp32 (mixed-precision safe) and checkpoints via the standard optimizer `state_dict`.

## Files
```
ember.py   # the optimizer + split_embedding_params — one file, that's the repo
```
