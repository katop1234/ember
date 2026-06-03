"""DeepSpeed ZeRO carve-out for Ember.  [RAW — adapt to your DeepSpeed version.]

DeepSpeed wraps ONE optimizer and ZeRO-shards its state. The trick is to keep the
embedding OUT of that optimizer (it's ~1 MB — nothing to shard, and ZeRO flat-shards
across row boundaries which breaks the row/col factoring). You step Ember yourself,
alongside the engine.

Clean for ZeRO-1 / ZeRO-2 (params replicated; only optimizer state / grads are sharded):

    from ember import Ember, split_embedding_params

    emb_params, other_params = split_embedding_params(model)
    base_opt = torch.optim.AdamW(other_params, lr=3e-4)          # NON-embedding only
    engine, _, _, _ = deepspeed.initialize(
        model=model, optimizer=base_opt, config=ds_config,
    )
    ember = Ember(emb_params, lr=1e-3)                           # replicated, you step it

    # train loop
    engine.backward(loss)        # grads for all params, incl. the embedding
    ember.step()                 # embedding step (not ZeRO-managed)
    engine.step()                # ZeRO step for everything else
    ember.zero_grad(set_to_none=True)

ZeRO-3 (params themselves sharded): the embedding param is partitioned + gathered on the
fly by DeepSpeed, so you can't step it from outside the engine as-is. Easiest fix: keep
the embedding module out of ZeRO-3 partitioning (exclude it from the wrapped module, or
register it as an external parameter) so each rank holds the full table, then carve out as
above. See DeepSpeed's zero.Init / register_external_parameter docs for the exact call in
your version.

This module is a thin convenience wrapper around that recipe.
"""
import torch

from ..ember import Ember
from ..param_utils import split_embedding_params


def build_deepspeed_carveout(model, base_optimizer_cls=torch.optim.AdamW,
                             base_kwargs=None, ember_kwargs=None):
    """Returns (base_optimizer, ember). Pass `base_optimizer` to deepspeed.initialize();
    step `ember` yourself each iteration (after engine.backward, around engine.step)."""
    base_kwargs = base_kwargs or {"lr": 3e-4}
    ember_kwargs = ember_kwargs or {"lr": 1e-3}
    emb_params, other_params = split_embedding_params(model)
    base_optimizer = base_optimizer_cls(other_params, **base_kwargs)
    ember = Ember(emb_params, **ember_kwargs)
    return base_optimizer, ember


class EmberDeepSpeedStepper:
    """Tiny helper so you don't forget to step/zero Ember alongside the engine.

        stepper = EmberDeepSpeedStepper(ember)
        ...
        engine.backward(loss)
        stepper.step()      # ember.step(); ember.zero_grad()
        engine.step()
    """
    def __init__(self, ember: Ember):
        self.ember = ember

    def step(self):
        self.ember.step()
        self.ember.zero_grad(set_to_none=True)
