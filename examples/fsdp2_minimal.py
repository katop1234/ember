"""Minimal FSDP2 example: Ember on the embedding, AdamW on everything else.

    torchrun --nproc_per_node=2 examples/fsdp2_minimal.py

The only Ember-specific lines are the param split and the two optimizers. `fully_shard`
shards the embedding row-wise (dim 0); Ember reads that sharding and does its one small
v_col all-reduce. Nothing else to configure.
"""
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

from ember import Ember, split_embedding_params

V, D = 4096, 256


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.lin = nn.Linear(D, V, bias=False)

    def forward(self, x):
        return self.lin(self.emb(x))


def main():
    dist.init_process_group("gloo")
    torch.manual_seed(0)
    model = Tiny()
    fully_shard(model.emb)            # embedding sharded row-wise
    fully_shard(model)

    emb_params, other_params = split_embedding_params(model)
    opt_emb = Ember(emb_params, lr=1e-3)                       # ~1 MB state, replicated
    opt_other = torch.optim.AdamW(other_params, lr=3e-4)      # your usual optimizer / ZeRO

    for step in range(5):
        x = torch.randint(0, V, (8, 16))
        loss = model(x).float().log_softmax(-1).mean().neg()
        loss.backward()
        opt_emb.step(); opt_other.step()
        opt_emb.zero_grad(); opt_other.zero_grad()
        if dist.get_rank() == 0:
            print(f"step {step}  loss {loss.item():.4f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
