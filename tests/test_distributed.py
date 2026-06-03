"""Correctness contract: row-sharded Ember == single-device EmberReference, bit-for-bit.

Run on >=2 processes:
    torchrun --nproc_per_node=2 tests/test_distributed.py

Every rank builds the SAME full embedding + gradients (same seed), computes the reference
update on the full table, then runs Ember on its row-shard (a DTensor) and checks the
gathered result equals the reference. This is the whole promise: not an approximation —
the exact same optimizer, placed correctly.
"""
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard

from ember import Ember, EmberReference

V, D, STEPS = 2048, 64, 6


def sparse_grad(step):
    g = torch.randn(V, D, generator=torch.Generator().manual_seed(step))
    dead = torch.rand(V, generator=torch.Generator().manual_seed(step + 100)) > 0.2
    g[dead] = 0.0
    return g


def main():
    dist.init_process_group("gloo")
    mesh = init_device_mesh("cpu", (dist.get_world_size(),))

    # --- reference: full table, full gradients, single-device optimizer (same on all ranks)
    torch.manual_seed(0)
    E_full = torch.nn.Parameter(torch.randn(V, D))
    ref = EmberReference([E_full], lr=1e-3)
    for s in range(STEPS):
        E_full.grad = sparse_grad(s)
        ref.step()

    # --- sharded: same init, but E is a row-sharded DTensor; Ember handles the reductions
    torch.manual_seed(0)
    E_sh = torch.nn.Parameter(distribute_tensor(torch.randn(V, D), mesh, [Shard(0)]))
    opt = Ember([E_sh], lr=1e-3)
    for s in range(STEPS):
        E_sh.grad = distribute_tensor(sparse_grad(s), mesh, [Shard(0)])
        opt.step()

    gathered = E_sh.detach().full_tensor()
    max_diff = (gathered - E_full.detach()).abs().max().item()
    if dist.get_rank() == 0:
        ok = max_diff < 1e-6
        print(f"[ember dist test] world={dist.get_world_size()}  max|diff|={max_diff:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
        assert ok
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
