"""Tests + demo.

    python test.py                                # single-device tests
    torchrun --nproc_per_node=2 test.py           # + sharded==single-device, + FSDP2 demo
"""
import os
import torch
import torch.nn as nn

from ember import Ember, split_embedding_params

V, D = 2048, 64


class _TinyLM(nn.Module):
    def __init__(self, vocab=V, dim=D):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        return self.head(self.emb(x))


def sparse_grad(seed):
    g = torch.randn(V, D, generator=torch.Generator().manual_seed(seed))
    g[torch.rand(V, generator=torch.Generator().manual_seed(seed + 99)) > 0.2] = 0.0
    return g


def _run(steps=8, lr=1e-3):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(V, D))
    opt = Ember([p], lr=lr)
    for s in range(steps):
        p.grad = sparse_grad(s)
        opt.step()
    return p, opt


# ---- single-device ----
def test_basics():
    p, opt = _run()
    st = opt.state[p]
    assert torch.isfinite(p).all()
    assert st["r"].shape == (V,) and st["c"].shape == (D,)        # factored V+D state
    assert st["r"].dtype == torch.float32                          # fp32 state
    assert st["r"].numel() + st["c"].numel() == V + D             # vs Adam's 2*V*D


def test_state_dict_roundtrip():
    _, opt = _run(steps=4)
    sd = opt.state_dict()
    p2 = torch.nn.Parameter(torch.randn(V, D))
    opt2 = Ember([p2], lr=1e-3)
    opt2.load_state_dict(sd)                                       # resume works
    for p in opt2.state:
        assert torch.equal(opt2.state[p]["c"], list(opt.state.values())[0]["c"])


# ---- distributed: sharded == single-device, bit-for-bit ----
def test_distributed():
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard

    mesh = init_device_mesh("cpu", (dist.get_world_size(),))
    torch.manual_seed(0)
    Efull = torch.nn.Parameter(torch.randn(V, D))
    ref = Ember([Efull], lr=1e-3)
    torch.manual_seed(0)
    Esh = torch.nn.Parameter(distribute_tensor(torch.randn(V, D), mesh, [Shard(0)]))
    sh = Ember([Esh], lr=1e-3)
    for s in range(6):
        Efull.grad = sparse_grad(s)
        Esh.grad = distribute_tensor(sparse_grad(s), mesh, [Shard(0)])
        ref.step(); sh.step()
    diff = (Esh.detach().full_tensor() - Efull.detach()).abs().max().item()
    if dist.get_rank() == 0:
        print(f"[distributed] world={dist.get_world_size()} max|diff|={diff:.2e} "
              f"{'PASS' if diff < 1e-6 else 'FAIL'}")
    assert diff < 1e-6


def demo_fsdp2():
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard

    torch.manual_seed(0)
    model = _TinyLM(V, D)
    fully_shard(model.emb); fully_shard(model)
    emb, other = split_embedding_params(model)
    opt_emb, opt_other = Ember(emb, lr=1e-3), torch.optim.AdamW(other, lr=3e-4)
    for step in range(3):
        loss = model(torch.randint(0, V, (8, 16))).float().log_softmax(-1).mean().neg()
        loss.backward()
        opt_emb.step(); opt_other.step()
        opt_emb.zero_grad(); opt_other.zero_grad()
        if dist.get_rank() == 0:
            print(f"[fsdp2 demo] step {step} loss {loss.item():.4f}")


if __name__ == "__main__":
    test_basics()
    test_state_dict_roundtrip()
    print("single-device tests passed")
    if "RANK" in os.environ:
        import torch.distributed as dist
        dist.init_process_group("gloo")
        test_distributed()
        demo_fsdp2()
        dist.destroy_process_group()
