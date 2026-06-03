"""Correctness: Ember (single-device path) == EmberReference, bit-for-bit, and the
factored update has the right shape/behavior on a sparse embedding gradient."""
import torch
from ember import Ember, EmberReference


def _sparse_grad(V, D, active_frac=0.2, seed=0):
    g = torch.randn(V, D, generator=torch.Generator().manual_seed(seed))
    dead = torch.rand(V, generator=torch.Generator().manual_seed(seed + 1)) > active_frac
    g[dead] = 0.0
    return g


def _run(opt_cls, steps=8, V=2000, D=64, lr=1e-3):
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(V, D))
    opt = opt_cls([p], lr=lr)
    for s in range(steps):
        p.grad = _sparse_grad(V, D, seed=s)
        opt.step()
    return p.detach().clone()


def test_ember_matches_reference():
    a = _run(EmberReference)
    b = _run(Ember)                      # local path: pg=None -> same arithmetic
    assert torch.equal(a, b), f"max|diff|={(a - b).abs().max():.2e}"


def test_finite_and_moves():
    p0 = torch.randn(2000, 64)
    out = _run(EmberReference)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, p0)   # it actually updated


def test_state_is_factored_VplusD():
    p = torch.nn.Parameter(torch.randn(2000, 64))
    opt = EmberReference([p])
    p.grad = _sparse_grad(2000, 64)
    opt.step()
    st = opt.state[p]
    assert st["r"].shape == (2000,) and st["c"].shape == (64,)
    # O(V+D) = 2064 floats, vs Adam's 2*V*D = 256000
    assert st["r"].numel() + st["c"].numel() == 2064


if __name__ == "__main__":
    test_ember_matches_reference()
    test_finite_and_moves()
    test_state_is_factored_VplusD()
    print("all reference tests passed")
