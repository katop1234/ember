"""Ember — single-file, single-device. Copy this one file into your project.

A drop-in optimizer for the token-embedding table: matches AdamW quality at O(V+D)
optimizer state instead of Adam's O(2*V*D). Stores a row x column factored second moment
and no first moment.

    from ember import Ember
    opt = Ember(model.get_input_embeddings().parameters(), lr=1e-3)

For distributed training (FSDP2 / tensor-parallel / DeepSpeed) install the package instead:
`pip install git+https://github.com/katop1234/ember.git` — same optimizer, sharding-aware.

Update, for embedding gradient G in R^{V x D} (sparse over V — only rows of tokens in the
batch get a gradient):

    r[i] <- b2*r[i] + (1-b2) * mean_j(G[i,j]^2)            # per-row 2nd moment (all rows)
    c[j] <- b2*c[j] + (1-b2) * mean_{i active}(G[i,j]^2)   # per-col 2nd moment (sparse-aware)
    r_hat, c_hat = bias-correct(r, c)
    v_hat[i,j] = r_hat[i] * c_hat[j] / mean(r_hat)         # Adafactor rank-1 reconstruction
    E <- E - lr * G / (sqrt(v_hat) + eps)
"""
import torch


class Ember(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta2=0.999, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, b2 = group["lr"], group["beta2"]
            eps, wd = group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise ValueError("Ember is for 2-D embedding tables (V x D).")
                g = p.grad
                V, D = p.shape
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["r"] = torch.zeros(V, dtype=p.dtype, device=p.device)
                    state["c"] = torch.zeros(D, dtype=p.dtype, device=p.device)
                state["t"] += 1
                t, r, c = state["t"], state["r"], state["c"]

                # per-row second moment (over all V rows; inactive rows just decay)
                r.mul_(b2).add_(g.pow(2).mean(dim=1), alpha=1 - b2)
                # per-col second moment, averaged over ACTIVE rows only (sparse-aware)
                n_active = (g.abs().sum(dim=1) > 0).sum().clamp(min=1).to(g.dtype)
                c.mul_(b2).add_(g.pow(2).sum(dim=0) / n_active, alpha=1 - b2)

                bc = 1 - b2 ** t
                r_hat, c_hat = r / bc, c / bc
                scale = r_hat.mean().clamp(min=1e-30)
                denom = (r_hat.unsqueeze(1) * c_hat.unsqueeze(0) / scale).sqrt().add_(eps)

                if wd != 0.0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(g, denom, value=-lr)
        return loss
