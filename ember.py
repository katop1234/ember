"""Ember — an O(V+D) optimizer for the token-embedding table.

Adam keeps two moments per weight, so for a V x D embedding its optimizer state is 2*V*D
(e.g. ~300 MB at a 50K vocab x 768 dim). Ember matches AdamW's quality on the embedding
while storing only V + D numbers (~1 MB) — about 1500x less.

How: an embedding's per-coordinate update scale factorizes. A token row's gradient scale
depends on how often that token appears (participation); a feature column's scale depends on
that feature. So instead of a full V x D second moment, Ember keeps a per-row vector r and a
per-col vector c and reconstructs the rank-1 estimate v[i,j] = r[i]*c[j]/mean(r) (the
Adafactor factorization). It uses no first moment.

Update, for embedding gradient G in R^{V x D} (sparse over V — only rows of tokens in the
batch get a gradient):

    r[i] <- b2*r[i] + (1-b2) * mean_j(G[i,j]^2)            # per-row 2nd moment, all rows
    c[j] <- b2*c[j] + (1-b2) * mean_{i active}(G[i,j]^2)   # per-col 2nd moment, active rows
    r_hat, c_hat = bias-correct(r, c)
    E <- E - lr * G / (sqrt(r_hat[i]*c_hat[j]/mean(r_hat)) + eps)

"Active rows only" for the column statistic: rows of tokens absent from the batch have zero
gradient and would otherwise dilute the column average toward zero.

Distributed: the state is ~1 MB, so replicate it — don't shard it. Single-device, DDP, and
ZeRO-1 work as-is. When the embedding table is sharded along rows (FSDP2 dim-0, or tensor /
vocab parallel), r is local to each rank's rows for free, and only the column factor c and
the global mean need one small all-reduce (D floats + a scalar) over the row-shard group.
That reduction is auto-detected for DTensor params (FSDP2); for other frameworks pass
`row_shard_group=<process group>`. The result is bit-identical to single device either way.
"""
import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor
except Exception:  # torch without DTensor
    DTensor = ()


def _row_shard_group(p):
    """Process group p's rows are sharded over (DTensor dim-0 shard), or None if local."""
    if isinstance(p, DTensor):
        for dim, pl in enumerate(p.placements):
            if pl.is_shard() and pl.dim == 0:
                return p.device_mesh.get_group(dim)
    return None


def _local(x):
    return x.to_local() if isinstance(x, DTensor) else x


class Ember(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta2=0.999, eps=1e-8, weight_decay=0.0,
                 row_shard_group=None):
        """row_shard_group: the process group the embedding rows are sharded over. Leave
        None for single-device / DDP / FSDP2 (auto-detected from DTensor). Set it for
        non-DTensor frameworks (Megatron tensor-parallel, etc.)."""
        defaults = dict(lr=lr, beta2=beta2, eps=eps, weight_decay=weight_decay,
                        row_shard_group=row_shard_group)
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
                pg = group.get("row_shard_group") or _row_shard_group(p)
                p_l, g_l = _local(p), _local(p.grad)
                if p_l.dim() != 2:
                    raise ValueError("Ember is for 2-D embedding tables (V x D).")
                Vloc, D = p_l.shape
                state = self.state[p]
                if not state:
                    state["t"] = 0  # state kept in fp32 (mixed-precision safe)
                    state["r"] = torch.zeros(Vloc, dtype=torch.float32, device=p_l.device)
                    state["c"] = torch.zeros(D, dtype=torch.float32, device=p_l.device)
                state["t"] += 1
                t, r, c = state["t"], state["r"], state["c"]
                g32 = g_l.float()

                # per-row 2nd moment: this rank's rows only -> local, no comms
                r.mul_(b2).add_(g32.pow(2).mean(dim=1), alpha=1 - b2)

                # per-col 2nd moment: sum local contribution, all-reduce the D-vector and the
                # active-row count over the row-shard group (no-op when not sharded)
                col_sum = g32.pow(2).sum(dim=0)
                n_active = (g32.abs().sum(dim=1) > 0).sum().float()
                if pg is not None:
                    dist.all_reduce(col_sum, group=pg)
                    dist.all_reduce(n_active, group=pg)
                c.mul_(b2).add_(col_sum / n_active.clamp(min=1), alpha=1 - b2)

                bc = 1 - b2 ** t
                r_hat, c_hat = r / bc, c / bc

                # global mean(r_hat) over all rows -> all-reduce sum & count (2 scalars)
                if pg is not None:
                    rsum = r_hat.sum()
                    rcnt = torch.tensor(float(r_hat.numel()), device=p_l.device)
                    dist.all_reduce(rsum, group=pg)
                    dist.all_reduce(rcnt, group=pg)
                    scale = (rsum / rcnt).clamp(min=1e-30)
                else:
                    scale = r_hat.mean().clamp(min=1e-30)

                denom = (r_hat.unsqueeze(1) * c_hat.unsqueeze(0) / scale).sqrt().add_(eps)
                if wd != 0.0:
                    p_l.mul_(1 - lr * wd)
                p_l.addcdiv_(g_l, denom.to(p_l.dtype), value=-lr)
        return loss
