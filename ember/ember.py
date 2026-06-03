"""Ember — distributed implementation.

Bit-identical to `ember_reference.py`, but correct when the embedding table is sharded
along rows (FSDP2 dim-0, or tensor/vocab parallel). The whole distributed story is two
facts:

  * The state is ~1 MB (V + D floats), so we DON'T shard it. Treat it like a LayerNorm
    parameter: replicated, kept consistent by the gradient all-reduce you already do.
    (Exclude it from your ZeRO optimizer partition — there's nothing to save.)

  * `r` (per-row) shards naturally with the rows — purely local, no communication.
    `c` (per-col) and `mean(r_hat)` aggregate over ALL rows, so when rows live on
    different ranks they need ONE small all-reduce over the row-shard group:
    `c` is D floats, the mean is a scalar. That's the entire communication cost.

If the param isn't row-sharded (plain Tensor / DP-replicated), this is exactly the
reference with zero extra communication.
"""
import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor
except Exception:  # older torch
    DTensor = ()


def _autodetect_row_shard_group(p):
    """The process group p is sharded over along dim 0 (DTensor only), or None."""
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
        """row_shard_group: process group the embedding rows are sharded over. Leave None
        for FSDP2/DTensor (auto-detected) or replicated/DDP (no sharding). Set it explicitly
        for frameworks that don't expose DTensor placements (Megatron tensor-parallel,
        DeepSpeed). Can also be set per param-group."""
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
                # explicit group (Megatron/DeepSpeed) overrides DTensor auto-detect (FSDP2)
                pg = group.get("row_shard_group") or _autodetect_row_shard_group(p)
                p_l, g_l = _local(p), _local(p.grad)
                if p_l.dim() != 2:
                    raise ValueError("Ember is for 2-D embedding tables (V x D).")
                Vloc, D = p_l.shape
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["r"] = torch.zeros(Vloc, dtype=p_l.dtype, device=p_l.device)
                    state["c"] = torch.zeros(D, dtype=p_l.dtype, device=p_l.device)
                state["t"] += 1
                t, r, c = state["t"], state["r"], state["c"]

                # per-row 2nd moment: this rank's rows only -> local, no comms
                r.mul_(b2).add_(g_l.pow(2).mean(dim=1), alpha=1 - b2)

                # per-col 2nd moment: sum this rank's contribution, then all-reduce the
                # D-vector + the active-row count over the row-shard group.
                col_sum = g_l.pow(2).sum(dim=0)                       # [D] local
                n_active = (g_l.abs().sum(dim=1) > 0).sum().to(p_l.dtype)
                if pg is not None:
                    dist.all_reduce(col_sum, group=pg)
                    dist.all_reduce(n_active, group=pg)
                c.mul_(b2).add_(col_sum / n_active.clamp(min=1), alpha=1 - b2)

                bc = 1 - b2 ** t
                r_hat, c_hat = r / bc, c / bc

                # global mean(r_hat) over ALL rows -> all-reduce sum & count (2 scalars)
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
                p_l.addcdiv_(g_l, denom, value=-lr)
        return loss
