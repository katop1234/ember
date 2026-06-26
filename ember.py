"""Ember — an O(V+D) optimizer for the token-embedding / LM-head table.

Adam keeps two moments per weight, so for a V x D embedding its optimizer state is 2*V*D
(e.g. ~300 MB at a 50K vocab x 768 dim). Ember matches AdamW's quality on the embedding
while storing only V + D numbers (~1 MB) — about 1500x less.

How: an embedding's per-coordinate update scale factorizes. A token row's gradient scale
depends on how often that token appears (participation); a feature column's scale depends on
that feature. So instead of a full V x D second moment, Ember keeps a per-row vector r and a
per-col vector c and reconstructs the rank-1 estimate v[i,j] = r[i]*c[j]/mean(r) (the
Adafactor factorization). It uses no first moment; bias correction and beta2=0.999 are
lifted straight from Adam.

One-line drop-in (hand it the *model* so it can route automatically):

    from ember import Ember
    # was: opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt = Ember(model, lr=1e-3)

Handed a **model**, Ember figures out which parameters to hook itself: it puts the factored
Ember update on the **token tables only** — every `nn.Embedding` weight plus the LM head — and
runs a normal **AdamW** on everything else (attention/MLP linears, norms, biases). Hidden
linear layers are *never* Embered. It returns a single optimizer with the usual
`step()/zero_grad()/state_dict()` API.

Handed an explicit parameter iterable instead of a model (e.g. you split yourself), Ember
applies the factored update to every 2-D tensor and a plain scaled-SGD step to non-2-D
tensors — use this only when you've already isolated the embedding params.

Update, for an embedding gradient G in R^{V x D} (sparse over V — only rows of tokens in the
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
import torch.nn as nn
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor
except Exception:  # torch without DTensor
    DTensor = ()

# 2-D parameters whose *name* marks them as the output token table (LM head). nn.Embedding
# weights are detected by module type and need no name match; these catch an untied LM head,
# which is an nn.Linear. Deliberately narrow so it can never match an attention/MLP linear.
_LM_HEAD_KEYS = ("lm_head", "output_embed", "output_projection", "embed_out")


def _row_shard_group(p):
    """Process group p's rows are sharded over (DTensor dim-0 shard), or None if local."""
    if isinstance(p, DTensor):
        for dim, pl in enumerate(p.placements):
            if pl.is_shard() and pl.dim == 0:
                return p.device_mesh.get_group(dim)
    return None


def _local(x):
    return x.to_local() if isinstance(x, DTensor) else x


def split_embedding_params(model, extra_names=(), lm_head=True):
    """Return (token_table_params, other_params) for a model.

    Token tables = the .weight of every `nn.Embedding`, plus (if `lm_head=True`) any 2-D
    parameter whose name marks it an LM head (lm_head / output_embed / ... + `extra_names`).
    Tied weights are de-duped. Attention/MLP linears, norms and biases go to `other`.
    """
    emb_ids = {id(m.weight) for m in model.modules()
               if isinstance(m, nn.Embedding) and m.weight is not None}
    keys = (_LM_HEAD_KEYS if lm_head else ()) + tuple(extra_names)
    emb, other, seen = [], [], set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        is_table = id(p) in emb_ids or (p.dim() == 2 and any(k in name.lower() for k in keys))
        (emb if is_table else other).append(p)
    return emb, other


class Ember(torch.optim.Optimizer):
    """Factored O(V+D) second-moment optimizer for token-table matrices.

    Two ways to construct:

    * `Ember(model, lr=1e-3)` — recommended. Auto-routes: token tables (every nn.Embedding +
      the LM head) get the factored Ember update; everything else gets a standard AdamW. One
      optimizer object, the usual `step/zero_grad/state_dict` API. Hidden linears are never
      Embered. `betas`/`eps`/`weight_decay` apply to the AdamW side; `body_lr` overrides the
      learning rate for the non-table params (defaults to `lr`).

    * `Ember(params, lr=1e-3)` — explicit param iterable (you've split yourself). The factored
      update is applied to every 2-D tensor and scaled-SGD to non-2-D tensors.
    """

    def __init__(self, model_or_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, beta2=None, body_lr=None, lm_head=True,
                 extra_table_names=(), row_shard_group=None):
        b2 = betas[1] if beta2 is None else beta2
        if not 0.0 <= b2 < 1.0:
            raise ValueError(f"Invalid beta2: {b2}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        defaults = dict(lr=lr, betas=betas, beta2=b2, eps=eps, weight_decay=weight_decay,
                        mode="ember", row_shard_group=row_shard_group)

        if isinstance(model_or_params, nn.Module):
            emb, other = split_embedding_params(model_or_params, extra_table_names, lm_head)
            if not emb:
                raise ValueError(
                    "Ember(model, ...) found no nn.Embedding (or LM-head) tables in the model. "
                    "Pass the embedding parameters explicitly: Ember(list_of_emb_params, ...).")
            groups = [dict(params=emb, mode="ember", lr=lr)]
            if other:
                groups.append(dict(params=other, mode="adamw",
                                   lr=lr if body_lr is None else body_lr))
            super().__init__(groups, defaults)
        else:
            # explicit param iterable -> single Ember group (factored on 2-D, SGD on non-2-D)
            super().__init__(model_or_params, defaults)

    # -- the factored token-table update (+ scaled-SGD fallback for stray non-2-D params) --
    def _ember_step(self, p, group):
        lr, b2 = group["lr"], group["beta2"]
        eps, wd = group["eps"], group["weight_decay"]
        pg = group.get("row_shard_group") or _row_shard_group(p)
        p_l, g_l = _local(p), _local(p.grad)

        if p_l.dim() != 2:                         # bias/norm/1-D fallback (explicit-params path)
            if wd != 0.0:
                p_l.mul_(1 - lr * wd)
            p_l.add_(g_l, alpha=-lr)
            return

        Vloc, D = p_l.shape
        state = self.state[p]
        if not state:
            state["t"] = 0
            state["r"] = torch.zeros(Vloc, dtype=torch.float32, device=p_l.device)
            state["c"] = torch.zeros(D, dtype=torch.float32, device=p_l.device)
        state["t"] += 1
        t, r, c = state["t"], state["r"], state["c"]
        g32 = g_l.float()

        r.mul_(b2).add_(g32.pow(2).mean(dim=1), alpha=1 - b2)          # per-row, local
        col_sum = g32.pow(2).sum(dim=0)
        n_active = (g32.abs().sum(dim=1) > 0).sum().float()
        if pg is not None:
            dist.all_reduce(col_sum, group=pg)
            dist.all_reduce(n_active, group=pg)
        c.mul_(b2).add_(col_sum / n_active.clamp(min=1), alpha=1 - b2)  # per-col, active rows

        bc = 1 - b2 ** t
        r_hat, c_hat = r / bc, c / bc
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

    # -- standard AdamW for the non-table parameters (body of the model) --
    def _adamw_step(self, p, group):
        lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
        b1, b2 = group["betas"][0], group["beta2"]
        g = p.grad
        state = self.state[p]
        if not state:
            state["t"] = 0
            state["m"] = torch.zeros_like(p, dtype=torch.float32)
            state["v"] = torch.zeros_like(p, dtype=torch.float32)
        state["t"] += 1
        t, m, v = state["t"], state["m"], state["v"]
        g32 = g.float()
        m.mul_(b1).add_(g32, alpha=1 - b1)
        v.mul_(b2).addcmul_(g32, g32, value=1 - b2)
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        if wd != 0.0:
            p.mul_(1 - lr * wd)
        p.addcdiv_(m_hat.to(p.dtype), (v_hat.sqrt().add_(eps)).to(p.dtype), value=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            mode = group.get("mode", "ember")
            for p in group["params"]:
                if p.grad is None:
                    continue
                if mode == "adamw":
                    self._adamw_step(p, group)
                else:
                    self._ember_step(p, group)
        return loss
