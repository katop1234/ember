"""CPU bare-test for Ember. No GPU. Run: python test_ember.py

Checks the drop-in promise: `Ember(model, lr=...)` auto-routes the token tables (nn.Embedding
+ LM head) to the factored Ember update and everything else to AdamW — hidden linears are
never Embered — trains (loss down, no NaN), and round-trips its state_dict.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ember import Ember, split_embedding_params

torch.manual_seed(0)
V, D = 64, 16


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, D)
        self.attn = nn.Linear(D, D)          # hidden linear: must NOT be Embered
        self.mlp = nn.Linear(D, D)           # hidden linear: must NOT be Embered
        self.lm_head = nn.Linear(D, V, bias=False)   # output token table -> Ember

    def forward(self, idx):
        h = torch.tanh(self.attn(self.embed_tokens(idx)))
        return self.lm_head(torch.tanh(self.mlp(h)))


idx = torch.randint(0, V, (32, 8))
tgt = torch.randint(0, V, (32, 8))

# --- 1. Ember(model) auto-routing: tables -> Ember, hidden linears -> AdamW ----------------
m = Toy()
opt = Ember(m, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
modes = {g["mode"] for g in opt.param_groups}
assert modes == {"ember", "adamw"}, f"expected ember+adamw groups, got {modes}"
ember_ids = {id(p) for g in opt.param_groups if g["mode"] == "ember" for p in g["params"]}
adamw_ids = {id(p) for g in opt.param_groups if g["mode"] == "adamw" for p in g["params"]}
assert id(m.embed_tokens.weight) in ember_ids, "embedding -> Ember"
assert id(m.lm_head.weight) in ember_ids, "lm_head -> Ember"
assert id(m.attn.weight) in adamw_ids, "attn linear must NOT be Embered"
assert id(m.mlp.weight) in adamw_ids, "mlp linear must NOT be Embered"
assert id(m.attn.bias) in adamw_ids, "biases -> AdamW"
print("[1] Ember(model) routes only nn.Embedding + lm_head to Ember; hidden linears -> AdamW")

# --- 2. trains, and state proves the routing (r/c on tables, m/v on the body) --------------
losses = []
for _ in range(60):
    loss = F.cross_entropy(m(idx).reshape(-1, V), tgt.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    assert torch.isfinite(loss), "NaN/inf loss"
    losses.append(loss.item())
assert losses[-1] < losses[0] - 0.1, f"loss did not fall: {losses[0]:.3f}->{losses[-1]:.3f}"
assert set(opt.state[m.embed_tokens.weight]) >= {"r", "c"}, "embedding should hold factored r/c"
assert "m" not in opt.state[m.embed_tokens.weight], "embedding should have NO first moment"
assert set(opt.state[m.attn.weight]) >= {"m", "v"}, "hidden linear should hold AdamW m/v"
assert "r" not in opt.state[m.attn.weight], "hidden linear must NOT have Ember r/c"
print(f"[2] train OK: loss {losses[0]:.3f}->{losses[-1]:.3f}; tables hold r/c, body holds m/v")

# --- 3. Adam-compatible constructor (betas tuple + beta2 override) --------------------------
o = Ember(m, lr=1e-3, betas=(0.9, 0.999))
assert all(g["beta2"] == 0.999 for g in o.param_groups), "betas[1] -> beta2"
o2 = Ember(m, lr=1e-3, betas=(0.9, 0.999), beta2=0.99)
assert all(g["beta2"] == 0.99 for g in o2.param_groups), "explicit beta2= wins"
print("[3] Adam-compatible constructor OK (betas tuple + beta2 override)")

# --- 4. explicit-params path still works (you split yourself) ------------------------------
m2 = Toy()
emb, other = split_embedding_params(m2)
assert id(m2.embed_tokens.weight) in {id(p) for p in emb} and id(m2.attn.weight) in {id(p) for p in other}
opt_emb, opt_other = Ember(emb, lr=1e-2), torch.optim.AdamW(other, lr=1e-2)
for _ in range(20):
    loss = F.cross_entropy(m2(idx).reshape(-1, V), tgt.reshape(-1))
    opt_emb.zero_grad(); opt_other.zero_grad(); loss.backward(); opt_emb.step(); opt_other.step()
    assert torch.isfinite(loss)
print("[4] explicit split (Ember on tables + AdamW on body) OK")

# --- 5. state_dict round-trips -------------------------------------------------------------
sd = opt.state_dict()
opt_new = Ember(m, lr=1e-2, betas=(0.9, 0.999))
opt_new.load_state_dict(sd)
s_old, s_new = opt.state[m.embed_tokens.weight], opt_new.state[m.embed_tokens.weight]
assert s_new["t"] == s_old["t"] and torch.allclose(s_new["r"], s_old["r"]) and torch.allclose(s_new["c"], s_old["c"])
print("[5] state_dict round-trip OK")

# --- 6. helpful error when a model has no token tables -------------------------------------
try:
    Ember(nn.Sequential(nn.Linear(4, 4)), lr=1e-3)
    raise AssertionError("should have raised: no embedding tables")
except ValueError as e:
    assert "no nn.Embedding" in str(e)
print("[6] clear error when model has no embedding tables")

print("\nALL TESTS PASSED")
