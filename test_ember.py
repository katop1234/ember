"""CPU bare-test for Ember. No GPU. Run: python test_ember.py

Checks the drop-in promise: Ember constructs with the *exact* torch.optim.Adam call
signature, trains a toy embedding model (loss down, no NaN), routes a whole model without
crashing on non-2-D params, and round-trips its state_dict.
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
        self.body = nn.Linear(D, D)          # has a 2-D weight + a 1-D bias
        self.lm_head = nn.Linear(D, V, bias=False)

    def forward(self, idx):
        return self.lm_head(torch.tanh(self.body(self.embed_tokens(idx))))


# --- 1. exact Adam constructor signature is accepted ---------------------------------------
m = Toy()
opt = Ember(m.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
# betas[1] became beta2; betas[0] ignored (Ember has no first moment)
assert opt.param_groups[0]["beta2"] == 0.999, "betas[1] should set beta2"
# explicit beta2= overrides betas[1]
opt_b2 = Ember(m.parameters(), lr=1e-3, betas=(0.9, 0.999), beta2=0.99)
assert opt_b2.param_groups[0]["beta2"] == 0.99, "explicit beta2= should win over betas[1]"
print("[1] Adam-compatible constructor OK (betas tuple + beta2 override)")

# --- 2. whole-model step does not crash on non-2-D params (bias) ---------------------------
idx = torch.randint(0, V, (32, 8))
tgt = torch.randint(0, V, (32, 8))
e0 = m.embed_tokens.weight.detach().clone()
bias0 = m.body.bias.detach().clone()
losses = []
for _ in range(60):
    loss = F.cross_entropy(m(idx).reshape(-1, V), tgt.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss), "NaN/inf loss"
    losses.append(loss.item())
assert losses[-1] < losses[0] - 0.1, f"loss did not decrease: {losses[0]:.3f}->{losses[-1]:.3f}"
assert not torch.allclose(m.embed_tokens.weight, e0), "Ember did not move the 2-D embedding"
assert not torch.allclose(m.body.bias, bias0), "non-2-D bias fallback did not move the bias"
print(f"[2] whole-model train OK: loss {losses[0]:.3f} -> {losses[-1]:.3f}, no NaN, 1-D bias moved")

# --- 3. recommended usage: Ember on embed/lm_head, AdamW on the body -----------------------
m2 = Toy()
emb, other = split_embedding_params(m2)
emb_ids = {id(p) for p in emb}
assert id(m2.embed_tokens.weight) in emb_ids, "embed -> Ember"
assert id(m2.lm_head.weight) in emb_ids, "lm_head -> Ember"
assert id(m2.body.weight) not in emb_ids, "body weight -> other"
opt_emb = Ember(emb, lr=1e-2)
opt_other = torch.optim.AdamW(other, lr=1e-2)
for _ in range(20):
    loss = F.cross_entropy(m2(idx).reshape(-1, V), tgt.reshape(-1))
    opt_emb.zero_grad(); opt_other.zero_grad()
    loss.backward()
    opt_emb.step(); opt_other.step()
    assert torch.isfinite(loss)
print("[3] split routing (Ember on token tables + AdamW on body) OK")

# --- 4. state_dict round-trips -------------------------------------------------------------
sd = opt.state_dict()
opt_new = Ember(m.parameters(), lr=1e-2, betas=(0.9, 0.999))
opt_new.load_state_dict(sd)
# the row/col factors and step count survive the round-trip
s_old = next(iter(opt.state.values()))
s_new = next(iter(opt_new.state.values()))
assert s_new["t"] == s_old["t"], "step count did not round-trip"
assert torch.allclose(s_new["r"], s_old["r"]), "row factor did not round-trip"
assert torch.allclose(s_new["c"], s_old["c"]), "col factor did not round-trip"
print("[4] state_dict round-trip OK")

# --- 5. one-line drop-in: literally the Adam call, swapped -----------------------------------
m3 = Toy()
# was: optimizer = torch.optim.Adam(m3.parameters(), lr=1e-3)
optimizer = Ember(m3.parameters(), lr=1e-3)
loss = F.cross_entropy(m3(idx).reshape(-1, V), tgt.reshape(-1))
optimizer.zero_grad(); loss.backward(); optimizer.step()
assert torch.isfinite(loss)
print("[5] one-line Adam->Ember drop-in OK")

print("\nALL TESTS PASSED")
