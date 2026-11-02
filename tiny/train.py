"""Train a ~5K-parameter character-level transformer on makemore's names.txt.

Vocabulary: '.' (start / end) and a..z. A name is fed as '.' + name, the targets are
name + '.', in a context of 16 (the longest name has 15 letters). The architecture is the one
the integer engine runs: token + position embeddings, 2 pre-norm blocks (RMSNorm, 2-head causal
attention, RMSNorm, SiLU MLP 16 -> 32 -> 16), a final RMSNorm and an untied head. No biases.

Run: uv run python tiny/train.py   (writes tiny/model.pt)
"""

import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CHARS = ".abcdefghijklmnopqrstuvwxyz"
V, T, D, H, FF, L = len(CHARS), 16, 16, 2, 32, 2
EPS = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * self.g


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = RMSNorm(D), RMSNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.up = nn.Linear(D, FF, bias=False)
        self.down = nn.Linear(FF, D, bias=False)

    def forward(self, x):
        B, t, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, -1)
        q, k, v = (z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, t, D))
        return x + self.down(F.silu(self.up(self.n2(x))))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.blocks = nn.ModuleList(Block() for _ in range(L))
        self.nf = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.head(self.nf(x))


def dataset():
    names = (HERE / "names.txt").read_text().split()
    random.Random(42).shuffle(names)
    X = torch.zeros(len(names), T, dtype=torch.long)
    Y = torch.full((len(names), T), -1, dtype=torch.long)
    for i, n in enumerate(names):
        ids = [CHARS.index(c) for c in n]
        X[i, 1:len(ids) + 1] = torch.tensor(ids)
        Y[i, :len(ids) + 1] = torch.tensor(ids + [0])
    n_val = 2000
    return (X[n_val:], Y[n_val:]), (X[:n_val], Y[:n_val])


def loss_of(model, X, Y):
    logits = model(X)
    return F.cross_entropy(logits.view(-1, V), Y.view(-1), ignore_index=-1)


def main(steps=6000, batch=256, lr=1e-2):
    torch.manual_seed(1337)
    (Xtr, Ytr), (Xva, Yva) = dataset()
    model = Tiny()
    print(f"params: {sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99))
    for step in range(steps + 1):
        lr_t = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr_t
        ix = torch.randint(0, len(Xtr), (batch,))
        loss = loss_of(model, Xtr[ix], Ytr[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            with torch.no_grad():
                print(f"step {step}: train {loss.item():.4f}, val {loss_of(model, Xva, Yva).item():.4f}")
    torch.save(model.state_dict(), HERE / "model.pt")


if __name__ == "__main__":
    main()
