"""M1 演習の完成版（解答）。

使い方:
  python solutions/m1/solution.py
すべての check が OK を表示すれば正しく動いています。
解説は solutions/m1/README.md を参照。
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---- Q1 -------------------------------------------------------------
def q1():
    a = torch.randn(64, 3)
    b = torch.randn(64, 8, 3)
    c = b.reshape(64, -1)
    d = a.unsqueeze(1)
    e = torch.randn(8).unsqueeze(-1)
    f = (b * torch.ones(64, 8, 1)).shape
    assert tuple(a.shape) == (64, 3)
    assert tuple(b.shape) == (64, 8, 3)
    assert tuple(c.shape) == (64, 24)
    assert tuple(d.shape) == (64, 1, 3)
    assert tuple(e.shape) == (8, 1)
    assert tuple(f) == (64, 8, 3)
    print("Q1: OK", tuple(a.shape), tuple(b.shape), tuple(c.shape),
          tuple(d.shape), tuple(e.shape), tuple(f))


# ---- Q2 -------------------------------------------------------------
def q2():
    torch.manual_seed(0)
    x = torch.randn(128, 1)
    y = 3.0 * x - 2.0 + 0.05 * torch.randn(128, 1)

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(200):
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    w, b = model.weight.item(), model.bias.item()
    assert abs(w - 3.0) < 0.2 and abs(b + 2.0) < 0.2
    print(f"Q2: OK w={w:.3f} b={b:.3f}")


# ---- Q3 -------------------------------------------------------------
def q3():
    emb = nn.Embedding(num_embeddings=50, embedding_dim=16)
    tokens = torch.tensor([3, 7, 1])              # int64
    out_a = emb(tokens)

    state = torch.tensor([0.3, 0.7, 0.0])
    idx = torch.tensor([1, 2, 3])
    out_b = state * idx.float()

    from vla_learn.utils.device import get_device
    dev = get_device()
    model = nn.Linear(3, 3).to(dev)
    x = torch.randn(4, 3).to(dev)
    out_c = model(x)

    assert tuple(out_a.shape) == (3, 16)
    assert tuple(out_b.shape) == (3,)
    assert tuple(out_c.shape) == (4, 3)
    print("Q3: OK", tuple(out_a.shape), tuple(out_b.shape), tuple(out_c.shape))


# ---- Q4 -------------------------------------------------------------
def q4():
    torch.manual_seed(0)
    x = torch.randn(128, 1)
    y = 2.0 * x + 1.0

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    losses = []
    first = None
    for _ in range(200):
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if first is None:
            first = loss.item()
    assert losses[-1] < 0.05 * first
    print(f"Q4: OK first={first:.4f} last={losses[-1]:.6f}")


# ---- Q5 -------------------------------------------------------------
def masked_mse(pred, target, mask=None):
    se = (pred - target) ** 2
    if mask is None:
        return se.mean()
    mask3 = mask.unsqueeze(-1).expand_as(se)
    return (se * mask3).sum() / mask3.sum().clamp(min=1.0)


def q5():
    torch.manual_seed(0)
    pred = torch.randn(4, 8, 3)
    target = torch.randn(4, 8, 3)
    assert torch.allclose(masked_mse(pred, target),
                          torch.nn.functional.mse_loss(pred, target))
    mask = torch.ones(4, 8)
    mask[:, -2:] = 0.0
    val = masked_mse(pred, target, mask)
    ref = ((pred - target) ** 2)[:, :6, :].mean()
    assert torch.allclose(val, ref)
    print("Q5: OK", float(val))


# ---- Q6 -------------------------------------------------------------
class ToyVLADataset(Dataset):
    def __init__(self, n=200):
        torch.manual_seed(0)
        self.states = torch.randn(n, 3)
        self.actions = torch.randn(n, 8, 3)

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return {"state": self.states[idx], "action": self.actions[idx]}


def q6():
    ds = ToyVLADataset(n=200)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    assert tuple(batch["state"].shape) == (32, 3)
    assert tuple(batch["action"].shape) == (32, 8, 3)
    print("Q6: OK", tuple(batch["state"].shape), tuple(batch["action"].shape))


# ---- Q7 -------------------------------------------------------------
def q7():
    torch.manual_seed(0)
    state = torch.randn(16, 3)
    target = torch.randn(16, 8, 3)

    model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 8 * 3))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    first = None
    for _ in range(2000):
        pred = model(state).view(-1, 8, 3)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if first is None:
            first = loss.item()
    last = loss.item()
    assert last < 0.05 * first, f"過学習できていない: {first:.4f} -> {last:.6f}"
    print(f"Q7: OK first={first:.4f} last={last:.6f}")


if __name__ == "__main__":
    q1(); q2(); q3(); q4(); q5(); q6(); q7()
    print("\nALL OK")
