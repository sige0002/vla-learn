"""M1 演習の雛形。

使い方:
  python exercises/m1/starter.py
（リポジトリのルートから実行してください。下の _bootstrap が src/ をパスに追加します。）

各問の TODO を埋め、対応する check_qN() が "OK" を表示すれば正解です。
本文: lessons/m1_pytorch.md / 解答: solutions/m1/README.md
"""
import sys
from pathlib import Path

# src/ を import パスに追加（pip install -e . 済みなら不要だが保険）
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ======================================================================
# Q1. shape 確認（実行して答え合わせ。まず紙に書いてから！）
# ======================================================================
def q1():
    a = torch.randn(64, 3)
    b = torch.randn(64, 8, 3)
    c = b.reshape(64, -1)
    d = a.unsqueeze(1)
    e = torch.randn(8).unsqueeze(-1)
    f = (b * torch.ones(64, 8, 1)).shape
    print("Q1:", a.shape, b.shape, c.shape, d.shape, e.shape, f)


# ======================================================================
# Q2. 穴埋め: 最小の学習ループ
# ======================================================================
def q2():
    torch.manual_seed(0)
    x = torch.randn(128, 1)
    y = 3.0 * x - 2.0 + 0.05 * torch.randn(128, 1)

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for step in range(200):
        pred = None        # TODO(1): 順伝播
        loss = None        # TODO(2): 損失
        # TODO(3): 勾配リセット
        # TODO(4): 逆伝播
        # TODO(5): 更新
        if loss is None:
            raise NotImplementedError("Q2 の TODO を埋めてください")

    w, b = model.weight.item(), model.bias.item()
    print(f"Q2: w={w:.3f} b={b:.3f}  (target w=3.0, b=-2.0)")
    assert abs(w - 3.0) < 0.2 and abs(b - (-2.0)) < 0.2, "Q2: まだ収束していません"
    print("Q2: OK")


# ======================================================================
# Q3. バグ修正: device / dtype 不一致（直したコードをここに書く）
# ======================================================================
def q3():
    # --- Q3-a: 埋め込みの索引は整数 ---
    emb = nn.Embedding(num_embeddings=50, embedding_dim=16)
    tokens = torch.tensor([3.0, 7.0, 1.0])     # TODO: 整数(int64)にする
    out_a = None                                # TODO: emb(tokens) を通す

    # --- Q3-b: 型をそろえてから演算 ---
    state = torch.tensor([0.3, 0.7, 0.0])       # float32
    idx = torch.tensor([1, 2, 3])               # int64
    out_b = None                                # TODO: 型をそろえて掛ける

    # --- Q3-c: モデルと入力を同じ device に ---
    from vla_learn.utils.device import get_device
    dev = get_device()
    model = nn.Linear(3, 3)                     # TODO: dev に載せる
    x = torch.randn(4, 3).to(dev)
    out_c = None                                # TODO: model(x)

    assert out_a is not None and out_b is not None and out_c is not None, \
        "Q3 の TODO を埋めてください"
    print("Q3: OK", out_a.shape, out_b.shape, out_c.shape)


# ======================================================================
# Q4. バグ修正: loss が下がらない 3 大バグを直す
# ======================================================================
def q4():
    torch.manual_seed(0)
    x = torch.randn(128, 1)
    y = 2.0 * x + 1.0

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    losses = []
    first = None
    for step in range(200):
        pred = model(x)
        loss = loss_fn(pred, y)
        # TODO(1): ここで毎ステップ何かをリセットする
        loss.backward()
        optimizer.step()
        # TODO(2): スカラを取り出して記録する
        losses.append(loss)        # ← これを直す
        if first is None:
            first = loss.item()
    print(f"Q4: first={first:.4f} last={float(losses[-1]):.6f}  mean={sum(losses)/len(losses):.6f}")
    assert float(losses[-1]) < 0.05 * first, "Q4: まだバグが残っています"
    print("Q4: OK")


# ======================================================================
# Q5. 小実装: masked_mse
# ======================================================================
def masked_mse(pred, target, mask=None):
    # TODO: 実装する（仕様は exercises/m1/README.md）
    raise NotImplementedError


def q5():
    torch.manual_seed(0)
    pred = torch.randn(4, 8, 3)
    target = torch.randn(4, 8, 3)
    # mask=None は通常の MSE と一致
    assert torch.allclose(masked_mse(pred, target),
                          torch.nn.functional.mse_loss(pred, target)), "Q5: mask=None が MSE と一致しません"
    # マスクで最後の 2 ステップを除外
    mask = torch.ones(4, 8)
    mask[:, -2:] = 0.0
    val = masked_mse(pred, target, mask)
    # 手計算（有効部分だけの平均）と一致するはず
    se = (pred - target) ** 2
    ref = (se[:, :6, :]).mean()
    assert torch.allclose(val, ref), "Q5: マスク計算が合いません"
    print("Q5: OK", float(val))


# ======================================================================
# Q6. 小実装: ToyVLADataset + DataLoader
# ======================================================================
class ToyVLADataset(Dataset):
    def __init__(self, n=200):
        # TODO: states[n,3], actions[n,8,3] を randn で作る（seed 固定）
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


def q6():
    ds = ToyVLADataset(n=200)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print("Q6:", batch["state"].shape, batch["action"].shape)
    assert batch["state"].shape == (32, 3)
    assert batch["action"].shape == (32, 8, 3)
    print("Q6: OK")


# ======================================================================
# Q7. 必須: 1 バッチに過学習させる
# ======================================================================
def q7():
    torch.manual_seed(0)
    state = torch.randn(16, 3)
    target = torch.randn(16, 8, 3)

    model = None        # TODO
    loss_fn = None      # TODO
    optimizer = None    # TODO

    first = None
    last = None
    for step in range(2000):
        # TODO: pred = model(state) を [16, 8, 3] に整形
        # TODO: loss を計算し、zero_grad -> backward -> step
        raise NotImplementedError("Q7 の TODO を埋めてください")

    print(f"Q7: first={first:.4f} last={last:.6f}")
    assert last < 0.05 * first, "Q7: 過学習できていない（配線にバグ）"
    print("Q7: OK 1 バッチに過学習できた = 学習機構は健全")


if __name__ == "__main__":
    # 完成したものから順にコメントを外して実行してください。
    q1()
    # q2()
    # q3()
    # q4()
    # q5()
    # q6()
    # q7()
