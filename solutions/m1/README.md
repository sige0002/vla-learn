# M1 解答と解説: PyTorch 速習

対応する演習: [`../../exercises/m1/README.md`](../../exercises/m1/README.md) ／ 本文: [`../../lessons/m1_pytorch.md`](../../lessons/m1_pytorch.md)

各問、**正解コード**と「**なぜその shape か / なぜ loss が下がるか**」の短い解説を付けます。
動く完成版は [`solution.py`](solution.py) にもまとめてあります（`python solutions/m1/solution.py`）。

---

## Q1.（shape 確認）

| 式 | shape | なぜ |
|----|-------|------|
| `a = torch.randn(64, 3)` | `[64, 3]` | 指定どおり。`[B, D]` の形。 |
| `b = torch.randn(64, 8, 3)` | `[64, 8, 3]` | `[B, C, A]`（バッチ 64・チャンク 8・行動 3）。 |
| `c = b.reshape(64, -1)` | `[64, 24]` | 先頭を 64 に固定、残り `8*3=24` を `-1` が自動計算。要素数 1536 は不変。 |
| `d = a.unsqueeze(1)` | `[64, 1, 3]` | 位置 1 に大きさ 1 の次元を挿入。 |
| `e = torch.randn(8).unsqueeze(-1)` | `[8, 1]` | `[8]` の末尾に 1 を足す。`masked_mse` で mask を `[B,C]→[B,C,1]` にするのと同じ操作。 |
| `f = (b * torch.ones(64, 8, 1)).shape` | `[64, 8, 3]` | `[64,8,1]` が末尾 1→3 にブロードキャストされ `b` と同形に。 |

> **要点**: `reshape(_, -1)` の `-1` は「残りを自動計算」。`unsqueeze` は大きさ 1 の次元追加。
> ブロードキャストは「末尾から見て等しい or 片方が 1」。VLA では shape を即答できることが最重要スキルです。

---

## Q2.（穴埋め）最小の学習ループ

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
x = torch.randn(128, 1)
y = 3.0 * x - 2.0 + 0.05 * torch.randn(128, 1)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model(x)                 # (1) 順伝播
    loss = loss_fn(pred, y)         # (2) 損失
    optimizer.zero_grad()           # (3) 勾配リセット
    loss.backward()                 # (4) 逆伝播
    optimizer.step()                # (5) 更新

print(model.weight.item(), model.bias.item())  # ≈ 3.0, -2.0
```

出力例（環境でぶれます）:

```text
3.0027 -2.0002
```

> **なぜ loss が下がるか**: `loss.backward()` が「loss を小さくするには各パラメータをどちらへ動かせばよいか」（勾配）を計算し、`optimizer.step()` がその逆方向へ `lr` だけ動かします。これを 200 回繰り返すと、`w→3, b→-2` に収束します。
> `zero_grad()` を入れる理由は、`.grad` が**加算される**ため。消さないと前ステップの勾配が混ざって発散します。

---

## Q3.（バグ修正）device / dtype の不一致

**Q3-a**: 埋め込み (`nn.Embedding`) の入力は**索引**なので整数 (int64) でなければなりません。float を渡すと `RuntimeError`。

```python
tokens = torch.tensor([3, 7, 1])          # int64（小数点を付けない）
# もしくは: tokens = torch.tensor([3.0, 7.0, 1.0]).long()
out_a = emb(tokens)                        # OK -> [3, 16]
```

> 本教材の `tokens` が int64 なのはこのため（埋め込み表の行を選ぶ索引だから）。

**Q3-b**: float32 と int64 はそのまま掛け算できません。**型をそろえて**から演算します。

```python
out_b = state * idx.float()                # int64 -> float32 にそろえる -> OK
```

**Q3-c**: 「テンソルとモデルは同じ device」。モデルも入力と同じ device へ載せます。

```python
from vla_learn.utils.device import get_device
dev = get_device()
model = nn.Linear(3, 3).to(dev)            # ← モデルを dev へ
x = torch.randn(4, 3).to(dev)
out_c = model(x)                           # 両方 dev なので OK
```

> **要点**: VLA で最も多い実行時エラーは dtype 不一致（特にトークンを float にしてしまう）と device 不一致です。`.float()` / `.long()` / `.to(dev)` の 3 つで大半が直ります。

---

## Q4.（バグ修正）「loss が下がらない」3 大バグ

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
x = torch.randn(128, 1)
y = 2.0 * x + 1.0

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

losses = []
for step in range(200):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()          # 修正(1): 毎ステップ勾配をリセット
    loss.backward()
    optimizer.step()
    losses.append(loss.item())     # 修正(2): スカラを取り出して記録
print(sum(losses) / len(losses))   # 修正(3): (2)を直せば軽く・正しく計算できる
```

3 つの問題:
1. **`zero_grad()` 忘れ**: `.grad` が加算され続け、勾配が膨らんで更新が暴れる。loss が下がらない最頻出バグ。
2. **テンソルをそのまま `list` に貯めた**: `loss`（テンソル）を貯めると**計算グラフが解放されずメモリを食う**。`loss.item()` で Python の `float` にして貯めるのが正解。
3. **(2) の波及**: `sum(losses)` が「テンソルの総和」になり計算グラフを連結して重く・不安定になっていた。`.item()` 済みなら普通の数値和で軽い。

> **なぜ loss が下がるようになるか**: (1) を直すと各ステップで「今この瞬間の勾配」だけを使って正しく更新できるようになり、`w→2, b→1` に収束します。

---

## Q5.（小実装）`masked_mse`

```python
import torch

def masked_mse(pred, target, mask=None):
    se = (pred - target) ** 2                  # [B, C, A]
    if mask is None:
        return se.mean()
    mask3 = mask.unsqueeze(-1).expand_as(se)   # [B, C] -> [B, C, 1] -> [B, C, A]
    return (se * mask3).sum() / mask3.sum().clamp(min=1.0)
```

解説:
- **なぜその shape か**: 誤差 `se` は `[B, C, A]`。マスクは `[B, C]` なので、`unsqueeze(-1)` で `[B, C, 1]` にしてから行動次元 `A` 方向へ広げる（`expand_as`）と、要素ごとに掛けられます。パディング位置（mask=0）の誤差はここで 0 になります。
- **なぜ有効要素数で割るか**: 平均は「合計 ÷ 個数」。パディングを除いた**有効要素数** `mask3.sum()` で割ることで、パディングの多寡に左右されない平均になります。`clamp(min=1.0)` は全部パディングのときの 0 除算を防ぐ保険です。
- これは [`functional.py`](../../src/vla_learn/functional.py) の実装と同じです。行動チャンクは末尾がパディングされうるので、損失でそこを無視するのが正しい扱いです（M3 で詳述）。

---

## Q6.（小実装）`ToyVLADataset` + DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader

class ToyVLADataset(Dataset):
    def __init__(self, n=200):
        torch.manual_seed(0)
        self.states  = torch.randn(n, 3)            # [N, 3]
        self.actions = torch.randn(n, 8, 3)         # [N, 8, 3]

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return {"state": self.states[idx],          # [3]
                "action": self.actions[idx]}        # [8, 3]

ds = ToyVLADataset(n=200)
loader = DataLoader(ds, batch_size=32, shuffle=True)
batch = next(iter(loader))
print(batch["state"].shape, batch["action"].shape)  # [32, 3] [32, 8, 3]
```

> **なぜその shape か**: `__getitem__` は **1 件**（`state [3]`, `action [8,3]`）を返します。`DataLoader` がそれを `batch_size=32` 件スタックするので、先頭に**バッチ次元 `B=32`** が付いて `[32, 3]`・`[32, 8, 3]` になります。dict を返すと DataLoader が**キーごとに**まとめてくれるのが便利な点です。これは [`SyntheticVLADataset`](../../src/vla_learn/datasets/synthetic_dataset.py) と同じ設計です。

---

## Q7.（実験・必須）1 バッチに過学習

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
state  = torch.randn(16, 3)
target = torch.randn(16, 8, 3)

model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 8 * 3))
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

first = None
for step in range(2000):
    pred = model(state).view(-1, 8, 3)      # [16, 24] -> [16, 8, 3]
    loss = loss_fn(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if first is None:
        first = loss.item()
last = loss.item()
print(f"first={first:.4f}  last={last:.6f}")
assert last < 0.05 * first
print("OK: 1 バッチに過学習できた = 学習機構は健全")
```

出力例（環境でぶれます）:

```text
first=1.0623  last=0.000007
OK: 1 バッチに過学習できた = 学習機構は健全
```

> **ステップ数について**: ランダムな target を完全に丸暗記するため、線形回帰（Q2）より多めの
> 2000 ステップ・`lr=5e-3` にしています。500 ステップ・`lr=1e-3` だと途中（loss≈0.25）で
> 下げ止まり、丸暗記しきれません。「下がりきらない＝バグ」と早合点せず、まずステップ数と
> `lr` を見直すのも実践的な勘どころです（モデル容量が十分なら、回せば必ず 0 へ向かいます）。

考察:
- **なぜランダムな target でも loss が 0 近くまで下がるのか**: 汎化は不要で、**固定された 16 件を丸暗記**するだけだから。入力 `state [16,3]` から出力 `[16,8,3]` への対応は、64 次元の隠れ層を持つ MLP には十分覚えられる容量があります。だから配線が正しければ loss はほぼ 0 まで落ちます。
- **これがテストである理由**: 賢さではなく**配線（パイプライン）の健全性**を測っています。落ちないなら、本文 1.6 のチェックリスト（`zero_grad` 忘れ／`lr` が極端／pred と target の shape 不一致／勾配が `detach`・`no_grad` で切れている／device・dtype 不一致）のどれかが原因です。
- 実装の [`tests/test_overfit_tiny_batch.py`](../../tests/test_overfit_tiny_batch.py) は、これとほぼ同じことを `TinyVLA` で行い「最後 < 最初 × 0.2」を要求しています。各章でこの確認を続けます。

## Q8.（shape 確認）Conv2d の出力サイズ

公式 `out = (in + 2*padding - kernel) // stride + 1` に代入するだけです。

1. `(64 + 2·1 - 3) // 2 + 1 = 32` → **`[8, 16, 32, 32]`**（チャンネルは `out_channels=16` に変わる）
2. `(32 + 2·1 - 3) // 2 + 1 = 16` → **`[8, 32, 16, 16]`**
3. `(64 + 2·0 - 5) // 1 + 1 = 60` → **`[4, 8, 60, 60]`**（padding=0 だと縁の分だけ縮む）
4. `[8, 32*16*16] = [8, 8192]` → `nn.Linear` の `in_features` は **8192**

> ポイント: `stride=2, kernel=3, padding=1` の組は「**ちょうど半分**」になる定番設定です（本教材の
> `ImageEncoder` はこれを 4 回重ねて 64→32→16→8→4）。`in_features` の数え間違いは
> M2 以降の最頻出バグなので、「畳み込みの最終 shape → flatten → Linear」を常に手で言えるようにしておきます。

## Q9.（小実装）保存 → 復元 → 同じ出力（state_dict）

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 24))
x = torch.randn(5, 3)
y1 = model(x)

torch.save(model.state_dict(), "m1_q9.pt")                    # TODO 1: 重みの辞書だけ保存

model2 = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 24))  # TODO 2: 同じ構造を作り直す
model2.load_state_dict(torch.load("m1_q9.pt"))                #          重みを流し込む
model2.eval()                                                  # TODO 3: 推論モード
y2 = model2(x)

assert torch.allclose(y1, y2), "復元後の出力が一致しない"
print("OK: 保存 → 復元 → 同じ出力")
```

考察:
- **なぜ state_dict か**: `torch.save(model)` はモデルオブジェクトを pickle するため、**クラス定義の
  import パスや PyTorch のバージョンに依存**して壊れやすい。「構造はコードで作り直し、重みだけ流し込む」
  state_dict 方式なら、コードさえあればどの環境でも復元できます。
- **`policy.pt` が重み以外に保存しているもの**（[`checkpoint.py`](../../src/vla_learn/training/checkpoint.py)）:
  ① モデル種別とコンストラクタ引数（`model_type` / `model_kwargs` — どのクラスをどう作り直すか）、
  ② トークナイザ語彙（同じ文字→同じ ID にするため）、③ 行動・状態の正規化統計（M3 の最重要点。
  これが欠けると「学習時と同じ前処理」が再現できず方策が静かに壊れます）。

### 発展

- **A**（`zero_grad` を消す）: 勾配が加算され、Adam の更新が過大になって loss が振動・発散しがちになります（下がっても不安定）。「鉄則の逆」を体感できます。
- **B**（サンプル数を変える）: 16→2 にするとより速く 0 へ。1024 に増やすと、同じ容量・同じステップ数では下がりきらない（「丸暗記」の難度がサンプル数とともに上がる）ことが見えます。

---

> **まとめ**: M1 の演習は「shape を読む・学習ループを正しく組む・頻出バグ（dtype/device/zero_grad/.item）を直す・Dataset を書く・1 バッチに過学習させる」を一通り体験しました。
> ここで身につけた配線が、M2 以降そのまま VLA の学習に育っていきます。次は [M2: 最小の模倣学習](../../lessons/m2_imitation.md) へ。
