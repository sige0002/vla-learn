# M1 演習: PyTorch 速習

対応する本文: [`../../lessons/m1_pytorch.md`](../../lessons/m1_pytorch.md)

この演習は本教材共通の **5 型**「**shape 確認 → 穴埋め → バグ修正 → 小実装 → 実験**」で構成しています。
1 問 1 概念。難易度の目安は 誘導 60% / 小実装 25% / 自由実験 15%。
**最後の Q7 は必ずやってください**（「1 バッチに過学習できるか」= 学習デバッグの鉄則）。

## 進め方

- 環境は [M0](../../lessons/m0_overview.md) のセットアップ済み（`uv sync` 済みで `uv run pytest` が通る）が前提です。
- すべて **CPU・コピペで動く** ように作っています。Python シェルか、雛形 [`starter.py`](starter.py) を使ってください。
- 雛形には `TODO` が入っています。`uv run python starter.py` で実行し、各問の `check_qN()` が `OK` を出せば正解です。
- 詰まったら本文の該当節に戻り、それでも分からなければ [`../../solutions/m1/README.md`](../../solutions/m1/README.md) を見ましょう。

---

## Q1.（shape 確認）形を手で言い当てる

次のテンソルの最終的な `shape` を、**コードを実行する前に紙に書いてから**確認してください。
`B`=バッチ, `C`=チャンク長, `A`=行動次元 の感覚を体に入れる問題です。

```python
import torch

a = torch.randn(64, 3)               # (1) a.shape = ?
b = torch.randn(64, 8, 3)            # (2) b.shape = ?
c = b.reshape(64, -1)                # (3) c.shape = ?   （-1 は自動計算）
d = a.unsqueeze(1)                   # (4) d.shape = ?
e = torch.randn(8).unsqueeze(-1)     # (5) e.shape = ?
f = (b * torch.ones(64, 8, 1)).shape # (6) f       = ?   （ブロードキャスト）
```

> ヒント: `reshape(64, -1)` は「先頭を 64 に固定し、残りはよしなに」。`unsqueeze(k)` は位置 `k` に大きさ 1 の次元を挿入。

---

## Q2.（穴埋め）最小の学習ループを完成させる

`y = 3x - 2` を当てる線形回帰です。`____` を埋めて、loss が下がるようにしてください（本文 1.5 と同じ 4 ステップ）。

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
    pred = ____                       # (1) 順伝播
    loss = ____                       # (2) 損失
    optimizer.____()                  # (3) 勾配リセット
    loss.____()                       # (4) 逆伝播
    optimizer.____()                  # (5) 更新

print(model.weight.item(), model.bias.item())  # 3.0, -2.0 に近づけば成功
```

---

## Q3.（バグ修正）device / dtype の不一致を直す

次の 3 つのスニペットはそれぞれ**実行時エラー**になります。原因を述べ、1〜2 行の修正で直してください。
（VLA 開発で最も頻繁に出会うエラー群です。）

**Q3-a（dtype 不一致）**: トークン ID を埋め込みに通したい。

```python
import torch
import torch.nn as nn

emb = nn.Embedding(num_embeddings=50, embedding_dim=16)
tokens = torch.tensor([3.0, 7.0, 1.0])     # ← ここが原因
out = emb(tokens)                          # RuntimeError が出る
```

> ヒント: 埋め込み表の索引は整数でなければなりません。本教材の `tokens` は int64 でした。

**Q3-b（dtype 不一致 in 演算）**: 状態と整数を足したい。

```python
import torch

state = torch.tensor([0.3, 0.7, 0.0])      # float32
idx   = torch.tensor([1, 2, 3])            # int64
mixed = state * idx                        # RuntimeError（型が違う）
```

**Q3-c（device 不一致）**: モデルと入力の device がずれている想定。次のコードを「**CPU/GPU どちらでも安全に動く**」形に直してください。

```python
import torch
import torch.nn as nn
from vla_learn.utils.device import get_device

dev = get_device()
model = nn.Linear(3, 3)                     # ← CPU に置かれたまま
x = torch.randn(4, 3).to(dev)
y = model(x)                                # dev が cuda だと不一致でエラー
```

---

## Q4.（バグ修正）「loss が下がらない」3 大バグ

次の学習ループは loss が下がりません（または途中で壊れます）。**3 か所**のバグを見つけて直してください。

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
    loss.backward()                 # バグ(1): あるものを呼び忘れている
    optimizer.step()
    losses.append(loss)             # バグ(2): 記録の仕方が良くない
print(sum(losses) / len(losses))    # バグ(2)の影響でここが壊れる/重い
```

> ヒント:
> - (1) `.grad` は加算される。毎ステップ何をリセットすべき？（本文 1.5「zero_grad を忘れると」）
> - (2) テンソルをそのまま `list` に貯めると計算グラフが残る。スカラを取り出す関数は？（本文 1.5 `.item()`）
> - (3) もう 1 つ、`losses` に貯めた値を `sum()` する行も、(2) を直せば自然に解決します。何が問題だったか言葉で説明してください。

---

## Q5.（小実装）`masked_mse` を自分で書く

本教材の損失 [`functional.py`](../../src/vla_learn/functional.py) の `masked_mse` を、**仕様だけ見て自力で実装**してください。
パディング（行動チャンク末尾の埋め）を損失から除外する MSE です。

**仕様**:
- 入力: `pred [B, C, A]`, `target [B, C, A]`, `mask [B, C]`（1=有効, 0=パディング, float32）。`mask` は `None` も許す。
- `mask is None` のとき: 全要素の平均二乗誤差を返す。
- `mask` があるとき: **mask=0 の (B,C) 位置を平均から除外**して二乗誤差の平均を返す。
- 返り値はスカラのテンソル。

```python
import torch

def masked_mse(pred, target, mask=None):
    # TODO: 実装する
    # ヒント: se = (pred - target) ** 2  # [B, C, A]
    #         mask を [B, C, 1] にして broadcast で掛け、有効要素数で割る
    ...

# 動作確認（mask=None なら nn.MSELoss と一致するはず）
pred   = torch.randn(4, 8, 3)
target = torch.randn(4, 8, 3)
assert torch.allclose(masked_mse(pred, target), torch.nn.functional.mse_loss(pred, target))
```

> ヒント: 有効要素数で割るとき 0 除算を避けるため `.clamp(min=1.0)` を使うと安全です。

---

## Q6.（小実装）`ToyVLADataset` と DataLoader を書く

状態 `[3]` と行動チャンク `[8, 3]` の組を返す `Dataset` を実装し、`DataLoader` で 1 バッチ取り出してください。

**仕様**:
- `__init__(self, n=200)`: `torch.manual_seed(0)` 後、`states [n, 3]` と `actions [n, 8, 3]` を `torch.randn` で作る。
- `__len__`: 件数 `n` を返す。
- `__getitem__(idx)`: `{"state": [3], "action": [8, 3]}` を返す。
- 作った Dataset を `DataLoader(batch_size=32, shuffle=True)` で包み、最初のバッチの `state`/`action` の shape を表示する。

期待する出力:

```text
torch.Size([32, 3]) torch.Size([32, 8, 3])
```

> ヒント: 本文 1.7 の `ToyVLADataset` がほぼそのまま使えます。`from torch.utils.data import Dataset, DataLoader`。
> 「先頭にバッチ次元 `B=32` が付く」ことを目で確認してください。

---

## Q7.（実験・必須）小さな線形回帰を「1 バッチに過学習」させる

**学習デバッグの鉄則**を体験します。本教材の全章でこの確認をします。

**お題**: 状態 `[3]` → 行動チャンク `[8, 3]` を出す小さな MLP を作り、**ランダムだが固定した 1 バッチ（16 サンプル）** に過学習させてください。

**要件**:
1. モデル: `nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 8*3))`。出力は `.view(-1, 8, 3)` で `[B, 8, 3]` に整形。
2. データ: `torch.manual_seed(0)` 後、`state [16, 3]` と `target [16, 8, 3]` を `randn` で 1 度だけ作り、**同じものを使い回す**。
3. 損失 `nn.MSELoss`、`Adam(lr=5e-3)` で **2000 ステップ**回す。
4. 最初の loss と最後の loss を表示し、**最後 < 最初 × 0.05** になっていることを確認する。

> ランダムな target を「丸暗記」させるので、線形回帰（Q2）より多くのステップが要ります。
> もし途中の loss が下げ止まったら、ステップ数を増やすか `lr` を上げてみてください（本文 1.6 のチェックリストも参照）。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
state  = torch.randn(16, 3)
target = torch.randn(16, 8, 3)

model = ...          # TODO
loss_fn = ...        # TODO
optimizer = ...      # TODO

first = None
for step in range(2000):
    pred = ...       # TODO: model(state) を [16, 8, 3] に整形
    loss = ...       # TODO
    # TODO: zero_grad -> backward -> step
    if first is None:
        first = loss.item()
last = loss.item()
print(f"first={first:.4f}  last={last:.6f}")
assert last < 0.05 * first, "過学習できていない（配線にバグ）"
print("OK: 1 バッチに過学習できた = 学習機構は健全")
```

**考察（解答で確認）**:
- なぜランダムな target でも loss が 0 近くまで下がるのか？
- もし下がらなかったら、本文 1.6 のチェックリストのどれを疑うか？

---

## Q8.（shape 確認）Conv2d の出力サイズを手で当てる

本文 1.4 の公式 `out = (in + 2*padding - kernel) // stride + 1` だけで、**実行する前に**答えてください。

1. `nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)` に `[8, 3, 64, 64]` を入れた出力 shape は？
2. 1 の出力に `nn.Conv2d(16, 32, 3, stride=2, padding=1)` を重ねた出力 shape は？
3. `nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=0)` に `[4, 3, 64, 64]` を入れた出力 shape は？
4. 2 の出力を `x.flatten(1)` すると shape は？ その次に置く `nn.Linear` の `in_features` はいくつ？

手で答えてから、実行して答え合わせ:

```python
import torch
import torch.nn as nn

x = torch.rand(8, 3, 64, 64)
print(nn.Conv2d(3, 16, 3, stride=2, padding=1)(x).shape)   # 1 の確認（以降も同様に）
```

---

## Q9.（小実装）保存 → 復元 → 同じ出力を確認する（state_dict）

本文 1.5 の state_dict の定石を、最小モデルで一往復します。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 24))
x = torch.randn(5, 3)
y1 = model(x)

# TODO 1: model の state_dict を "m1_q9.pt" に保存する
# TODO 2: 同じ構造の model2 を作り、保存した重みを読み込む
# TODO 3: model2 を eval モードにして y2 = model2(x) を計算する

assert torch.allclose(y1, y2), "復元後の出力が一致しない"
print("OK: 保存 → 復元 → 同じ出力")
```

**考察（解答で確認）**:
- なぜ `torch.save(model)`（モデル丸ごと）ではなく `state_dict` を保存するのが定石か？
- 本教材の `policy.pt`（[`checkpoint.py`](../../src/vla_learn/training/checkpoint.py)）は、重みの他に何を保存しているか。3 つ挙げてください。

---

### 発展（任意）

- **A**: Q7 で `optimizer.zero_grad()` を**わざと消す**と loss がどうなるか観察してください（鉄則の逆方向の確認）。
- **B**: Q7 のサンプル数を 16 → 2 に減らすと、何ステップで下がりますか。逆に 1024 に増やすと？（過学習のしやすさが「サンプル数」に依存することを体感）。
