# 演習 M3: 行動表現とデータ

対応する本文: [`../../lessons/m3_data_actions.md`](../../lessons/m3_data_actions.md)

型は「**shape 確認 → 穴埋め → バグ修正 → 小実装 → 実験**」。1 問 1 概念。
解答は [`../../solutions/m3/README.md`](../../solutions/m3/README.md)。まず自分で手を動かしてから見てください。

準備（共通）:

```python
import numpy as np
import torch
from torch.utils.data import DataLoader
from vla_learn.envs import all_instruction_strings
from vla_learn.datasets import (
    generate_episodes, build_normalizers, SyntheticVLADataset,
    CharTokenizer, Normalizer, extract_action_chunk,
)
from vla_learn.functional import masked_mse
from vla_learn.utils import set_seed
```

---

## Q1（shape 確認）`SyntheticVLADataset` の 1 サンプル

`SyntheticVLADataset` を `chunk_len=8` で作り、`ds[0]` の各キーの shape と dtype を
**コードを動かす前に**言い当ててください。次に DataLoader で `batch_size=4` にしたとき、
各テンソルの shape がどう変わるかも答えてください。

- `image` : `[?, ?, ?]`（→ batch では `[?, ?, ?, ?]`）
- `state` : `[?]`（→ `[?, ?]`）
- `tokens`: `[?]`（→ `[?, ?]`）
- `action`: `[?, ?]`（→ `[?, ?, ?]`）
- `pad_mask`: `[?]`（→ `[?, ?]`）

```python
eps = generate_episodes(n_episodes=30, seed=0)
tok = CharTokenizer.from_corpus(all_instruction_strings())
an, sn = build_normalizers(eps)
ds = SyntheticVLADataset(eps, tok, chunk_len=8, action_normalizer=an, state_normalizer=sn)
# print して確認
```

> ヒント: 本文 4 節の表。画像 `[B,C,H,W]`、行動チャンク `[B,T,A]`、ベクトルは `[B,D]`。

---

## Q2（shape 確認）`extract_action_chunk` の終端パディング

`chunk_len=4`、エピソード長 `T=6` のとき、`t=2, 4, 5` での `chunk` と `pad_mask` の shape、
そして `pad_mask` の中身（どこが 1 でどこが 0 か）を**手で**書き出してください。
その後コードで確認します。

```python
actions = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
for t in (2, 4, 5):
    chunk, pad_mask = extract_action_chunk(actions, t, chunk_len=4)
    print(t, chunk.shape, pad_mask)
```

> ヒント: 本文 2.3 の図。`n_valid = min(chunk_len, T - t)`、足りない分は最後の行動で埋め `pad_mask=0`。

---

## Q3（穴埋め）正規化の往復と推論での逆正規化

`____` を埋めて、(1) 行動を正規化して学習用テンソルにし、(2) モデル出力（ここではダミー）を
**環境に渡す前に逆正規化**する流れを完成させてください。

```python
eps = generate_episodes(n_episodes=100, seed=0)
actions = np.concatenate([ep["actions"] for ep in eps], axis=0)   # [N, 3]

norm = Normalizer.____(actions)            # (a) 統計を推定するメソッドは？
z = norm.____(actions[:5])                 # (b) 正規化（平均0分散1の空間へ）
print("正規化後の平均(おおよそ0):", z.mean(0))

# --- 推論側のダミー: モデルは正規化空間で z_hat を出すと仮定 ---
z_hat = torch.zeros(8, 3)                   # [chunk_len=8, 3] のダミー出力（正規化空間）
a_for_env = norm.____(z_hat)               # (c) 環境に渡すために元の空間へ戻す
print("環境に渡す行動の例:", a_for_env[0])  # 生の [dx, dy, grip] スケール
```

(b) と (c) を**逆にしてしまうと**何が起きるか、本文 1.4 の図を使って 1 行で説明してください。

> ヒント: `fit` / `normalize` / `denormalize`。学習は normalize、推論出力は denormalize。

---

## Q4（バグ修正）正規化の往復が一致しない

下のコードは「正規化 → 逆正規化で元に戻る」はずなのに、`往復誤差` が大きく出ます。
**2 つのバグ**を見つけて直してください。

```python
eps = generate_episodes(n_episodes=50, seed=0)
actions = np.concatenate([ep["actions"] for ep in eps], axis=0)   # [N, 3]

# バグ1: 行動ではなく状態の統計で行動を正規化している
states = np.concatenate([ep["agent"] for ep in eps], axis=0)
norm = Normalizer.fit(states)              # ← ?

a = actions[:5]
z = norm.normalize(a)
# バグ2: 逆正規化のつもりで normalize をもう一度呼んでいる
back = norm.normalize(z)                    # ← ?

err = np.abs(back - a).max()
print("往復誤差(最大):", err)               # 大きい値が出てしまう
```

直した上で、往復誤差が `1e-5` 未満になることを確認してください。
さらに「**行動の統計で行動を正規化しないと何が問題か**」を 1〜2 行で。

> ヒント: 正規化は**対象データ自身**の `mean/std` で。戻すのは `denormalize`。本文 1 節。

---

## Q5（小実装）`extract_action_chunk` を自分で書く

`vla_learn` の実装を見ずに、`extract_action_chunk(actions, t, chunk_len)` を **20〜40 行**で
書いてください。仕様（本文 2.2）:

- 入力: `actions` `[T, A]`、`t`（開始時刻）、`chunk_len`。
- 出力: `chunk` `[chunk_len, A]`（float32）と `pad_mask` `[chunk_len]`（1=有効, 0=パディング）。
- 終端で足りない分は **`actions[T-1]`（最後の行動）で埋め**、その位置の `pad_mask=0`。

書けたら、本物の実装と**出力が一致するか**を `t=0,3,5`（`T=6, chunk_len=4`）で照合してください。

```python
def my_extract_action_chunk(actions, t, chunk_len):
    T, A = actions.shape
    # ここを実装
    ...
    return chunk, pad_mask

# 照合
from vla_learn.datasets import extract_action_chunk as ref
actions = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
for t in (0, 3, 5):
    c1, m1 = my_extract_action_chunk(actions, t, 4)
    c2, m2 = ref(actions, t, 4)
    assert np.allclose(c1, c2) and np.allclose(m1, m2), f"t={t} で不一致"
print("OK: 実装が一致しました")
```

> ヒント: `n_valid = min(chunk_len, T - t)`。`chunk[:n_valid] = actions[t:t+n_valid]`、残りを埋める。

---

## Q6（小実装）`CharTokenizer` で語彙と固定長を確かめる

`all_instruction_strings()` から `CharTokenizer` を作り、次を確認する短いコードを書いてください。

1. `vocab_size` と `max_len` を表示。
2. 任意の指示文を `encode` し、**長さが常に `max_len`** であること（短い文でも PAD で埋まる）。
3. `decode(encode(s)) == s` が成り立つこと（PAD=0 は復元時に除外される）。
4. `encode` の戻り値 ID 列の**末尾**に `0`（PAD）が並ぶことを目視。

「ID 列の**並び**が情報を持つ」とはどういうことか、本文 3 節の語順の例（「赤を青ゴール」vs「青を赤ゴール」）を
1 行で要約してください（[M4](../../lessons/m4_tiny_vla_mse.md) の `TextEncoder` への伏線）。

> ヒント: 本文 3 節のコードがほぼ答え。`tok.encode(s)` の長さは常に `tok.max_len`。

---

## Q7（実験）`masked_mse` でパディングを除外する効果

`pad_mask` を使う `masked_mse` と、使わない素朴 MSE で、
**パディング位置だけ予測が大きく外れている**ケースを作り、値が変わることを観察してください。

```python
pred   = torch.zeros(1, 4, 3)
target = torch.zeros(1, 4, 3)
target[0, 3, :] = 100.0                 # パディング位置(t=3)だけ巨大な値
mask   = torch.tensor([[1., 1., 1., 0.]])  # t=3 はパディング

naive  = ((pred - target) ** 2).mean()
masked = masked_mse(pred, target, mask)
print("素朴 MSE :", naive.item())
print("masked   :", masked.item())
```

- 2 つの値が**なぜ違うのか**を、`masked_mse` の「**有効ステップ数で割る**」実装（本文 5 節）に即して説明。
- もし `mask` を渡し忘れたら学習にどんな悪影響が出るか、1 行で。

---

## Q8（必須・学習デバッグの鉄則）正しい dict で 1 バッチ過学習

`SyntheticVLADataset` から `batch_size=16` の **1 バッチ**を取り、簡単な回帰モデルで
**そのバッチだけ**に過学習できることを確認してください。ここでは画像・言語を使わず、
`state[3] → action[8,3]` を当てる小さな MLP で十分です（行動チャンクを平らにして出力）。

```python
set_seed(0)
eps = generate_episodes(n_episodes=8, seed=0)
tok = CharTokenizer.from_corpus(all_instruction_strings())
an, sn = build_normalizers(eps)
ds = SyntheticVLADataset(eps, tok, chunk_len=8, action_normalizer=an, state_normalizer=sn)
batch = next(iter(DataLoader(ds, batch_size=16, shuffle=True)))

import torch.nn as nn
model = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 8 * 3))  # -> [B, 24]
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

first = None
for _ in range(400):
    pred = model(batch["state"]).reshape(-1, 8, 3)     # [B, 8, 3] に整形
    loss = masked_mse(pred, batch["action"], batch["pad_mask"])
    opt.zero_grad(); loss.backward(); opt.step()
    if first is None:
        first = loss.item()
print(f"{first:.4f} -> {loss.item():.6f}")
assert loss.item() < 0.2 * first
```

- なぜ `reshape(-1, 8, 3)` が必要か（`masked_mse` が期待する shape は？）。
- `state` だけで `action[8,3]` を完全再現できる**この 1 バッチ**と、汎化（[M4](../../lessons/m4_tiny_vla_mse.md) で画像・言語が要る理由）の違いを 1 行で。

> ヒント: `masked_mse(pred, target, mask)` は `pred,target:[B,C,A]`, `mask:[B,C]`。本文 5 節と 6 章。

---

### 提出のしかた（自習用チェックリスト）

- [ ] Q1〜Q2: shape を**動かす前に**言い当てた（batch 次元の付き方も）
- [ ] Q3: 正規化→逆正規化の役割を理解し、逆にすると壊れる理由を言えた
- [ ] Q4: 2 バグを直し往復誤差 `< 1e-5`
- [ ] Q5: 自作 `extract_action_chunk` が本物と一致
- [ ] Q6: 固定長・復元・PAD を確認、語順の意味を要約
- [ ] Q7: masked と素朴 MSE の差を実装に即して説明
- [ ] Q8: 正しい dict・正しい shape で 1 バッチ過学習（`< 0.2 * first`）
