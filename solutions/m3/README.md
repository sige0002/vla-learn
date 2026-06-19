# 解答 M3: 行動表現とデータ

問題: [`../../exercises/m3/README.md`](../../exercises/m3/README.md) ／ 本文: [`../../lessons/m3_data_actions.md`](../../lessons/m3_data_actions.md)

各解答に「**正解コード**」と「**なぜその shape か / なぜ loss が下がるか**」の短い説明を付けます。

---

## A1（shape 確認）

```python
import numpy as np, torch
from torch.utils.data import DataLoader
from vla_learn.envs import all_instruction_strings
from vla_learn.datasets import generate_episodes, build_normalizers, SyntheticVLADataset, CharTokenizer

eps = generate_episodes(n_episodes=30, seed=0)
tok = CharTokenizer.from_corpus(all_instruction_strings())
an, sn = build_normalizers(eps)
ds = SyntheticVLADataset(eps, tok, chunk_len=8, action_normalizer=an, state_normalizer=sn)

s = ds[0]
for k, v in s.items():
    print(k, tuple(v.shape), v.dtype)
batch = next(iter(DataLoader(ds, batch_size=4, shuffle=True)))
for k, v in batch.items():
    print(k, tuple(v.shape))
```

| キー | `ds[0]` | batch（B=4） | dtype |
|------|---------|--------------|-------|
| `image` | `(3, 64, 64)` | `(4, 3, 64, 64)` | float32 |
| `state` | `(3,)` | `(4, 3)` | float32 |
| `tokens` | `(max_len,)` | `(4, max_len)` | int64 |
| `action` | `(8, 3)` | `(4, 8, 3)` | float32 |
| `pad_mask` | `(8,)` | `(4, 8)` | float32 |

**なぜこの shape か**: `__getitem__` は 1 サンプルを返す。DataLoader が `batch_size` 個積んで
**先頭に `B`** を作る。画像は `[B,C,H,W]`、行動チャンクは `[B,T,A]`（T=8, A=3）、ベクトルは `[B,D]`。
`tokens` は `int64`（埋め込み層の index に使うため整数）。`max_len` は語彙コーパスの最長文長。

---

## A2（shape 確認）

```python
import numpy as np
from vla_learn.datasets import extract_action_chunk
actions = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
for t in (2, 4, 5):
    c, m = extract_action_chunk(actions, t, chunk_len=4)
    print(t, c.shape, m)
```

| t | `chunk.shape` | `pad_mask` |
|---|---------------|------------|
| 2 | `(4, 3)` | `[1. 1. 1. 1.]`（`n_valid=min(4, 6-2)=4`、全有効） |
| 4 | `(4, 3)` | `[1. 1. 0. 0.]`（`n_valid=2`、残り 2 はパディング） |
| 5 | `(4, 3)` | `[1. 0. 0. 0.]`（`n_valid=1`、残り 3 はパディング） |

**なぜこの shape か**: `chunk` は常に `[chunk_len, A]` の固定形（足りない分は `actions[T-1]` で埋める）。
`pad_mask` は `n_valid=min(chunk_len, T-t)` 個だけ 1、残りは 0。固定形にすることで DataLoader が積める。

---

## A3（穴埋め）

```python
import numpy as np, torch
from vla_learn.datasets import generate_episodes, Normalizer

eps = generate_episodes(n_episodes=100, seed=0)
actions = np.concatenate([ep["actions"] for ep in eps], axis=0)

norm = Normalizer.fit(actions)          # (a) fit: 統計(mean/std)を推定
z = norm.normalize(actions[:5])         # (b) normalize: 正規化
print("正規化後の平均(おおよそ0):", z.mean(0))

z_hat = torch.zeros(8, 3)               # 正規化空間のダミー出力
a_for_env = norm.denormalize(z_hat)     # (c) denormalize: 元の空間へ
print("環境に渡す行動の例:", a_for_env[0])
```

- (a) `fit` (b) `normalize` (c) `denormalize`。
- **逆にしたら**: 学習で `denormalize`、推論で `normalize` を使うと、環境に**正規化空間の小さな値**が
  渡って動かない／学習で**生の大きな差分**を教師にして不安定。本文 1.4 の図の通り、
  「学習=normalize、推論出力=denormalize」を守る。

---

## A4（バグ修正）

```python
import numpy as np
from vla_learn.datasets import generate_episodes, Normalizer

eps = generate_episodes(n_episodes=50, seed=0)
actions = np.concatenate([ep["actions"] for ep in eps], axis=0)

norm = Normalizer.fit(actions)          # 修正1: 行動の統計で行動を正規化
a = actions[:5]
z = norm.normalize(a)
back = norm.denormalize(z)              # 修正2: 戻すのは denormalize
print("往復誤差(最大):", np.abs(back - a).max())   # < 1e-5
```

- 修正1: `Normalizer.fit(states)` ではなく `fit(actions)`。**対象データ自身**の `mean/std` で正規化する。
- 修正2: `normalize` を二度ではなく `denormalize` で戻す。

行動を状態の統計で正規化すると問題: スケール（`grip` は 0/1、`dx,dy` は ±0.04）が**合わず**、
正規化後の平均 0・分散 1 が崩れる。学習が不安定になり、推論での逆正規化もズレる。

**なぜ往復が一致するか**: `denormalize(normalize(x)) = (x-m)/s * s + m = x`（数学的に厳密に元へ戻る。
浮動小数の丸めだけ `~1e-7` 残る）。

---

## A5（小実装）

```python
import numpy as np
from vla_learn.datasets import extract_action_chunk as ref

def my_extract_action_chunk(actions, t, chunk_len):
    T, A = actions.shape
    chunk = np.zeros((chunk_len, A), dtype=np.float32)
    pad_mask = np.zeros((chunk_len,), dtype=np.float32)
    n_valid = min(chunk_len, T - t)
    chunk[:n_valid] = actions[t:t + n_valid]
    pad_mask[:n_valid] = 1.0
    if n_valid < chunk_len:
        chunk[n_valid:] = actions[T - 1]   # 最後の行動で埋める
    return chunk, pad_mask

actions = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
for t in (0, 3, 5):
    c1, m1 = my_extract_action_chunk(actions, t, 4)
    c2, m2 = ref(actions, t, 4)
    assert np.allclose(c1, c2) and np.allclose(m1, m2), f"t={t} で不一致"
print("OK: 実装が一致しました")
```

**なぜこの shape か**: 出力は常に `[chunk_len, A]` と `[chunk_len]`。`n_valid` を境に「有効＝実データ」
「無効＝最後の行動で埋め（`pad_mask=0`）」に分ける。固定形なので後段の DataLoader/モデルが扱える。

---

## A6（小実装）

```python
from vla_learn.envs import all_instruction_strings
from vla_learn.datasets import CharTokenizer

tok = CharTokenizer.from_corpus(all_instruction_strings())
print("vocab_size:", tok.vocab_size, " max_len:", tok.max_len)

for s in ["青のブロックを青のゴールに置いて", "赤ブロックをつかんで緑ゴールへ"]:
    ids = tok.encode(s)
    assert len(ids) == tok.max_len            # 常に固定長
    assert tok.decode(ids) == s               # PAD(0) を除いて復元
    print(len(s), "->", len(ids), "末尾:", ids[-4:])  # 末尾は 0 が並ぶことが多い
print("OK")
```

語順の要約: ID 列は**並び**を保つので、「赤を青ゴールへ」と「青を赤ゴールへ」は**違う ID 順**になる。
これを単純な平均で潰すと両者が同じベクトルになり grounding 不能。だから M4 の `TextEncoder` は
位置埋め込み + Transformer で語順を区別する。

**なぜ固定長か**: モデルにバッチで入れるには長さを揃える必要がある。`encode` は不足を PAD(0)、超過を
切り詰めて `max_len` に固定する。`PAD_ID=0` は埋め込みで「無視する印」として使う。

---

## A7（実験）

```python
import torch
from vla_learn.functional import masked_mse

pred   = torch.zeros(1, 4, 3)
target = torch.zeros(1, 4, 3)
target[0, 3, :] = 100.0                 # パディング位置だけ巨大
mask   = torch.tensor([[1., 1., 1., 0.]])

naive  = ((pred - target) ** 2).mean()
masked = masked_mse(pred, target, mask)
print("素朴 MSE :", naive.item())       # 大きい（巨大な誤差を平均に含む）
print("masked   :", masked.item())      # 0（パディング位置を除外）
```

出力例:

```text
素朴 MSE : 2500.0
masked   : 0.0
```

（検算: `t=3` の 3 要素が誤差 100 → 二乗 10000 が 3 つ。全要素数 `1*4*3=12` で割ると `30000/12=2500`。
masked は有効要素 `1*3*3=9` で割るが、誤差はすべてパディング位置にあるので合計 0 → `0.0`。）

説明: 素朴 MSE は全 `B*C*A` 要素で割るので、パディング位置 `t=3` の巨大誤差まで平均に入る。
`masked_mse` は `pad_mask` を掛けてから**有効要素数で割る**（`mask3.sum()` で正規化）ので、
存在しないステップは一切寄与しない。mask を渡し忘れると、終端パディングの再現に学習が引っ張られ、
**実在ステップの精度が犠牲**になる。

---

## A8（必須・1 バッチ過学習）

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from vla_learn.envs import all_instruction_strings
from vla_learn.datasets import generate_episodes, build_normalizers, SyntheticVLADataset, CharTokenizer
from vla_learn.functional import masked_mse
from vla_learn.utils import set_seed

set_seed(0)
eps = generate_episodes(n_episodes=8, seed=0)
tok = CharTokenizer.from_corpus(all_instruction_strings())
an, sn = build_normalizers(eps)
ds = SyntheticVLADataset(eps, tok, chunk_len=8, action_normalizer=an, state_normalizer=sn)
batch = next(iter(DataLoader(ds, batch_size=16, shuffle=True)))

model = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 8 * 3))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
first = None
for _ in range(400):
    pred = model(batch["state"]).reshape(-1, 8, 3)
    loss = masked_mse(pred, batch["action"], batch["pad_mask"])
    opt.zero_grad(); loss.backward(); opt.step()
    if first is None:
        first = loss.item()
print(f"{first:.4f} -> {loss.item():.6f}")
assert loss.item() < 0.2 * first
print("OK")
```

出力例: `0.9985 -> 0.096665` 程度（最初の 20% を下回る。値はぶれる）。

- `reshape(-1, 8, 3)` が必要な理由: `masked_mse` は `pred,target:[B,C,A]`、`mask:[B,C]` を期待する。
  `Linear` の出力 `[B, 24]` を `[B, 8, 3]` に整形して形を合わせる。
- 1 バッチ過学習 vs 汎化: この 16 サンプルは `state` だけで**丸暗記**できるので loss はほぼ 0 まで落ちる。
  だが**未知の局面**を解くには「どのブロックをどこへ」を知る必要があり、`state[3]` だけでは足りない。
  だから M4 では**画像 + 言語**を加える（過学習できること＝学習機構が健全、汎化できること＝情報が十分、は別問題）。

**なぜ loss が下がるべきか**: 1 バッチ過学習は「モデル容量 + 最適化 + 損失 + shape」が正しければ必ず通る
健全性チェック。`masked_mse` がパディングを正しく除外し、`reshape` で shape が合っていれば、
Adam が 16 サンプルの入出力対応を暗記して loss を 0 近くまで下げる。下がらなければどこかにバグ。
