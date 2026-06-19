# 演習 M2: 最小の模倣学習

対応する本文: [`../../lessons/m2_imitation.md`](../../lessons/m2_imitation.md)

各問は **1 問 1 概念**です。型は本文と同じ「**shape 確認 → 穴埋め → バグ修正 → 小実装 → 実験**」。
解答は [`../../solutions/m2/README.md`](../../solutions/m2/README.md) にあります。まず自分で手を動かしてから見てください。

準備（共通）:

```bash
# 仮想環境を有効化してパッケージを入れた状態で（README 参照）
python   # 対話モードで進めると楽です
```

```python
import numpy as np
import torch
import torch.nn as nn
from vla_learn.datasets import generate_episodes
from vla_learn.envs import Tabletop2DEnv, expert_action
from vla_learn.utils import set_seed
```

---

## Q1（shape 確認）エキスパートのデモの形

`generate_episodes` を呼び、最初のエピソードについて次を**手で言い当ててから**コードで確認してください。

1. `ep["actions"]` の shape は `[?, ?]`。それぞれの軸は何を表す？
2. `ep["agent"]` の shape は？ 1 行（`ep["agent"][t]`）の 3 要素は何？
3. `expert_action(world)` が返す配列の shape と、3 要素の意味は？

```python
eps = generate_episodes(n_episodes=10, seed=0)
ep = eps[0]
# ここで print して確認
```

> ヒント: 行動は `[dx, dy, grip_cmd]`、状態は `[ax, ay, gripper]`。本文 2 節。

---

## Q2（shape 確認）バッチにしたときの形

3 節の `state→action` 学習で、`DataLoader(TensorDataset(X, Y), batch_size=256)` から取り出した
1 バッチ `(xb, yb)` の shape をそれぞれ答えてください。`X, Y` は全時刻を縦に積んだ `[N, 3]` です。
最後のバッチだけ `256` にならないことがあるのはなぜ？

> ヒント: DataLoader は先頭に `B` を作る。N が 256 で割り切れないと最後が端数。

---

## Q3（穴埋め）`state → action` の MLP と学習ループ

次のコードの `____` を埋めて、`state[3] → action[3]` の模倣学習を完成させてください。

```python
set_seed(0)
eps = generate_episodes(n_episodes=200, seed=0)
states  = np.concatenate([ep["agent"]   for ep in eps], axis=0)
actions = np.concatenate([ep["actions"] for ep in eps], axis=0)
X = torch.from_numpy(states).float()
Y = torch.from_numpy(actions).float()

class StateToAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, ____),          # (a) 出力次元は？
        )
    def forward(self, s):
        return self.net(s)

from torch.utils.data import TensorDataset, DataLoader
loader = DataLoader(TensorDataset(X, Y), batch_size=256, shuffle=True)
model = StateToAction()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(10):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        ____            # (b) 勾配をリセット
        ____            # (c) 逆伝播
        ____            # (d) パラメータ更新
print("最終 loss:", loss.item())
```

> ヒント: 出力は行動の次元数。最適化 3 点セットは `opt.zero_grad()` → `loss.backward()` → `opt.step()`。本文 3 節。

---

## Q4（バグ修正）閉ループで「行動を逆正規化し忘れた」ような壊れ方を直す

下のミニ閉ループには **3 つのバグ**があります。本文 4 節を参考に直してください。
（このモデルは正規化していない素朴版なので「逆正規化忘れ」ではなく、もっと基本的なミスです。）

```python
@torch.no_grad()
def mini_closed_loop_buggy(model, n_episodes=10, seed=1000):
    # (1) eval モードにしていない
    succ = 0
    for k in range(n_episodes):
        env = Tabletop2DEnv(seed=seed + k)
        obs = env.reset()
        done = False
        while not done:
            state = torch.from_numpy(obs["state"]).float()      # (2) バッチ次元が無い [3]
            a = model(state).numpy()
            obs, _, done, info = env.step(a)
        # (3) 最後の info ではなく毎回成功判定しようとして変数が未定義のことがある
        succ += int(info.get("success", False))
    return succ / n_episodes
```

直すべき 3 点を述べ、修正版を書いてください。

> ヒント:
> - (1) 推論前に `model.eval()`。
> - (2) `nn.Linear` は `[B, in]` を期待。`unsqueeze(0)` で `[1,3]` にし、出力は `squeeze(0)`。
> - (3) `info` は `while` の中で必ず定義されるが、`n_episodes=0` や途中 break に弱い。ここでは
>   「ループ内で必ず1回は step する」前提を保ちつつ、`info` を関数冒頭で初期化しておくと安全。

---

## Q5（小実装）`image → action` の最小 CNN を書く

`[B, 3, 64, 64] → [B, 3]` の CNN 回帰 `ImageToAction` を **30〜50 行**で実装してください。要件:

- 畳み込みを 3 段重ねて `64 → 32 → 16 → 8` と解像度を半分ずつ落とす（`stride=2`）。
- 最後に `Flatten` → `Linear` 2 段で `[B, 3]` を出す。
- 出力層に活性化を**付けない**（理由も一言）。

`forward` に `torch.zeros(2, 3, 64, 64)` を通して、出力が `[2, 3]` になることを確認してください。

> ヒント: 本文 5.1 の `ImageToAction` が答えの一例。`32*8*8 = 2048` が Flatten 後の次元。

---

## Q6（実験・本章の主役）ノイズの有無で閉ループ成功率を比べる

本文 5.1 の `train_image_bc` と `closed_loop_success` を使い、次を比較してください。

1. `action_noise=0.00` で学習したモデルの閉ループ成功率。
2. `action_noise=0.03` で学習したモデルの閉ループ成功率。
3. 余裕があれば `0.06`, `0.10` も。**ノイズを上げ続けると成功率はどうなる**？

各設定で**評価シードは固定**して公平に比べること。結果を 1 行でまとめ、
「**なぜノイズありが（多くの場合）強いのか**」を本文 4〜5 節の言葉（分布シフト / 誤差蓄積 / リカバリ）で説明してください。

> 注意: CPU では 1 設定あたり数十秒〜数分。時間がなければ `n_episodes=150, epochs=4` に落として傾向を見る。
> 数値は強くぶれるので、**絶対値ではなく傾向**（ノイズありが同等以上）を見ること。

---

## Q7（実験）「お手本は失敗の直し方を見せない」を観察する

`generate_episodes(action_noise=0.0)` のデータで、エピソードごとに
「`agent[t]` がエキスパート軌道からどれだけ広がっているか」を雑に測ってみましょう。例えば
**各時刻の `ax` の分散**や、**1 ステップあたりの移動量 `|dx|+|dy|` の分布**を、
`action_noise=0.0` と `0.03` で比べてください。ノイズありの方が**訪問状態が広がっている**ことを数字で示し、
それが Q6 の結果（崩れにくさ）とどう繋がるかを 2〜3 行で書いてください。

> ヒント: `np.concatenate([ep["agent"] for ep in eps])` の列ごとの `std`、
> `np.abs(actions[:, :2]).sum(1)` のヒストグラム的な統計（min/mean/max）など、簡単な指標で十分。

---

## Q8（必須・学習デバッグの鉄則）1 バッチに過学習できるか

`state→action` モデル（Q3）で、**32 サンプルの 1 バッチだけ**を 300 ステップ学習し、
loss が最初の **20% 未満**まで下がることを確認してください。下がらない場合に疑う点を 3 つ挙げること。

```python
from torch.utils.data import TensorDataset, DataLoader
set_seed(0)
one = next(iter(DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)))
xb, yb = one
# ここで model を作り直し、300 ステップ回して first/last loss を比較
```

> ヒント: 本文 6 節。`assert loss.item() < 0.2 * first`。疑う点は zero_grad 忘れ / detach / shape / train 状態。

---

### 提出のしかた（自習用チェックリスト）

- [ ] Q1〜Q2: shape を**コードを動かす前に**言い当てた
- [ ] Q3: loss が下がった（epoch が進むほど小さく）
- [ ] Q4: 3 バグを直し、成功率が `0%` 以外で出た
- [ ] Q5: `[2,3]` を確認、活性化を付けない理由を言えた
- [ ] Q6: noise 有無を**同一評価シード**で比較し、傾向を説明した
- [ ] Q7: 訪問状態の広がりを数字で示した
- [ ] Q8: 1 バッチ過学習に成功（`< 0.2 * first`）
