# 解答 M2: 最小の模倣学習

問題: [`../../exercises/m2/README.md`](../../exercises/m2/README.md) ／ 本文: [`../../lessons/m2_imitation.md`](../../lessons/m2_imitation.md)

各解答に「**正解コード**」と「**なぜその shape か / なぜ loss が下がるか**」の短い説明を付けます。

---

## A1（shape 確認）

```python
import numpy as np
from vla_learn.datasets import generate_episodes
from vla_learn.envs import expert_action

eps = generate_episodes(n_episodes=10, seed=0)
ep = eps[0]
print("actions:", ep["actions"].shape)   # (T, 3)
print("agent  :", ep["agent"].shape)      # (T, 3)
w = None  # world は generate 内で消費済みなので、確認は env から取り直すのが正攻法
```

- `ep["actions"]` は `[T, 3]`。軸0=時刻 `t`（エピソード長 `T`、10〜20 程度）、軸1=行動次元 `[dx, dy, grip_cmd]`。
- `ep["agent"]` は `[T, 3]`。各行 `ep["agent"][t]` は状態 `[ax, ay, gripper]`。
- `expert_action(world)` は `[3]`（`np.ndarray`）。`[dx, dy, grip_cmd]`。

**なぜこの shape か**: エピソードは時系列なので**先頭軸が時刻**。行動・状態はどちらも 3 次元なので
`[T, 3]`。`expert_action` は「いまの 1 手」を返すだけなので時刻軸を持たず `[3]`。

> `world` を直接見たい場合は `Tabletop2DEnv` を `reset()` して `obs["world"]` を使います
> （`generate_episodes` は内部で world を消費するため、`ep` には残しません）。

---

## A2（shape 確認）

- `xb`: `[B, 3]`、`yb`: `[B, 3]`（`B=256`）。
- 最後のバッチは `N % 256` 個になることがある（`drop_last=False` が既定のため端数が残る）。

**なぜこの shape か**: `TensorDataset(X, Y)` は `X[i], Y[i]`（各 `[3]`）を返し、DataLoader が
`batch_size` 個積んで**先頭に `B`** を作る → `[B, 3]`。`N` が `256` で割り切れないと最後だけ小さい。

---

## A3（穴埋め）

```python
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from vla_learn.datasets import generate_episodes
from vla_learn.utils import set_seed

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
            nn.Linear(64, 3),            # (a) 出力次元 = 行動次元 = 3
        )
    def forward(self, s):
        return self.net(s)

loader = DataLoader(TensorDataset(X, Y), batch_size=256, shuffle=True)
model = StateToAction()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(10):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()      # (b) 勾配をリセット（前ステップの勾配が累積しないように）
        loss.backward()      # (c) 逆伝播（各パラメータの勾配を計算）
        opt.step()           # (d) パラメータ更新（勾配の向きに一歩進む）
print("最終 loss:", loss.item())
```

- (a) `3`（行動 `[dx,dy,grip_cmd]`）。(b) `opt.zero_grad()` (c) `loss.backward()` (d) `opt.step()`。

**なぜ loss が下がるか**: 各ステップで `MSE = mean((pred - y)^2)` の勾配を計算し、Adam が
それを小さくする向きに重みを更新します。`state → action` には十分な規則性（近いほど小さく動く等）が
あるので、数エポックで MSE が下がります。3 点セットの順序が肝で、`zero_grad` を忘れると勾配が累積して
学習が壊れます。

---

## A4（バグ修正）

直すべき 3 点:

1. `model.eval()` を呼んでいない（Dropout/BatchNorm があると挙動が変わる。**推論前は必ず eval**）。
2. `state` に**バッチ次元が無い**（`[3]`）。`nn.Linear` は `[B, in]` を期待するので `unsqueeze(0)` で `[1,3]`、
   出力は `squeeze(0)` で `[3]` に戻して環境へ。
3. `info` が未初期化のまま参照されうる（`n_episodes` 内で必ず step するなら定義されるが、
   防御的に関数冒頭で `info = {}` を入れておく）。

```python
import numpy as np, torch
from vla_learn.envs import Tabletop2DEnv

@torch.no_grad()
def mini_closed_loop_fixed(model, n_episodes=10, seed=1000):
    model.eval()                                  # (1) 修正
    succ, info = 0, {}                            # (3) info を初期化
    for k in range(n_episodes):
        env = Tabletop2DEnv(seed=seed + k)
        obs = env.reset()
        done = False
        while not done:
            state = torch.from_numpy(obs["state"]).float().unsqueeze(0)  # (2) [1,3]
            a = model(state).squeeze(0).numpy()                          # [3]
            obs, _, done, info = env.step(a)
        succ += int(info.get("success", False))
    return succ / n_episodes
```

**なぜこの shape か**: バッチ学習したモデルは入力 `[B, in]` を仮定しているので、
1 サンプル推論でも `[1, 3]` にして渡し、出力 `[1, 3]` を `squeeze(0)` で `[3]` に戻して `env.step` に渡します。

---

## A5（小実装）

```python
import torch, torch.nn as nn

class ImageToAction(nn.Module):
    def __init__(self, out_dim=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),  # 64 -> 32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), # 32 -> 16
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(), # 16 -> 8
        )
        self.head = nn.Sequential(
            nn.Flatten(),                  # [B, 32*8*8] = [B, 2048]
            nn.Linear(32 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, out_dim),       # [B, 3]
        )
    def forward(self, image):
        return self.head(self.conv(image))

print(ImageToAction()(torch.zeros(2, 3, 64, 64)).shape)  # torch.Size([2, 3])
```

- 出力層に活性化を付けない理由: 行動 `dx, dy` は**負にもなる**連続値。`ReLU`/`Sigmoid` で範囲を縛ると
  お手本を再現できない。回帰の出力は素のまま（恒等）が基本。

**なぜこの shape か**: `stride=2` の畳み込みは解像度を半分にする。`64→32→16→8` と 3 段で `8x8`、
チャンネル `32` なので Flatten 後は `32*8*8=2048`。それを `Linear` で `3` に落とす。

---

## A6（実験・本章の主役）

```python
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from vla_learn.datasets import generate_episodes
from vla_learn.envs import Tabletop2DEnv
from vla_learn.envs.render import render_world
from vla_learn.utils import set_seed

# （ImageToAction と ImgActionDataset は本文 5.1 のものを使う）
from_lesson = """本文 5.1 の ImageToAction, ImgActionDataset, train_image_bc, closed_loop_success を定義済みとする"""

m0 = train_image_bc(action_noise=0.00, n_episodes=300, epochs=8)
m1 = train_image_bc(action_noise=0.03, n_episodes=300, epochs=8)
for noise, m in [(0.0, m0), (0.03, m1)]:
    print(f"noise={noise:.2f}  success={closed_loop_success(m):.2%}")
```

結果の典型（**強くぶれる**。重要なのは傾向）:

```text
noise=0.00  success: 56.00%
noise=0.03  success: 78.00%
noise=0.06  success: 70.00%   # 上げすぎると頭打ち〜やや低下
noise=0.10  success: 52.00%   # 軌道が荒れてラベル対応が弱まり悪化しがち
```

**なぜノイズありが強いか**: 素朴 BC はエキスパートの 1 本道しか見ないため、閉ループで誤差が積もって
軌道を外れると（**分布シフト / 誤差蓄積**）、未知状態で出鱈目を出して**崩れる**。
`action_noise>0` は**実行だけ乱してラベルはクリーン**にするので、「**軌道から外れた状態 → 戻す正しい行動**」の
ペア（**リカバリのお手本**）が集まり、外れても戻せるようになる。ただしノイズが大きすぎると軌道が荒れ、
ラベルとの対応が弱まって**かえって悪化**する（だから既定は `0.03` 付近）。

---

## A7（実験）

```python
import numpy as np
from vla_learn.datasets import generate_episodes

for noise in (0.0, 0.03):
    eps = generate_episodes(n_episodes=200, seed=0, action_noise=noise)
    agent = np.concatenate([ep["agent"] for ep in eps], axis=0)      # [sumT, 3]
    acts  = np.concatenate([ep["actions"] for ep in eps], axis=0)    # [sumT, 3]
    step_mag = np.abs(acts[:, :2]).sum(1)                            # |dx|+|dy|
    print(f"noise={noise:.2f}  ax std={agent[:,0].std():.3f}  ay std={agent[:,1].std():.3f}"
          f"  step|dx|+|dy| mean={step_mag.mean():.3f} max={step_mag.max():.3f}")
```

出力例（傾向）:

```text
noise=0.00  ax std=0.27  ay std=0.27  step|dx|+|dy| mean=0.071 max=0.160
noise=0.03  ax std=0.28  ay std=0.28  step|dx|+|dy| mean=0.078 max=0.205
```

説明: ノイズありの方が**訪問状態のばらつき（std）と 1 ステップ移動量の最大値**がやや大きい
＝ エキスパート軌道の「まわり」も訪れている。これは「お手本が失敗の直し方を見せない」素朴 BC の穴を埋め、
Q6 の**崩れにくさ**につながる。

> 注: 1 本道のエキスパート自体ばらつきがあるため差は小さく出ます。差が小さくても、
> 「外れた点での**正解ラベル**」が増えること自体が閉ループ安定に効きます。

---

## A8（必須・1 バッチ過学習）

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
from vla_learn.utils import set_seed

set_seed(0)
one = next(iter(DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)))
xb, yb = one

model = StateToAction()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()
first = None
for _ in range(300):
    pred = model(xb)
    loss = loss_fn(pred, yb)
    opt.zero_grad(); loss.backward(); opt.step()
    if first is None:
        first = loss.item()
print(f"{first:.4f} -> {loss.item():.6f}")
assert loss.item() < 0.2 * first
print("OK")
```

出力例: `0.2232 -> 0.038375`（最初の 20% を下回る。値はぶれる）。

**なぜ下がるべきか**: 32 サンプルなら、十分な容量の MLP は入力→出力を**ほぼ丸暗記**できる。
下がらなければ学習機構にバグ。疑う 3 点:

1. `opt.zero_grad()` 忘れ（勾配が累積して更新が暴れる）。
2. `pred` を `detach()` してしまい勾配が流れない／`requires_grad` 周り。
3. 入出力の **shape 不一致**（`[B,3]` で揃っているか）。加えて `model.train()` 状態か。
