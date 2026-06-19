# 解答 M5: flow matching 化

問題: [`../../exercises/m5/README.md`](../../exercises/m5/README.md) ／ 本文: [`../../lessons/m5_flow_matching.md`](../../lessons/m5_flow_matching.md)

各解答に「**正解コード**」と「**なぜその shape か / なぜ loss が下がるか**」の短い説明を付けます。
（下の shape・数値は実際に動かして確認したものです。flow の loss 値は乱数でぶれます。）

共通の準備とヘルパは [`../../exercises/m5/README.md`](../../exercises/m5/README.md) の冒頭と同じ（`make_batch` を使います）。

---

## A1（shape 確認）`SinusoidalTimeEmbedding`

```python
import torch
from vla_learn.models.flow_head import SinusoidalTimeEmbedding

emb = SinusoidalTimeEmbedding(dim=64)
tau = torch.rand(5)            # [5]
out = emb(tau)
print(tuple(out.shape))        # (5, 64)
```

| 入力 `tau` | 出力 |
|---|---|
| `[5]` | `[5, 64]` |

**なぜこの shape か**: `forward` は各 `τ` を `half = dim//2` 個の周波数で展開し、`sin` 半分・`cos` 半分を
`torch.cat([sin, cos], dim=-1)` で連結します。だから出力の最終次元は `half + half = dim`。
入力のバッチ次元 `[B]` はそのまま `[B, dim]` の先頭に残ります。

- `dim` が偶数の理由: `sin` と `cos` を **同数** 並べて `dim` にするため。奇数だと半分に割れず `assert dim % 2 == 0` で落ちます。
- `time_dim=32` にすると出力は `[5, 32]`。

---

## A2（shape 確認）`velocity` の連結入力と出力

```python
import torch
from vla_learn.models.flow_head import FlowVLA
# make_batch は問題文冒頭のヘルパ
batch, tok = make_batch(batch_size=4)
model = FlowVLA(vocab_size=tok.vocab_size, chunk_len=8)
h = model.encode(batch["image"], batch["state"], batch["tokens"])
tau = torch.rand(h.shape[0])
v = model.velocity(batch["action"], tau, h)
print("h:", tuple(h.shape), " v:", tuple(v.shape))
print("vnet in_dim:", model.vnet[0].in_features)
```

出力:

```text
h: (4, 256)  v: (4, 8, 3)
vnet in_dim: 344
```

| 部品 | shape | 要素数 |
|---|---|---|
| `a.flatten(1)` | `[4, 24]` | `C*A = 8*3 = 24` |
| `h` | `[4, 256]` | `hidden = 256` |
| `time_embed(tau)` | `[4, 64]` | `time_dim = 64` |
| 連結 `x` | `[4, 344]` | **`in_dim = 24 + 256 + 64 = 344`** |
| `velocity` 出力 | `[4, 8, 3]` | — |

**なぜこの shape か**: 速度ネット `vnet` は `Linear(in_dim, ...)` から始まる MLP。入力は
「いまの行動 `a` を平らにしたもの（24）＋条件 `h`（256）＋時刻埋め込み（64）」の連結なので `in_dim=344`。
最後の `Linear` 出力は `[B, 24]` ですが、`.view(B, chunk_len, action_dim)` で `[B, 8, 3]` に **整形**して返すので、
**行動チャンクと同じ形**になります。式: `in_dim = C*A + hidden + time_dim`。

---

## A3（shape 確認）`flow_loss` の途中テンソル

```python
batch, tok = make_batch(batch_size=4)
model = FlowVLA(vocab_size=tok.vocab_size, chunk_len=8)
loss = model.flow_loss(batch["image"], batch["state"], batch["tokens"],
                       batch["action"], batch["pad_mask"])
print(loss.shape, loss.item())     # () すなわちスカラ
```

| テンソル | shape |
|---|---|
| `a0 = torch.randn_like(a1)` | `[4, 8, 3]` |
| `tau = torch.rand(B)` | `[4]` |
| `tau[:, None, None]` | `[4, 1, 1]` |
| `a_tau` | `[4, 8, 3]` |
| `v_target = a1 - a0` | `[4, 8, 3]` |
| `flow_loss` 戻り値 | `()`（スカラ） |

- `tau[:, None, None]` は `[B, 1, 1]`。`a0`（`[B, C, A]`）に掛けると、長さ 1 の `C`・`A` 次元が
  それぞれ `8`・`3` へ **ブロードキャスト** され、**サンプルごとに 1 つの `τ`** がチャンク全体へ効きます。
- `flow_loss` がスカラを返す理由: 最後の `masked_mse` が要素ごとの二乗誤差を **有効ステップで平均**して
  1 個の数にするから（[M3](../../lessons/m3_data_actions.md) の `masked_mse`）。backward できるのは
  スカラ損失だけなので、これで正しい。

**なぜ loss がこの形か**: 学習は「速度を当てる」回帰なので、`v_pred` と `v_target`（ともに `[B,C,A]`）の
L2 をマスク平均してスカラにする。`pad_mask` で終端パディングを除外する作法は MSE 版と同じ。

---

## A4（穴埋め）`flow_loss`

```python
import torch
from vla_learn.functional import masked_mse

def my_flow_loss(model, image, state, tokens, action, pad_mask=None):
    h = model.encode(image, state, tokens)        # [B, hidden]
    a1 = action                                    # [B, C, A]
    a0 = torch.randn_like(a1)                       # (a) a1 と同じ形のノイズ
    tau = torch.rand(a1.shape[0], device=a1.device)
    a_tau = (1 - tau)[:, None, None] * a0 + tau[:, None, None] * a1   # (b) 経路上の点
    v_pred = model.velocity(a_tau, tau, h)
    v_target = a1 - a0                              # (c) まっすぐな道の速度（一定）
    return masked_mse(v_pred, v_target, pad_mask)
```

- (a) `torch.randn_like(a1)`（標準正規・同形状） (b) `... * a1` (c) `a1 - a0`。
- `v_target = a1 - a0` の理由: 道は `a0`（`τ=0`）から `a1`（`τ=1`）へ進む。経路 `a_τ=(1-τ)a0+τa1` を
  `τ` で微分すると `d a_τ/dτ = a1 - a0`。これが「進む向きと速さ」。`a0 - a1` だと **逆走**して
  推論時にノイズへ向かってしまう。

**なぜ loss が下がるか**: ネットが「経路上のどこにいても、`a1` へ向かう一定速度 `a1-a0` を出す」ように
近づくほど二乗誤差が縮む。多数の `(a0, τ)` 標本で平均的に当てると、推論で `a0` から `a1` 付近へ運ぶ
速度場が手に入る。

---

## A5（バグ修正）`τ` のブロードキャスト忘れ

実行すると、`(1 - tau) * a0` で `tau:[B]` と `a0:[B,C,A]` の形が合わず、典型的には
`RuntimeError`（broadcast 不可）になります（`C`/`A` がたまたま `B` と一致すると黙って壊れる）。

修正（1 行）:

```python
def fixed_flow_loss(model, image, state, tokens, action, pad_mask=None):
    h = model.encode(image, state, tokens)
    a1 = action
    a0 = torch.randn_like(a1)
    tau = torch.rand(a1.shape[0], device=a1.device)
    a_tau = (1 - tau)[:, None, None] * a0 + tau[:, None, None] * a1   # ← [:, None, None] を追加
    v_pred = model.velocity(a_tau, tau, h)
    v_target = a1 - a0
    return masked_mse(v_pred, v_target, pad_mask)
```

確認:

```python
batch, tok = make_batch(batch_size=4)
model = FlowVLA(vocab_size=tok.vocab_size, chunk_len=8)
loss = fixed_flow_loss(model, batch["image"], batch["state"], batch["tokens"],
                       batch["action"], batch["pad_mask"])
print(loss.shape, loss.item())     # () のスカラ。本物の flow_loss と同じ形
```

- 原因: `tau` は `[B]`（1 次元）、`a0`/`a1` は `[B, C, A]`（3 次元）。次元数が違うので末尾合わせの
  ブロードキャストが成立しない。`tau[:, None, None]` で `[B, 1, 1]` にすると `C`/`A` 方向へ放送できる。
- 黙って混線する危険: もし `C==B` や `A==B` だと、PyTorch が **意図しない軸**で形を合わせてしまい、
  エラーなしで「バッチとチャンクが混ざった」無意味な `a_τ` を作る。エラーが出ないぶん発見が遅れる
  ——だから「次元を 1 にして明示的に放送」する癖が大事。

**なぜ shape を合わせると下がるか**: `a_τ` が正しく「サンプルごと・チャンク全体に同じ `τ`」で作られて
初めて、`v_target=a1-a0` との対応が意味を持つ。混線していると教師信号が壊れ、loss は下がらない。

---

## A6（バグ修正）`eval` / `a=a+v*dt` / `no_grad`

```python
import torch

@torch.no_grad()                          # 修正3: 推論全体で勾配を作らない
def fixed_sample(model, image, state, tokens, n_steps=10):
    model.eval()                          # 修正1: 評価モードにする
    h = model.encode(image, state, tokens)
    B = h.shape[0]
    a = torch.randn(B, model.chunk_len, model.action_dim, device=h.device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        tau = torch.full((B,), i * dt, device=h.device)
        a = a + model.velocity(a, tau, h) * dt     # 修正2: 速度を足して a を前進
    return a                               # [B, C, A]（正規化空間）
```

本物と一致するか（同じ seed で初期ノイズを揃える）:

```python
batch, tok = make_batch(batch_size=4)
model = FlowVLA(vocab_size=tok.vocab_size, chunk_len=8)
from vla_learn.utils import set_seed
set_seed(0); a1 = fixed_sample(model, batch["image"], batch["state"], batch["tokens"], n_steps=10)
set_seed(0); a2 = model.sample(batch["image"], batch["state"], batch["tokens"], n_steps=10)
print("一致:", torch.allclose(a1, a2), " shape:", tuple(a1.shape))   # True (4, 8, 3)
```

- `@torch.no_grad()`: 推論では勾配計算グラフを作らない。メモリと速度を節約し、誤って学習しない。
- `self.eval()`（`model.eval()`）: Dropout / BatchNorm 等を **評価時の挙動**に切り替える
  （本モデルには無いが、習慣として必須。[M1](../../lessons/m1_pytorch.md)）。
- 戻り値は **正規化空間**。環境に渡す前に `action_norm.denormalize(...)` で生の `[dx,dy,grip]` に戻す
  （`PolicyWrapper` が担当。本文 4 節）。これを忘れると正規化空間の小さな値が環境に渡り、動かない。

**なぜ正しく生成できるか**: `a = a + v·dt` は確率フロー ODE のオイラー前進。`τ=0` のノイズから
`dt` 刻みで速度場に沿って進めると、`τ=1` で目標分布のサンプルに到達する。3 修正のどれが欠けても
（更新しない／勾配を残す／モードを誤る）正しいサンプルにならない。

---

## A7（小実装＋実験）Euler `sample` と `flow_steps`

```python
import time, torch
from vla_learn.models.flow_head import FlowVLA
from vla_learn.utils import set_seed

def my_sample(model, image, state, tokens, n_steps=10):
    model.eval()
    with torch.no_grad():
        h = model.encode(image, state, tokens)
        B = h.shape[0]
        a = torch.randn(B, model.chunk_len, model.action_dim, device=h.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            tau = torch.full((B,), i * dt, device=h.device)
            a = a + model.velocity(a, tau, h) * dt
    return a

batch, tok = make_batch(batch_size=4)
set_seed(0); model = FlowVLA(vocab_size=tok.vocab_size, chunk_len=8)
for k in (1, 5, 10, 50):
    set_seed(0)                        # 同じ初期ノイズで比較
    t0 = time.time()
    a = my_sample(model, batch["image"], batch["state"], batch["tokens"], n_steps=k)
    ms = (time.time() - t0) * 1000
    print(f"flow_steps={k:2d}  out={tuple(a.shape)}  mean={a.mean():+.3f} std={a.std():.3f}  {ms:.1f} ms")
```

出力イメージ（数値はぶれます。**未学習モデルなので生成の絶対値に意味はなく、傾向を見ます**）:

```text
flow_steps= 1  out=(4, 8, 3)  mean=... std=...   ~?  ms
flow_steps= 5  out=(4, 8, 3)  mean=... std=...   ~5x ms
flow_steps=10  out=(4, 8, 3)  mean=... std=...  ~10x ms
flow_steps=50  out=(4, 8, 3)  mean=... std=...  ~50x ms
```

- **時間はほぼ線形**に増える: `sample` は速度ネットを `n_steps` 回呼ぶので、`flow_steps` に比例して重くなる。
- 学習後の評価（任意）: `python scripts/eval_policy.py --ckpt checkpoints/flow/policy.pt` を
  `flow_steps` 違いで比べると、**少なすぎる（1 など）と粗くて成功率が落ち**、ある程度（5〜10）で実用域、
  そこから増やしても **改善は頭打ち**になりがち。理由: 本タスクの rectified flow は **直線パス** で
  速度が一定に近く、曲がっていないので少ない刻みでも誤差が小さい。曲がった軌道ほど多ステップが効く。

**なぜ出力が `[B,8,3]` か**: `sample` は行動チャンク（`chunk_len=8, action_dim=3`）を生成する。
ループで更新する `a` は最初から `[B, 8, 3]`、速度も `[B, 8, 3]` なので、足し合わせても形は不変。

---

## A8（必須・1 バッチ過学習：平均で見る）

```python
import numpy as np, torch
from torch.utils.data import DataLoader
from vla_learn.envs import all_instruction_strings
from vla_learn.datasets import generate_episodes, build_normalizers, SyntheticVLADataset, CharTokenizer
from vla_learn.models.flow_head import FlowVLA
from vla_learn.utils import set_seed

set_seed(0)
eps = generate_episodes(n_episodes=8, seed=0)
tok = CharTokenizer.from_corpus(all_instruction_strings())
an, sn = build_normalizers(eps)
ds = SyntheticVLADataset(eps, tok, chunk_len=8, action_normalizer=an, state_normalizer=sn)
batch = next(iter(DataLoader(ds, batch_size=16, shuffle=True)))

model = FlowVLA(vocab_size=tok.vocab_size, chunk_len=8)
model.train()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
for step in range(800):
    loss = model.flow_loss(batch["image"], batch["state"], batch["tokens"],
                           batch["action"], batch["pad_mask"])
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

early = float(np.mean(losses[:50]))
late  = float(np.mean(losses[-50:]))
print(f"early={early:.4f} -> late={late:.4f}")
assert late < 0.5 * early, "1 バッチを暗記できていない（学習機構を疑う）"
print("OK")
```

出力イメージ: `early=1.53 -> late=0.70` 程度（**値はぶれます**。300 ステップで既に半減、800 ステップで
さらに下がります。生の 1 ステップ loss は上下に揺れます）。

- **平均で見る理由**: `flow_loss` は毎ステップ `τ~U(0,1)` と `a0~N(0,I)` を引き直すので、同じバッチ・同じ重みでも
  loss が確率的に揺れる（MSE 版は教師が固定なので素直に下がる）。だから 1 ステップの増減ではなく
  **移動平均（ここでは前後 50 ステップ平均）の低下**で「暗記できているか」を判定する（本文 5.2 / 7）。
- `late < 0.5 * early` が**通らない**ときに疑う配線（例）:
  1. `opt.zero_grad()` 忘れ（勾配が累積して発散・停滞）。
  2. `a_tau` の `tau[:, None, None]` 忘れ（教師信号が壊れている → A5）。
  3. `model.train()` 忘れ／`no_grad` の誤用で勾配が流れていない。
- 何を**保証し / しないか**: 1 バッチ過学習が通る＝「`encode → velocity → flow_loss → backward → step` の
  配線と shape は健全」。**保証しないのは汎化**——16 サンプルを暗記できても、未知局面で正しい行動を出せるかは別問題
  （[M3](../../lessons/m3_data_actions.md) Q8 と同じ論点）。汎化には十分なデータ量・エポックと、
  画像・言語による grounding（[M4](../../lessons/m4_tiny_vla_mse.md)）が要る。

**なぜ loss が下がるべきか**: 固定した 1 バッチに対し、ネットは「この `h` ではこの `a1` へ向かう速度」を
（`τ`/`a0` を平均して）覚えるだけの容量がある。最適化・損失・shape が正しければ Adam が平均誤差を縮める。
下がらなければモデルやデータではなく **学習ループ側のバグ**を疑うのが鉄則。
