# 演習 M5: flow matching 化

対応する本文: [`../../lessons/m5_flow_matching.md`](../../lessons/m5_flow_matching.md)

型は「**shape 確認 → 穴埋め → バグ修正 → 小実装 → 実験**」。1 問 1 概念。
解答は [`../../solutions/m5/README.md`](../../solutions/m5/README.md)。まず自分で手を動かしてから見てください。

この章のポイント: **M4 との差分は「行動ヘッドと損失」だけ**。エンコーダ（`VLABackbone`）は共有です。
flow の loss は `τ` と `a0` を乱数で引くので **値が揺れます**——「移動平均で下がるか」で見ます。

準備（共通）:

```python
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from vla_learn.envs import all_instruction_strings
from vla_learn.datasets import (
    generate_episodes, build_normalizers, SyntheticVLADataset, CharTokenizer,
)
from vla_learn.models.flow_head import FlowVLA, SinusoidalTimeEmbedding
from vla_learn.functional import masked_mse
from vla_learn.utils import set_seed
```

小さなバッチを 1 つ作るヘルパ（多くの問題で使います）:

```python
def make_batch(n_episodes=8, batch_size=4, chunk_len=8, seed=0):
    set_seed(seed)
    eps = generate_episodes(n_episodes=n_episodes, seed=seed)
    tok = CharTokenizer.from_corpus(all_instruction_strings())
    an, sn = build_normalizers(eps)
    ds = SyntheticVLADataset(eps, tok, chunk_len=chunk_len,
                             action_normalizer=an, state_normalizer=sn)
    batch = next(iter(DataLoader(ds, batch_size=batch_size, shuffle=True)))
    return batch, tok
```

---

## Q1（shape 確認）`SinusoidalTimeEmbedding` の入出力

`time_dim=64` で `SinusoidalTimeEmbedding` を作り、`tau`（バッチ 5 個分）を入れたときの
出力 shape を、**コードを動かす前に**言い当ててください。

- 入力 `tau` : `[?]`
- 出力 : `[?, ?]`

```python
emb = SinusoidalTimeEmbedding(dim=64)
tau = torch.rand(5)            # [5]  ~ U(0,1)
out = emb(tau)
print(out.shape)
```

- なぜ `dim` は偶数でなければならないか（`forward` の最後の `torch.cat` を見て）1 行で。
- `time_dim` を 64 から 32 に変えたら出力 shape はどうなるか。

> ヒント: 本文 3.1。`sin` 半分・`cos` 半分を連結するので `dim = half + half`。

---

## Q2（shape 確認）`velocity` の連結入力 `x` と出力

`FlowVLA`（既定: `chunk_len=8, action_dim=3, hidden=256, time_dim=64`）の `velocity(a, tau, h)` で、
内部の連結ベクトル `x = torch.cat([a.flatten(1), h, time_embed(tau)], dim=-1)` の shape と、
`velocity` の **出力** shape を手で求めてください。

- `a` : `[B, C, A] = [B, 8, 3]`
- `a.flatten(1)` : `[?, ?]`
- `h` : `[?, ?]`
- `time_embed(tau)` : `[?, ?]`
- 連結後 `x` : `[?, ?]`  ← この第 2 次元 `in_dim` を数値で
- `velocity` 出力 : `[?, ?, ?]`

```python
batch, tok = make_batch(batch_size=4)
model = FlowVLA(vocab_size=tok.vocab_size, chunk_len=8)
h = model.encode(batch["image"], batch["state"], batch["tokens"])  # [B, hidden]
tau = torch.rand(h.shape[0])
v = model.velocity(batch["action"], tau, h)
print("h:", tuple(h.shape), " v:", tuple(v.shape))
print("vnet 期待 in_dim =", model.vnet[0].in_features)
```

- `in_dim` を `C, A, hidden, time_dim` の式で書き、数値（既定値で）を求めてください。
- 出力が `[B, 24]` ではなく `[B, 8, 3]` で返るのは、`velocity` の最後で何をしているからか。

> ヒント: 本文 3.2 の表。`in_dim = C*A + hidden + time_dim = 24 + 256 + 64`。`.view(B, 8, 3)` で整形。

---

## Q3（shape 確認）`flow_loss` の途中テンソル `a_tau` / `v_target`

`flow_loss` の内部で作られる次のテンソルの shape を答えてください（`B=4, C=8, A=3`）。

- `a0 = torch.randn_like(a1)` : `[?, ?, ?]`
- `tau = torch.rand(B)` : `[?]`
- `a_tau = (1-tau)[:,None,None]*a0 + tau[:,None,None]*a1` : `[?, ?, ?]`
- `v_target = a1 - a0` : `[?, ?, ?]`
- `flow_loss` の戻り値 : スカラ（`[]`）か `[?]` か？

確認:

```python
batch, tok = make_batch(batch_size=4)
model = FlowVLA(vocab_size=tok.vocab_size, chunk_len=8)
loss = model.flow_loss(batch["image"], batch["state"], batch["tokens"],
                       batch["action"], batch["pad_mask"])
print(loss.shape, loss.item())     # スカラ（要素 1 個）
```

- `tau[:, None, None]` の shape は何か。これを `a0`（`[B,C,A]`）に掛けるとどう放送されるか 1 行で。
- `flow_loss` が **スカラ**を返すのはなぜか（`masked_mse` が最後に何をするか、本文 2.2 と [M3](../../lessons/m3_data_actions.md)）。

> ヒント: `tau[:, None, None]` は `[B, 1, 1]`。`masked_mse` は平均してスカラにする。

---

## Q4（穴埋め）`flow_loss` を自分で書く

`____` を埋めて `flow_loss` 相当を完成させてください（本文 2.2 そのもの）。

```python
def my_flow_loss(model, image, state, tokens, action, pad_mask=None):
    h = model.encode(image, state, tokens)        # [B, hidden]
    a1 = action                                    # [B, C, A] 正規化済みの目標
    a0 = torch.____(a1)                            # (a) a1 と同じ形のノイズ
    tau = torch.rand(a1.shape[0], device=a1.device)            # [B] ~ U(0,1)
    a_tau = (1 - tau)[:, None, None] * a0 + tau[:, None, None] * ____   # (b) 経路上の点
    v_pred = model.velocity(a_tau, tau, h)         # [B, C, A]
    v_target = ____ - ____                         # (c) まっすぐな道の速度（一定）
    return masked_mse(v_pred, v_target, pad_mask)
```

```python
batch, tok = make_batch(batch_size=4)
set_seed(0); model = FlowVLA(vocab_size=tok.vocab_size, chunk_len=8)
mine = my_flow_loss(model, batch["image"], batch["state"], batch["tokens"],
                    batch["action"], batch["pad_mask"])
print(mine.item())   # 正の有限値が出れば OK（乱数で毎回少し変わる）
```

- (a) ノイズの作り方、(b) 経路 `a_τ` の式、(c) 速度の教師、をそれぞれ答えてください。
- `v_target = a1 - a0` であって `a0 - a1` ではない理由を、`τ=0→1` の向き（[本文 2.1](../../lessons/m5_flow_matching.md)）で 1 行。

> ヒント: `randn_like` / `a1` / `a1 - a0`。道は `a0(τ=0)` から `a1(τ=1)` へ進む。

---

## Q5（バグ修正）`τ` のブロードキャスト忘れ

下の `flow_loss` は、`τ` を `[:, None, None]` で整形し忘れています（よくあるバグ）。
**何が起きるか**を確かめ、**1 行直して**正しく動かしてください。

```python
def buggy_flow_loss(model, image, state, tokens, action, pad_mask=None):
    h = model.encode(image, state, tokens)
    a1 = action                                    # [B, C, A]
    a0 = torch.randn_like(a1)
    tau = torch.rand(a1.shape[0], device=a1.device)            # [B]
    a_tau = (1 - tau) * a0 + tau * a1               # ← バグ: tau は [B]、a0/a1 は [B,C,A]
    v_pred = model.velocity(a_tau, tau, h)
    v_target = a1 - a0
    return masked_mse(v_pred, v_target, pad_mask)
```

```python
batch, tok = make_batch(batch_size=4)
model = FlowVLA(vocab_size=tok.vocab_size, chunk_len=8)
try:
    buggy_flow_loss(model, batch["image"], batch["state"], batch["tokens"],
                    batch["action"], batch["pad_mask"])
except Exception as e:
    print("エラー:", type(e).__name__, e)
```

- まずエラー（または黙って壊れた結果）を観察し、原因を述べてください。
- `tau` を `[B]` → `[B, 1, 1]` にする 1 行を入れて直し、`flow_loss` の本物と**形が一致**することを確認。
- もし `C` や `A` がたまたま `B` と同じ値だと、エラーにならず **黙って混線** することがあります。
  なぜ危険か 1 行で。

> ヒント: 本文 2.3。`tau[:, None, None]` で `[B,1,1]` にしてから掛ける。

---

## Q6（バグ修正）推論モード・no_grad・正規化空間の取り違え

下の自作 `sample` には **3 つ**の問題があります。本文 4 節と照らして全部直してください。

```python
def buggy_sample(model, image, state, tokens, n_steps=10):
    # 問題1: eval にしていない（学習用の挙動のまま推論している）
    h = model.encode(image, state, tokens)
    B = h.shape[0]
    a = torch.randn(B, model.chunk_len, model.action_dim, device=h.device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        tau = torch.full((B,), i * dt, device=h.device)
        # 問題2: 速度を足していない（a を更新していない）
        v = model.velocity(a, tau, h)
    # 問題3: ループ内で勾配が作られている（推論なのに no_grad で囲っていない）
    return a
```

- 問題 1〜3 を直し、`FlowVLA.sample` と **同じ結果**（同じ `seed` で）になることを確認してください。
- `@torch.no_grad()` と `self.eval()` を付ける意味を、それぞれ 1 行で（[M1](../../lessons/m1_pytorch.md) 参照）。
- 直した `sample` の戻り値は **正規化空間** です。これを環境に渡す前に何をするか（本文 4 節）。

> ヒント: 本文 4 節の `sample` がそのまま答え。`a = a + v * dt`、関数全体を `@torch.no_grad()` に。

---

## Q7（小実装＋実験）Euler `sample` を書いて `flow_steps` を比べる

(1) `model` の `velocity` だけを使い、`sample` を **自分で 15〜30 行**で書いてください
（`FlowVLA.sample` を見ずに、本文 4 節の仕様から）。
(2) 書けたら、学習済みでない初期化モデルでも構わないので、`flow_steps=1, 5, 10, 50` で
**生成された行動チャンクがどう変わるか**（スケール・ばらつき）と **1 回の `sample` にかかる時間** を観察します。

```python
def my_sample(model, image, state, tokens, n_steps=10):
    model.eval()
    with torch.no_grad():
        h = model.encode(image, state, tokens)
        B = h.shape[0]
        a = torch.randn(B, model.chunk_len, model.action_dim, device=h.device)
        # ここを実装（dt 刻みで n_steps 回、a = a + v*dt）
        ...
    return a

batch, tok = make_batch(batch_size=4)
set_seed(0); model = FlowVLA(vocab_size=tok.vocab_size, chunk_len=8)
import time
for k in (1, 5, 10, 50):
    set_seed(0)   # 同じ初期ノイズで比べる
    t0 = time.time()
    a = my_sample(model, batch["image"], batch["state"], batch["tokens"], n_steps=k)
    dt = time.time() - t0
    print(f"flow_steps={k:2d}  out={tuple(a.shape)}  mean={a.mean():+.3f} std={a.std():.3f}  {dt*1000:.1f} ms")
```

- `flow_steps` を増やすと **時間がどう増えるか**（線形か）。理由を「速度ネットを何回呼ぶか」で説明。
- （任意・学習後）`scripts/train_flow.py` で学習したチェックポイントを使い、
  `python scripts/eval_policy.py --ckpt checkpoints/flow/policy.pt` を `flow_steps` 違いで比べると、
  **成功率や final_distance がどう動くか**を観察してください（**重い学習は必須ではありません**）。
  本タスクの直線パスでは、ステップを増やしても改善が早めに頭打ちになりがち——なぜか 1 行で。

> ヒント: 本文 4.1。直線パスは曲がっていないので、少ないステップでもそれなりに当たる。

---

## Q8（必須・学習デバッグの鉄則）1 バッチ過学習（揺れるので平均で見る）

`FlowVLA` で **1 バッチだけ**を繰り返し学習し、`flow_loss` が下がることを確認してください。
**注意: flow の loss は `τ`/`a0` の乱数で毎ステップ揺れます。** 1 ステップごとの増減ではなく、
**最初 50 ステップの平均 → 最後 50 ステップの平均** で判定します。

```python
set_seed(0)
batch, tok = make_batch(n_episodes=8, batch_size=16, seed=0)
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

- なぜ **平均で見る**必要があるのか（MSE の loss と何が違うか、本文 5.2 / 7）。
- `late < 0.5 * early` が**通らない**とき、まず疑う配線を 3 つ挙げてください
  （`zero_grad` / `[:, None, None]` / `.train()` などのどれか）。
- この「1 バッチ過学習」が通ることは何を保証し、何は**保証しない**か（汎化との違い、[M3](../../lessons/m3_data_actions.md) Q8 と同じ論点）。

> ヒント: 本文 7 節。`encode → velocity → flow_loss → backward → step` の配線チェック。揺れても平均が下がれば正常。

---

### 提出のしかた（自習用チェックリスト）

- [ ] Q1: `SinusoidalTimeEmbedding` の入出力 `[B] → [B, time_dim]`、`dim` が偶数の理由
- [ ] Q2: `velocity` の `in_dim=344`（`24+256+64`）、出力 `[B,8,3]` を言い当てた
- [ ] Q3: `a_tau`/`v_target` が `[B,C,A]`、`flow_loss` がスカラである理由
- [ ] Q4: `flow_loss` を穴埋め（`randn_like` / `a1` / `a1-a0`）、向きの理由を説明
- [ ] Q5: `[:, None, None]` 忘れバグを 1 行で修正、黙って混線する危険を説明
- [ ] Q6: `eval` / `a=a+v*dt` / `no_grad` の 3 バグ修正、正規化空間→逆正規化を理解
- [ ] Q7: Euler `sample` を自作、`flow_steps` で時間・生成の質の変化を観察
- [ ] Q8: 1 バッチ過学習を**平均で**確認（`late < 0.5*early`）、汎化との違いを説明
