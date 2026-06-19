# 演習 M4: 最小 VLA を自作する（MSE 回帰版 TinyVLA）

対応する本文: [`../../lessons/m4_tiny_vla_mse.md`](../../lessons/m4_tiny_vla_mse.md)

型は「**shape 確認 → 穴埋め → バグ修正 → 小実装 → 実験**」。1 問 1 概念。
解答は [`../../solutions/m4/README.md`](../../solutions/m4/README.md)。まず自分で手を動かしてから見てください。

準備（共通）:

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vla_learn.envs import all_instruction_strings
from vla_learn.datasets import (
    generate_episodes, build_normalizers, SyntheticVLADataset, CharTokenizer,
)
from vla_learn.models import TinyVLA, count_parameters
from vla_learn.models.image_encoder import ImageEncoder, FiLM
from vla_learn.models.text_encoder import TextEncoder
from vla_learn.models.state_encoder import StateEncoder
from vla_learn.training.losses import masked_mse   # = vla_learn.functional.masked_mse
from vla_learn.utils import set_seed, get_device
```

学習・評価をコマンドで回す問題（Q7）は、先に `uv sync`（または `export PYTHONPATH=src`）を済ませ、`uv run python scripts/...` で実行してください。

---

## Q1（shape 確認）各エンコーダと融合の出力 shape

`B=4`, トークン長 `L=17`, `vocab_size=30` とします。次の各テンソルの shape を **コードを動かす前に** 言い当ててください。

- `TextEncoder(vocab_size=30)(tokens)` … `l` : `[?, ?]`
- `ImageEncoder(out_dim=128, cond_dim=128)(image, cond=l)` … `v` : `[?, ?]`
- `StateEncoder(out_dim=64)(state)` … `s` : `[?, ?]`
- `torch.cat([v, l, s], dim=-1)` … 融合前 : `[?, ?]`
- `VLABackbone(...).forward(...)` の戻り `h` : `[?, ?]`
- `TinyVLA(vocab_size=30, chunk_len=8)(image, state, tokens)` … 行動チャンク : `[?, ?, ?]`

```python
B, L, VOCAB = 4, 17, 30
image  = torch.rand(B, 3, 64, 64)
state  = torch.rand(B, 3)
tokens = torch.randint(0, VOCAB, (B, L))
# 上の各行を実際に動かして確認
```

最後に **融合次元（concat の幅）が `320` になる足し算** を、`128 + 128 + 64` の形で説明してください。

> ヒント: 本文 1 節の図と表。画像/言語の `out_dim` は既定 128、状態は 64。`chunk_len=8`, `action_dim=3`。

---

## Q2（shape 確認）行動チャンク `[B, C, 3]` の意味

`TinyVLA(chunk_len=8)` の出力は `[B, 8, 3]` です。この **3 つの軸がそれぞれ何か** を答えてください。

- 軸 0（`B`）は？
- 軸 1（`8`）は？（単位は？ 値を変えるとどの引数か？）
- 軸 2（`3`）は？（中身 `[dx, dy, grip_cmd]` のそれぞれの意味）

さらに、`head = nn.Linear(hidden, chunk_len * action_dim)` の出力 `[B, 24]` を `[B, 8, 3]` にするのに使う
**1 行のコード** を書いてください（本文 3 節の `forward` 参照）。

> ヒント: `out.view(-1, chunk_len, action_dim)`。`24 = 8 * 3`。

---

## Q3（穴埋め）FiLM の forward を完成させる

FiLM は条件ベクトル `cond` から **チャンネルごとの (scale, shift)** を作り、特徴マップ `x:[B,C,H,W]` を変調します。
`____` を埋めて、`vla_learn` の実装と同じ振る舞いにしてください。

```python
class MyFiLM(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, ____)   # (a) 出力次元は？（scale と shift の両方）

    def forward(self, x, cond):                # x:[B,C,H,W], cond:[B,cond_dim]
        gamma, beta = self.to_scale_shift(cond).chunk(2, dim=-1)  # それぞれ [B, C]
        # (b) gamma, beta を [B,C,1,1] にして x をチャンネルごとに変調
        return x * (1 + gamma[:, :, ____, ____]) + beta[:, :, ____, ____]

# 確認: 入出力 shape が変わらないこと
f = MyFiLM(cond_dim=128, num_channels=32)
x = torch.randn(4, 32, 16, 16); cond = torch.randn(4, 128)
print(tuple(f(x, cond).shape))   # (4, 32, 16, 16) になるはず
```

- (a) なぜ `num_channels` の **2 倍** が要るのか、1 行で。
- (b) `gamma[:, :, None, None]` の `None` 2 つは何のため（broadcast）か、1 行で。
- `1 + gamma` と書く（`gamma` 単体でなく）のはなぜ嬉しいか、初期化の観点で 1 行で。

> ヒント: 本文 教訓 3。`chunk(2, dim=-1)` で `[B, 2C]` を `[B,C],[B,C]` に割る。`None` は次元追加。

---

## Q4（バグ修正）学習ループが動かない 3 つのバグ

下の学習ループには **3 つのバグ** があります（device / eval / 正規化の代表的なつまずき）。
本文 4 節と照らして見つけ、`success_rate` 評価まで通るように直してください。

```python
set_seed(0)
device = get_device()
eps = generate_episodes(n_episodes=60, seed=0, action_noise=0.03)
tok = CharTokenizer.from_corpus(all_instruction_strings())
an, sn = build_normalizers(eps)
ds = SyntheticVLADataset(eps, tok, chunk_len=8, action_normalizer=an, state_normalizer=sn)
loader = DataLoader(ds, batch_size=64, shuffle=True)

model = TinyVLA(vocab_size=tok.vocab_size, chunk_len=8).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# バグ1: model は device 上にあるが、batch を device に移していない
for epoch in range(3):
    for batch in loader:
        pred = model(batch["image"], batch["state"], batch["tokens"])
        loss = masked_mse(pred, batch["action"], batch["pad_mask"])
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {epoch}  loss={loss.item():.4f}")

# バグ2: 評価前に eval モードへ切り替えていない（推論の作法）
from vla_learn.evaluation.rollout import PolicyWrapper, evaluate_policy

# バグ3: PolicyWrapper に正規化器を渡していない（逆正規化できず環境が壊れる）
policy = PolicyWrapper(model, tok, model_type="mse", device=device)   # ← 引数が足りない
print(evaluate_policy(policy, n_episodes=5))
```

3 つを直したら、評価が例外なく走り `success_rate` が表示されること（値の大小は問わない）を確認してください。
**それぞれのバグを放置すると何が起きるか** を 1 行ずつ説明してください。

> ヒント: `batch = {k: v.to(device) for k, v in batch.items()}` / `model.eval()`（`PolicyWrapper` は内部で `.eval()` するが、推論の作法として明示） / `PolicyWrapper(model, tok, an, sn, ...)`（`action_norm, state_norm` が必須）。

---

## Q5（バグ修正）逆正規化を忘れて環境が破綻する

下は「学習済みっぽいモデル」を環境で 1 ステップ動かすミニ rollout です。**行動を逆正規化せずに** `env.step` へ渡しているため、
環境が桁違いの行動を受け取ります。本文 6.1 を見て、`env.step` に渡す直前で逆正規化するよう直してください。

```python
from vla_learn.envs import Tabletop2DEnv

set_seed(0)
eps = generate_episodes(n_episodes=40, seed=0)
tok = CharTokenizer.from_corpus(all_instruction_strings())
an, sn = build_normalizers(eps)
model = TinyVLA(vocab_size=tok.vocab_size, chunk_len=8).eval()

env = Tabletop2DEnv(max_steps=48)
obs = env.reset(seed=0)

img    = torch.from_numpy(np.ascontiguousarray(obs["image"]))[None]
state  = torch.from_numpy(sn.normalize(obs["state"].astype(np.float32)))[None]  # 入力 state は正規化
tokens = torch.tensor([tok.encode(obs["instruction"])], dtype=torch.long)

with torch.no_grad():
    chunk = model(img, state, tokens)[0]    # [8, 3]（正規化空間のまま！）

# バグ: 正規化空間の行動をそのまま環境へ渡している
a0 = chunk[0].numpy()
obs, _, done, info = env.step(a0)           # ← ここを直す
print("1 ステップ後の state:", obs["state"])
```

直したら、正規化空間の `chunk[0]` と、逆正規化後の `an.denormalize(chunk)[0]` を **両方 print して比べて** ください。
学習が進んだ方策ほど正規化空間の出力は `±1〜2` に達し、これを生の `dx,dy`（標準偏差 `~0.065`、本文/M3）として
そのまま渡すと **ワールドスケールに対して桁違いに大きく**、環境内でクリップされて挙動が壊れます。逆正規化が「正しい縮尺」へ戻す役割です。

> ヒント: `chunk_raw = an.denormalize(chunk)`、`a0 = chunk_raw[0].numpy()`。`PolicyWrapper.predict_chunk` がやっているのと同じこと。
> （未学習モデルだと head がほぼゼロ初期化のため正規化出力も小さく差が出にくい点に注意。差が際立つのは **学習後** です。）

---

## Q6（小実装）行動ヘッドと最小 forward を自作する

`VLABackbone` は与えられたものとして、そこから先の **行動ヘッド** を自分で書いて `TinyVLA` 相当を完成させてください。

```python
class MyTinyVLA(nn.Module):
    def __init__(self, vocab_size, chunk_len=8, action_dim=3, hidden=256, **backbone_kwargs):
        super().__init__()
        from vla_learn.models.tiny_vla import VLABackbone
        self.backbone = VLABackbone(vocab_size, hidden=hidden, **backbone_kwargs)
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        # (a) hidden -> chunk_len*action_dim の全結合ヘッドを 1 行で
        self.head = ____

    def forward(self, image, state, tokens):
        h = self.backbone(image, state, tokens)   # [B, hidden]
        # (b) head を通し、[B, chunk_len, action_dim] に整形して返す（2〜3 行）
        ...

# 動作確認: shape と過学習能力
set_seed(0)
tok = CharTokenizer.from_corpus(all_instruction_strings())
m = MyTinyVLA(vocab_size=tok.vocab_size, chunk_len=8)
B = 4
out = m(torch.rand(B,3,64,64), torch.rand(B,3), torch.randint(0,tok.vocab_size,(B,17)))
assert out.shape == (B, 8, 3), out.shape
print("OK forward:", tuple(out.shape), " params:", f"{count_parameters(m):,}")
```

- `head` の出力次元はいくつで、なぜ `view` で `[B, 8, 3]` に戻すのか（本文 3 節）。
- `**backbone_kwargs` を残しておくと、後の実験（Q8）で何が嬉しいか 1 行で。

> ヒント: `nn.Linear(hidden, chunk_len * action_dim)`、`self.head(h).view(-1, self.chunk_len, self.action_dim)`。

---

## Q7（実験 = 必須）スクリプトで学習し、閉ループ成功率を測る

設定ファイルとスクリプトで TinyVLA を学習し、**閉ループ評価** まで通してください。

まずスモークで配線確認（1〜2 分）:

```bash
uv run python scripts/train_mse.py --config configs/smoke.json
```

次に本番設定（CPU で数分）。学習後に自動で評価まで走ります:

```bash
uv run python scripts/train_mse.py --config configs/m4_mse.json
uv run python scripts/eval_policy.py --ckpt checkpoints/mse/policy.pt --n-episodes 100
```

確認すること:

1. 学習ログの `パラメータ数` と最終 `loss`（`0.7` 付近から `0.05〜0.06` 付近まで下がるか）。
2. `eval_policy.py` の `success_rate`（**およそ 7〜8 割** が目安。**環境・乱数でぶれます**）。
3. `success_rate` がエキスパート（100%）に届かないのはなぜか、本文 7 節（distribution shift）の言葉で 1〜2 行。

（任意）`uv run python scripts/demo_rollout.py --ckpt checkpoints/mse/policy.pt --out assets/rollout.png` で
ロールアウトを目で見て、指示の色のブロックが運ばれているか確認してください（matplotlib が必要）。

> ヒント: CLI 引数は config より優先（例 `--epochs 30 --n-episodes 1500`）。出力例は本文 5・6 節。

---

## Q8（実験 = 必須）3 つの設計教訓を「壊して」成功率の変化を見る

本文 2 節の 3 教訓のうち **2 つ** を実際に無効化し、既定（flatten + FiLM）と **成功率・grounding** を比べます。
`TinyVLA(...)` は `**backbone_kwargs` を素通しするので、引数を変えるだけで切り替わります。

学習関数を 1 つ用意し、3 条件で学習 → 評価します（CPU で各数分。`m4_mse.json` 相当の規模を推奨）:

```python
import json
from vla_learn.training.config import load_config
from vla_learn.training.trainer import run_training

def train_and_eval(tag, **backbone_kwargs):
    # configs/m4_mse.json をベースに out_dir だけ変える
    cfg = load_config("configs/m4_mse.json", out_dir=f"checkpoints/m4_{tag}")
    # ※ backbone_kwargs（image_pool / condition_vision）を TinyVLA に渡すには
    #   下の「実装メモ」を読み、run_training を使う代わりに自前ループで回すか、
    #   小規模設定（n_episodes を 600〜1000、epochs を 15〜20 程度）で比較してください。
    ...

# 比較する 3 条件:
#  (A) 既定         : image_pool="flatten", condition_vision=True
#  (B) avg          : image_pool="avg"       （位置情報を捨てる）
#  (C) FiLM 無し    : condition_vision=False  （言語で視覚を変調しない）
```

実装メモ: `run_training` は `TinyVLA(vocab_size=..., chunk_len=...)` を既定 backbone で作るため、
`image_pool` / `condition_vision` を変えるには **本文 4.2 の自前学習ループ**（`TinyVLA(..., image_pool=..., condition_vision=...)`）を
使うのが簡単です。学習後に `PolicyWrapper` + `evaluate_policy(n_episodes=50〜100)` で成功率を出してください。
**そのまま動く雛形**は卒業課題②の [`../m6/ablation.py`](../m6/ablation.py)（穴埋め 3 か所）にあります。この Q8 と中身は同じなので、
自前ループを 1 から書くのが大変なら、そちらを埋めて 3 条件を比較しても構いません。

観察と考察:

1. (B) avg と (C) FiLM 無しは、(A) 既定に比べて **`success_rate` が下がる** はずです（どれだけ下がるかは規模・乱数でぶれます）。
   それぞれの **3 つの数字**（A/B/C の `success_rate`）を並べて報告してください。
2. **学習 `loss` は 3 条件で大差ない／むしろ下がる** ことがあるのに、`success_rate` は (A) が高い——これはなぜか。
   本文 2 節（空間情報・grounding）と 7 節（loss ≠ 成功）の言葉で説明してください。
3. (C) FiLM 無しで「**指示の色を変えても運ぶ対象が変わらない**」（grounding 崩壊）を、`demo_rollout.py` か
   自前の rollout で **同じ初期配置・異なる指示** を 2 回流して目視確認してください（任意だが推奨）。

> ヒント: `image_pool="avg"` も `condition_vision=False` も `TinyVLA(vocab_size=..., chunk_len=8, image_pool=..., condition_vision=...)` で渡せます（本文 3 節、テスト `test_avg_pool_variant_runs` 参照）。

---

## Q9（必須・学習デバッグの鉄則）TinyVLA を 1 バッチに過学習させる

学習ループ・shape・損失・最適化が正しいかを、**小さな 1 バッチに過学習できるか** で確かめます（本文 8 節）。

```python
set_seed(0)
eps = generate_episodes(n_episodes=8, seed=0)
tok = CharTokenizer.from_corpus(all_instruction_strings())
an, sn = build_normalizers(eps)
ds = SyntheticVLADataset(eps, tok, 8, an, sn)
batch = next(iter(DataLoader(ds, batch_size=16, shuffle=True)))

model = TinyVLA(vocab_size=tok.vocab_size, chunk_len=8)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
first = None
for i in range(200):
    pred = model(batch["image"], batch["state"], batch["tokens"])
    loss = masked_mse(pred, batch["action"], batch["pad_mask"])
    opt.zero_grad(); loss.backward(); opt.step()
    if first is None:
        first = loss.item()
print(f"first={first:.4f}  last={loss.item():.4f}")
assert loss.item() < 0.2 * first, "1 バッチに過学習できていない（どこかにバグ）"
print("OK")
```

- なぜ「1 バッチに過学習できること」が **学習機構の健全性** の証明になるのか、1〜2 行で。
- これが通っても本番の `success_rate` が 100% にならないのはなぜか（汎化と過学習の違い、本文 7 節）。

> ヒント: リポジトリの [`../../tests/test_overfit_tiny_batch.py`](../../tests/test_overfit_tiny_batch.py) と同じ趣旨。`uv run pytest -k overfit` でも走ります。

---

### 提出のしかた（自習用チェックリスト）

- [ ] Q1〜Q2: 各エンコーダ出力・融合幅 320・行動チャンク `[B,8,3]` の各軸を **動かす前に** 言い当てた
- [ ] Q3: FiLM の `2*C`・`None` broadcast・`1+gamma` の意味を説明し、入出力 shape 不変を確認
- [ ] Q4: device / eval / 正規化器の 3 バグを直し、評価が走った
- [ ] Q5: 逆正規化を入れ、`dx,dy` が常識的スケールになった
- [ ] Q6: 行動ヘッド付き `MyTinyVLA` を完成（`[B,8,3]`）
- [ ] Q7: `train_mse.py` + `m4_mse.json` で学習し、`success_rate`（7〜8 割目安）を確認
- [ ] Q8: avg / FiLM 無しで `success_rate` が下がることを 3 条件比較で観察し、loss≠成功を説明
- [ ] Q9: TinyVLA が 1 バッチに過学習（`< 0.2 * first`）
