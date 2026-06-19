# 解答 M4: 最小 VLA を自作する（MSE 回帰版 TinyVLA）

問題: [`../../exercises/m4/README.md`](../../exercises/m4/README.md) ／ 本文: [`../../lessons/m4_tiny_vla_mse.md`](../../lessons/m4_tiny_vla_mse.md)

各解答に「**正解コード**」と「**なぜその shape か / なぜ loss が下がるか / なぜ成功率が変わるか**」の短い説明を付けます。

共通 import は演習 [`../../exercises/m4/README.md`](../../exercises/m4/README.md) の「準備」と同じです。

---

## A1（shape 確認）

```python
import torch
from vla_learn.envs import all_instruction_strings
from vla_learn.datasets import CharTokenizer
from vla_learn.models import TinyVLA, count_parameters
from vla_learn.models.image_encoder import ImageEncoder
from vla_learn.models.text_encoder import TextEncoder
from vla_learn.models.state_encoder import StateEncoder

B, L, VOCAB = 4, 17, 30
image  = torch.rand(B, 3, 64, 64)
state  = torch.rand(B, 3)
tokens = torch.randint(0, VOCAB, (B, L))

l = TextEncoder(vocab_size=VOCAB)(tokens)               # (4, 128)
v = ImageEncoder(out_dim=128, cond_dim=128)(image, cond=l)  # (4, 128)
s = StateEncoder(out_dim=64)(state)                     # (4, 64)
fused = torch.cat([v, l, s], dim=-1)                    # (4, 320)
out = TinyVLA(vocab_size=VOCAB, chunk_len=8)(image, state, tokens)  # (4, 8, 3)
for name, t in [("l", l), ("v", v), ("s", s), ("fused", fused), ("out", out)]:
    print(name, tuple(t.shape))
```

| テンソル | shape | 理由 |
|----------|-------|------|
| `l`（言語） | `(4, 128)` | `TextEncoder` の `out_dim` 既定 128。`[B, txt_dim]` |
| `v`（視覚） | `(4, 128)` | `ImageEncoder` の `out_dim=128`。`[B, img_dim]` |
| `s`（状態） | `(4, 64)` | `StateEncoder` の `out_dim=64`。`[B, state_dim]` |
| `fused` | `(4, 320)` | `concat([v,l,s])` = `128+128+64` |
| `h`（融合 MLP 出力） | `(4, 256)` | `Fusion` が `320 → hidden=256` |
| `out`（行動チャンク） | `(4, 8, 3)` | `head` が `256 → 8*3`、`view` で `[B,8,3]` |

**融合次元 320 の足し算**: `画像 128 + 言語 128 + 状態 64 = 320`。`VLABackbone` はこの 320 次元ベクトルを
`Fusion`（`Linear(320,256)→ReLU→Linear(256,256)→ReLU`）に通して `h[B,256]` を作ります。

---

## A2（shape 確認）

行動チャンク `[B, 8, 3]` の各軸:

- 軸 0 `B`: **バッチ**（同時に処理するサンプル数。DataLoader が付ける）。
- 軸 1 `8`: **chunk_len**（一度にまとめて予測する未来ステップ数。単位は「ステップ」）。`TinyVLA(chunk_len=...)` で変わる。
- 軸 2 `3`: **action_dim** = `[dx, dy, grip_cmd]`。
  - `dx, dy`: グリッパの移動量（ワールド座標の差分、各軸 `±MAX_STEP=0.08` 程度）。
  - `grip_cmd`: グリッパ指令（`0.0`=開く / `1.0`=閉じる、`>=0.5` で閉と解釈）。

`[B, 24]` → `[B, 8, 3]` への整形（本文 3 節の `forward`）:

```python
return out.view(-1, self.chunk_len, self.action_dim)   # 24 = 8 * 3
```

**なぜ view で戻すか**: `head = nn.Linear(hidden, chunk_len*action_dim)` は 1 本のベクトル `[B,24]` を出すだけなので、
「8 ステップ × 3 次元」という意味を持たせるために `view` で軸を分けます（メモリはそのまま、見え方だけ変える）。

---

## A3（穴埋め）

```python
import torch
import torch.nn as nn

class MyFiLM(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * num_channels)   # (a) scale と shift で 2 倍

    def forward(self, x, cond):                # x:[B,C,H,W], cond:[B,cond_dim]
        gamma, beta = self.to_scale_shift(cond).chunk(2, dim=-1)      # [B,C], [B,C]
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]  # (b) [B,C,1,1]

f = MyFiLM(cond_dim=128, num_channels=32)
x = torch.randn(4, 32, 16, 16); cond = torch.randn(4, 128)
print(tuple(f(x, cond).shape))   # (4, 32, 16, 16)
```

- (a) **2 倍が要る理由**: チャンネルごとに **scale（gamma）と shift（beta）の 2 つ** を作るため。`chunk(2)` で半分ずつに割る。
- (b) **`None` 2 つ**: `gamma:[B,C]` を `[B,C,1,1]` にして、空間次元 `H,W` 方向に **broadcast**（全画素に同じチャンネル係数を掛ける）するため。
- **`1 + gamma` の意味**: 学習初期は `Linear` 出力（gamma,beta）が 0 付近なので、`1+gamma≈1`, `beta≈0` となり
  **FiLM は恒等写像から始まる**。最初は視覚特徴を壊さず、学習が進むにつれて言語による変調を獲得できる（初期化が安定）。

**なぜ shape が変わらないか**: FiLM はチャンネルごとのアフィン変換なので、`[B,C,H,W]` の各要素を係数倍・加算するだけ。
形は不変で、中身（特徴の強弱）だけが言語に応じて変わります。

---

## A4（バグ修正）

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

model.train()
for epoch in range(3):
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}   # 修正1: batch も device へ
        pred = model(batch["image"], batch["state"], batch["tokens"])
        loss = masked_mse(pred, batch["action"], batch["pad_mask"])
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {epoch}  loss={loss.item():.4f}")

from vla_learn.evaluation.rollout import PolicyWrapper, evaluate_policy
model.eval()                                              # 修正2: 推論前に eval モード
# 修正3: action_norm, state_norm を渡す（逆正規化のため必須）
policy = PolicyWrapper(model, tok, an, sn, model_type="mse", device=device)
print(evaluate_policy(policy, n_episodes=5))
```

放置した場合に起きること:

- **バグ1（device 不一致）**: GPU 利用時、モデルは `cuda`・入力は `cpu` のままで `RuntimeError`（型/デバイス不一致）。CPU だけなら表面化しないが、移植性のため必ず揃える。
- **バグ2（eval 忘れ）**: 推論を学習モードのまま回すと Dropout/BatchNorm がある場合に挙動がブレる。本モデルには無いが、**推論は `eval()`** が鉄則（`PolicyWrapper` も内部で `.eval()` する）。
- **バグ3（正規化器を渡さない）**: `PolicyWrapper(model, tok, model_type=..., device=...)` は `action_norm, state_norm` が位置引数として不足し `TypeError`。仮に動いても逆正規化できず、環境に正規化空間の行動が渡って破綻する。

**なぜこれで通るか**: 3 修正で「入力が正しい device」「推論の作法」「逆正規化の経路」が揃い、`evaluate_policy` が
`{success_rate, mean_final_distance, mean_steps, n_episodes}` を返します（少エピソード・少エポックなので値自体は低くて構いません）。

---

## A5（バグ修正）

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
state  = torch.from_numpy(sn.normalize(obs["state"].astype(np.float32)))[None]
tokens = torch.tensor([tok.encode(obs["instruction"])], dtype=torch.long)

with torch.no_grad():
    chunk = model(img, state, tokens)[0]       # [8, 3]（正規化空間）

chunk_raw = an.denormalize(chunk)              # 修正: 逆正規化してから環境へ
a0 = chunk_raw[0].numpy()
print("正規化空間 chunk[0]:", chunk[0].numpy())   # 学習後は ±1〜2 に達しうる
print("逆正規化 a0      :", a0)                   # dx,dy は ±0.08 程度のワールドスケール
obs, _, done, info = env.step(a0)
print("1 ステップ後の state:", obs["state"])
```

- **なぜ逆正規化が要るか**: 学習は正規化空間（平均 0・分散 1）で行うため、モデル出力もその空間。
  環境 `step` が期待するのは **生のワールド差分**（`dx,dy` は標準偏差 `~0.065`、`MAX_STEP=0.08`。M3 本文の実測 `[0.065, 0.064, ...]`）。
  逆正規化 `denormalize` で「正しい縮尺」へ戻す。これは `PolicyWrapper.predict_chunk` がやっているのと同じこと。
- **放置すると**: 学習が進んだ方策ほど正規化出力が `±1〜2` になり、それを生の差分として渡すと環境内で
  `MAX_STEP` にクリップされて毎ステップ最大移動・暴れる、あるいは方向が破綻して掴めない。
  （注意: **未学習モデル**は head がほぼ 0 初期化で正規化出力も小さいため差が出にくい。差が際立つのは学習後。）

---

## A6（小実装）

```python
import torch
import torch.nn as nn
from vla_learn.envs import all_instruction_strings
from vla_learn.datasets import CharTokenizer
from vla_learn.models import count_parameters
from vla_learn.utils import set_seed

class MyTinyVLA(nn.Module):
    def __init__(self, vocab_size, chunk_len=8, action_dim=3, hidden=256, **backbone_kwargs):
        super().__init__()
        from vla_learn.models.tiny_vla import VLABackbone
        self.backbone = VLABackbone(vocab_size, hidden=hidden, **backbone_kwargs)
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.head = nn.Linear(hidden, chunk_len * action_dim)    # (a) 256 -> 24

    def forward(self, image, state, tokens):
        h = self.backbone(image, state, tokens)                 # [B, hidden]
        out = self.head(h)                                      # [B, 24]
        return out.view(-1, self.chunk_len, self.action_dim)    # (b) [B, 8, 3]

set_seed(0)
tok = CharTokenizer.from_corpus(all_instruction_strings())
m = MyTinyVLA(vocab_size=tok.vocab_size, chunk_len=8)
B = 4
out = m(torch.rand(B,3,64,64), torch.rand(B,3), torch.randint(0,tok.vocab_size,(B,17)))
assert out.shape == (B, 8, 3), out.shape
print("OK forward:", tuple(out.shape), " params:", f"{count_parameters(m):,}")
```

出力例: `OK forward: (4, 8, 3)  params: 422,168`（語彙サイズでぶれます）。

- **`head` の出力次元**: `chunk_len * action_dim = 8 * 3 = 24`。`view(-1, 8, 3)` で「8 ステップ × 3 次元」に意味づけして戻す。
- **`**backbone_kwargs` を残す利点**: `image_pool="avg"` や `condition_vision=False` をそのまま `VLABackbone` に渡せるので、
  Q8 の 3 教訓アブレーションを **同じクラスのまま** 切り替えられる。

**なぜこの shape か**: 行動チャンクは `[B, C, A]` が約束（[M3](../../lessons/m3_data_actions.md)・`masked_mse` の期待形）。
`Linear` は 1 ベクトルしか出せないので、`C*A` 次元を出して `view` で `[C, A]` に割るのが定石です。

---

## A7（実験・スクリプト学習）

```bash
# 配線確認（小規模）
uv run python scripts/train_mse.py --config configs/smoke.json
# 本番（CPU 数分）→ 学習後に自動評価
uv run python scripts/train_mse.py --config configs/m4_mse.json
uv run python scripts/eval_policy.py --ckpt checkpoints/mse/policy.pt --n-episodes 100
```

学習ログ・評価の例（数値はぶれます）:

```text
[model] mse | パラメータ数 = 422,168
[train] epoch   0  loss=0.70218
[train] epoch  29  loss=0.05604
[eval] success_rate=0.760  final_dist=0.160  steps=27.4
...
==== 評価結果 ====
  success_rate: 0.76
  mean_final_distance: 0.16
  mean_steps: 27.4
  n_episodes: 100
```

確認した点:

1. パラメータ数は約 `0.42M`。`loss` は `~0.7 → ~0.056`（下がるのが正常）。
2. `success_rate` は **およそ 7〜8 割**（`≈0.76`）。**環境・乱数で数ポイントはぶれる**。
3. **エキスパート（100%）に届かない理由**: 閉ループでは自分の予測の小さなズレが
   「お手本が通らなかった状態」を生み、そこでの誤差が積み重なる（**distribution shift**、本文 7 節）。
   `masked_mse` は各ステップの平均誤差しか測らないので、この連鎖的破綻は損失に現れない。

**なぜ loss が下がるか**: 教師（正規化済み行動チャンク）に対する回帰なので、Adam が `pred→action` の対応を学べば
`masked_mse` は単調に下がる。ただし **loss の低さは閉ループ成功を保証しない**（3 項参照）。

---

## A8（実験・3 教訓のアブレーション）

`run_training` は既定 backbone で `TinyVLA` を作るため、`image_pool` / `condition_vision` を変えるには
**自前学習ループ**（本文 4.2）で `TinyVLA(..., image_pool=..., condition_vision=...)` を使うのが簡単です。

```python
import torch
from torch.utils.data import DataLoader
from vla_learn.datasets import generate_episodes, build_normalizers, SyntheticVLADataset, CharTokenizer
from vla_learn.envs import all_instruction_strings
from vla_learn.models import TinyVLA
from vla_learn.training.losses import masked_mse
from vla_learn.evaluation.rollout import PolicyWrapper, evaluate_policy
from vla_learn.utils import set_seed, get_device

def train_and_eval(tag, n_episodes=800, epochs=15, eval_n=80, **backbone_kwargs):
    set_seed(0)
    device = get_device()
    eps = generate_episodes(n_episodes=n_episodes, seed=0, action_noise=0.03)
    tok = CharTokenizer.from_corpus(all_instruction_strings())
    an, sn = build_normalizers(eps)
    ds = SyntheticVLADataset(eps, tok, 8, an, sn)
    loader = DataLoader(ds, batch_size=128, shuffle=True)

    model = TinyVLA(vocab_size=tok.vocab_size, chunk_len=8, **backbone_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    last = None
    for _ in range(epochs):
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            loss = masked_mse(model(b["image"], b["state"], b["tokens"]), b["action"], b["pad_mask"])
            opt.zero_grad(); loss.backward(); opt.step()
            last = loss.item()
    pol = PolicyWrapper(model, tok, an, sn, "mse", device)
    met = evaluate_policy(pol, n_episodes=eval_n, n_objects=3, n_goals=2, exec_horizon=4)
    print(f"[{tag:10s}] loss={last:.4f}  success_rate={met['success_rate']:.3f}")
    return met["success_rate"]

a = train_and_eval("default")                          # flatten + FiLM
b = train_and_eval("avg",   image_pool="avg")          # 位置情報を捨てる
c = train_and_eval("noFiLM", condition_vision=False)   # 言語で視覚を変調しない
print(f"\nsuccess_rate  A(default)={a:.3f}  B(avg)={b:.3f}  C(noFiLM)={c:.3f}")
```

観察（**具体値は規模・乱数で大きくぶれます**。傾向を見るのが目的で、断定はしません）:

1. 期待される傾向は **A（既定）> B（avg）, A > C（noFiLM）**。avg は位置情報を、noFiLM は grounding を失うため
   `success_rate` が下がります（学習を長く・データを増やすほど差は安定して見えます。短い学習だと 3 条件とも低く出て差が埋もれることがあります）。
2. **loss は大差ないのに success が違う理由**:
   - **avg**: 平均プーリングで「画面のどこに対象があるか」が消える。MSE は教師との平均一致を学べても、
     **方向（`dx,dy`）を決める空間手がかり** を欠くため閉ループで的を外す（本文 教訓 1）。
   - **noFiLM**: 言語が視覚を変調しないので、画像エンコーダは「どの色を運ぶか」を見られない。
     concat の `l` だけでは対象選択の steering が弱く、**grounding が崩壊**（本文 教訓 3）。
   - いずれも **loss（平均誤差）には現れにくいが閉ループ成功率に効く**（本文 7 節「loss ≠ 成功」）。
3. **grounding 目視**: noFiLM モデルで「同じ初期配置・異なる指示（例: 赤→ / 青→）」を 2 回 rollout すると、
   **運ぶ対象が指示で切り替わらない** ことが多い（指示を無視して同じブロックへ向かう）。FiLM 有りだと対象が切り替わります。

**なぜ成功率が変わるか（まとめ）**: 3 つの設計（空間保持・語順・言語条件付け）は、いずれも
「**正しい対象を、正しい方向へ**」運ぶための情報経路です。どれを切っても情報が欠け、MSE は最小化できても
**閉ループでタスクを達成できなくなる**。これが「部品ごとに役割がある」ことの実感です。

---

## A9（必須・1 バッチ過学習）

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
assert loss.item() < 0.2 * first
print("OK")
```

出力例（数値はぶれます。`first` は概ね 0.6〜1.0、`last` はほぼ 0 まで落ちる）: `first=1.0097  last=0.0000`（`first` の 1/5 未満）。

- **健全性の証明になる理由**: モデル容量・最適化・損失・shape が正しければ、たった 16 サンプルは **丸暗記** できる。
  下がらなければ、その 4 つのどこか（device/shape/損失/勾配）にバグがある、と切り分けられる。
- **本番が 100% にならない理由**: 1 バッチ過学習は「**暗記できる**（学習機構が健全）」を示すだけ。
  本番は **未知の局面** を解く＝**汎化** が要り、distribution shift（本文 7 節）もある。
  「過学習できる」と「汎化できる」は別問題で、後者には情報（画像・言語）と頑健化（`action_noise`・`exec_horizon`）が要る。

**なぜ loss が下がるべきか**: `masked_mse` がパディングを正しく除外し、`forward` の shape が `[B,8,3]` で揃っていれば、
Adam が 16 サンプルの入出力対応を暗記して loss を 0 近くまで落とす。これはリポジトリの
[`../../tests/test_overfit_tiny_batch.py`](../../tests/test_overfit_tiny_batch.py)（`test_mse_overfits_one_batch`、`< 0.2 * first` を assert）と同じ判定です。
