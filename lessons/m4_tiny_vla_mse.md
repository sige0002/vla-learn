# M4: 最小 VLA を自作する（MSE 回帰版 TinyVLA）

> この章のゴール:
> - `画像 + 言語 + 状態 → 行動チャンク` を出す **TinyVLA**（MSE 回帰版）を、3 つのエンコーダ + 融合 MLP +
>   行動ヘッドの **部品を組んで** 自作する。各テンソルの **shape を完全に把握** する。
> - VLA を「動く」ものにするための **3 つの設計教訓** を、実装と結びつけて理解する:
>   (1) 画像は **flatten** で空間情報を保つ（avg だと位置が消える）、(2) 言語は **Transformer + 位置埋め込み** で
>   語順を区別する（平均プーリングの落とし穴）、(3) **FiLM** で言語が視覚を変調する（`condition_vision=False` だと grounding が崩壊）。
> - `masked_mse` で学習ループを回し、`scripts/train_mse.py` + `configs/m4_mse.json` で学習、
>   `scripts/eval_policy.py` で **閉ループ評価（成功率）** まで通す。
> - 「**損失は下がるのに動かすと失敗**」という現象を、[M2](m2_imitation.md) の distribution shift と結びつけて理解する。
>
> 前提: [M1](m1_pytorch.md)（テンソル・`nn.Module`・学習ループ・Dataset/DataLoader）、
> [M2](m2_imitation.md)（エキスパート・`generate_episodes`・素朴 BC と distribution shift）、
> [M3](m3_data_actions.md)（正規化・行動チャンク・トークナイザ・`SyntheticVLADataset` の各 shape・`masked_mse`）。
> 所要時間: 90〜120 分（学習自体は CPU で数分。重い学習は必須ではありません）。

---

## 0. この章の位置づけ — ついに「自作 VLA」

ここが本教材の山場です。[M3](m3_data_actions.md) までで **入力と出力の整え方**（正規化・チャンク・トークン化）が揃いました。
この章では、その `SyntheticVLADataset` をそのまま学習データにして、

```text
画像 image[3,64,64] + 言語 instruction + 状態 state[3]  ──▶  行動チャンク action[8,3]
```

を出す小さな方策 (policy) を **スクラッチで** 組み、学習し、環境で動かして成功率を測ります。

実装は [`../src/vla_learn/models/tiny_vla.py`](../src/vla_learn/models/tiny_vla.py) の `TinyVLA` と `VLABackbone` です。
本物の VLA（SmolVLA / π0 など）も骨格は同じ「**3 つの感覚を融合して行動を出す**」です。
M4 では行動ヘッドを「ただの全結合 + MSE」にします（決定論的方策）。[M5](m5_flow_matching.md) では、この **ヘッドだけを flow matching に差し替え** ます。

> 座学とのつながり: 既習の VLM は「画像 + 言語 → テキスト」でした。VLA は出力が **行動** に変わるだけで、
> 「複数モダリティを 1 本のベクトルに融合する」発想は同じです。M4 はその最小形です。

---

## 1. 全体像 — VLABackbone（3 エンコーダ + 融合）

まず鳥瞰図です。`TinyVLA` は **`VLABackbone`（条件ベクトル `h` を作る）** と **`head`（`h` → 行動チャンク）** の 2 段構成です。

```text
                     ┌─────────────────────────────────────────────┐
 tokens [B,L] ──────▶│ TextEncoder   : 埋め込み+位置埋め込み+Transformer │──▶ l [B,128]
                     └─────────────────────────────────────────────┘        │
                                                       l を条件に ▼ (FiLM)   │
 image [B,3,64,64] ─▶┌─────────────────────────────────────────────┐        │
                     │ ImageEncoder  : CNN(64→32→16→8→4) + FiLM × 2  │──▶ v [B,128]
                     └─────────────────────────────────────────────┘        │
                                                                            │
 state [B,3] ───────▶┌─────────────────────────────────────────────┐        │
                     │ StateEncoder  : 小さな MLP                     │──▶ s [B,64] │
                     └─────────────────────────────────────────────┘        │
                                                                            ▼
                          concat([v, l, s]) = [B, 320] ──▶ Fusion MLP ──▶ h [B,256]
                                                                            │
                                       ┌────────────────────────────────────┘
                                       ▼
                         head: Linear(256 → 8*3) ──▶ view ──▶ action [B,8,3]
```

各テンソルの shape（`B`=バッチ、`L`=トークン長、`C`=chunk_len=8、`A`=action_dim=3）:

| 段 | 入力 | 出力 | 出力 shape |
|----|------|------|-----------|
| TextEncoder | `tokens [B,L]` | 言語ベクトル `l` | `[B, 128]` |
| ImageEncoder（`cond=l`） | `image [B,3,64,64]` | 視覚ベクトル `v` | `[B, 128]` |
| StateEncoder | `state [B,3]` | 状態ベクトル `s` | `[B, 64]` |
| concat | `[v, l, s]` | 融合前ベクトル | `[B, 320]` |
| Fusion MLP | `[B, 320]` | 条件ベクトル `h` | `[B, 256]` |
| head | `h [B,256]` | 行動チャンク | `[B, 8, 3]` |

> 融合次元 `320 = 128(画像) + 128(言語) + 64(状態)` です。この足し算は手で言えるようにしておきましょう（演習で問います）。

`VLABackbone.forward` の順番が大事です。実体（[`tiny_vla.py`](../src/vla_learn/models/tiny_vla.py)）はこうです:

```python
def forward(self, image, state, tokens):
    l = self.text_encoder(tokens)          # [B, txt_dim] … まず言語を符号化
    v = self.image_encoder(image, cond=l)  # [B, img_dim] … その言語で視覚を条件付け（FiLM）
    s = self.state_encoder(state)          # [B, state_dim]
    h = torch.cat([v, l, s], dim=-1)       # [B, img+txt+state] = [B, 320]
    return self.fusion(h)                  # [B, hidden] = [B, 256]
```

ポイントは **言語 `l` を先に作り、それを画像エンコーダに `cond` として渡す** ことです。
こうして「言語が視覚の見え方を変える」（= 後述の FiLM）を実現します。

### 1.1 実際に shape を確かめる

部品の出力 shape を、手を動かして確認しましょう。これは演習でも問います。

```python
import torch
from vla_learn.models.image_encoder import ImageEncoder
from vla_learn.models.text_encoder import TextEncoder
from vla_learn.models.state_encoder import StateEncoder

B, L, VOCAB = 4, 17, 30
image  = torch.rand(B, 3, 64, 64)
state  = torch.rand(B, 3)
tokens = torch.randint(0, VOCAB, (B, L))

txt = TextEncoder(vocab_size=VOCAB)             # out_dim=128（既定）
img = ImageEncoder(out_dim=128, cond_dim=128)   # FiLM 有効（cond_dim を渡す）
stt = StateEncoder(out_dim=64)

l = txt(tokens)            # 先に言語
v = img(image, cond=l)     # その言語で視覚を条件付け
s = stt(state)
print("text :", tuple(l.shape))   # (4, 128)
print("image:", tuple(v.shape))   # (4, 128)
print("state:", tuple(s.shape))   # (4, 64)
print("fused:", l.shape[-1] + v.shape[-1] + s.shape[-1])  # 320
```

出力例:

```text
text : (4, 128)
image: (4, 128)
state: (4, 64)
fused: 320
```

---

## 2. 3 つの設計教訓 — なぜこの作りなのか

VLA を「とりあえず動くニューラルネット」にするだけなら、画像も言語も雑に潰してしまえます。
しかしそれだと **タスクを成功させられません**。この章でいちばん大事なのは、次の 3 つの「効かせ方」です。
いずれも `TinyVLA` のコンストラクタ引数で **オン/オフを切り替えられ**、消すと成功率がはっきり落ちることを演習で体験します。

### 教訓 1: 画像は flatten で「位置」を保つ（avg は位置を捨てる）

このタスクは「**指定色のブロックの“場所”へ動く**」ものです。だから視覚特徴は「**どこに何があるか**」を保たねばなりません。
[`image_encoder.py`](../src/vla_learn/models/image_encoder.py) の `ImageEncoder` は、CNN で `64→32→16→8→4` と空間を縮めた最後の特徴マップ `[B,64,4,4]` を、

- `pool="flatten"`（既定）: そのまま **flatten** して `[B, 64*4*4]=[B,1024]` にする → **4×4 の空間配置が残る**
- `pool="avg"`（比較版）: GlobalAveragePooling で `[B,64]` に潰す → **位置が消える**（「何があるか」は残るが「どこか」が消える）

```python
# image_encoder.py（抜粋）
if self.pool == "flatten":
    x = x.flatten(1)            # [B, 64*4*4] … 空間配置を保持
else:
    x = self.gap(x).flatten(1) # [B, 64]      … 位置を平均で消す（比較用）
return self.fc(x)              # [B, out_dim]
```

直感: avg は「画面のどこかに赤がある」までしか言えません。`[dx, dy]`（どちらへ動くか）を出すには位置が要るので、
avg 版は方向の手がかりを失い、**成功率が落ちます**（[演習 Q8](../exercises/m4/README.md) で実測します）。

> 本物の VLA は事前学習済み ViT（SigLIP 等）を使い、パッチごとのトークン列で空間情報を保ちます。
> ここで flatten が果たす役割は、その「**空間を潰さない**」ことの最小版です。

### 教訓 2: 言語は語順を区別する（平均プーリングの落とし穴 → Transformer + 位置埋め込み）

最も罠にはまりやすいのが言語です。「埋め込みを平均するだけ」にすると、

> 「**赤** のブロックを **青** のゴールに置いて」 と 「**青** のブロックを **赤** のゴールに置いて」

は **同じ文字の集合** なので **同じベクトル** になってしまいます。どちらの色が「運ぶ対象」か区別できず、grounding（言語と対象の対応付け）が原理的に不能になります。
そこで [`text_encoder.py`](../src/vla_learn/models/text_encoder.py) の `TextEncoder` は **位置埋め込み (positional embedding) + 1 層の Transformer エンコーダ** で語順を考慮します:

```python
# text_encoder.py（抜粋）
pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)  # [B, L]
x = self.token_embed(tokens) + self.pos_embed(pos)   # 位置情報を足す
pad_mask = tokens == PAD_ID                           # PAD は無視
x = self.encoder(x, src_key_padding_mask=pad_mask)    # 1 層 Transformer
keep = (~pad_mask).float().unsqueeze(-1)
pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)  # PAD を除いた平均
return self.fc(pooled)                                # [B, out_dim]
```

「最後は平均してるじゃないか」と思うかもしれませんが、Transformer を通した後の各位置ベクトルは
**周囲の語と位置を織り込んだ文脈表現** なので、語順の違いがちゃんと反映されます（素の埋め込みの平均とは別物です）。

> 座学とのつながり: 本物の VLA が言語に Transformer を使うのと同じ発想です。ここでは「語順を捨てない」ことが要点です。

### 教訓 3: FiLM で言語が視覚を変調する（`condition_vision=False` だと grounding 崩壊）

3 つの感覚を **ただ concat する** だけでは、実は弱いのです。「赤を運べ」と言われたとき、視覚側が
**赤を探す目つき** になってほしい。これを実現するのが **FiLM (Feature-wise Linear Modulation)** です。
言語ベクトル `cond` から、特徴マップの **チャンネルごとの (scale, shift)** を作って畳み込み特徴を変調します:

```python
# image_encoder.py（抜粋）
class FiLM(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, x, cond):  # x:[B,C,H,W], cond:[B,cond_dim]
        gamma, beta = self.to_scale_shift(cond).chunk(2, dim=-1)        # [B,C],[B,C]
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
```

`ImageEncoder` は中間 2 箇所（`[B,32,16,16]` と `[B,64,8,8]` の後）に FiLM を挿し、`forward(image, cond=l)` のときだけ効かせます。
`VLABackbone` の `condition_vision` でオン/オフします（[`tiny_vla.py`](../src/vla_learn/models/tiny_vla.py)）:

```python
cond_dim = txt_dim if condition_vision else None
self.image_encoder = ImageEncoder(out_dim=img_dim, pool=image_pool, cond_dim=cond_dim)
```

- `condition_vision=True`（既定）: 言語が視覚を変調 → 「名指しされた色」に視覚を向けられる。
- `condition_vision=False`（比較版）: `cond_dim=None` になり FiLM を作らないため、**画像エンコーダは言語をまったく見ません**。
  言語は concat 時の `l` としてしか入らず、**対象選択（どの色を運ぶか）の steering がほぼ効かなくなり、grounding が崩壊** します。

> 重要教訓（実測の主旨）: 開発時、FiLM を外すと「**言語で対象を選べる率がほぼ 0**」になった一方、FiLM を入れると
> 同じ画像でも指示の色を切り替えると対象が切り替わるようになりました。「concat してあるから言語は届いている**はず**」は
> 通用しません。**どこで** 効かせるか（早い段階で視覚を変調する）が grounding を左右します。

### 3 つまとめ

| 教訓 | 引数 | 既定（良い） | 比較版（悪化） | 失われるもの |
|------|------|--------------|----------------|--------------|
| 空間情報 | `image_pool` | `"flatten"` | `"avg"` | 「どこに」あるか（方向が出せない） |
| 語順 | （`TextEncoder` 内蔵） | Transformer+位置埋め込み | 平均プーリング | 「赤→青」と「青→赤」の区別 |
| 言語条件付け | `condition_vision` | `True`（FiLM） | `False` | 言語による対象選択（grounding） |

これら無しだと **損失は下がっても成功率が落ちます**。だから 3 つとも「効かせる」のが既定値なのです。

---

## 3. TinyVLA 本体 — head と forward

`VLABackbone` が条件ベクトル `h [B,256]` を作ったら、あとは行動チャンクに変換するだけです（[`tiny_vla.py`](../src/vla_learn/models/tiny_vla.py)）:

```python
class TinyVLA(nn.Module):
    def __init__(self, vocab_size, chunk_len=8, action_dim=3, hidden=256, **backbone_kwargs):
        super().__init__()
        self.backbone = VLABackbone(vocab_size, hidden=hidden, **backbone_kwargs)
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.head = nn.Linear(hidden, chunk_len * action_dim)   # 256 → 8*3 = 24

    def forward(self, image, state, tokens):
        h = self.backbone(image, state, tokens)                 # [B, 256]
        out = self.head(h)                                      # [B, 24]
        return out.view(-1, self.chunk_len, self.action_dim)    # [B, 8, 3]
```

`head` は `256 → chunk_len*action_dim = 24` の **ただの全結合** で、`view` で `[B, 8, 3]` に整形します。
これが「決定論的に行動チャンクを 1 つ出す」MSE 版の正体です。`**backbone_kwargs` で `image_pool` や `condition_vision` を素通しできる点に注目（教訓の実験で使います）。

### 3.1 forward と parameter 数を確認

```python
import torch
from vla_learn.models import TinyVLA, count_parameters
from vla_learn.datasets import CharTokenizer
from vla_learn.envs import all_instruction_strings

tok = CharTokenizer.from_corpus(all_instruction_strings())
model = TinyVLA(vocab_size=tok.vocab_size, chunk_len=8)
print("vocab_size =", tok.vocab_size)
print("params     =", f"{count_parameters(model):,}")

B = 4
image  = torch.rand(B, 3, 64, 64)
state  = torch.rand(B, 3)
tokens = torch.randint(0, tok.vocab_size, (B, 17))
out = model(image, state, tokens)
print("forward out:", tuple(out.shape))   # (4, 8, 3)
```

出力例（パラメータ数は語彙サイズ等でぶれます）:

```text
vocab_size = 30
params     = 422,168
forward out: (4, 8, 3)
```

> 規模感: 約 **0.4M パラメータ** の「Tiny」VLA です（テストでも `50_000 < n < 5_000_000` を確認しています）。
> CPU で数分学習でき、本物の VLA（数億〜数十億）の **同じ部品の縮小版** として全工程を体験できます。
> 参考までに `image_pool="avg"` だと約 0.30M、`condition_vision=False` だと約 0.40M になります（位置や FiLM の重みが減るため）。

---

## 4. 学習ループ — masked_mse で教師あり回帰

### 4.1 損失: なぜ masked_mse か（[M3](m3_data_actions.md) の復習）

教師は「正規化済みの行動チャンク」`action [B,8,3]` です。`TinyVLA` の出力 `pred [B,8,3]` をこれに近づけます。
ただし [M3](m3_data_actions.md) で見たとおり、チャンク末尾は **パディング** されていることがあるので、
[`../src/vla_learn/functional.py`](../src/vla_learn/functional.py) の `masked_mse` で `pad_mask=0` のステップを損失から外します。
学習コードでは **`from vla_learn.training.losses import masked_mse`** で使います（中身は `functional` のものを再公開しているだけ）:

```python
# functional.py（全文に近い抜粋）
def masked_mse(pred, target, mask=None):     # pred,target: [B,C,A]  mask: [B,C]
    se = (pred - target) ** 2                # [B, C, A]
    if mask is None:
        return se.mean()
    mask3 = mask.unsqueeze(-1).expand_as(se) # [B, C, A] に拡張
    return (se * mask3).sum() / mask3.sum().clamp(min=1.0)  # 有効ステップで平均
```

### 4.2 学習ループの中核（自分で書けるようになるのが目標）

学習ループ本体は [`../src/vla_learn/training/trainer.py`](../src/vla_learn/training/trainer.py) の `run_training` です。
心臓部だけ抜き出すと、[M1](m1_pytorch.md) で習った標準形そのものです:

```python
import torch
from torch.utils.data import DataLoader
from vla_learn.datasets import (
    generate_episodes, build_normalizers, SyntheticVLADataset, CharTokenizer,
)
from vla_learn.envs import all_instruction_strings
from vla_learn.models import TinyVLA
from vla_learn.training.losses import masked_mse
from vla_learn.utils import set_seed, get_device

set_seed(0)
device = get_device()                                   # CPU/GPU 自動

# --- データ（M2/M3 の部品をそのまま）---
episodes = generate_episodes(n_episodes=200, seed=0, action_noise=0.03)
tok = CharTokenizer.from_corpus(all_instruction_strings())
action_norm, state_norm = build_normalizers(episodes)
ds = SyntheticVLADataset(episodes, tok, chunk_len=8,
                         action_normalizer=action_norm, state_normalizer=state_norm)
loader = DataLoader(ds, batch_size=64, shuffle=True)

# --- モデル・最適化 ---
model = TinyVLA(vocab_size=tok.vocab_size, chunk_len=8).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- 学習ループ ---
model.train()                                            # 学習モード（重要）
for epoch in range(5):                                   # デモ用に 5 epoch
    running, nb = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}   # device を揃える（重要）
        pred = model(batch["image"], batch["state"], batch["tokens"])  # [B,8,3]
        loss = masked_mse(pred, batch["action"], batch["pad_mask"])
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item(); nb += 1
    print(f"epoch {epoch}  loss={running/nb:.5f}")
```

出力例（数値はぶれます。下がっていれば OK）:

```text
epoch 0  loss=0.61432
epoch 1  loss=0.18044
epoch 2  loss=0.10231
epoch 3  loss=0.07515
epoch 4  loss=0.06120
```

初心者がつまずく 3 点（演習のバグ修正でも問います）:

- **device を揃える**: `batch` のテンソルをモデルと同じ device に `.to(device)` する。忘れると `cpu` と `cuda` の不一致で落ちます。
- **`.train()` / `.eval()`**: 学習中は `model.train()`。推論・評価では `model.eval()`（このモデルに Dropout/BN は無いものの、習慣として徹底）。
- **正規化**: 教師 `action` は **正規化済み**（`SyntheticVLADataset` が `action_normalizer` で変換済み）。推論時は逆正規化して環境へ（次節）。

> [M5](m5_flow_matching.md) との関係: この学習ループで **MSE と flow の違いは「モデル構築」と「損失計算」の 2 箇所だけ** です。
> `trainer.py` の `_compute_loss` は `model_type=="flow"` のとき `model.flow_loss(...)` を呼び、そうでなければ `masked_mse(...)` を使います。
> いまは「ヘッドと損失を差し替えるだけで枠組みは共通」と覚えておけば十分です。

---

## 5. スクリプトで学習する — train_mse.py + configs/m4_mse.json

実運用では Python を直書きせず、設定ファイルとスクリプトで回します。

```bash
# セットアップは M0 の `uv sync` で完了済み（vla_learn が editable 導入されている）。
# 以降のコマンドは `uv run python ...` で実行する。
# uv を使わない場合のフォールバックだけ、src を import パスに見せる:
export PYTHONPATH=src
```

スモークテスト（ごく小規模で配線確認。1〜2 分）。`configs/smoke.json` は `n_episodes=60, epochs=3` 程度です:

```bash
uv run python scripts/train_mse.py --config configs/smoke.json
```

本番設定（`configs/m4_mse.json`）。中身は次のとおりです:

```json
{
  "model_type": "mse",
  "n_episodes": 1500,
  "n_objects": 3,
  "n_goals": 2,
  "chunk_len": 8,
  "epochs": 30,
  "batch_size": 128,
  "lr": 0.001,
  "eval_episodes": 100,
  "exec_horizon": 4,
  "seed": 0,
  "out_dir": "checkpoints/mse"
}
```

```bash
uv run python scripts/train_mse.py --config configs/m4_mse.json
# 一部だけ上書きしたいとき（CLI 引数が config より優先）
uv run python scripts/train_mse.py --config configs/m4_mse.json --epochs 30 --n-episodes 1500
```

`run_training` は最後に **学習済み方策を保存し、そのまま閉ループ評価** まで行います。出力例（数値はぶれます）:

```text
[setup] device=cpu  model_type=mse
[data] 1500 episodes 生成 (action_noise=0.03)
[data] 12030 学習サンプル / vocab=30
[model] mse | パラメータ数 = 422,168
[train] epoch   0  loss=0.70218
[train] epoch  10  loss=0.09531
[train] epoch  29  loss=0.05604
[save] checkpoints/mse/policy.pt
[eval] success_rate=0.760  final_dist=0.160  steps=27.4
```

> 実測の目安（Claude が CPU で計測。`n_objects=3, n_goals=2, action_noise=0.03, 1500ep×30epoch, exec_horizon=4`）:
> 学習 loss はおよそ `0.7 → 0.056` まで下がり、**閉ループ成功率はおよそ 7〜8 割**（`success_rate≈0.76`）、`final_distance≈0.16` でした。
> ただし **環境・乱数で数ポイントはぶれます**。`success_rate` が 0.7 前後〜0.8 台に入っていれば想定どおりです（解析エキスパートは 100% 成功なので、その下にギャップがあるのは正常です）。

---

## 6. 閉ループ評価 — 損失ではなく成功率を見る

「損失が下がった」は **お手本との一致度** にすぎません。本当に知りたいのは「環境で動かしてタスクを成功できるか」です。
[`../src/vla_learn/evaluation/rollout.py`](../src/vla_learn/evaluation/rollout.py) がこれを担います。

### 6.1 PolicyWrapper — obs から行動チャンクへ（正規化・device 込み）

`PolicyWrapper` は学習済みモデルを「`obs → 行動チャンク（生の値）`」に変換します。要点は **state を正規化して入れ、出力を逆正規化して返す** ことです:

```python
# rollout.py（PolicyWrapper.predict_chunk 抜粋）
state_np = self.state_norm.normalize(obs["state"].astype(np.float32))   # 入力 state は正規化
...
a = self.model(img, state, tokens)               # [1,C,3]（正規化空間）
a = self.action_norm.denormalize(a)[0].cpu().numpy()  # 生の行動に戻して返す
```

> [M3](m3_data_actions.md) の最重要点の実物です。**逆正規化を忘れると、環境は ±2 のような巨大な `dx,dy` を受け取り破綻** します。
> 学習は正規化空間、環境とのやり取りは生の空間、という分担を `PolicyWrapper` が守っています。

### 6.2 receding horizon と exec_horizon

行動チャンクは「**chunk を予測 → 先頭 `exec_horizon` ステップだけ実行 → 観測し直して再予測**」を繰り返して使います（receding horizon, 後退ホライズン）:

```python
# rollout.py（rollout_episode 抜粋）
while not done:
    chunk = policy.predict_chunk(obs)             # [C, 3]
    for k in range(min(exec_horizon, chunk.shape[0])):
        obs, _, done, info = env.step(chunk[k])   # 先頭 exec_horizon ステップだけ実行
        if done:
            break
```

`exec_horizon=4` なら「8 ステップ予測して 4 ステップ実行し、また観測し直す」です。**全 8 ステップを実行しきらない** のがコツ（理由は次節）。

### 6.3 評価を走らせる

学習で保存した `checkpoints/mse/policy.pt` を読み込んで評価します:

```bash
uv run python scripts/eval_policy.py --ckpt checkpoints/mse/policy.pt --n-episodes 100
```

出力例（数値はぶれます）:

```text
==== 評価結果 ====
  success_rate: 0.76
  mean_final_distance: 0.16
  mean_steps: 27.4
  n_episodes: 100
```

`success_rate` が「指定色のブロックを指定色のゴールへ運べた割合」です。**7〜8 割** あたりが目安です。

> 目で見たいときは `uv run python scripts/demo_rollout.py --ckpt checkpoints/mse/policy.pt --out assets/rollout.png`
> でロールアウトを画像グリッドに保存できます（matplotlib が必要）。指示文と成功可否が表示されます。

---

## 7. 「損失は下がるのに動かすと失敗」 — distribution shift の再来

ここは VLA でいちばん大事な落とし穴です。学習 loss が `0.056` まで下がっても、成功率は 100% にはなりません。なぜか。

[M2](m2_imitation.md) で見た **distribution shift（分布シフト）** が原因です。学習データは **エキスパートが通った軌道** だけ。
ところが本番では、自分の予測がわずかにズレ、**お手本が一度も通らなかった状態** に入ります。そこでの行動は学習していないのでさらにズレ、誤差が積み重なって失敗します。`masked_mse` は「各ステップの平均誤差」を測るだけで、**この連鎖的なズレ（閉ループでの破綻）は測れません**。だから「loss は低いのに失敗」が起きます。

この教材が打っている **2 つの緩和策** を、引数と結びつけて押さえましょう:

- **`action_noise`（既定 0.03）**: データ生成時にエキスパート行動へ小さなノイズを混ぜます（[M2](m2_imitation.md) の DAgger 風の発想）。
  すると「少しズレた状態」も訓練分布に入り、本番で軌道を外しても **復帰** しやすくなります。`0.0` にすると見た目の loss は下がりやすいのに、
  閉ループは脆くなりがちです（演習の実験ネタ）。
- **`exec_horizon`（既定 4）**: 8 ステップ全部を実行しきらず、**4 ステップで観測し直す** ことで、古い観測に基づく行動の積み重ねを断ち、ズレを早めにリセットします。
  大きくする（例 8）ほど再観測が減って distribution shift に弱くなり、小さすぎる（例 1）と再観測は増えますが推論回数が増え、チャンクの滑らかさの利点が薄れます。

> まとめると、**MSE の低さ ≠ タスク成功**。閉ループ成功率こそが VLA の通信簿です。
> [M5](m5_flow_matching.md) の flow matching は、行動分布の **多峰性**（同じ状況で複数の正解経路がある場合）を扱える点で、
> この決定論 MSE 版より頑健になり得ます。まずは「MSE 版で 7〜8 割」を体で覚えるのが目的です。

---

## 8. 章末チェック — 1 バッチに過学習できるか

学習ループ・shape・損失・最適化のどれかにバグがあると、**そもそも学習が回っていません**。これを最速で炙り出すのが
「**小さな 1 バッチに過学習できるか**」テストです（機械学習デバッグの鉄則）。正しければ loss は限りなく 0 に近づきます。

```python
import torch
from torch.utils.data import DataLoader
from vla_learn.datasets import (
    generate_episodes, build_normalizers, SyntheticVLADataset, CharTokenizer,
)
from vla_learn.envs import all_instruction_strings
from vla_learn.models import TinyVLA
from vla_learn.training.losses import masked_mse
from vla_learn.utils import set_seed

set_seed(0)
eps = generate_episodes(n_episodes=8, seed=0)
tok = CharTokenizer.from_corpus(all_instruction_strings())
an, sn = build_normalizers(eps)
ds = SyntheticVLADataset(eps, tok, 8, an, sn)
batch = next(iter(DataLoader(ds, batch_size=16, shuffle=True)))   # 固定の 1 バッチ

model = TinyVLA(vocab_size=tok.vocab_size, chunk_len=8)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
first = None
for i in range(200):
    pred = model(batch["image"], batch["state"], batch["tokens"])
    loss = masked_mse(pred, batch["action"], batch["pad_mask"])
    opt.zero_grad(); loss.backward(); opt.step()
    if first is None:
        first = loss.item()
print(f"first={first:.4f}  last={loss.item():.4f}  (last は first の 1/5 未満が目安)")
```

出力例（数値はぶれます）:

```text
first=0.6149  last=0.0087  (last は first の 1/5 未満が目安)
```

> これはリポジトリの [`../tests/test_overfit_tiny_batch.py`](../tests/test_overfit_tiny_batch.py)（`test_mse_overfits_one_batch`）と同じ趣旨です。
> 学習ループを自作したら、まずこの確認をしてから本番データに進むのが安全です。`uv run pytest -k overfit` でも走らせられます。
> なお `run_training` には `overfit_one_batch=True` という設定もあり、1 バッチだけを繰り返し学習するモードを `TrainConfig` から使えます。

---

## 9. まとめ

- **TinyVLA = VLABackbone（3 エンコーダ + 融合 MLP）+ head（全結合）**。`forward(image,state,tokens) → [B,8,3]`。
  融合は `concat([v,l,s]) = [B,320] → Fusion → h[B,256] → head → [B,8,3]`。約 0.4M パラメータ。
- **3 つの設計教訓**:
  (1) 画像は `image_pool="flatten"` で **空間情報を保持**（avg は位置を捨て、方向が出せない）。
  (2) 言語は **Transformer + 位置埋め込み** で **語順を区別**（平均プーリングだと「赤→青」と「青→赤」が同一表現）。
  (3) **FiLM** で言語が視覚を変調（`condition_vision=False` だと言語をほぼ無視し grounding が崩壊）。
  いずれも消すと **損失は下がっても成功率が落ちる**。
- **学習**: `masked_mse(pred, action, pad_mask)` を Adam で最小化。`from vla_learn.training.losses import masked_mse`。
  device を揃える / `.train()`/`.eval()` / 正規化を忘れない。
- **スクリプト**: `uv run python scripts/train_mse.py --config configs/m4_mse.json` で学習 + 自動評価。
  `uv run python scripts/eval_policy.py --ckpt checkpoints/mse/policy.pt` で閉ループ評価。実測の目安は **成功率およそ 7〜8 割**（ぶれる）。
- **閉ループ評価**: `PolicyWrapper`（state 正規化・行動逆正規化・device）＋ receding horizon（`exec_horizon=4`）。
  **損失の低さ ≠ 成功**。
- **distribution shift**: 「loss は低いのに失敗」は [M2](m2_imitation.md) の分布シフトの再来。`action_noise` と `exec_horizon` で緩和する。
- **デバッグ**: 何よりまず **1 バッチに過学習** できるか確認する。

## 次の章へ

MSE 版 TinyVLA で「画像 + 言語 + 状態 → 行動チャンク」を自作し、閉ループ成功率まで測れました。
次の [M5](m5_flow_matching.md) では、この `TinyVLA` の **行動ヘッドだけを flow matching ヘッドに差し替え** て `FlowVLA` を作ります。
`VLABackbone`（3 エンコーダ + 融合）も学習ループの骨格も **そのまま流用** し、変わるのは「ヘッドと損失」だけ —
まさに「同じ部品の付け替え」を体験します。既習の flow/拡散の概念が、ここで実装に接続されます。

→ 演習は [`../exercises/m4/README.md`](../exercises/m4/README.md)、解答は [`../solutions/m4/README.md`](../solutions/m4/README.md)。
