#import "../lib/template.typ": *

= PyTorch 速習 <m1>

#goal[
  - *テンソル (tensor)* を作り、`shape` / `dtype` / `device` を読めるようになる。
  - *ブロードキャスト (broadcasting)* の規則を理解し、形の違う配列の演算を予測できる。
  - *自動微分 (autograd)*: `requires_grad` / `backward()` / `.grad` / `no_grad` を使える。
  - `nn.Module` / `nn.Linear` / `nn.Sequential` でモデルを定義できる。
  - 損失と optimizer (Adam / `zero_grad` / `backward` / `step`) で*学習ループ*を書ける。
  - `Dataset` / `DataLoader`（`__len__` / `__getitem__` / バッチ化 / `shuffle`）を使える。
  - *「1 バッチに過学習できるか」* という学習デバッグの鉄則を理解する。
]

この章は *PyTorch の基礎に集中* します。diffusion / VLM の座学は既知前提なので触れません。
代わりに、後の章で実際に使う *本書の題材*（状態 $[3]$・行動チャンク $[T, 3]$・小さな回帰）に
寄せて練習します。本章のコードはすべて *CPU・コピペで動く* ように書いています（対話シェルか
短い `.py` を `python xxx.py` で実行）。

#theory[
  この章で握る道具（tensor / autograd / `nn.Module` / 学習ループ / `DataLoader`）は、最終的に
  下図の VLA を組み立てるための部品です。いま全体像が分からなくて大丈夫です。「この層は
  `nn.Linear`」「この矢印は `forward`」「学習は loss を `backward`」——と後で対応が取れるよう、
  完成形を一度だけ眺めておきましょう。
]

#fig("/figures/architecture.png", caption: [本章で握る道具が組み上がる先（M4 / M5 の完成形）。
  3 入力をエンコードし融合して行動チャンクを出す。各エンコーダは `nn.Module`、矢印は `forward`、
  学習は損失の `backward`。実装は #raw("src/vla_learn/models/tiny_vla.py")。], width: 92%)

== テンソル (tensor) ―― PyTorch の基本データ

テンソルは「GPU でも動く、自動微分に対応した多次元配列」です。NumPy の `ndarray` とほぼ同じ
感覚で使えます。

=== 作る

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])    # リストから（状態 state[3] のイメージ）
z = torch.zeros(8, 3)                 # 0 埋め [8, 3]（行動チャンク [T=8, A=3] のイメージ）
o = torch.ones(2, 3)                  # 1 埋め
r = torch.randn(4, 3)                 # 標準正規分布から [4, 3]
e = torch.arange(6)                   # 0,1,2,3,4,5
print(a)                              # tensor([1., 2., 3.])
print(z.shape, r.shape)               # torch.Size([8, 3]) torch.Size([4, 3])
```

=== shape（形）

`shape` は各次元の大きさです。VLA では shape を読めることが最重要スキルになります（観測画像
$[B, C, H, W]$、行動 $[B, T, A]$ など）。

```python
import torch

x = torch.randn(4, 8, 3)        # 例: バッチ4, チャンク長8, 行動次元3 = [B, T, A]
print(x.shape)                  # torch.Size([4, 8, 3])
print(x.shape[0], x.ndim)       # 4 3   （バッチサイズ と 次元数）

flat = x.reshape(4, 24)         # 形を変える（要素数は不変）[4, 8*3] = [4, 24]
print(flat.shape)               # torch.Size([4, 24])

s = torch.randn(3)              # [3]
print(s.unsqueeze(0).shape)             # [1, 3]  先頭にバッチ次元を足す
print(s.unsqueeze(0).squeeze(0).shape)  # [3]     大きさ1の次元を消す
```

#note[
  *読み方の規約（本書共通）*
  - $B$ = バッチサイズ、$T$ または $C$ = チャンク長（時間ステップ数）、$A$ = 行動次元 (=3)、
    $D$ = 特徴次元。
  - 画像は $[B, C, H, W]$（C=チャンネル=3, H=W=64）。
  - 行動チャンクは $[B, T, A]$（例 $[64, 8, 3]$）、状態は $[B, D]$（例 $[64, 3]$）。
]

=== dtype（型）

`dtype` は数値の型です。*float と int を混ぜると壊れる*ので注意します。

```python
import torch

f = torch.tensor([1.0, 2.0])    # 既定は float32
i = torch.tensor([1, 2])        # 整数リストなら int64
print(f.dtype, i.dtype)         # torch.float32 torch.int64

g = torch.zeros(3, dtype=torch.float32)
h = i.float()                   # int64 -> float32
j = f.long()                    # float32 -> int64
print(g.dtype, h.dtype, j.dtype)  # torch.float32 torch.float32 torch.int64
```

#pitfall[
  *本書での型の約束*（`src/vla_learn/datasets/synthetic_dataset.py` の `__getitem__` より）:
  画像・状態・行動・pad_mask は *float32*、言語のトークン ID (`tokens`) は *int64* です。
  トークン ID が int64 なのは、後で `nn.Embedding`（埋め込み表）の*索引*として使うから——
  索引は整数でなければなりません。ここを float にすると埋め込みでエラーになります。
]

=== device（CPU / GPU）

テンソルは CPU か GPU のどちらかに乗っています。*鉄則: 演算する者どうしは同じ device に置く*
（混在は実行時エラー）。本書は CPU 完結なので基本は `cpu` のままで困りません。

```python
import torch
from vla_learn.utils.device import get_device

x = torch.randn(2, 3)           # 既定は CPU
print(x.device)                 # cpu

dev = get_device()              # GPU があれば cuda、無ければ cpu
x = x.to(dev)                   # その device へ移す
print(x.device)
```

#readcode("src/vla_learn/utils/device.py", target: "get_device")[
  `get_device()` は `torch.cuda.is_available()` を見て `cuda` か `cpu` を返すだけの数行です。
  ファイル冒頭コメントに「テンソルとモデルは同じ device に置く」という鉄則が書いてあります。
  本書の学習・評価スクリプトはみなこのヘルパで device を取ります。
]

=== NumPy との行き来

環境 (`envs/`) は NumPy で世界を作り、学習は PyTorch です。両者は橋渡しできます。

```python
import numpy as np
import torch

arr = np.array([0.3, 0.7, 0.0], dtype=np.float32)   # state [ax, ay, gripper]
t = torch.from_numpy(arr)        # NumPy -> tensor（メモリ共有）
print(t, t.dtype)                # tensor([0.3000, 0.7000, 0.0000]) torch.float32

back = t.numpy()                 # tensor -> NumPy
print(type(back))                # <class 'numpy.ndarray'>
```

実際、後の章で作る `SyntheticVLADataset.__getitem__` は `torch.from_numpy(...)` で NumPy 画像・
状態・行動を tensor に変換して返しています（M3）。

== ブロードキャスト (broadcasting)

形の違うテンソルどうしの演算を、PyTorch が*自動で形を合わせて*計算してくれる仕組みです。
規則は「*末尾の次元から見て、大きさが等しい or どちらかが 1 なら OK*」。1 の側が引き伸ばされます。

```python
import torch

x = torch.randn(4, 3)                 # [B=4, D=3]
b = torch.tensor([10.0, 20.0, 30.0])  # [3]  -> [1, 3] とみなされ各行に足される
print((x + b).shape)                  # torch.Size([4, 3])

col = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # [4, 1] -> 各列にブロードキャスト
print((x + col).shape)                # torch.Size([4, 3])
```

本書で実際に使われる例です。パディング位置の誤差を 0 にする `masked_mse`（行動チャンクの損失）
で、`[B, C]` のマスクを `[B, C, 1]` にしてから二乗誤差 `[B, C, A]` に掛けます。

```python
import torch

se   = torch.randn(4, 8, 3) ** 2    # 二乗誤差 [B, C, A]
mask = torch.ones(4, 8)             # pad_mask [B, C]（1=有効, 0=パディング）
mask3 = mask.unsqueeze(-1)          # [B, C, 1]
masked = se * mask3                 # [B,C,1] が [B,C,3] にブロードキャストされる
print(masked.shape)                 # torch.Size([4, 8, 3])
```

`unsqueeze(-1)` で `[B, C]` を `[B, C, 1]` にしてから掛けると、行動次元 $A=3$ 方向に同じマスクが
伸びて、パディング位置の誤差をまとめて 0 にできます。*「次元を 1 にして broadcast」は頻出
パターン*です。

#readcode("src/vla_learn/functional.py", target: "masked_mse")[
  この `unsqueeze(-1)` で broadcast するパターンが実際に使われている場所
  （`mask.unsqueeze(-1).expand_as(se)`）。行動チャンクは長さが足りない分を 0 で*パディング*し、
  `pad_mask` でそこを損失から除外します。M4 の学習損失そのもので、学習側からは
  `from vla_learn.training.losses import masked_mse` で使います。broadcast の練習として今のうちに
  眺めておくと M4 がスムーズです。
]

#pitfall[
  *よくある罠*: `[8, 3]` と `[3, 8]` は broadcast できません（末尾から見て 3 ≠ 8 かつ 1 でも
  ない）。「形が合わない」エラーの大半はこれです。落ち着いて `.shape` を print しましょう。
]

== 自動微分 (autograd)

ニューラルネットの学習は「損失を下げる方向にパラメータを少し動かす」の繰り返しです。その
「方向」= *勾配 (gradient)* を、PyTorch が*自動で*計算してくれるのが autograd です。座学で
出てきた誤差逆伝播 (backpropagation) を、手で微分せずに使えます。

=== requires_grad / backward / .grad

```python
import torch

x = torch.tensor([2.0], requires_grad=True)  # 「微分の対象」として追跡される
y = x ** 2 + 3 * x          # y = x^2 + 3x
y.backward()                # dy/dx を計算（逆伝播）
print(x.grad)               # dy/dx = 2x + 3 = 2*2+3 = 7  -> tensor([7.])
```

ポイント:
- `requires_grad=True` のテンソルから計算した結果には、その履歴（計算グラフ）が記録されます。
- `loss.backward()` を呼ぶと、グラフを逆向きにたどって各 `requires_grad=True` のテンソルの
  `.grad` に勾配が貯まります。
- *`.grad` は加算されていく*（上書きではない）。だから毎ステップ `zero_grad()` でリセットが
  必要です（後述。忘れると壊れます）。

=== no_grad（勾配を切る）

推論時や勾配が要らない処理では `torch.no_grad()` で追跡を止めます。*メモリと速度の節約*に
なり、また「パラメータ更新の式に勾配を混ぜない」ために必須です。

```python
import torch

w = torch.tensor([1.0], requires_grad=True)
with torch.no_grad():
    w2 = w * 5              # この計算は追跡されない
print(w2.requires_grad)    # False
```

#readcode("src/vla_learn/models/tiny_vla.py", target: "TinyVLA.predict")[
  実装では `TinyVLA.predict` が `@torch.no_grad()` デコレータで推論し、評価ループ
  (`evaluation/rollout.py`) でも勾配を切って行動を予測します。*学習中は勾配を追跡、推論中は切る*、
  と覚えてください。同じファイルの `forward` と並べて読むと「学習用と推論用の入口」の違いが
  分かります。
]

== nn.Module / nn.Linear / nn.Sequential

毎回パラメータを手で持つのは大変です。PyTorch では *`nn.Module`* を継承してモデルを作ると、
パラメータ管理・device 移動・学習/評価モード切替などを面倒見てくれます。

=== nn.Linear（全結合層）

`nn.Linear(in, out)` は $y = x W^top + b$ を計算する層です。入力の*最後の次元*を `in` から
`out` に変えます。

```python
import torch
import torch.nn as nn

lin = nn.Linear(3, 64)          # 状態 state[*, 3] -> 特徴[*, 64]
s = torch.randn(4, 3)           # [B=4, 3]
out = lin(s)
print(out.shape)                # torch.Size([4, 64])
print(lin.weight.shape, lin.bias.shape)  # torch.Size([64, 3]) torch.Size([64])
```

これは本書の `StateEncoder`（状態 $[*, 3]$ → $[*, 64]$）と同じ発想です（M4）。

=== nn.Module を継承する

```python
import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, in_dim=3, hidden=32, out_dim=3):
        super().__init__()                      # ← 必ず最初に呼ぶ
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):                       # 順伝播を書く
        h = self.act(self.fc1(x))
        return self.fc2(h)

model = TinyMLP()
x = torch.randn(4, 3)                            # [B, 3]
y = model(x)                                     # model(x) は forward(x) を呼ぶ
print(y.shape)                                   # torch.Size([4, 3])

n_params = sum(p.numel() for p in model.parameters())  # パラメータは自動で集まる
print("params:", n_params)
```

#pitfall[
  モデルを呼ぶときは `model.forward(x)` ではなく *`model(x)`* と書きます（フックなどが正しく
  動くため）。また `super().__init__()` を忘れると、`self.fc1 = nn.Linear(...)` を代入しても
  パラメータが登録されず、`model.parameters()` が空になって学習できません（最頻出の初心者バグ）。
]

#readcode("src/vla_learn/models/tiny_vla.py", target: "count_parameters")[
  上の `sum(p.numel() for p in model.parameters())` と同じ書き方が `count_parameters` です
  （`requires_grad` のものだけ数える点だけ違う）。これで `TinyVLA` が約 0.42M パラメータ、と
  測れます。モデル規模をすぐ把握できるので、自作モデルを作るたびに呼ぶ習慣をつけましょう。
]

=== nn.Sequential（層を直列に並べる）

分岐のない単純な積み重ねは `nn.Sequential` が簡潔です。

```python
import torch
import torch.nn as nn

mlp = nn.Sequential(
    nn.Linear(3, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 3),
)
print(mlp(torch.randn(4, 3)).shape)              # torch.Size([4, 3])
```

#readcode("src/vla_learn/models/tiny_vla.py", target: "VLABackbone")[
  実際、VLA 本体 `VLABackbone` の融合 MLP は
  `nn.Sequential(nn.Linear(...), nn.ReLU(inplace=True), nn.Linear(...), nn.ReLU(inplace=True))`
  で書かれています（本章の書き方そのまま）。`forward` で 3 つのエンコーダ出力を `torch.cat` して
  この MLP に通し、条件ベクトル $h$ を作ります。冒頭の図の「融合」がこのクラスです。
]

== 学習ループ（損失と optimizer）

ここが PyTorch の心臓部です。*毎ステップ次の 4 つを順番に呼ぶ*だけ、と覚えてください。

```text
   for ステップ in 学習:
     ① pred = model(x)              # 順伝播（予測）
     ② loss = 損失(pred, target)     # どれだけ間違ったか
     ③ optimizer.zero_grad()        # 前回の勾配を消す（忘れると加算され壊れる）
        loss.backward()             # 逆伝播（.grad を計算）
     ④ optimizer.step()             # パラメータを勾配方向に更新
```

=== 最小の完全な例: 小さな線形回帰

「$y = 2x + 1$ を当てる」を、本書の道具（`nn.Linear`・MSE・Adam）で学習します。*そのまま
コピペで動きます。*

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# 1) データ（答え y = 2x + 1。少しノイズを足す）
x = torch.randn(128, 1)                 # [N=128, 1]
y = 2.0 * x + 1.0 + 0.05 * torch.randn(128, 1)

# 2) モデル・損失・optimizer
model = nn.Linear(1, 1)                  # 直線 y = w x + b を学習
loss_fn = nn.MSELoss()                   # 平均二乗誤差
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# 3) 学習ループ
for step in range(200):
    pred = model(x)                      # ① 順伝播  [128, 1]
    loss = loss_fn(pred, y)              # ② 損失（スカラ）
    optimizer.zero_grad()               # ③ 勾配リセット
    loss.backward()                     #    逆伝播
    optimizer.step()                    # ④ 更新
    if step % 50 == 0:
        print(f"step {step:3d}  loss = {loss.item():.4f}")

w = model.weight.item()
b = model.bias.item()
print(f"学習結果: y ≈ {w:.2f} x + {b:.2f}  (正解は 2.00 x + 1.00)")
```

出力例（数値は環境でぶれます）:

```text
step   0  loss = 10.1834
step  50  loss = 0.7888
step 100  loss = 0.0052
step 150  loss = 0.0017
学習結果: y ≈ 2.00 x + 1.00  (正解は 2.00 x + 1.00)
```

各行の意味:
- *`nn.MSELoss()`*: 予測と正解の差の二乗の平均。回帰の定番（本書 M4 も MSE です）。
- *`torch.optim.Adam(model.parameters(), lr=...)`*: 更新方法。`model.parameters()` を渡すことで
  「どのテンソルを更新するか」を optimizer に教えます。`lr`（学習率）は 1 歩の大きさ。
- *`loss.item()`*: スカラのテンソルから Python の `float` を取り出します。`print` や記録には
  `.item()` を使う（テンソルのまま貯めると計算グラフが残りメモリを食う/壊れる原因に）。

#pitfall[
  *`zero_grad()` を忘れるとどうなる？* `.grad` は加算され続けるので、勾配がどんどん膨らみ、
  更新が暴れて loss が下がりません。「loss が下がらない」ときに真っ先に疑う定番バグです。
]

== 「1 バッチに過学習できるか」――学習デバッグの鉄則

VLA に限らず、学習コードを書いたら*最初に必ず確認すべきこと*があります。

#theory[
  *鉄則*: モデルと学習ループが正しければ、*小さな 1 バッチ（数十サンプル）には必ず過学習できる*。
  過学習すらできないなら、損失・shape・最適化・勾配のどこかにバグがある。
]

なぜか。汎化（未知データへの一般化）は難しい問題ですが、「*手元の固定された数十サンプルを
丸暗記する*」のは、配線が正しければニューラルネットには簡単なはずだからです。つまり「1 バッチに
過学習できるか」は *モデルの賢さのテストではなく、配線（パイプライン）の健全性テスト*です。
大きなデータで延々と回す前に、これで素早くバグを切り分けます。

確認のしかた:
+ データを*ごく少数*（例 16 サンプル）に固定する（`shuffle` は切る or 同じバッチを使い続ける）。
+ 同じバッチで*何百ステップ*も学習を回す。
+ *loss がほぼ 0 近くまで下がれば合格*。下がらなければバグを探す。

上の線形回帰の例も、実は 128 サンプルを固定して回しているので「1 バッチ過学習」に近い形でした。
loss が小さく下がりましたね。これが「配線は正しい」というシグナルです。

#readcode("tests/test_overfit_tiny_batch.py", target: "test_mse_overfits_one_batch")[
  本書の「健全性テスト」の実体。`TinyVLA` を 1 バッチ（16 サンプル）で 200 ステップ学習し、
  *loss が最初の 20% 未満まで下がること* (`loss.item() < 0.2 * first`) を `assert` します。
  `next(iter(DataLoader(ds, batch_size=16, shuffle=True)))` で 1 バッチを取り出す書き方（次節）も
  ここで使われています。各章の演習でも必ず 1 問、この確認を入れます。
]

うまく下がらないときのチェックリスト:
- `optimizer.zero_grad()` を呼んでいるか（最頻出バグ）。
- 学習率 `lr` が極端でないか（大きすぎると発散、小さすぎると動かない。まず `1e-3` 付近から）。
- 予測と正解の *shape が一致*しているか（ズレていると broadcast で意図しない損失になる）。
- モデルの出力に `model.parameters()` が勾配でつながっているか（`detach()` / `no_grad` で
  切っていないか）。
- 入力・モデルが同じ *device / dtype* か。

== Dataset と DataLoader

データが増えると、「どう 1 件を取り出すか」と「どうミニバッチにまとめるか」を分けて書きたく
なります。PyTorch では *`Dataset`* が前者、*`DataLoader`* が後者を担当します。

=== Dataset: `__len__` と `__getitem__`

`Dataset` は 2 つのメソッドさえ実装すればよい、というルールです。

- `__len__(self)` … データ件数を返す。
- `__getitem__(self, idx)` … `idx` 番目の 1 件を返す（dict やタプル）。

本書の題材に寄せた最小例（状態 $[3]$ → 行動チャンク $[8, 3]$ の組を返す擬似データセット）:

```python
import torch
from torch.utils.data import Dataset

class ToyVLADataset(Dataset):
    def __init__(self, n=100, chunk_len=8, action_dim=3):
        torch.manual_seed(0)
        self.states  = torch.randn(n, 3)                       # [N, 3]
        self.actions = torch.randn(n, chunk_len, action_dim)   # [N, 8, 3]

    def __len__(self):
        return self.states.shape[0]                            # 件数

    def __getitem__(self, idx):
        return {
            "state":  self.states[idx],                        # [3]
            "action": self.actions[idx],                       # [8, 3]
        }

ds = ToyVLADataset()
print(len(ds))                       # 100
sample = ds[0]
print(sample["state"].shape, sample["action"].shape)   # torch.Size([3]) torch.Size([8, 3])
```

#readcode("src/vla_learn/datasets/synthetic_dataset.py", target: "SyntheticVLADataset.__getitem__")[
  実物の `SyntheticVLADataset` も同じ形で、`__getitem__` が
  `{"image":[3,64,64], "state":[3], "tokens":[L], "action":[8,3], "pad_mask":[8]}` を返します
  （型は画像・状態・行動・pad_mask が float32、tokens が int64）。この章の `ToyVLADataset` を
  「本物の観測」に差し替えたものが M3 の到達点です。
]

=== DataLoader: バッチ化と shuffle

`DataLoader` は `Dataset` を包み、*ミニバッチにまとめて (`batch_size`)*、*毎エポック順番を
シャッフル (`shuffle=True`)* して、繰り返し取り出せるようにします。1 件が $[3]$ でも、
`batch_size=16` で取り出すと先頭に*バッチ次元が付いて* $[16, 3]$ になります。

```python
import torch
from torch.utils.data import DataLoader

loader = DataLoader(ds, batch_size=16, shuffle=True)

batch = next(iter(loader))           # 最初のバッチを 1 つ取り出す
print(batch["state"].shape)          # torch.Size([16, 3])   ← 先頭に B=16 が付く
print(batch["action"].shape)         # torch.Size([16, 8, 3])

for batch in loader:                 # 学習ではこう回す（1 エポック = 全データを一巡）
    s = batch["state"]               # [16, 3]
    a = batch["action"]              # [16, 8, 3]
    # ここで model(...) と loss.backward() ...
    pass
```

ポイント:
- *`batch_size`*: 一度に何件まとめるか。大きいほど勾配が安定し速いが、メモリを食う。
- *`shuffle=True`*: 毎エポック並び替える。順序の偏りで学習がゆがむのを防ぐ。*学習データは
  True、評価データは False* が定石。
- *辞書もまとめてくれる*: `__getitem__` が dict を返すと、DataLoader が*キーごとにスタック*して
  `{"state": [B,3], "action": [B,8,3]}` にしてくれます（賢い）。

#note[
  *「1 バッチ過学習」との合わせ技*: `batch = next(iter(loader))` で 1 バッチだけ取り出し、それを
  使い回して数百ステップ回せば、前節の健全性テストがそのまま書けます。本書の
  `tests/test_overfit_tiny_batch.py` もまさに
  `next(iter(DataLoader(ds, batch_size=16, shuffle=True)))` で 1 バッチを取り出しています。
]

=== 学習ループ + DataLoader（まとめ）

これまでの部品を全部つなぐと、本書で何度も書く形になります。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.manual_seed(0)
ds = ToyVLADataset(n=256)                              # 上で定義した擬似データ
loader = DataLoader(ds, batch_size=32, shuffle=True)

# 状態[3] -> 行動チャンク[8,3] を出すごく小さなモデル
model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 8 * 3))
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    running = 0.0
    for batch in loader:
        s = batch["state"]                            # [B, 3]
        a = batch["action"]                           # [B, 8, 3]
        pred = model(s).view(-1, 8, 3)                # [B, 24] -> [B, 8, 3]
        loss = loss_fn(pred, a)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item()
    print(f"epoch {epoch}  mean loss = {running / len(loader):.4f}")
```

ここでは答え `a` がランダムなので「賢く当てる」のは無理ですが、配線
（`Dataset → DataLoader → model → loss → backward → step`）が一巡することを確認できます。
本物のデータ（M3 以降）に差し替えれば、そのまま VLA の学習になります。

#summary[
  - *tensor*: 多次元配列。`shape`・`dtype`・`device` を読めることが最重要。float と int、
    CPU と GPU は混ぜない。
  - *broadcast*: 末尾の次元から見て「等しい or 片方が 1」なら自動で形が合う。`unsqueeze(-1)` で
    1 を作って掛けるのは頻出パターン（`masked_mse`）。
  - *autograd*: `requires_grad=True` → 計算 → `loss.backward()` で `.grad` に勾配が貯まる。
    推論は `no_grad`。`.grad` は加算されるので `zero_grad` が必要。
  - *nn.Module / nn.Linear / nn.Sequential*: モデルを定義しパラメータを自動管理。呼び出しは
    `model(x)`。`super().__init__()` を忘れない。
  - *学習ループ*: `pred → loss → zero_grad → backward → step` の 4 つ。`Adam` に
    `model.parameters()` を渡す。記録は `.item()`。
  - *1 バッチ過学習*: 学習コードを書いたら最初に確認する鉄則。配線の健全性テスト。
  - *Dataset / DataLoader*: `__len__` と `__getitem__` で 1 件を定義 → `DataLoader` でバッチ化・
    `shuffle`。バッチ化で先頭に $B$ 次元が付く。
  - 道具がそろいました。次章 M2 で「状態 → 行動」「画像 → 行動」をこの学習ループで回帰し、
    *なぜ素朴な模倣はロールアウトで崩れるのか*（行動チャンクが必要になる理由）を体験します。
]
