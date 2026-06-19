#import "../lib/template.typ": *

= 全体像とセットアップ <m0>

#goal[
  - VLA (Vision-Language-Action) が「何を入力に、何を出力するもの」かを一言で言えるようになる。
  - ロボット模倣学習 (imitation learning) の流れと、本書で繰り返し出てくる用語
    (action / state / episode / 行動チャンク / policy / rollout) を理解する。
  - 本書の完成物 *Tiny Tabletop 2D Pick-and-Place* が何をするタスクか分かる。
  - 開発環境 (uv) を用意し、テストと小さな学習が手元で回ることを確認する。
  - M0 → M6 の地図を持って、次に何を学ぶか分かる。
]

本書は「まず動かす → 理解する」方針です。数式から入るのではなく、小さな VLA を *自分の手で*
作って学習・評価し、最後に SmolVLA / π0 など本物の VLA を「同じ部品の大規模版」として読みます。
PyTorch は知らなくても大丈夫です（次章 M1 でやさしく入門します）。diffusion / VLM の座学
（概念・数式）は既習前提ですが、M0 では使いません。

== VLA とは何か（一文で）

*VLA とは「画像 (Vision) と言語指示 (Language) を入力に取り、ロボットの行動 (Action) を出力する
モデル」* です。
画像分類は「画像 → ラベル」、VLM は「画像 + テキスト → テキスト」でした。VLA はその出力を
*テキストではなく「行動」* に変えたもの、と捉えると分かりやすいです。「ロボットの目と耳と手を
つなぐモデル」だと思ってください。

#fig("/figures/architecture.png", caption: [VLA の骨格。画像・言語・状態の 3 入力をそれぞれ
  エンコードし、言語で視覚を条件付け (FiLM) してから融合し、行動ヘッドで *行動チャンク* を出す。
  本書ではこの図の全部品を自作する。完成形の実装は #raw("src/vla_learn/models/tiny_vla.py") の
  #raw("VLABackbone")。], width: 100%)

#theory[
  既習の VLM（画像とテキストを融合して表現を作る部分）は、VLA の「入力を理解する側」にほぼ
  そのまま流用できます。VLA で新しく要るのは、その表現から *連続値の行動を生成するヘッド* です。
  本書では後半 M5 で、座学の flow matching / 拡散をこの「行動ヘッド」として実装回収します。
]

== ロボット模倣学習 (imitation learning) の流れ

本書は *模倣学習 (imitation learning)*、特に *行動クローニング (behavior cloning)* という最も
基本的な方法を使います。強化学習のように「試行錯誤して報酬を最大化」するのではなく、
*「お手本 (expert) の行動をそっくり真似する」* 教師あり学習です。流れは 4 ステップです。

#table(
  columns: (auto, 1fr),
  [ステップ], [本書での自作内容],
  [1. お手本を集める],
  [ルールベースの *解析エキスパート* (`expert_action`) がタスクを 100% 成功させ、その軌跡を
    記録する。ニューラルネットは使わない。世界の状態を直接読めるので毎回成功する。],
  [2. データセットにする],
  [記録した軌跡を `(画像, 状態, 言語, 行動チャンク)` の組に整形する（M3）。],
  [3. 方策を学習する],
  [VLA に「観測 → お手本の行動」を回帰 (MSE) で真似させる（M4）。さらに flow matching 版
    （M5）に発展させる。],
  [4. 評価する],
  [学習した方策を環境で実際に動かし（*ロールアウト rollout*）、成功率を測る。],
)

#readcode("src/vla_learn/envs/expert.py", target: "expert_action")[
  ステップ 1 の「お手本」を作る心臓部。`if not holding: ブロックへ近づいて掴む / else: ゴールへ
  運んで置く` という単純な状態機械です。20 行ほどなので、最初に読むと「学習が何を真似するのか」が
  具体的につかめます。NN を一切使わず *ワールド状態を直接読む* から 100% 成功する、という点が
  「お手本は賢くてよい（カンニングしてよい）」という模倣学習の発想そのものです。
]

#theory[
  *行動チャンク (action chunking)* は VLA 頻出の重要概念なので直感だけ掴んでおきましょう。
  1 手ずつ予測すると、予測の小さなブレが毎ステップ積み重なって軌道が崩れがちです。そこで
  「今の観測から未来 8 手をまとめて予測 → そのうち数手だけ実行 → また観測して予測」とすると
  軌道が安定します。SmolVLA や π0 など本物の VLA でも標準的に使われます（M3 で実装します）。
]

== タスクの仕様

最終的にあなたが自作・学習・評価できるようになるのは、次の小さな VLA です。

- *タスク*: 2D 平面（テーブルを真上から見た図）で、*言語指示* に従い、*指定色のブロックを
  指定色のゴールへ運ぶ* Pick-and-Place（つかんで置く）。
- *指示は日本語*。例:「青のブロックを青のゴールに置いて」「赤ブロックをつかんで黄ゴールへ」。
- *観測* は画像・状態・言語指示の 3 つ。*行動* は $[d_x, d_y, "grip"]$ の 3 次元で、一度に未来
  8 ステップ分（行動チャンク）を予測します。

#table(
  columns: (auto, auto, 1fr),
  [要素], [形], [意味],
  [image], [$[3, 64, 64]$], [真上から見たテーブルの RGB 画像（値域 $0..1$）],
  [state], [$[3]$], [グリッパの $(a_x, a_y)$ 位置と開閉 grip（固有受容感覚 proprioception）],
  [instruction], [str], [日本語の指示文（例「青のブロックを青のゴールに置いて」）],
  [action], [$[8, 3]$], [未来 8 手の $(d_x, d_y, "grip")$。$d_x, d_y$ は各軸 $plus.minus 0.08$ に
    クリップ、grip は $0.0$=開 / $1.0$=閉（$gt.eq 0.5$ で閉）],
)

#fig("/figures/env_samples.png", caption: [Tiny Tabletop 2D の観測例グリッド。色つきブロック
  （塗り円）を指定色のゴール（リング）へ運ぶ。白がグリッパ（開=リング / 閉=塗り円）。実装は
  #raw("src/vla_learn/envs/render.py") の #raw("render_world") と
  #raw("src/vla_learn/envs/tabletop2d.py")。], width: 100%)

これらの形・単位は #raw("src/vla_learn/constants.py") に一元化されています（`IMG_SIZE=64`,
`ACTION_DIM=3`, `STATE_DIM=3`, `MAX_STEP=0.08`, `SUCCESS_RADIUS=0.12`, `DEFAULT_CHUNK_LEN=8`）。
環境・データ・モデル・学習・評価のあいだで shape がズレないための「唯一の真実の源」です。

#pitfall[
  この環境の「成功」は、*対象ブロックが対象ゴールの半径 (`SUCCESS_RADIUS=0.12`) 内に入った時点*
  で判定します（`Tabletop2DEnv._is_success` は距離だけを見ます）。掴んだまま運べば成功で、実機
  タスクのように「グリッパを開いて*離す*」ことまでは要求しません。物理エンジンも無く当たり判定は
  距離計算だけ——本物の VLA の骨格は保ちつつ、CPU で完結するよう思い切って単純化しています。
]

このタスクの良いところは、*実機もシミュレータ（物理エンジン）も要らず、NumPy だけで世界が完結
する* ことです。画像はルールに従って描画され、当たり判定も単純な距離計算なので、手元の CPU
だけで本物の VLA と同じ骨格（画像・言語・状態を融合 → 行動チャンクを生成）を最後まで体験できます。

#table(
  columns: (auto, auto, 1fr, auto),
  [版], [モデル名], [行動ヘッド], [学ぶ章],
  [MSE（回帰）版], [`TinyVLA`（約 0.42M）], [全結合で行動を直接回帰、損失は MSE], [M4],
  [flow matching 版], [`FlowVLA`（約 0.58M）], [rectified flow で行動を生成（座学の実装回収）], [M5],
)

#note[
  *数値の注意*: 成功率やパラメータ数は乱数・環境差でぶれます。「学習で loss が下がり、成功率が
  上がる」という*傾向*を見てください。目安として解析エキスパート約 100%、`TinyVLA`(MSE) の成功率は
  約 0.76、`FlowVLA` は約 1.0、パラメータ数は `TinyVLA` 約 0.42M・`FlowVLA` 約 0.58M です
  （CPU で数分学習可能）。皆さんの環境で多少前後します。
]

== 環境の入口を読む

まずこの 1 ファイルを読むと、観測・行動・成功の定義が一望できます。

#readcode("src/vla_learn/envs/tabletop2d.py", target: "Tabletop2DEnv.step")[
  環境の `reset` / `step` はここ。`step(action)` がグリッパを動かし（`dx, dy` を `MAX_STEP` に
  クリップ）、掴み判定（最近傍の物体が `GRASP_RADIUS` 以内ならグリッパを閉じて追従）と成功判定
  （`_is_success`）を行い、`(obs, reward, done, info)` を返します。Gym 風の最小 API です。
]

```python
from vla_learn.envs import Tabletop2DEnv

env = Tabletop2DEnv(seed=0)
obs = env.reset()
print(obs["instruction"])        # 例: 青のブロックを青のゴールに置いて
print(obs["image"].shape)        # (3, 64, 64)
print(obs["state"])              # [ax, ay, gripper]  float32

# お手本(エキスパート)で 1 手進めてみる
from vla_learn.envs import expert_action
action = expert_action(obs["world"])             # [dx, dy, grip]
obs, reward, done, info = env.step(action)
print(action, reward, done, info["success"])
```

#pitfall[
  テンソルとモデルは同じ `device` に置く、画像は `[B, 3, 64, 64]`・行動チャンクは `[B, 8, 3]`——
  この「形」と「device」を最初に押さえると以降がぐっと楽になります（M1 で練習します）。
]

== 環境構築（uv）

#note[
  *重要*: この教材は *CPU だけ・実機なし・物理エンジンなし* で完結します。GPU は不要です。
]

パッケージと仮想環境の管理に #link("https://docs.astral.sh/uv/")[uv] を使います。uv は「速い・
1 ツールで完結・`uv.lock` で全員が同じバージョンを再現できる」のが利点で、`pip` と `venv` を
別々に覚えなくても `uv sync` 一発で環境がそろいます。

```bash
# 0) uv を入れる（未導入なら）。入っていればスキップ。
#    macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
#    Windows (PowerShell):
#    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 1) リポジトリのルートに移動
cd vla_learn

# 2) 依存を同期する。これ 1 つで
#    「.venv の作成 → PyTorch(CPU版) の導入 → vla_learn の editable 導入 → pytest 導入」
#    まで全部やってくれる。
uv sync

# 任意) 可視化(matplotlib) も使いたい場合は extra を足す
uv sync --extra viz
```

#note[
  *`uv sync` は何をしている?* カレントの `pyproject.toml` と `uv.lock` を読み、ロックされた
  バージョンで `.venv` を作り直します。本パッケージ `vla_learn` は *editable* で導入されるので、
  どこからでも `import vla_learn` が通り、`src/` を書き換えればすぐ反映されます。PyTorch は
  *CPU 専用 index* から入る設定（`pyproject.toml` の `[tool.uv.sources]`）なので、GPU/CUDA は
  不要・余計な手動 `pip install` も不要です。
  *`uv run`* は `uv run python ...` のように使うと、必要なら環境を自動同期してから実行します
  （`.venv` を手で activate しなくてよい）。
]

=== 動作確認 1: テストが通るか（数十秒）

```bash
uv run pytest -q
```

テストには「環境とエキスパート」「データの shape」「正規化」「モデルの forward」「*1 バッチに
過学習できるか*」が含まれます。最後の項目は学習機構が健全かを確かめる鉄則のテストで、本書で
繰り返し登場します（M1 で考え方を導入します）。全部 `passed` と出れば環境は OK です。

#readcode("tests/test_overfit_tiny_batch.py", target: "test_mse_overfits_one_batch")[
  「1 バッチ過学習」テストの実体。`TinyVLA` を 16 サンプルの 1 バッチで 200 ステップ学習し、
  *loss が最初の 20% 未満まで下がること* を `assert` します。これが本書の「健全性テスト」で、
  各章の演習にも必ず 1 問入ります。中身は M1 で全部読み解きます。
]

=== 動作確認 2: 小さく学習が回るか（1 分程度）

```bash
uv run python scripts/train_mse.py --config configs/smoke.json
```

`configs/smoke.json` は *ごく小規模な設定*（60 エピソード・3 epoch）で、「学習ループが最後まで
回るか」だけを確かめるためのものです（成功率を競う設定ではありません）。エラーなく学習が進み、
最後にチェックポイントが `checkpoints/smoke/` に保存されれば、ひととおりの配線（データ生成 →
学習 → 保存）が動いています。本番の学習・評価（`configs/m4_mse.json` など）は M4・M5 で扱います。

#pitfall[
  *困ったとき*
  - `ModuleNotFoundError: No module named 'vla_learn'` → `uv sync` を実行したか、コマンドを
    `uv run` 経由で動かしているか確認。
  - `ModuleNotFoundError: No module named 'torch'` → `uv sync` 未実行か、`uv run` を付け忘れ。
  - `command not found: uv` → uv 自体が未導入。手順 0 のインストールコマンドを実行。
]

== 本書の歩き方とリポジトリ地図

本文・演習・実装は次のように分かれています。

#table(
  columns: (auto, 1fr),
  [ディレクトリ], [中身],
  [`lessons/`], [Markdown の短縮版教材（M0〜M6）。閲覧用。enriched 版はこの Typst 本],
  [`book/`], [この Typst 教科書（`chapters/`・`figures/`・`lib/template.typ`）],
  [`exercises/`], [章ごとの演習（`mX/README.md` と雛形 `.py`）],
  [`solutions/`], [演習の解答と解説],
  [`src/vla_learn/`], [検証済みの実装（`envs` / `datasets` / `models` / `training` / `evaluation`）],
  [`scripts/`], [`make_dataset` / `train_mse` / `train_flow` / `eval_policy` など],
  [`configs/`], [学習設定（`m4_mse.json`, `m5_flow.json`, `smoke.json`）],
  [`tests/`], [pytest（shape・正規化・forward・1 バッチ過学習）],
)

各章は「*本文を読む → 演習 (`exercises/mX/`) を解く → 解答 (`solutions/mX/`) で答え合わせ*」の順で
進めます。演習はどの章も「*shape 確認 → 穴埋め → バグ修正 → 小実装 → 実験*」の 5 型で構成され、
毎章「1 バッチに過学習できるか」を確認します。本書では各章に *実装を読む* 枠（緑）を置き、
「次に開くべき `src/` のファイルと関数」を必ず指します。図も「飾り」ではなく、対応する実装
ファイルへの地図として読んでください（各図キャプションに src パスを書いています）。

== 学習マップ（M0 → M6）

#table(
  columns: (auto, auto, 1fr),
  [章], [テーマ], [主に学ぶこと],
  [*M0*（この章）], [全体像とセットアップ],
  [VLA とは / action・state・episode・chunk / 完成物デモ / 環境構築],
  [M1], [PyTorch 速習], [tensor・autograd・`nn.Module`・学習ループ・Dataset/DataLoader],
  [M2], [最小の模倣学習], [状態 → 行動・画像 → 行動の回帰、なぜ素朴な模倣は崩れるか],
  [M3], [行動表現とデータ], [正規化・時間窓・*行動チャンク*・LeRobot 風データ辞書],
  [M4], [最小 VLA をスクラッチ], [画像 + 言語 + 状態 → 行動チャンク（MSE 版）を自作して学習・評価],
  [M5], [flow matching 化], [行動ヘッドを生成モデル (rectified flow) へ。座学の実装回収],
  [M6], [LeRobot と有名 VLA],
  [自作データの LeRobot export / SmolVLA 精読 + π0・GR00T・OpenVLA・MolmoAct 概観 / 卒業課題],
)

== 用語の整理（最重要）

本書で繰り返し出てくる言葉です。ここで一度そろえておきます。

#table(
  columns: (auto, auto, 1fr),
  [用語], [英語], [本書での意味],
  [行動], [action], [エージェントが 1 ステップで取る出力。ここでは $[d_x, d_y, "grip"]$ の 3 次元],
  [状態], [state],
  [エージェント自身が感じ取れる内部状態（固有受容感覚 proprioception）。ここでは
    $[a_x, a_y, "gripper"]$ の 3 次元],
  [観測], [observation], [モデルへの入力一式。ここでは 画像 + 状態 + 言語指示],
  [エピソード], [episode], [タスク 1 回分の連続した記録（reset から done まで）の時系列],
  [方策], [policy], [「観測を入れると行動を返す」関数そのもの。学習する VLA は方策],
  [行動チャンク], [action chunking],
  [次の 1 手だけでなく未来の数ステップ分の行動をまとめて予測すること。本書では 8 ステップ分],
  [ロールアウト], [rollout], [学習した方策を環境で 1 エピソード分、最後まで動かすこと],
)

#summary[
  - *VLA = 画像 + 言語（+ 状態）→ 行動* を出すモデル。VLM の出力を「行動」に置き換えたもの。
  - 本書は *模倣学習（行動クローニング）*: お手本を集める → データ化 → 回帰で真似 → ロールアウト
    で評価、の 4 ステップ。
  - 完成物は *Tiny Tabletop 2D Pick-and-Place*。CPU・物理エンジン不要で本物の VLA の骨格を自作。
  - 重要用語: action / state / observation / episode / policy / 行動チャンク / rollout。
  - 環境構築は `uv sync` → `uv run pytest` → `configs/smoke.json` で学習が回る確認まで。
  - 次章 M1 で道具（PyTorch）を握ります。tensor・autograd・学習ループ・Dataset/DataLoader を、
    本書の題材（状態 $[3]$ や行動チャンク $[T, 3]$）に寄せて入門します。
]
