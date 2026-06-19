# 小さな VLA を自作して学ぶ（日本語・PyTorch 入門教材）

> Vision-Language-Action (VLA) を「読む」だけでなく「作って」理解するための、ハンズオン教材です。
> **実機なし・物理エンジンなし・CPU だけ**で、小さな VLA をスクラッチ実装し、最終的に
> **flow matching** 版（pi0 / SmolVLA と同じ系統）まで自作します。

## 🎯 この教材のゴール

最終的にあなたは、次の「小さな VLA」を **PyTorch で自作・学習・評価**できるようになります。

- タスク（Capstone）: **Tiny Tabletop 2D Pick-and-Place** — 2D 平面で
  「**青のブロックを青のゴールに置いて**」のような**言語指示**に従い、**画像**を見て
  対象を運ぶ。
- モデル: 画像 + 言語 + 状態 → **行動チャンク**（8 ステップ分の行動）を出力する VLA。
  - M4: MSE（回帰）版 `TinyVLA`（約 0.42M パラメータ）
  - M5: **flow matching** 版 `FlowVLA`（あなたが知っている拡散/フローの座学を実装で回収）

### 実測の目安（CPU・本リポジトリの既定設定）

| 指標 | 目安 |
|------|------|
| 解析エキスパート（お手本）の成功率 | 約 100% |
| M4 MSE 版 `TinyVLA`（約 0.42M params）の閉ループ成功率 | **およそ 7〜8 割**（ある検証で 0.76） |
| M5 flow 版 `FlowVLA`（約 0.58M params）の閉ループ成功率 | **ほぼ 10 割**（ある検証で 1.00、flow_steps=5/10） |
| 学習時間（CPU, 1500 エピソード × 30〜35 エポック） | 数分程度 |

> 数値は乱数・PC 差でぶれます（参考値）。大事なのは「学習で loss が下がり、閉ループ成功率が上がる」傾向、そして
> **同じ部品（FiLM 付き backbone）のまま行動ヘッドを MSE → flow matching に差し替えられる**こと
> （perception と行動生成の分離）を体感することです。
> なお本書の toy タスクは単峰（正解がほぼ一意）なので、ここで flow が同等以上に動くのは主に経験的な安定性です。
> π0 / SmolVLA など実用 VLA で flow（生成的な行動ヘッド）が効くのは、**多峰性・連続制御・行動チャンク生成**が
> 効く場面で、その差は本書の単峰タスクより大きく出ます（M5 で詳説）。
> M4 では、**画像の空間情報の保持**・**言語の語順の区別**・**FiLM による言語条件付け**の 3 つが
> そろって初めて grounding（どの色を運ぶか）が機能する、という設計上の勘所も体験します。

## 👤 想定読者

- VLA は初心者。プログラミングは「関数が書ける」程度。
- **PyTorch は初心者**（テンソル・自動微分・学習ループからやさしく説明します）。
- **diffusion / VLM の座学（数式・概念）は知っている**。本教材はその知識を「実装」へ橋渡しします。

## 🗺️ 学習マップ（M0 → M6）

| 章 | テーマ | 主に学ぶこと |
|----|--------|--------------|
| [M0](lessons/m0_overview.md) | 全体像とセットアップ | VLA とは / action・state・episode・chunk / 完成物デモ / 環境構築 |
| [M1](lessons/m1_pytorch.md) | PyTorch 速習 | tensor・autograd・`nn.Module`・学習ループ・Dataset/DataLoader |
| [M2](lessons/m2_imitation.md) | 最小の模倣学習 | 状態→行動・画像→行動の回帰、**なぜ素朴な模倣は崩れるか** |
| [M3](lessons/m3_data_actions.md) | 行動表現とデータ | 正規化・時間窓・**行動チャンク**・LeRobot 風データ辞書 |
| [M4](lessons/m4_tiny_vla_mse.md) | 最小 VLA をスクラッチ | 画像+言語+状態→行動チャンク（MSE 版）を自作して学習・評価 |
| [M5](lessons/m5_flow_matching.md) | flow matching 化 | 行動ヘッドを生成モデル（rectified flow）へ。座学の実装回収 |
| [M6](lessons/m6_lerobot_and_models.md) | LeRobot と有名 VLA | 自作データの LeRobot export / SmolVLA 精読 + π0・GR00T・OpenVLA・MolmoAct 概観 / 卒業課題 |

各章には**模擬問題（演習）**があります → [`exercises/`](exercises/) と解答 [`solutions/`](solutions/)。

## 📖 2 つの版（Markdown と PDF）

| 版 | 用途 | 入口 |
|----|------|------|
| **Markdown**（`lessons/*.md`） | GitHub 上でさっと読む・リンクをたどる | 上の学習マップ |
| **図表入り PDF**（Typst） | 図・数式・「実装を読む」案内つきで腰を据えて読む | [`book/`](book/) を `uv` でビルド |

```bash
# 図表入り PDF をビルド（システム TeX 不要・uv だけ。pip の typst パッケージを使用）
uv run --extra viz  python scripts/make_figures.py    # 図を生成（初回）
uv run --extra book python scripts/build_book.py      # → book/build/book.pdf と章別 mX.pdf
```

> PDF 版は図（環境レンダ・forward パイプライン・行動チャンク・flow 経路・成功率など）と、
> 各章の「次に読むべき実装ファイル（自作 `src/vla_learn/` と有名 VLA の公式 repo）」への案内を
> 加えた enriched 版です。詳細は [`book/README.md`](book/README.md)。

## ⚡ セットアップ（[uv](https://docs.astral.sh/uv/) を使います）

本教材はパッケージ/環境管理に **uv** を使います。`uv sync` 一発で「仮想環境の作成 →
PyTorch（CPU 版）の導入 → 本パッケージの editable インストール」までを自動で行います。

```bash
# 0) uv 未導入なら入れる（既にあればスキップ）
#    macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
#    Windows (PowerShell):  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 1) 依存を同期（.venv 作成・PyTorch CPU 版・vla_learn の editable 導入・pytest まで全自動）
uv sync

# 任意) 可視化(matplotlib)も使うなら extra を足す
uv sync --extra viz

# 任意) 演習を Jupyter で解くなら（推奨）。notebooks/ が練習場になります
uv sync --extra notebook
uv run jupyter lab        # → notebooks/playground.ipynb から
```

> **演習・練習コードはどこに書く?** → **Jupyter ノートブックがおすすめ**です（shape・画像・loss が
> その場で見える）。[`notebooks/playground.ipynb`](notebooks/playground.ipynb) をコピーして章ごとに使ってください。
> 詳しくは [`notebooks/README.md`](notebooks/README.md) / [`exercises/README.md`](exercises/README.md)。

> `uv sync` は `uv.lock` に固定されたバージョンで環境を再現します。PyTorch は CPU 専用 index
> から取得する設定（`pyproject.toml` の `[tool.uv.sources]`）なので、GPU/CUDA は不要です。
> 個別に `pip install` する必要はありません。

動作確認（数十秒）:

```bash
uv run pytest -q                                              # テストが通れば環境 OK
uv run python scripts/train_mse.py --config configs/smoke.json   # 小さく学習が回るか
```

> `uv run <コマンド>` は、必要なら環境を自動同期してから実行します。`.venv` を手で
> activate する必要はありません（したい場合は `source .venv/bin/activate` も可）。

## 🚀 クイックスタート（Capstone を動かす）

```bash
# 1) MSE 版 VLA を学習（CPU で数分）
uv run python scripts/train_mse.py --config configs/m4_mse.json

# 2) 学習した方策を閉ループ評価（成功率を測る）
uv run python scripts/eval_policy.py --ckpt checkpoints/mse/policy.pt --n-episodes 100

# 3) flow matching 版に発展
uv run python scripts/train_flow.py --config configs/m5_flow.json
uv run python scripts/eval_policy.py --ckpt checkpoints/flow/policy.pt --n-episodes 100

# 4) ロールアウトを画像で確認（matplotlib 必要: uv sync --extra viz）
uv run python scripts/demo_rollout.py --ckpt checkpoints/mse/policy.pt --out assets/rollout.png
```

> 数値（成功率など）は乱数・環境差でぶれます。傾向（学習で loss が下がり、成功率が上がる）を見てください。

## 📁 リポジトリ構成

```text
vla_learn/
├── lessons/        # 教材本文（M0〜M6, 日本語 Markdown）
├── exercises/      # 模擬問題（章ごと）
├── solutions/      # 解答と解説
├── src/vla_learn/  # 検証済みの実装（envs / datasets / models / training / evaluation）
├── scripts/        # make_dataset / train_mse / train_flow / eval_policy / export_lerobot / demo_rollout
├── configs/        # 学習設定（m4_mse.json, m5_flow.json, smoke.json）
├── tests/          # pytest（shape・正規化・forward・1バッチ過学習）
└── docs/           # 用語集・カリキュラム概要
```

## 🧩 模擬問題について

各章の演習は「**shape 確認 → 穴埋め → バグ修正 → 小実装 → 実験**」の 5 型で構成。
1 問 1 概念で、毎章「**1 バッチに過学習できるか**」を確認します（学習デバッグの鉄則）。
解答は正解コードに加えて「**なぜその shape か / なぜ loss が下がるか**」を説明します。

## 🔭 有名 VLA とのつながり

本教材で自作する部品は、有名 VLA の縮小版です。M6 で次の対応を学びます。

- **SmolVLA**（精読）: VLM + 状態トークン + action expert + **flow matching** + action chunking。
- **π0 / π0.5**: flow matching による連続行動生成（本教材 M5 の大規模版）。
- **OpenVLA**: 行動を**離散トークン**化して自己回帰生成（M5 のもう一つの選択肢）。
- **GR00T N1**: ヒューマノイド向け dual-system（遅い VLM + 速い行動）。
- **MolmoAct**: 行動の「推論（トレース）」を取り入れた方式。

## 🙏 設計について

本教材の基本方針（章構成・Capstone・from scratch と LeRobot の境界・演習の型）は、
**Claude（Anthropic）と OpenAI Codex CLI を交えて設計方針を議論**したうえで確定しました。

## ライセンス

MIT License（[LICENSE](LICENSE)）。
