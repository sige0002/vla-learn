# カリキュラム設計のねらい（指導者・自習者向け）

このドキュメントは、教材全体の**設計思想**と**各章の役割**をまとめたものです。
「なぜこの順番なのか」「どこを自作し、どこを既存ライブラリに任せるのか」を理解しておくと、
迷ったときに立ち返れます。

## 基本方針

1. **最初から本物の VLA を動かさない。** まず小さい VLA を自分で作り、その後で
   SmolVLA や π0 を「同じ部品が大規模化したもの」として読む。これが最短の理解ルートです。
2. **到達点（Capstone）を 1 つに固定する。** あれこれ手を広げず、
   「Tiny Tabletop 2D Pick-and-Place VLA」を完成させることに集中します。
3. **実機・物理エンジン・GPU は不要。** 合成データと小さなモデルで、ノート PC の CPU でも回ります。
4. **まず動かす → 理解する。** コードはコピペで動く完結性を重視します。
5. **既習の座学（diffusion / VLM）を実装で回収する。** 概念の再講義はせず、実装への接続に注力します。

## from scratch と LeRobot の境界線

| 自作する（理解の核心） | LeRobot に任せる（規格・運用） |
|---|---|
| Dataset / DataLoader、正規化、時間窓、行動チャンク | LeRobotDataset 形式の理解 |
| 画像 / 言語 / 状態エンコーダ、方策 forward | 自作データの LeRobot 形式 export |
| 損失、学習ループ、rollout 評価 | 既存ロボットデータセットの読み込み |
| flow matching ヘッド | SmolVLA など既存 policy の構成読解・比較 |

「理解すべきものは自作、実務で再利用すべきものは LeRobot」。最初から `lerobot-train` に
全面依存したり、巨大な policy を写経したりはしません。

## 各章の役割

- **M0 全体像**: VLA とは何か、action / state / episode / chunk、完成物のデモ、環境構築。
- **M1 PyTorch 速習**: tensor・autograd・`nn.Module`・学習ループ・Dataset/DataLoader。**ここが土台**。
- **M2 最小の模倣学習**: 「状態→行動」「画像→行動」の回帰。**素朴な模倣がなぜ閉ループで崩れるか**
  （distribution shift / 誤差蓄積）と、ノイズ注入による対策を体験。
- **M3 行動表現とデータ**: 正規化、時間窓、**行動チャンク**、LeRobot 風データ辞書。
- **M4 最小 VLA をスクラッチ**: 画像 + 言語 + 状態 → 行動チャンク（MSE 版）。
  - 重要教訓 1: 画像エンコーダで **空間情報を保持**しないと「場所へ向かう」を学べない。
  - 重要教訓 2: 言語エンコーダは **語順を区別**できないと「どの色を運ぶか」を学べない。
- **M5 flow matching 化**: 行動ヘッドを rectified flow へ。**既習の拡散/フロー座学を実装で回収**。
- **M6 LeRobot と有名 VLA**: 自作データの export、SmolVLA 精読、π0 / GR00T / OpenVLA / MolmoAct 概観、卒業課題。

## 模擬問題の設計

- 固定型: **shape 確認 → 穴埋め → バグ修正 → 小実装 → 実験**。1 問 1 概念。
- 各章で必ず「**1 バッチに過学習できるか**」を確認（学習機構のデバッグの鉄則）。
- 難易度比率の目安: 誘導 60% / 小実装 25% / 自由実験 15%。
- 解答は「正解コード」＋「**なぜその shape か / なぜ loss が下がるか**」。

## 有名 VLA の扱い

- **精読**: SmolVLA（LeRobot 直結、小型 OSS、VLM + 状態トークン + action expert + flow matching +
  action chunking が一通り入っており、自作 VLA と対応づけやすい）。
- **概観**: OpenVLA（離散トークン自己回帰）、π0 / π0.5（flow matching）、
  GR00T N1（dual-system humanoid）、MolmoAct（行動推論）。
- 各 VLA は「入力 → 表現/トークン化 → 行動ヘッド（離散 or flow/diffusion）→ 学習」の軸で、
  自作 `TinyVLA` / `FlowVLA` との**対応表**として読みます。

> 設計方針は Claude（Anthropic）と OpenAI Codex CLI の議論を踏まえて確定しました。
