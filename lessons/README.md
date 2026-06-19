# 章インデックス（M0 → M6）

この教材は、小さな VLA（Vision-Language-Action）を **PyTorch でスクラッチ自作**し、最後に
flow matching 版まで作るハンズオンです。下の表の順に進めてください。各章の演習は
[`../exercises/`](../exercises/)、解答は [`../solutions/`](../solutions/) にあります。

教材全体の方針や「なぜこの順番か」は [`../docs/curriculum.md`](../docs/curriculum.md)、
用語の定義は [`../docs/glossary.md`](../docs/glossary.md) を参照してください。

## 章一覧

| 章 | テーマ | 一言 | 所要時間の目安 |
|----|--------|------|----------------|
| [M0](m0_overview.md) | 全体像とセットアップ | VLA とは何か、action / state / episode / chunk、完成物デモ、環境構築 | 30〜45 分 |
| [M1](m1_pytorch.md) | PyTorch 速習 | tensor・autograd・`nn.Module`・学習ループ・Dataset/DataLoader。**ここが土台** | 90〜120 分 |
| [M2](m2_imitation.md) | 最小の模倣学習 | 状態→行動・画像→行動の回帰、**なぜ素朴な模倣は閉ループで崩れるか** | 60〜90 分 |
| [M3](m3_data_actions.md) | 行動表現とデータ | 正規化・時間窓・**行動チャンク**・LeRobot 風データ辞書 | 60〜90 分 |
| [M4](m4_tiny_vla_mse.md) | 最小 VLA をスクラッチ | 画像 + 言語 + 状態 → 行動チャンク（MSE 版 `TinyVLA`）を自作・学習・評価 | 90〜150 分 |
| [M5](m5_flow_matching.md) | flow matching 化 | 行動ヘッドを rectified flow（`FlowVLA`）へ。座学の拡散/フローを実装で回収 | 90〜150 分 |
| [M6](m6_lerobot_and_models.md) | LeRobot と有名 VLA | 自作データの LeRobot export / SmolVLA 精読 + π0・GR00T・OpenVLA・MolmoAct 概観 / 卒業課題 | 90〜120 分 |

> 所要時間は目安です。コードを写経して実際に動かすか、読むだけかで大きく変わります。
> CPU だけで完結しますが、学習（M4・M5）は数分かかります。

## 学習順序のすすめ

- **基本は M0 から順番に。** 各章が前章の部品を前提にしています（特に M1 → M2 → M3 → M4 → M5 は積み上げ式）。
- **PyTorch が初めての方は M1 を飛ばさないでください。** tensor・autograd・学習ループ・Dataset/DataLoader を
  ここで固めておかないと、M4 以降のコードが読めなくなります。
- **diffusion / flow の座学がある方も、M5 は「概念の再講義」ではなく「実装への接続」**です。M4（MSE 版）を
  動かしてから M5 に進むと、差分（行動ヘッドの差し替え）が明確になります。
- 各章の章末「次の章へ」に従えば、迷わず進めます。

## つまずいたら戻る先

| 症状・つまずき | まず戻る章 |
|---|---|
| tensor の形 `[B, C, H, W]` / `[B, T, A]` が読めない、autograd や `.backward()` が曖昧 | [M1](m1_pytorch.md) |
| 学習ループ・`optimizer.zero_grad()` / `.step()`・`.train()` / `.eval()` が曖昧 | [M1](m1_pytorch.md) |
| Dataset / DataLoader の役割、バッチがどう作られるか分からない | [M1](m1_pytorch.md) |
| 「損失は下がるのに動かすと失敗する」が腑に落ちない（distribution shift / 誤差蓄積） | [M2](m2_imitation.md) |
| 正規化・逆正規化、行動チャンク、`pad_mask` の意味が曖昧 | [M3](m3_data_actions.md) |
| 画像エンコーダの空間情報、言語エンコーダの語順・grounding でつまずく | [M4](m4_tiny_vla_mse.md) |
| 速度場・rectified flow・Euler 積分の実装が追えない | [M5](m5_flow_matching.md) |
| 用語そのものが分からない | [用語集](../docs/glossary.md) |

困ったら、まず**1 バッチに過学習できるか**（各章の演習に必ずあります）を確認すると、
学習機構のどこが壊れているか切り分けられます。

## 関連リンク

- 演習の歩き方 → [`../exercises/README.md`](../exercises/README.md)
- 解答の使い方 → [`../solutions/README.md`](../solutions/README.md)
- 用語集 → [`../docs/glossary.md`](../docs/glossary.md)
- カリキュラム設計のねらい → [`../docs/curriculum.md`](../docs/curriculum.md)
- 実装本体 → [`../src/vla_learn/`](../src/vla_learn/)
