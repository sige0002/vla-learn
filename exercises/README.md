# 演習の歩き方（exercises）

各章（[`../lessons/`](../lessons/)）には**模擬問題**があります。手を動かして初めて身につくよう、
1 問 1 概念・短時間で解ける粒度にしてあります。解答は [`../solutions/`](../solutions/) にあります。

## どこに自分のコードを書くか（おすすめ: Jupyter）

この `exercises/mX/README.md` は**問題文**です。**あなたの答え・実験コードは別の場所**に書きます。
shape や画像・loss がその場で見える **Jupyter ノートブックが演習に最適**なので、まずはこちらを推奨します。

```bash
uv sync --extra notebook      # 初回のみ（notebook 用の依存を入れる）
uv run jupyter lab            # 起動 → ブラウザで開く
```

- 練習場のテンプレート → [`../notebooks/playground.ipynb`](../notebooks/playground.ipynb)
  （環境表示 → VLA を forward →「1 バッチ過学習」まで動く最小例）。
- **章ごとにコピー**して使うと整理しやすいです（例: `notebooks/m1.ipynb` を作って M1 の問題を解く）。
- 詳しい起動方法・進め方は [`../notebooks/README.md`](../notebooks/README.md) を参照。

Jupyter を使わない場合は、各 `mX/` の雛形 `.py`（例 [`m1/starter.py`](m1/starter.py)）をコピーして埋め、
`uv run python あなたのファイル.py`（または `PYTHONPATH=src python ...`）で実行しても構いません。

## 固定の 5 型（毎章おおむねこの順）

演習は次の 5 つの型で構成されます。**やさしい確認 → 自分で書く**へと段階的に進みます。

1. **shape 確認** — `[B, C, H, W]`（画像）・`[B, T, A]`（行動チャンク）・`[B, D]`（特徴）などの形を、
   コードを実行する前に手で言い当てます。VLA 実装でいちばん効く基礎体力です。
2. **穴埋め** — 5〜15 行のコードの `____` を埋めます。誘導つきで、書く場所が限定されています。
3. **バグ修正** — わざと壊したコードを直します。よくあるのは device / dtype 不一致、`detach` 漏れ、
   正規化・逆正規化忘れ、`.train()` / `.eval()` 切り替え忘れ、形状ミスなど。
4. **小実装** — 30〜80 行で関数やクラスを自分で書きます（例: 行動チャンク抽出、損失、簡単なエンコーダ）。
5. **実験** — 自由課題。chunk 長を変える、画像にノイズを足す、指示を言い換える、MSE と flow を比べる、など。
   手を動かして「どう変わるか」を観察します。

難易度の配分はおおよそ **誘導 60% / 小実装 25% / 自由実験 15%** です。

## 各章へのリンク

| 章 | 演習 | 対応する本文 |
|----|------|--------------|
| M1 PyTorch 速習 | [`m1/`](m1/) | [`../lessons/m1_pytorch.md`](../lessons/m1_pytorch.md) |
| M2 最小の模倣学習 | [`m2/`](m2/) | [`../lessons/m2_imitation.md`](../lessons/m2_imitation.md) |
| M3 行動表現とデータ | [`m3/`](m3/) | [`../lessons/m3_data_actions.md`](../lessons/m3_data_actions.md) |
| M4 最小 VLA（MSE 版） | [`m4/`](m4/) | [`../lessons/m4_tiny_vla_mse.md`](../lessons/m4_tiny_vla_mse.md) |
| M5 flow matching 化 | [`m5/`](m5/) | [`../lessons/m5_flow_matching.md`](../lessons/m5_flow_matching.md) |

> M0（全体像・セットアップ）と M6（LeRobot・有名 VLA・卒業課題）は読み物と実践が中心です。
> M6 の手を動かす課題は本文の「卒業課題」を参照してください。

各 `mX/` には問題の `README.md` と、必要に応じて穴埋め・小実装用の雛形 `.py` が入っています。

## 答え合わせ（solutions/）の使い方

1. **まず自分で解く。** 詰まっても、すぐ解答を見ないでください。型 1〜3（shape・穴埋め・バグ修正）は
   特に「自分の頭で形を追う」ことが目的です。
2. 解けたら、または 15〜20 分ほど詰まったら、対応する [`../solutions/mX/`](../solutions/) と突き合わせます。
3. 解答には正解コードだけでなく「**なぜその shape になるか / なぜ loss が下がるべきか**」の短い説明が
   付いています。コードが合っていても、この説明と自分の理解がずれていないか確認してください。

解答の詳しい使い方は [`../solutions/README.md`](../solutions/README.md) を参照してください。

## 「1 バッチ過学習」の意義（毎章必須）

各章に必ず「**小さな 1 バッチだけで学習を回し、loss がほぼ 0 まで下がるか**」を確認する課題があります。
これは学習機構が正しく組めているかを切り分ける、最も簡単で強力なデバッグ法です。

- **下がる**: forward → 損失 → `backward` → `optimizer.step()` の配線は概ね正しい、と判断できます。
- **下がらない**: モデルやデータではなく**学習ループ側の不具合**を疑います。よくある原因は
  `optimizer.zero_grad()` 忘れ、勾配が切れている（`detach` / `no_grad` の誤用）、`.train()` になっていない、
  正規化していない、損失の形・マスクの取り違えなど。

つまり「汎化する前に、まず暗記できることを確かめる」ステップです。これが通らないうちに
データ量やエポックを増やしても無駄なので、各章で最初に確認します。

## 関連リンク

- 章インデックス → [`../lessons/README.md`](../lessons/README.md)
- 解答 → [`../solutions/README.md`](../solutions/README.md)
- 用語集 → [`../docs/glossary.md`](../docs/glossary.md)
