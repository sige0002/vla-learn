# notebooks — あなたの練習場（Jupyter）

**演習や実験のコードは、基本ここ（Jupyter ノートブック）に書くのがおすすめ**です。
テンソルの shape や画像、loss の変化がその場で見えるので、手を動かして学ぶのに向いています。

## 使い方

```bash
# リポジトリのルートで（初回のみ）notebook 用の依存を入れる
uv sync --extra notebook

# JupyterLab を起動（ブラウザが開く）
uv run jupyter lab
```

カーネルは `.venv` の Python なので、`import vla_learn` がそのまま通ります（editable 導入済み）。

## まずは playground から

- [`playground.ipynb`](playground.ipynb) … 環境の表示 → 小さな VLA を forward → 「1 バッチ過学習」までの
  最小例が動くテンプレートです。上から実行すれば、教材の部品の使い方が一通りつかめます。
- **章ごとにコピーして使う**と整理しやすいです（例: `playground.ipynb` を複製して `m1.ipynb`,
  `m4_experiment.ipynb` …）。`exercises/mX/README.md` の問題をそのノートで解いていきましょう。

## 進め方の例

1. `exercises/m1/README.md` を開く。
2. このフォルダに `m1.ipynb` を作る（`playground.ipynb` をコピー）。
3. 問題ごとにセルを足して解く。`____` 穴埋めや「1 バッチ過学習」をその場で実行して確認。
4. 詰まったら `lessons/m1_pytorch.md`、答え合わせは `solutions/m1/`。

## Jupyter を使いたくない場合

プレーンな `.py` でも構いません。`exercises/mX/` の雛形 `.py`（例 [`../exercises/m1/starter.py`](../exercises/m1/starter.py)）を
コピーして埋め、次のように実行します:

```bash
uv run python my_practice.py        # uv 経由（推奨）
# もしくは
PYTHONPATH=src python my_practice.py
```

## メモ

- あなたが作ったノートブック（`m1.ipynb` など）はあなたのものです。コミットしてもしなくても自由です。
- Jupyter の自動保存フォルダ `.ipynb_checkpoints/` は `.gitignore` 済みなので無視して構いません。
- 図を使うセルは `matplotlib` を使います（`--extra notebook` に含まれています）。
