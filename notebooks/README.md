# notebooks — Jupyter で解く（練習場・演習・解答）

**演習は Jupyter で解くのがいちばんスムーズ**です。問題と答えも、そのまま開いて使える
ノートブックを用意してあります（`.md` の演習・解答から自動生成。正本は `.md` の方です）。

## 起動

```bash
uv sync --extra notebook        # 初回のみ（jupyterlab/ipykernel/matplotlib）
uv run jupyter lab              # ブラウザで開く
```

カーネルは `.venv` の Python。`import vla_learn` がそのまま通ります（editable 導入済み）。

## 3 種類のノートブック

| 種類 | 置き場所 | 使い方 |
|------|----------|--------|
| **練習場** | [`playground.ipynb`](playground.ipynb) | 環境表示→VLA forward→1バッチ過学習 が動く土台。自由実験用 |
| **演習（問題）** | [`exercises/mX.ipynb`](exercises/) | **あなたが解く作業ノート**。問題の説明セル＋答えを書く空コードセル |
| **解答** | [`solutions/mX.ipynb`](solutions/) | **実行できる参考解答**（上から実行で確認できる）＋「なぜそうなるか」 |

> ノートが古い/作り直したいときは `uv run --extra notebook python scripts/make_exercise_notebooks.py` で再生成できます
> （正本の `exercises/*.md` `solutions/*.md` から作ります）。

## おすすめの進め方（例: M1）

1. `lessons/m1_pytorch.md`（または PDF の M1）を読む。
2. **`notebooks/exercises/m1.ipynb` を開く**。先頭のセットアップセルを実行 → 各問の空セルに答えを書いて実行。
   - これは *あなたのローカルコピー* なので、自由に書き換えて構いません。
3. 詰まったら同じ章の lesson に戻る。
4. **答え合わせは `notebooks/solutions/m1.ipynb`**（実行して結果を見比べる）か、`solutions/m1/README.md`（解説）。

`____` 穴埋めや「1 バッチに過学習できるか」も、その場で実行して loss の下がり方を目で確認できます。

## 完全に動く一括スクリプトもあります

抜粋ではなく「最初から最後まで通る」コードを動かしたいとき:

- M1: [`../solutions/m1/solution.py`](../solutions/m1/solution.py) … ノート内で `%load ../solutions/m1/solution.py` でも読めます。
- M6 卒業課題: [`../exercises/m6/ablation.py`](../exercises/m6/ablation.py)（3勘所アブレーション）、
  [`../exercises/m6/discrete_head.py`](../exercises/m6/discrete_head.py)（OpenVLA 流の離散ヘッド）。

```bash
uv run python solutions/m1/solution.py     # 全チェックが OK と出れば正解
```

## Jupyter を使わない場合

プレーンな `.py` でも構いません。`exercises/mX/` の `.md` を見ながら自分の `.py` に書き、
`uv run python あなたのファイル.py`（または `PYTHONPATH=src python ...`）で実行します。

## メモ

- `.ipynb_checkpoints/` は `.gitignore` 済み。気にしなくて OK。
- 図を使うセルは `matplotlib`（`--extra notebook` に同梱）。
