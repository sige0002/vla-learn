# vla_learn

日本語の PyTorch ハンズオン教材。Tiny Tabletop 2D を題材に、小さな VLA を CPU で実装・学習・評価する。

- 学習の入口は [README.md](README.md)。章は `lessons/`、演習・解答は `exercises/` / `solutions/`、共通実装は `src/`、設定は `configs/`、検証は `tests/`。
- Python 3.10+、環境管理は `uv`。依存と CPU PyTorch の選択は [pyproject.toml](pyproject.toml) が正本。教材の修正を理由に GPU や実機を必須にしない。
- 教材変更では対象章の説明・演習・解答と、実行されるコードを一致させる。解答を演習側へ無条件に埋め込まない。
- 対象の検証は `uv run pytest tests/<対象> -q`、全体の回帰は `uv run pytest -q`。説明だけの変更では例とリンクを確認し、学習を再実行しない。
- モデル構造や学習処理を変える場合は短い実行から確認し、既存の `checkpoints/` と測定結果を保持する。長時間学習・外部モデル取得は依頼範囲に含まれるときに行う。
