"""演習・解答の Markdown から、そのまま解ける Jupyter ノートブックを生成する。

ねらい: 「Jupyter を使うとき、演習問題と答えはどこに？」を解消する。
  exercises/mX/README.md  → notebooks/exercises/mX.ipynb（問題＝説明セル + 解答を書く空コードセル）
  solutions/mX/README.md  → notebooks/solutions/mX.ipynb（解答＝説明セル + 実行できるコードセル）

Markdown の ```python ブロックはコードセルに、それ以外（散文・```bash・表）はマークダウンセルにします。
canonical（正本）は Markdown の方です。問題文を直したら、このスクリプトで再生成してください。

使い方:
  uv run --extra notebook python scripts/make_exercise_notebooks.py
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ["m1", "m2", "m3", "m4", "m5", "m6"]

# 各ノートの先頭に置く共通セットアップ（vla_learn を確実に import）
SETUP = '''# --- セットアップ: vla_learn を import できるようにする（uv の editable 導入があれば自動で通る）---
import sys, pathlib
try:
    import vla_learn
except ModuleNotFoundError:
    for _d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:
        if (_d / "src" / "vla_learn").exists():
            sys.path.insert(0, str(_d / "src")); break
    import vla_learn
import numpy as np
import torch
print("vla_learn", vla_learn.__version__, "| torch", torch.__version__)'''

_FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def md_to_cells(text: str) -> list:
    """Markdown を nbformat セル列へ。```python はコードセル、他はマークダウンセル。"""
    cells = []
    pos = 0
    buf: list[str] = []

    def flush_md():
        s = "".join(buf).strip("\n")
        if s.strip():
            cells.append(nbf.v4.new_markdown_cell(s))
        buf.clear()

    for m in _FENCE.finditer(text):
        lang = (m.group(1) or "").lower()
        code = m.group(2).rstrip("\n")
        if lang in ("python", "py"):
            buf.append(text[pos:m.start()])
            flush_md()
            cells.append(nbf.v4.new_code_cell(code))
        else:
            # python 以外のフェンスはマークダウンに残す（そのまま表示）
            buf.append(text[pos:m.end()])
        pos = m.end()
    buf.append(text[pos:])
    flush_md()
    return cells


def build_exercise_nb(ch: str) -> nbf.NotebookNode | None:
    src = ROOT / "exercises" / ch / "README.md"
    if not src.exists():
        return None
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            f"# 演習 {ch.upper()}（あなたの作業用ノート）\n\n"
            f"- これは **あなたが解くための作業ノート** です。空のコードセルに答えを書いて実行してください（自由に書き換え OK）。\n"
            f"- 問題の全文・図は [`../../lessons/{ch}_*.md`](../../lessons/) と "
            f"[`../../exercises/{ch}/README.md`](../../exercises/{ch}/README.md) にもあります。\n"
            f"- 答え合わせは [`solutions/{ch}.ipynb`](../solutions/{ch}.ipynb)（実行できる解答）か "
            f"[`../../solutions/{ch}/README.md`](../../solutions/{ch}/README.md)。\n"
            f"- まず下のセットアップセルを実行してから始めましょう。"
        ),
        nbf.v4.new_code_cell(SETUP),
    ]
    cells += md_to_cells(src.read_text(encoding="utf-8"))
    # 末尾に自由作業セル
    cells.append(nbf.v4.new_markdown_cell("---\n### 自由に試す\n下のセルで自由に実験してください。"))
    cells.append(nbf.v4.new_code_cell("# ここに書く\n"))
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    return nb


def build_solution_nb(ch: str) -> nbf.NotebookNode | None:
    src = ROOT / "solutions" / ch / "README.md"
    if not src.exists():
        return None
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            f"# 解答 {ch.upper()}（参考・実行できる版）\n\n"
            f"- 各コードセルは解答の抜粋です。**上から順に実行**すれば多くは動きますが、抜粋ゆえ前のセルや"
            f"演習ノートの文脈を前提にする場合があります（その時は必要な行を補ってください）。\n"
            f"- 完全に一括で動く例があるものは: `solutions/m1/solution.py`（`%load solutions/m1/solution.py` で読み込める）、"
            f"`exercises/m6/ablation.py` / `discrete_head.py`。\n"
            f"- 解説の正本は [`../../solutions/{ch}/README.md`](../../solutions/{ch}/README.md)。"
        ),
        nbf.v4.new_code_cell(SETUP),
    ]
    cells += md_to_cells(src.read_text(encoding="utf-8"))
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    return nb


def main() -> None:
    (ROOT / "notebooks" / "exercises").mkdir(parents=True, exist_ok=True)
    (ROOT / "notebooks" / "solutions").mkdir(parents=True, exist_ok=True)
    n = 0
    for ch in CHAPTERS:
        enb = build_exercise_nb(ch)
        if enb is not None:
            out = ROOT / "notebooks" / "exercises" / f"{ch}.ipynb"
            nbf.write(enb, str(out)); n += 1
            print(f"[nb] {out.relative_to(ROOT)}")
        snb = build_solution_nb(ch)
        if snb is not None:
            out = ROOT / "notebooks" / "solutions" / f"{ch}.ipynb"
            nbf.write(snb, str(out)); n += 1
            print(f"[nb] {out.relative_to(ROOT)}")
    print(f"done. {n} notebooks generated.")


if __name__ == "__main__":
    main()
