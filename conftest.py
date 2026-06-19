"""pytest 用の設定。src/ をインポートパスに追加する。

`pip install -e .` をしていなくても `pytest` が動くようにするための保険です。
"""
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
