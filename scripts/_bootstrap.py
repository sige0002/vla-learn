"""scripts/ を `pip install -e .` なしでも実行できるように src/ をパスに足す。

各スクリプトの冒頭で `import _bootstrap` するだけで使えます。
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
