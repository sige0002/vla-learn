"""実行デバイス（CPU / GPU）の選択ヘルパ。

この教材は CPU だけで完結しますが、GPU があれば自動で使えるようにしておきます。
「テンソルとモデルは同じ device に置く」というのが PyTorch の鉄則です（M1 で学びます）。
"""
from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """利用可能な最良の device を返す。

    Args:
        prefer: "cpu" / "cuda" を明示したい場合に指定。None なら自動選択。
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
