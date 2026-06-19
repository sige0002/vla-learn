"""再現性のための乱数シード設定。

機械学習では「毎回同じ結果」が得られると、デバッグや課題の答え合わせが楽になります。
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Python / NumPy / PyTorch の乱数シードをまとめて固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU 教材なので CUDA 側は任意。GPU を使う場合のために一応書いておく。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
