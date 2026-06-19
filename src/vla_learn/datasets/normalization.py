"""行動・状態の正規化 (normalization)。

ニューラルネットは「平均 0・分散 1 くらい」の入力/出力で最も学習が安定します。
ここでは各次元ごとに (x - mean) / std で標準化し、逆変換 (x * std + mean) も用意します。

  - 学習: 行動を正規化してから損失を計算（dy などの小さな差分を扱いやすくする）
  - 推論: モデル出力を逆正規化してから環境に渡す（生のワールド座標差分に戻す）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class Normalizer:
    """次元ごとの平均/標準偏差をもつ標準化器。NumPy / Torch 両対応。"""

    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.where(self.std < eps, 1.0, self.std)  # 0 割り防止

    @classmethod
    def fit(cls, data: np.ndarray) -> "Normalizer":
        """data: [N, D]。各列の mean/std を推定。"""
        data = np.asarray(data, dtype=np.float32).reshape(-1, data.shape[-1])
        return cls(data.mean(axis=0), data.std(axis=0))

    def normalize(self, x):
        if isinstance(x, torch.Tensor):
            m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            s = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return (x - m) / s
        return (np.asarray(x, np.float32) - self.mean) / self.std

    def denormalize(self, x):
        if isinstance(x, torch.Tensor):
            m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            s = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return x * s + m
        return np.asarray(x, np.float32) * self.std + self.mean

    # --- 保存 / 読み込み ---
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist()}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(d["mean"], np.float32), np.array(d["std"], np.float32))
