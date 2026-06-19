"""可視化ヘルパ（任意・matplotlib が必要）。

学習曲線や rollout の様子を絵にして理解を助けます。matplotlib は必須ではないので、
import はこのモジュール内（関数内）で行い、無くても他の機能は動くようにしています。
"""
from __future__ import annotations

import numpy as np


def save_image_grid(images: np.ndarray, path: str, ncol: int = 8) -> None:
    """images: [N, 3, H, W]（0..1）を 1 枚のグリッド画像として保存する。"""
    import matplotlib.pyplot as plt  # 関数内 import（任意依存）

    n = len(images)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow))
    axes = np.atleast_2d(axes)
    for i in range(nrow * ncol):
        ax = axes[i // ncol, i % ncol]
        ax.axis("off")
        if i < n:
            ax.imshow(np.transpose(images[i], (1, 2, 0)))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_loss(history: list[float], path: str) -> None:
    """損失の履歴を折れ線グラフにして保存する。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(history)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
