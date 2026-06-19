"""パッケージ内のどこからでも安全に使える純粋関数（他モジュールに依存しない）。

ここに置くことで models と training の間で循環 import が起きないようにしています。
"""
from __future__ import annotations

import torch


def masked_mse(
    pred: torch.Tensor,    # [B, C, A]
    target: torch.Tensor,  # [B, C, A]
    mask: torch.Tensor | None = None,  # [B, C]（1=有効, 0=パディング）
) -> torch.Tensor:
    """パディングを除外した平均二乗誤差。

    pad_mask=0 のステップ（エピソード終端のパディング）を平均から外します。
    """
    se = (pred - target) ** 2  # [B, C, A]
    if mask is None:
        return se.mean()
    mask3 = mask.unsqueeze(-1).expand_as(se)  # [B, C, A]
    return (se * mask3).sum() / mask3.sum().clamp(min=1.0)
