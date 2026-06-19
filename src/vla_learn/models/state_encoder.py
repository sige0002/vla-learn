"""状態エンコーダ（小さな MLP）。グリッパの固有受容状態 [ax, ay, gripper] を符号化。

入力: [B, state_dim] → 出力: [B, out_dim]
本物の VLA でも、ロボットの関節角などの low-dim 状態は MLP で埋め込むのが一般的です。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import STATE_DIM


class StateEncoder(nn.Module):
    def __init__(self, in_dim: int = STATE_DIM, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)  # [B, out_dim]
