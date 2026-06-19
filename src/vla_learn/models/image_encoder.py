"""画像エンコーダ（小さな CNN）。VLA の「目」にあたる部分。

入力: [B, 3, 64, 64]  → 出力: [B, out_dim]
本物の VLA は事前学習済み ViT（SigLIP など）を使いますが、ここでは
仕組みを理解するために、小さな畳み込みネットを自前で組みます。

【設計上の 2 つの勘所】
1. 空間情報の保持: 「対象の“場所”へ動く」タスクなので、最後に GlobalAveragePooling で
   位置を潰すと方向を出せません。既定は特徴マップを flatten して空間配置を保ちます
   （pool="avg" にすると位置を捨てる比較版。性能が落ちることを実験で確認できます）。
2. 言語による条件付け (FiLM): 「どの色を運ぶか」を画像だけからは決められません。
   言語ベクトルから (scale, shift) を作り、畳み込み特徴をチャンネルごとに変調します
   （FiLM: Feature-wise Linear Modulation）。これで「名指しされた色」に視覚を向けられます。
   cond=None なら条件付けなし（言語を無視する比較版＝grounding が壊れることを体験できる）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import IMG_SIZE


class FiLM(nn.Module):
    """条件ベクトル cond から per-channel の (scale, shift) を作って特徴を変調する。"""

    def __init__(self, cond_dim: int, num_channels: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], cond: [B, cond_dim]
        gamma, beta = self.to_scale_shift(cond).chunk(2, dim=-1)  # [B,C],[B,C]
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ImageEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = 128,
        in_ch: int = 3,
        img_size: int = IMG_SIZE,
        pool: str = "flatten",
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        # 64 -> 32 -> 16 -> 8 -> 4 と空間サイズを半分ずつ落としていく（ブロックごとに分ける）
        self.block1 = self._block(in_ch, 16, 4)   # [B,16,32,32]
        self.block2 = self._block(16, 32, 8)       # [B,32,16,16]
        self.block3 = self._block(32, 64, 8)       # [B,64,8,8]
        self.block4 = self._block(64, 64, 8)       # [B,64,4,4]

        # 言語条件付け（FiLM）。中間 2 箇所に挿入して、言語で視覚特徴を変調する。
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.film2 = FiLM(cond_dim, 32)
            self.film3 = FiLM(cond_dim, 64)

        self.pool = pool
        feat_hw = img_size // 16  # stride2 を 4 回 → 1/16
        if pool == "flatten":
            in_features = 64 * feat_hw * feat_hw
        elif pool == "avg":
            self.gap = nn.AdaptiveAvgPool2d(1)
            in_features = 64
        else:
            raise ValueError(f"unknown pool={pool!r}")
        self.fc = nn.Linear(in_features, out_dim)
        self.out_dim = out_dim

    @staticmethod
    def _block(cin: int, cout: int, groups: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1),
            nn.GroupNorm(groups, cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # image: [B, 3, 64, 64]（値域 0..1 を中心化）
        x = image * 2.0 - 1.0
        x = self.block1(x)                       # [B,16,32,32]
        x = self.block2(x)                       # [B,32,16,16]
        if self.cond_dim is not None and cond is not None:
            x = self.film2(x, cond)              # 言語で変調
        x = self.block3(x)                       # [B,64,8,8]
        if self.cond_dim is not None and cond is not None:
            x = self.film3(x, cond)              # 言語で変調
        x = self.block4(x)                       # [B,64,4,4]
        if self.pool == "flatten":
            x = x.flatten(1)                     # [B, 64*4*4]
        else:
            x = self.gap(x).flatten(1)           # [B, 64]
        return self.fc(x)                        # [B, out_dim]
