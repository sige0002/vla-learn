"""最小 VLA 本体（M4）。画像 + 言語 + 状態 → 行動チャンク。

構成は本物の VLA と同じ「3 つの感覚を融合して行動を出す」骨格です:

    image ─▶ ImageEncoder ─┐
    tokens ▶ TextEncoder ──┼─▶ concat ─▶ Fusion MLP ─▶ h ─▶ ActionHead ─▶ action_chunk
    state ─▶ StateEncoder ─┘                          (条件ベクトル)

M4 では ActionHead を「単純な全結合（回帰）」にして MSE で学習します（決定論的方策）。
M5 ではこの h を条件として flow matching ヘッドに差し替えます（同じ部品の付け替え）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import ACTION_DIM, DEFAULT_CHUNK_LEN
from .image_encoder import ImageEncoder
from .state_encoder import StateEncoder
from .text_encoder import TextEncoder


class VLABackbone(nn.Module):
    """3 つのエンコーダ + 融合 MLP。条件ベクトル h（[B, hidden]）を作る共通部品。"""

    def __init__(
        self,
        vocab_size: int,
        img_dim: int = 128,
        txt_dim: int = 128,
        state_dim: int = 64,
        hidden: int = 256,
        image_pool: str = "flatten",
        condition_vision: bool = True,
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, out_dim=txt_dim)
        # 言語ベクトルで視覚を条件付け（FiLM）。condition_vision=False で無効化（比較用）。
        cond_dim = txt_dim if condition_vision else None
        self.image_encoder = ImageEncoder(out_dim=img_dim, pool=image_pool, cond_dim=cond_dim)
        self.state_encoder = StateEncoder(out_dim=state_dim)
        fused_in = img_dim + txt_dim + state_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.hidden = hidden

    def forward(self, image: torch.Tensor, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        l = self.text_encoder(tokens)        # [B, txt_dim] … 先に言語を符号化
        v = self.image_encoder(image, cond=l)  # [B, img_dim] … 言語で条件付けした視覚
        s = self.state_encoder(state)        # [B, state_dim]
        h = torch.cat([v, l, s], dim=-1)      # [B, img+txt+state]
        return self.fusion(h)                # [B, hidden]


class TinyVLA(nn.Module):
    """MSE（回帰）版の最小 VLA。h から行動チャンクを直接出力する。"""

    def __init__(
        self,
        vocab_size: int,
        chunk_len: int = DEFAULT_CHUNK_LEN,
        action_dim: int = ACTION_DIM,
        hidden: int = 256,
        **backbone_kwargs,
    ) -> None:
        super().__init__()
        self.backbone = VLABackbone(vocab_size, hidden=hidden, **backbone_kwargs)
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.head = nn.Linear(hidden, chunk_len * action_dim)

    def forward(self, image: torch.Tensor, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        h = self.backbone(image, state, tokens)        # [B, hidden]
        out = self.head(h)                             # [B, C*action_dim]
        return out.view(-1, self.chunk_len, self.action_dim)  # [B, C, action_dim]

    @torch.no_grad()
    def predict(self, image: torch.Tensor, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """推論用。勾配を切って行動チャンクを返す（正規化空間のまま）。"""
        self.eval()
        return self.forward(image, state, tokens)


def count_parameters(model: nn.Module) -> int:
    """学習対象パラメータ数を数える（モデル規模の把握に）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
