"""言語エンコーダ。VLA の「耳/言葉」にあたる部分。

入力: tokens [B, L]（int64, 0=PAD） → 出力: [B, out_dim]

【重要 — 語順を区別できること】
最初は「埋め込みを平均するだけ」にしたくなりますが、それだと致命的な問題があります:
平均プーリングは語順を無視するため、
  「赤のブロックを青のゴールに置いて」 と 「青のブロックを赤のゴールに置いて」
が**同じ文字の集合**＝同じベクトルになり、どちらの色が“運ぶ対象”か区別できません。
そこで、位置埋め込み (positional embedding) + 1 層の Transformer エンコーダで
**語順を考慮**します（本物の VLA が言語に Transformer を使うのと同じ発想）。
最後に PAD を除いた平均でプーリングします。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..datasets.tokenizer import PAD_ID


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        out_dim: int = 128,
        max_len: int = 48,
        nhead: int = 4,
        ff_dim: int = 128,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(max_len, embed_dim)  # 位置埋め込み（語順の情報）
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim,
            batch_first=True, dropout=0.0,
        )
        # enable_nested_tensor=False: PAD マスク使用時の警告を抑制（パラメータは不変）
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.fc = nn.Linear(embed_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, L]
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)  # [B, L]
        x = self.token_embed(tokens) + self.pos_embed(pos)  # [B, L, embed_dim]

        pad_mask = tokens == PAD_ID                          # [B, L]（True=PAD を無視）
        x = self.encoder(x, src_key_padding_mask=pad_mask)   # [B, L, embed_dim]

        # PAD を除いた平均プーリング
        keep = (~pad_mask).float().unsqueeze(-1)             # [B, L, 1]
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)  # [B, embed_dim]
        return self.fc(pooled)                               # [B, out_dim]
