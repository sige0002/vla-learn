"""課題⑤（発展）: 離散トークン化ヘッドの雛形（穴埋め）。

ねらい（本文 C-1 / OpenVLA の発想を最小限だけ自作する）:
  連続値の行動を K ビンに量子化 → 「各 (ステップ, 次元) ごとの K クラス分類」として解く。
  TinyVLA(連続回帰) / FlowVLA(連続生成) に対し、第 3 の行動表現＝離散 を手で作る。

重要:
  - OpenVLA のような「1 トークンずつの自己回帰」は再現しません。ここでは全ステップ同時に分類する簡易版。
  - 評価の主軸は「1 バッチ過学習で配線が正しいか」。重い学習はしません。
  - 同じ条件ベクトル h（VLABackbone の出力）の上に、ヘッドだけ差し替える点を体感するのが目的です。

使い方:
  python exercises/m6/discrete_head.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vla_learn.datasets import (
    CharTokenizer,
    SyntheticVLADataset,
    build_normalizers,
    generate_episodes,
)
from vla_learn.envs import all_instruction_strings
from vla_learn.models import VLABackbone, count_parameters
from vla_learn.utils import set_seed

CHUNK_LEN = 8
ACTION_DIM = 3
K = 21          # ビン数（奇数だと中心 0 を表現できて扱いやすい）
A_RANGE = 3.0   # 正規化済み行動の想定レンジ [-A_RANGE, A_RANGE] を K ビンに割る


def quantize(a: torch.Tensor) -> torch.Tensor:
    """正規化済み行動 a (任意 shape, float) を [0, K-1] のビン index へ量子化する。

    [-A_RANGE, A_RANGE] を K 等分。範囲外は端に丸める（clamp）。
    """
    # ------------------------------------------------------------
    # 【穴埋め 1/3】a を [0, K-1] の long テンソルに変換して返す。
    #   手順例:
    #     1) a を [-A_RANGE, A_RANGE] に clamp
    #     2) [0, 1] へ線形変換: (a + A_RANGE) / (2*A_RANGE)
    #     3) [0, K-1] へ拡大して round → long → clamp(0, K-1)
    # ------------------------------------------------------------
    raise NotImplementedError("quantize を実装してください")


def dequantize(idx: torch.Tensor) -> torch.Tensor:
    """ビン index [0, K-1] を、そのビン中心の連続値へ戻す（quantize の逆）。"""
    centers = torch.linspace(-A_RANGE, A_RANGE, K, device=idx.device)  # [K]
    return centers[idx]


class DiscreteVLA(nn.Module):
    """VLABackbone はそのまま、ヘッドを K クラス分類に差し替えた VLA。"""

    def __init__(self, vocab_size: int, chunk_len: int = CHUNK_LEN,
                 action_dim: int = ACTION_DIM, n_bins: int = K, hidden: int = 256) -> None:
        super().__init__()
        self.backbone = VLABackbone(vocab_size, hidden=hidden)
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.n_bins = n_bins
        # ------------------------------------------------------------
        # 【穴埋め 2/3】ヘッドを定義する。
        #   出力は (ステップ × 次元 × ビン) ぶんのロジット。
        #   ヒント: nn.Linear(hidden, chunk_len * action_dim * n_bins)
        # ------------------------------------------------------------
        self.head = ____

    def forward(self, image, state, tokens) -> torch.Tensor:
        h = self.backbone(image, state, tokens)               # [B, hidden]
        logits = self.head(h)                                 # [B, C*A*K]
        return logits.view(-1, self.chunk_len, self.action_dim, self.n_bins)  # [B,C,A,K]


def discrete_loss(logits: torch.Tensor, action: torch.Tensor,
                  pad_mask: torch.Tensor) -> torch.Tensor:
    """cross-entropy（pad_mask=0 の位置は除外）。

    logits: [B,C,A,K], action: [B,C,A]（正規化済み）, pad_mask: [B,C]
    """
    target = quantize(action)                                 # [B,C,A] long（正解ビン）
    B, C, A, Kk = logits.shape
    logits_flat = logits.reshape(B * C * A, Kk)               # [N, K]
    target_flat = target.reshape(B * C * A)                   # [N]
    # 各 (b,c,a) を有効/無効に展開（pad_mask は [B,C] → [B,C,A] へ）
    valid = pad_mask[:, :, None].expand(B, C, A).reshape(B * C * A) > 0.5
    # ------------------------------------------------------------
    # 【穴埋め 3/3】valid な位置だけで cross-entropy を計算して返す。
    #   ヒント: nn.functional.cross_entropy(logits_flat[valid], target_flat[valid])
    # ------------------------------------------------------------
    return ____


def main() -> None:
    """1 バッチ過学習で配線を確認する（重い学習はしない）。"""
    set_seed(0)
    eps = generate_episodes(n_episodes=8, seed=0)
    tok = CharTokenizer.from_corpus(all_instruction_strings())
    an, sn = build_normalizers(eps)
    ds = SyntheticVLADataset(eps, tok, CHUNK_LEN, an, sn)
    batch = next(iter(DataLoader(ds, batch_size=16, shuffle=True)))

    model = DiscreteVLA(vocab_size=tok.vocab_size, chunk_len=CHUNK_LEN)
    print(f"[model] discrete | params={count_parameters(model):,} | K={K}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    first = None
    model.train()
    for i in range(300):
        logits = model(batch["image"], batch["state"], batch["tokens"])
        loss = discrete_loss(logits, batch["action"], batch["pad_mask"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    print(f"[overfit] cross-entropy {first:.3f} -> {loss.item():.3f}")
    ok = loss.item() < 0.2 * first
    print("配線 OK（過学習できた）" if ok else "まだ下がりきっていません。穴埋めを見直しましょう")


if __name__ == "__main__":
    main()
