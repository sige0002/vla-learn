"""Flow Matching 版の最小 VLA（M5）。

M4 の決定論的な回帰ヘッドを、条件付き生成モデル（flow matching / rectified flow）に
置き換えたものです。pi0 / SmolVLA などの最新 VLA が採用している方式の最小版です。

直感（既習の diffusion 座学とのつながり）:
  - ノイズ a0 ~ N(0, I) から目標行動 a1 へ向かう「まっすぐな道」を考える:
        a_τ = (1-τ) a0 + τ a1            （τ: 0→1）
  - この道の速度は v* = da_τ/dτ = a1 - a0（一定）。
  - ネットワークは「今いる a_τ と時刻 τ と条件 h」から速度 v を予測するよう学習:
        L = || v_pred(a_τ, τ, h) - (a1 - a0) ||^2
  - 推論は a0~N(0,I) から τ=0→1 へ Euler 積分し、p(a1|h) からのサンプルを生成する
    （学習に使った特定の a1 が出てくるのではなく、条件 h のもとでの行動分布から引く）。
  diffusion の「ノイズ除去」を、確率フローの「速度場の積分」として実装した形です。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..constants import ACTION_DIM, DEFAULT_CHUNK_LEN
from ..functional import masked_mse
from .tiny_vla import VLABackbone


class SinusoidalTimeEmbedding(nn.Module):
    """連続時刻 τ∈[0,1] を [B, dim] のベクトルに変換（Transformer の位置符号と同じ発想）。"""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        # tau: [B]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(1000.0) * torch.arange(half, device=tau.device) / (half - 1)
        )  # [half]
        ang = tau[:, None] * freqs[None, :] * (2 * math.pi)  # [B, half]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B, dim]


class FlowVLA(nn.Module):
    """条件付き flow matching ヘッドを持つ VLA。"""

    def __init__(
        self,
        vocab_size: int,
        chunk_len: int = DEFAULT_CHUNK_LEN,
        action_dim: int = ACTION_DIM,
        hidden: int = 256,
        time_dim: int = 64,
        velocity_hidden: int = 256,
        **backbone_kwargs,
    ) -> None:
        super().__init__()
        self.backbone = VLABackbone(vocab_size, hidden=hidden, **backbone_kwargs)
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        in_dim = chunk_len * action_dim + hidden + time_dim
        self.vnet = nn.Sequential(
            nn.Linear(in_dim, velocity_hidden), nn.ReLU(inplace=True),
            nn.Linear(velocity_hidden, velocity_hidden), nn.ReLU(inplace=True),
            nn.Linear(velocity_hidden, chunk_len * action_dim),
        )

    # --- 速度場 v(a_τ, τ | h) ---
    def velocity(self, a: torch.Tensor, tau: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # a: [B, C, A], tau: [B], h: [B, hidden] → v: [B, C, A]
        B = a.shape[0]
        x = torch.cat([a.flatten(1), h, self.time_embed(tau)], dim=-1)
        return self.vnet(x).view(B, self.chunk_len, self.action_dim)

    def encode(self, image, state, tokens) -> torch.Tensor:
        return self.backbone(image, state, tokens)  # [B, hidden]

    # --- 学習: flow matching 損失 ---
    def flow_loss(self, image, state, tokens, action, pad_mask=None) -> torch.Tensor:
        h = self.encode(image, state, tokens)
        a1 = action                          # [B, C, A]（正規化済み目標）
        a0 = torch.randn_like(a1)            # ノイズ
        tau = torch.rand(a1.shape[0], device=a1.device)        # [B] ~ U(0,1)
        a_tau = (1 - tau)[:, None, None] * a0 + tau[:, None, None] * a1
        v_pred = self.velocity(a_tau, tau, h)
        v_target = a1 - a0                   # まっすぐな道の速度（一定）
        return masked_mse(v_pred, v_target, pad_mask)

    # --- 推論: τ=0→1 を Euler 積分して行動を生成 ---
    @torch.no_grad()
    def sample(self, image, state, tokens, n_steps: int = 10) -> torch.Tensor:
        self.eval()
        h = self.encode(image, state, tokens)
        B = h.shape[0]
        a = torch.randn(B, self.chunk_len, self.action_dim, device=h.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            tau = torch.full((B,), i * dt, device=h.device)
            a = a + self.velocity(a, tau, h) * dt
        return a  # [B, C, A]（正規化空間）
