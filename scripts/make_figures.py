"""教材(Typst)用の図を生成する（データ駆動・uv で再現可能）。

使い方:
  uv run --extra viz python scripts/make_figures.py            # 全図を book/figures/ に生成
  uv run --extra viz python scripts/make_figures.py --quick    # 学習を伴う図を小さめに

出力先: book/figures/*.png（Typst から #image で読み込む）

方針: 図のラベルは最小限（英語/数式）にし、日本語の説明は Typst のキャプション側に書く。
ただし環境サンプルの指示文など日本語が要る図は Noto Sans CJK JP を使う。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# --- 日本語フォント（あれば設定。無ければ既定のまま）---
from matplotlib import font_manager

for _cand in ["Noto Sans CJK JP", "Noto Sans CJK JP Regular", "IPAexGothic"]:
    try:
        font_manager.findfont(_cand, fallback_to_default=False)
        plt.rcParams["font.family"] = _cand
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parents[1] / "book" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path.relative_to(OUT.parents[1])}")


# ----------------------------------------------------------------------
def fig_env_samples() -> None:
    """環境の観測サンプル（タスクの見た目）。"""
    import textwrap
    from vla_learn.envs import Tabletop2DEnv

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.6))
    for i, ax in enumerate(axes.flat):
        env = Tabletop2DEnv(seed=i)
        obs = env.reset()
        ax.imshow(np.transpose(obs["image"], (1, 2, 0)))
        ax.set_title("\n".join(textwrap.wrap(obs["instruction"], 13)), fontsize=8.5, pad=4)
        ax.axis("off")
    fig.suptitle("Tiny Tabletop 2D：観測画像 [3,64,64] と言語指示", fontsize=13)
    fig.subplots_adjust(wspace=0.15, hspace=0.35, top=0.9)
    _save(fig, "env_samples.png")


def fig_architecture() -> None:
    """VLA の forward パイプライン（M0/M4/M5 共通の地図）。"""
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, fc, fontsize=9, tc="black"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="#333", lw=1.2, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=tc, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.4), zorder=1)

    # 入力
    box(0.2, 4.4, 2.1, 0.9, "image\n[B,3,64,64]", "#dbe7ff")
    box(0.2, 2.7, 2.1, 0.9, "tokens\n[B,L]", "#dbe7ff")
    box(0.2, 1.0, 2.1, 0.9, "state\n[B,3]", "#dbe7ff")
    # エンコーダ
    box(3.0, 4.4, 2.4, 0.9, "ImageEncoder\n(CNN+flatten)", "#e9f0ff", 8.5)
    box(3.0, 2.7, 2.4, 0.9, "TextEncoder\n(Transformer)", "#e9f0ff", 8.5)
    box(3.0, 1.0, 2.4, 0.9, "StateEncoder\n(MLP)", "#e9f0ff", 8.5)
    # FiLM（言語→視覚）
    ax.annotate("FiLM\n(言語で視覚を条件付け)", xy=(4.2, 5.3), xytext=(4.2, 5.85),
                ha="center", fontsize=8, color="#2e7d4f",
                arrowprops=dict(arrowstyle="-|>", color="#2e7d4f", lw=1.2))
    arrow(4.2, 3.6, 4.2, 4.4)  # text -> image (FiLM)
    # 融合
    box(6.2, 2.7, 2.0, 2.6, "concat\n+\nFusion MLP\n→ h [B,256]", "#fff0d9", 9)
    for yy in (4.85, 3.15, 1.45):
        arrow(2.3, yy, 3.0, yy)
    arrow(5.4, 4.85, 6.2, 4.4)
    arrow(5.4, 3.15, 6.2, 3.9)
    arrow(5.4, 1.45, 6.2, 3.2)
    # ヘッド
    box(9.0, 3.7, 2.6, 1.2, "MSE ヘッド (M4)\nLinear → 回帰", "#e7f6ea", 8.7)
    box(9.0, 1.3, 2.6, 1.2, "flow ヘッド (M5)\n速度場 v を積分", "#e7f6ea", 8.7)
    arrow(8.2, 4.3, 9.0, 4.3)
    arrow(8.2, 3.4, 9.0, 1.9)
    # 出力
    ax.text(11.7, 4.3, "→ action\n  chunk\n  [B,8,3]", ha="left", va="center", fontsize=8.5)
    ax.text(11.7, 1.9, "→ action\n  chunk\n  [B,8,3]", ha="left", va="center", fontsize=8.5)
    ax.text(6.0, 0.2, "同じ backbone（青→橙）を共有し、右端のヘッドだけ MSE↔flow で差し替える",
            ha="center", fontsize=9, color="#555")
    ax.set_title("TinyVLA / FlowVLA の forward：3入力→エンコーダ→FiLM融合→行動ヘッド→行動チャンク", fontsize=11)
    _save(fig, "architecture.png")


def fig_action_chunking() -> None:
    """行動チャンクと pad_mask、receding horizon の概念図。"""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    C = 8
    valid = 5  # 終端付近で 5 ステップ有効、3 ステップ pad の例
    for t in range(C):
        is_pad = t >= valid
        color = "#cdd5e0" if is_pad else "#4a7fd6"
        ax.add_patch(plt.Rectangle((t, 0), 0.9, 1, color=color))
        ax.text(t + 0.45, 0.5, f"a{t}", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        ax.text(t + 0.45, -0.35, "1" if not is_pad else "0", ha="center", fontsize=10)
    # 実行ホライズン
    ax.add_patch(plt.Rectangle((0, 1.15), 4 * 1.0 - 0.1, 0.3, color="#e07a3f"))
    ax.text(2.0, 1.3, "実行する先頭4手 (exec_horizon=4)", ha="center", va="center",
            color="white", fontsize=9)
    ax.text(-0.4, 0.5, "chunk", ha="right", va="center", fontsize=10)
    ax.text(-0.4, -0.35, "pad_mask", ha="right", va="center", fontsize=10)
    ax.set_xlim(-2.2, C + 0.2)
    ax.set_ylim(-0.7, 1.7)
    ax.axis("off")
    ax.set_title("行動チャンク [B,8,3]：先頭だけ実行→再観測（receding horizon）、終端は pad_mask=0", fontsize=11)
    _save(fig, "action_chunking.png")


def fig_flow_path() -> None:
    """rectified flow：直線経路 a_τ=(1-τ)a0+τa1 と Euler サンプリング。"""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # 左: 複数サンプルの直線経路（ノイズ a0 → 目標 a1）
    ax = axes[0]
    a1 = np.array([1.2, 0.8])
    for _ in range(12):
        a0 = rng.normal(size=2)
        taus = np.linspace(0, 1, 20)
        pts = (1 - taus)[:, None] * a0 + taus[:, None] * a1
        ax.plot(pts[:, 0], pts[:, 1], color="#4a7fd6", alpha=0.5, lw=1)
        ax.scatter(*a0, color="#888", s=12)
    ax.scatter(*a1, color="#e0432f", s=80, zorder=5, label="目標 $a_1$")
    ax.scatter([], [], color="#888", s=12, label="ノイズ $a_0\\sim N(0,I)$")
    ax.set_title("直線経路 $a_\\tau=(1-\\tau)a_0+\\tau a_1$（速度 $v^*=a_1-a_0$ は一定）")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlabel("$a^{(1)}$"); ax.set_ylabel("$a^{(2)}$"); ax.grid(alpha=0.3)

    # 右: Euler 積分でノイズから生成（flow_steps の刻み）
    ax = axes[1]
    for steps, col in [(2, "#bbb"), (5, "#7aa8e0"), (20, "#1f4e9c")]:
        a = np.array([-1.5, 1.6])  # 同じ初期ノイズ
        dt = 1.0 / steps
        xs, ys = [a[0]], [a[1]]
        for i in range(steps):
            v = a1 - a  # 教師: 目標へ向かう（ここでは説明用に target-current で近似）
            a = a + v * dt
            xs.append(a[0]); ys.append(a[1])
        ax.plot(xs, ys, "-o", color=col, ms=4, label=f"flow_steps={steps}")
    ax.scatter(*a1, color="#e0432f", s=80, zorder=5)
    ax.set_title("Euler 積分 $a\\leftarrow a+v\\,dt$：刻みを増やすと滑らか")
    ax.legend(fontsize=9); ax.set_xlabel("$a^{(1)}$"); ax.set_ylabel("$a^{(2)}$"); ax.grid(alpha=0.3)
    _save(fig, "flow_path.png")


def _train_quick(n_episodes=600, epochs=18):
    """図用に小さく学習し、(model, history, tok, an, sn) を返す。"""
    from torch.utils.data import DataLoader
    from vla_learn.datasets import (CharTokenizer, SyntheticVLADataset,
                                     build_normalizers, generate_episodes)
    from vla_learn.envs import all_instruction_strings
    from vla_learn.models import TinyVLA
    from vla_learn.training.losses import masked_mse
    from vla_learn.utils import set_seed

    set_seed(0)
    eps = generate_episodes(n_episodes, seed=0, action_noise=0.03)
    tok = CharTokenizer.from_corpus(all_instruction_strings())
    an, sn = build_normalizers(eps)
    ds = SyntheticVLADataset(eps, tok, 8, an, sn)
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    model = TinyVLA(vocab_size=tok.vocab_size, chunk_len=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = []
    model.train()
    for _ in range(epochs):
        for b in loader:
            loss = masked_mse(model(b["image"], b["state"], b["tokens"]), b["action"], b["pad_mask"])
            opt.zero_grad(); loss.backward(); opt.step()
            hist.append(loss.item())
    return model, hist, tok, an, sn


def fig_loss_and_rollout(quick: bool) -> None:
    """学習曲線 + 学習済み方策のロールアウト・フィルムストリップ。"""
    from vla_learn.envs import Tabletop2DEnv
    from vla_learn.evaluation.rollout import PolicyWrapper

    model, hist, tok, an, sn = _train_quick(*( (250, 8) if quick else (700, 20) ))

    # 学習曲線
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(hist, color="#1f4e9c", lw=0.8)
    ax.set_yscale("log"); ax.set_xlabel("ステップ"); ax.set_ylabel("MSE 損失 (log)")
    ax.set_title("TinyVLA の学習曲線（1 バッチではなく全データ）"); ax.grid(alpha=0.3)
    _save(fig, "loss_curve.png")

    # ロールアウト・フィルムストリップ
    pol = PolicyWrapper(model, tok, an, sn, "mse")
    env = Tabletop2DEnv(max_steps=48)
    obs = env.reset(seed=12345)
    frames = [obs["image"]]
    done = False
    while not done:
        chunk = pol.predict_chunk(obs)
        for k in range(4):
            obs, _, done, info = env.step(chunk[k]); frames.append(obs["image"])
            if done: break
    idx = np.linspace(0, len(frames) - 1, 8).astype(int)
    fig, axes = plt.subplots(1, 8, figsize=(14, 2.1))
    for j, (ax, fi) in enumerate(zip(axes, idx)):
        ax.imshow(np.transpose(frames[fi], (1, 2, 0))); ax.axis("off")
        ax.set_title(f"t={fi}", fontsize=8)
    fig.suptitle(f"学習した TinyVLA のロールアウト（{'成功' if info['success'] else '失敗'}）  指示: {env.world.instruction}", fontsize=11)
    _save(fig, "rollout_filmstrip.png")


def fig_success_bars() -> None:
    """成功率の比較（検証で得た代表値。乱数でぶれる旨は本文で明記）。"""
    labels = ["expert\n(お手本)", "TinyVLA\n(MSE)", "FlowVLA\n(flow)", "MSE\n(avg pool)", "MSE\n(no FiLM)"]
    vals = [1.00, 0.76, 1.00, 0.05, 0.21]  # 代表値（ablation は傾向）
    colors = ["#888", "#4a7fd6", "#1f7a3f", "#d98b3f", "#d9543f"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel("閉ループ成功率")
    ax.set_title("設計の3勘所を壊すと成功率が落ちる（代表値・環境/乱数でぶれる）")
    ax.axhline(0, color="k", lw=0.6)
    _save(fig, "success_bars.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="学習を伴う図を小さめに")
    p.add_argument("--only", type=str, default=None, help="env|chunk|flow|loss|bars のどれかだけ")
    args = p.parse_args()

    jobs = {
        "env": fig_env_samples,
        "arch": fig_architecture,
        "chunk": fig_action_chunking,
        "flow": fig_flow_path,
        "bars": fig_success_bars,
        "loss": lambda: fig_loss_and_rollout(args.quick),
    }
    if args.only:
        jobs[args.only]()
    else:
        for fn in [fig_env_samples, fig_architecture, fig_action_chunking, fig_flow_path, fig_success_bars]:
            fn()
        fig_loss_and_rollout(args.quick)
    print("done.")


if __name__ == "__main__":
    main()
