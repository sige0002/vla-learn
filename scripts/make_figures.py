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
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.7); ax.axis("off")

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
    ax.annotate("FiLM\n(言語で視覚を条件付け)", xy=(4.2, 5.3), xytext=(4.2, 6.0),
                ha="center", va="bottom", fontsize=8, color="#2e7d4f",
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


def _mixture_velocity(a: np.ndarray, tau: float, mus, sigmas, weights) -> np.ndarray:
    """学習が完璧なときの周辺速度場 v(a,τ)=E[a1-a0 | a_τ=a] の閉形式（ガウス混合の目標分布）。

    a0~N(0,I), a1~Σ_k w_k N(μ_k, σ_k^2 I), a_τ=(1-τ)a0+τa1 のとき、成分 k のもとで
      a_τ ~ N(τμ_k, s_k^2 I),  s_k^2=(1-τ)^2+(τσ_k)^2
      E[a1|a_τ=a,k] = μ_k + (τσ_k^2/s_k^2)(a-τμ_k),  E[a0|a_τ=a,k] = ((1-τ)/s_k^2)(a-τμ_k)
    を成分の事後確率 r_k で混ぜる。FlowVLA の v_pred が近づいていく「正解の速度場」。
    """
    a = np.asarray(a, dtype=float)
    D = a.shape[-1]
    logr, e1, e0 = [], [], []
    for mu, sg, w in zip(mus, sigmas, weights):
        mu = np.asarray(mu, dtype=float)
        s2 = (1.0 - tau) ** 2 + (tau * sg) ** 2
        diff = a - tau * mu
        logr.append(np.log(w) - 0.5 * float(diff @ diff) / s2 - 0.5 * D * np.log(s2))
        e1.append(mu + (tau * sg ** 2 / s2) * diff)
        e0.append(((1.0 - tau) / s2) * diff)
    logr = np.asarray(logr); logr -= logr.max()
    r = np.exp(logr); r /= r.sum()
    return sum(rk * (e1k - e0k) for rk, e1k, e0k in zip(r, e1, e0))


def _euler_path(a0: np.ndarray, n_steps: int, mus, sigmas, weights) -> np.ndarray:
    """a0 から周辺速度場を n_steps の Euler で積分した経路 [(n_steps+1), D]。"""
    a = np.asarray(a0, dtype=float).copy()
    dt = 1.0 / n_steps
    pts = [a.copy()]
    for i in range(n_steps):
        a = a + _mixture_velocity(a, i * dt, mus, sigmas, weights) * dt
        pts.append(a.copy())
    return np.stack(pts)


def fig_flow_path() -> None:
    """rectified flow：直線経路（学習時）と、多峰目標での Euler 積分（推論時）。"""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # 左: 学習時に使う「条件付き」直線経路（ノイズ a0 → 目標 a1、速度は一定）
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
    ax.set_title("学習時: 直線経路 $a_\\tau=(1-\\tau)a_0+\\tau a_1$（$v^*=a_1-a_0$ は一定）", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlabel("$a^{(1)}$"); ax.set_ylabel("$a^{(2)}$"); ax.grid(alpha=0.3)

    # 右: 推論時の Euler 積分。目標が 2 峰だと「平均の速度場」は途中で曲がるので、
    #     粗い刻みは真の経路（十分細かい積分）からずれる。
    ax = axes[1]
    mus = [np.array([1.5, 1.2]), np.array([1.3, -1.2])]
    sigmas = [0.25, 0.25]
    weights = [0.5, 0.5]
    for mu in mus:  # 目標分布（2 峰）を等高線円で表示
        for k_sig, alpha in [(1.0, 0.35), (2.0, 0.15)]:
            ax.add_patch(plt.Circle(mu, k_sig * sigmas[0], fill=False,
                                    color="#e0432f", alpha=alpha, lw=1.2))
    ax.scatter([m[0] for m in mus], [m[1] for m in mus], color="#e0432f", s=60,
               zorder=5, label="目標分布の 2 つの峰")

    a0 = np.array([-1.4, 0.35])  # 同じ初期ノイズから刻みだけ変えて積分
    exact = _euler_path(a0, 400, mus, sigmas, weights)
    ax.plot(exact[:, 0], exact[:, 1], "--", color="#333", lw=1.6, zorder=4,
            label="真の経路（十分細かい積分）")
    for steps, col in [(1, "#c9a03c"), (2, "#bbb"), (5, "#7aa8e0"), (20, "#1f4e9c")]:
        path = _euler_path(a0, steps, mus, sigmas, weights)
        ax.plot(path[:, 0], path[:, 1], "-o", color=col, ms=4, lw=1.3,
                label=f"flow_steps={steps}")
    ax.scatter(*a0, color="#555", s=40, zorder=5)
    ax.annotate("$a_0$", a0, textcoords="offset points", xytext=(-14, -4), fontsize=10)
    ax.set_title("推論時: 目標が多峰だと速度場が曲がる\n→ 粗い Euler は真の経路からずれる", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("$a^{(1)}$"); ax.set_ylabel("$a^{(2)}$"); ax.grid(alpha=0.3)
    fig.tight_layout()
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
    ax.set_yscale("log"); ax.set_xlabel("学習ステップ（ミニバッチ）"); ax.set_ylabel("MSE 損失 (log)")
    ax.set_title("TinyVLA の学習曲線（ミニバッチごとの損失を全ステップ分）"); ax.grid(alpha=0.3)
    _save(fig, "loss_curve.png")

    # ロールアウト・フィルムストリップ（成功エピソードを探して載せる。
    # 学習不足などで全滅した場合のみ最後の試行を載せ、キャプションに「失敗」と出す）
    pol = PolicyWrapper(model, tok, an, sn, "mse")
    best = None  # (frames, info, instruction, seed)
    for seed in range(12345, 12345 + 30):
        env = Tabletop2DEnv(max_steps=48)
        obs = env.reset(seed=seed)
        frames = [obs["image"]]
        instruction = env.world.instruction
        done = False
        while not done:
            chunk = pol.predict_chunk(obs)
            for k in range(4):
                obs, _, done, info = env.step(chunk[k]); frames.append(obs["image"])
                if done: break
        if best is None or (info["success"] and not best[1]["success"]):
            best = (frames, info, instruction, seed)
        # 開始時点でほぼ成功している自明なエピソードは図として意味がないので除外
        if info["success"] and len(frames) >= 12:
            best = (frames, info, instruction, seed)
            break
    frames, info, instruction, seed = best
    print(f"[fig] rollout: seed={seed} success={info['success']} len={len(frames)}")
    idx = np.linspace(0, len(frames) - 1, 8).astype(int)
    fig, axes = plt.subplots(1, 8, figsize=(14, 2.1))
    for ax, fi in zip(axes, idx):
        ax.imshow(np.transpose(frames[fi], (1, 2, 0))); ax.axis("off")
        ax.set_title(f"t={fi}", fontsize=8)
    fig.suptitle(f"学習した TinyVLA のロールアウト（{'成功' if info['success'] else '失敗'}）  指示: {instruction}", fontsize=11)
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


def fig_train_loop() -> None:
    """PyTorch の学習ループ 1 周（M1 の心臓部）。"""
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, title, code, fc):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="#333",
                                   lw=1.2, zorder=2))
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.30, code, ha="center", va="center",
                fontsize=8.5, family="monospace", color="#1f4e9c", zorder=3)

    def arrow(x1, y1, x2, y2, **kw):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5,
                                    connectionstyle=kw.get("cs", "arc3,rad=0")), zorder=1)

    # 時計回りの 6 ステップ
    box(0.3, 4.4, 2.8, 1.2, "1. バッチを取り出す", "for batch in loader:", "#dbe7ff")
    box(3.7, 4.4, 2.8, 1.2, "2. forward（予測）", "pred = model(x)", "#e9f0ff")
    box(7.1, 4.4, 2.8, 1.2, "3. 損失を計算", "loss = criterion(pred, y)", "#fff0d9")
    box(7.1, 0.6, 2.8, 1.2, "4. 勾配をリセット", "opt.zero_grad()", "#fdeeee")
    box(3.7, 0.6, 2.8, 1.2, "5. backward（自動微分）", "loss.backward()", "#e7f6ea")
    box(0.3, 0.6, 2.8, 1.2, "6. パラメータ更新", "opt.step()", "#e7f6ea")
    arrow(3.1, 5.0, 3.7, 5.0)
    arrow(6.5, 5.0, 7.1, 5.0)
    arrow(8.5, 4.4, 8.5, 1.8)
    arrow(7.1, 1.2, 6.5, 1.2)
    arrow(3.7, 1.2, 3.1, 1.2)
    arrow(1.0, 1.8, 1.0, 4.4)
    ax.text(0.85, 3.1, "次のバッチへ", ha="center", va="center",
            fontsize=8.5, color="#555", rotation=90)
    ax.text(5.1, 3.0, "この 6 手順の繰り返しが「学習」のすべて（1 epoch = 全バッチ一巡）。\n"
                      "2→3 が計算グラフを作り、5 が全パラメータの勾配 (.grad) を書き込み、\n"
                      "6 が勾配の逆方向へ少しだけ動かす（lr が歩幅）",
            ha="center", va="center", fontsize=9, color="#333",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f6f8fa", ec="#bbb"))
    ax.set_title("PyTorch の学習ループ（M1）：この 1 周を暗記ではなく「理由つき」で書けるようになる", fontsize=11)
    _save(fig, "train_loop.png")


def fig_covariate_shift() -> None:
    """素朴な模倣学習が閉ループで崩れる理由（M2: 分布シフトと誤差の複利）。"""
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    # 左: 教師データの「通り道」から一歩外れると戻れない、の模式図
    ax = axes[0]
    t = np.linspace(0, 1, 60)
    base = np.stack([t, 0.45 + 0.25 * np.sin(2.4 * t)], axis=1)
    for off in np.linspace(-0.05, 0.05, 7):  # 教師デモの束（訓練分布）
        ax.plot(base[:, 0], base[:, 1] + off, color="#9db8dd", lw=1.2, alpha=0.7)
    ax.fill_between(base[:, 0], base[:, 1] - 0.08, base[:, 1] + 0.08,
                    color="#dbe7ff", alpha=0.5, zorder=0)
    # 方策のロールアウト: 一歩ごとに小さな誤差 → 分布の外に出ると誤差が加速
    pos = base[0].copy()
    path = [pos.copy()]
    for i in range(1, 30):
        target = base[min(i * 2, 59)]
        drift = np.array([0.0, -0.004 * i])          # 累積するバイアス誤差
        noise = rng.normal(0, 0.004, size=2)
        pos = pos + (target - pos) * 0.45 + drift + noise
        path.append(pos.copy())
    path = np.stack(path)
    ax.plot(path[:, 0], path[:, 1], "-o", color="#c0392b", ms=3, lw=1.5,
            label="学習した方策の閉ループ実行")
    ax.plot([], [], color="#9db8dd", label="教師デモ（訓練分布）")
    ax.annotate("小さな誤差で\n見たことのない状態へ", xy=(path[14, 0], path[14, 1]),
                xytext=(0.28, 0.08), fontsize=8.5, color="#c0392b",
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.1))
    ax.annotate("教師データが無い領域では\n出力が当てにならず誤差が複利で増える",
                xy=(path[-1, 0], path[-1, 1]), xytext=(0.52, 0.75), fontsize=8.5,
                color="#c0392b",
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.1))
    ax.set_xlim(-0.02, 1.05); ax.set_ylim(0, 0.9)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("閉ループでは自分の出力が次の入力になる（模式図）", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # 右: 誤差の複利 vs 対策（模式カーブ）
    ax = axes[1]
    steps = np.arange(0, 40)
    ax.plot(steps, 0.01 * np.exp(steps * 0.12), color="#c0392b", lw=2,
            label="素朴な BC（誤差が複利で成長）")
    ax.plot(steps, 0.01 + 0.0035 * steps, color="#e07a3f", lw=2,
            label="+ ノイズ注入データ（戻り方を学ぶ）")
    ax.plot(steps, 0.01 + 0.0035 * np.sqrt(steps), color="#2e7d4f", lw=2,
            label="+ 再観測（receding horizon）")
    ax.set_xlabel("閉ループのステップ数"); ax.set_ylabel("軌道の誤差（模式）")
    ax.set_ylim(0, 0.6); ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    ax.set_title("対策: 誤差が増える前に観測し直す・戻り方を教える", fontsize=10)
    fig.tight_layout()
    _save(fig, "covariate_shift.png")


def fig_normalization() -> None:
    """行動の各次元はスケールが桁違い → 正規化で揃える（M3）。実データで描く。"""
    from vla_learn.datasets import build_normalizers, generate_episodes

    eps = generate_episodes(120, seed=0)
    actions = np.concatenate([ep["actions"] for ep in eps], axis=0)  # [N,3]
    an, _ = build_normalizers(eps)
    normed = an.normalize(actions)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    names = ["dx", "dy", "grip_cmd"]
    colors = ["#4a7fd6", "#2e7d4f", "#e07a3f"]
    for ax, data, title in [
        (axes[0], actions, "正規化前: dx/dy は ±0.08、grip は 0/1 と桁が違う"),
        (axes[1], normed, "正規化後: すべて平均 0・分散 1 くらいに揃う"),
    ]:
        for d in range(3):
            ax.hist(data[:, d], bins=60, histtype="step", lw=1.8,
                    color=colors[d], label=names[d], density=True)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("値"); ax.set_ylabel("密度 (log)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle("行動の正規化（実データ）: スケールがずれたまま MSE を取ると大きい次元だけが損失を支配する", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "normalization.png")


def fig_spatial_pooling() -> None:
    """勘所 1: average pooling は「どこにあるか」を消す。flatten は位置を保つ（M4）。"""
    fig, axes = plt.subplots(2, 3, figsize=(11, 4.6),
                             gridspec_kw={"width_ratios": [1, 2.4, 0.5]})
    for row, (cy, cx) in enumerate([(1, 1), (4, 4)]):
        fm = np.zeros((6, 6)); fm[cy, cx] = 1.0  # 対象がある場所だけ反応した特徴マップ
        ax = axes[row, 0]
        ax.imshow(fm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"特徴マップ {row + 1}（対象が{'左上' if row == 0 else '右下'}）", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[row, 1]
        ax.imshow(fm.reshape(1, -1), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"flatten → 36 次元ベクトル（index {cy * 6 + cx} が光る＝位置が残る）", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[row, 2]
        ax.imshow([[fm.mean()]], cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"avg pool\n→ {fm.mean():.3f}", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("勘所 1: 対象の位置が違っても avg pool は同じ値（位置が消える）。"
                 "「対象へ動く」タスクでは flatten で空間配置を残す", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "spatial_pooling.png")


def fig_word_order() -> None:
    """勘所 2: bag-of-chars（平均プーリング）は語順を区別できない（M4）。"""
    s1 = "赤のブロックを青のゴールに置いて"
    s2 = "青のブロックを赤のゴールに置いて"
    chars = sorted(set(s1) | set(s2))
    c1 = np.array([s1.count(c) for c in chars])
    c2 = np.array([s2.count(c) for c in chars])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax = axes[0]
    x = np.arange(len(chars))
    ax.bar(x - 0.2, c1, width=0.4, color="#c0392b", label=f"「{s1}」")
    ax.bar(x + 0.2, c2, width=0.4, color="#4a7fd6", label=f"「{s2}」")
    ax.set_xticks(x); ax.set_xticklabels(chars, fontsize=10)
    ax.set_ylabel("文字の出現回数")
    ax.set_yticks([0, 1, 2])
    ax.legend(fontsize=8.5)
    ax.set_title("文字の多重集合は完全に一致 → 語順を見ない平均プーリングでは同じベクトルになる", fontsize=9.5)

    ax = axes[1]
    ax.axis("off")
    for row, (s, col) in enumerate([(s1, "#c0392b"), (s2, "#4a7fd6")]):
        y = 0.72 - row * 0.42
        ax.text(0.0, y + 0.14, f"指示 {row + 1}", fontsize=9, color=col, transform=ax.transAxes)
        for i, ch in enumerate(s):
            emph = ch in ("赤", "青")
            ax.text(0.03 + i * 0.058, y, ch, fontsize=12,
                    fontweight="bold" if emph else "normal",
                    color=("#c0392b" if ch == "赤" else "#4a7fd6") if emph else "#333",
                    transform=ax.transAxes)
            if row == 0:
                ax.text(0.03 + i * 0.058, y - 0.14, str(i), fontsize=7, color="#999",
                        transform=ax.transAxes)
    ax.text(0.0, 0.02, "位置 0 と 7 の「赤/青」が入れ替わっているだけ。位置埋め込み + Transformer は\n"
                       "「どの位置に何があるか」を見るので、運ぶ色と置く色を区別できる（M4 勘所 2）",
            fontsize=9, color="#333", transform=ax.transAxes)
    ax.set_title("同じ文字集合・違う意味 — 区別には語順（位置）の情報が必要", fontsize=9.5)
    fig.tight_layout()
    _save(fig, "word_order.png")


def fig_temporal_ensemble() -> None:
    """チャンク境界の段差と temporal ensembling（ACT）の概念図（M4）。"""
    rng = np.random.default_rng(2)
    T, C, EH = 24, 8, 4
    true = 0.6 * np.sin(np.linspace(0, 2.2 * np.pi, T + C))
    # 各時刻 t0 でチャンクを予測（チャンクごとに系統誤差オフセットが乗る、を模擬）
    chunk_pred = {}
    for t0 in range(T):
        off = rng.normal(0, 0.10)
        chunk_pred[t0] = true[t0:t0 + C] + off + rng.normal(0, 0.02, C)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=True)

    # 左: 素朴な receding horizon（exec_horizon=4 ごとに切り替え → 境界に段差）
    ax = axes[0]
    ax.plot(np.arange(T), true[:T], "--", color="#999", lw=1.5, label="真にやりたい軌道")
    for t0 in range(0, T, EH):
        seg = chunk_pred[t0][:EH]
        ax.plot(np.arange(t0, t0 + EH), seg, "-o", color="#4a7fd6", ms=4, lw=1.6)
        if t0 > 0:
            prev = chunk_pred[t0 - EH][:EH][-1]
            ax.plot([t0 - 1, t0], [prev, seg[0]], color="#c0392b", lw=2.2, zorder=5)
    ax.plot([], [], "-o", color="#4a7fd6", ms=4, label="実行する行動（4 手ごとに新チャンク）")
    ax.plot([], [], color="#c0392b", lw=2.2, label="チャンク境界の段差")
    ax.set_title("素朴な receding horizon: 切り替えの瞬間がつながる保証はない", fontsize=9.5)
    ax.set_xlabel("時刻 t"); ax.set_ylabel("行動（1 次元で図示）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 右: temporal ensembling（毎ステップ予測し、t をカバーする予測を指数重み平均）
    ax = axes[1]
    m = 0.25
    ens = np.zeros(T)
    for t in range(T):
        preds, ws = [], []
        for t0 in range(max(0, t - C + 1), t + 1):  # t0 昇順 = 古い予測から
            i = t0 - max(0, t - C + 1)              # i=0 が最も古い予測
            preds.append(chunk_pred[t0][t - t0])
            ws.append(np.exp(-m * i))
        ens[t] = np.average(preds, weights=ws)
    ax.plot(np.arange(T), true[:T], "--", color="#999", lw=1.5, label="真にやりたい軌道")
    for t0 in range(0, T, 2):  # 重なり合うチャンク予測（間引いて表示）
        ax.plot(np.arange(t0, min(t0 + C, T)), chunk_pred[t0][:min(C, T - t0)],
                color="#9db8dd", lw=0.7, alpha=0.6)
    ax.plot(np.arange(T), ens, "-o", color="#2e7d4f", ms=4, lw=1.8,
            label="重なった予測の指数重み平均")
    ax.plot([], [], color="#9db8dd", lw=0.7, label="毎ステップのチャンク予測（重なる）")
    ax.set_title("temporal ensembling (ACT): 毎ステップ予測して平均 → 滑らか", fontsize=9.5)
    ax.set_xlabel("時刻 t"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "temporal_ensemble.png")


def fig_mse_vs_flow() -> None:
    """MSE は多峰の教師で「平均」に潰れる。flow は峰ごとに生成できる（M5 の動機）。"""
    rng = np.random.default_rng(0)
    mus, sigmas, weights = [np.array([-1.2]), np.array([1.2])], [0.18, 0.18], [0.5, 0.5]
    n = 4000
    comp = rng.random(n) < weights[1]
    data = np.where(comp, rng.normal(1.2, 0.18, n), rng.normal(-1.2, 0.18, n))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    # 左: 教師データ（2 峰）と MSE の答え（平均）
    ax = axes[0]
    ax.hist(data, bins=70, color="#9db8dd", density=True)
    ax.axvline(data.mean(), color="#c0392b", lw=2.5)
    ax.annotate("MSE の最適解 = 平均\n（どちらの峰でもない行動）", xy=(data.mean(), 0.6),
                xytext=(-0.9, 0.95), fontsize=9, color="#c0392b",
                arrowprops=dict(arrowstyle="-|>", color="#c0392b"))
    ax.set_title("教師の行動が 2 通りある場合（例: 左右どちらから回るか）", fontsize=9.5)
    ax.set_xlabel("行動 $a$"); ax.set_ylabel("密度")

    # 中: flow の軌跡 τ: 0→1（ノイズごとにどちらかの峰へ流れる）
    ax = axes[1]
    taus = np.linspace(0, 1, 60)
    for a0 in rng.normal(size=40):
        path = [np.array([a0])]
        for i in range(len(taus) - 1):
            v = _mixture_velocity(path[-1], taus[i], mus, sigmas, weights)
            path.append(path[-1] + v * (taus[i + 1] - taus[i]))
        path = np.concatenate(path)
        ax.plot(taus, path, color="#1f4e9c" if path[-1] > 0 else "#e07a3f",
                lw=0.8, alpha=0.6)
    ax.set_title("flow: ノイズ $a_0$ ごとに軌跡がどちらかの峰へ向かう", fontsize=9.5)
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel("行動 $a$"); ax.grid(alpha=0.3)

    # 右: 生成された行動の分布（2 峰が再現される）
    ax = axes[2]
    samples = []
    for a0 in rng.normal(size=1500):
        a = np.array([float(a0)])
        n_steps = 40; dt = 1.0 / n_steps
        for i in range(n_steps):
            a = a + _mixture_velocity(a, i * dt, mus, sigmas, weights) * dt
        samples.append(a[0])
    ax.hist(np.asarray(samples), bins=70, color="#8fbf9a", density=True, label="flow の生成")
    ax.hist(data, bins=70, histtype="step", color="#555", density=True, label="教師の分布")
    ax.axvline(0.0, color="#c0392b", lw=2, ls="--", label="MSE の答え（平均）")
    ax.legend(fontsize=8)
    ax.set_title("flow の生成分布は 2 峰を再現する", fontsize=9.5)
    ax.set_xlabel("行動 $a$"); ax.set_ylabel("密度")
    fig.tight_layout()
    _save(fig, "mse_vs_flow.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="学習を伴う図を小さめに")
    p.add_argument("--only", type=str, default=None,
                   help="env|arch|chunk|flow|bars|trainloop|shift|norm|pool|order|msevsflow|loss のどれかだけ")
    args = p.parse_args()

    jobs = {
        "env": fig_env_samples,
        "arch": fig_architecture,
        "chunk": fig_action_chunking,
        "flow": fig_flow_path,
        "bars": fig_success_bars,
        "trainloop": fig_train_loop,
        "shift": fig_covariate_shift,
        "norm": fig_normalization,
        "pool": fig_spatial_pooling,
        "order": fig_word_order,
        "ensemble": fig_temporal_ensemble,
        "msevsflow": fig_mse_vs_flow,
        "loss": lambda: fig_loss_and_rollout(args.quick),
    }
    if args.only:
        jobs[args.only]()
    else:
        for name, fn in jobs.items():
            if name != "loss":
                fn()
        fig_loss_and_rollout(args.quick)
    print("done.")


if __name__ == "__main__":
    main()
