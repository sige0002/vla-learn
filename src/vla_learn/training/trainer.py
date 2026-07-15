"""学習ループ本体（MSE / flow 共通）。

この 1 ファイルで「データ生成 → トークナイザ/正規化 → Dataset/DataLoader →
モデル → 最適化ループ → 評価 → 保存」までの VLA 学習の全工程をたどれます。
M4（mse）と M5（flow）の違いは、(1) モデル構築 と (2) 損失の計算 の 2 箇所だけです。
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..datasets.normalization import Normalizer
from ..datasets.synthetic_dataset import (
    SyntheticVLADataset,
    build_normalizers,
    generate_episodes,
)
from ..datasets.tokenizer import CharTokenizer
from ..envs.tabletop2d import all_instruction_strings
from ..evaluation.rollout import PolicyWrapper, evaluate_policy
from ..models.flow_head import FlowVLA
from ..models.tiny_vla import TinyVLA, count_parameters
from ..utils.device import get_device
from ..utils.seed import set_seed
from .checkpoint import save_policy
from .config import TrainConfig
from .losses import masked_mse


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def _compute_loss(model, model_type: str, batch: dict) -> torch.Tensor:
    """モデル種別に応じて損失を計算する（ここが M4 と M5 の本質的な違い）。"""
    if model_type == "flow":
        return model.flow_loss(
            batch["image"], batch["state"], batch["tokens"], batch["action"], batch["pad_mask"]
        )
    pred = model(batch["image"], batch["state"], batch["tokens"])  # [B,C,A]
    return masked_mse(pred, batch["action"], batch["pad_mask"])


def run_training(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"[setup] device={device}  model_type={cfg.model_type}")

    # --- データ生成 ---
    episodes = generate_episodes(
        cfg.n_episodes, cfg.n_objects, cfg.n_goals, seed=cfg.seed, action_noise=cfg.action_noise
    )
    print(f"[data] {len(episodes)} episodes 生成 (action_noise={cfg.action_noise})")

    # --- トークナイザ・正規化（全指示文から語彙、学習データから統計） ---
    tokenizer = CharTokenizer.from_corpus(all_instruction_strings())
    action_norm, state_norm = build_normalizers(episodes)

    # --- Dataset / DataLoader ---
    ds = SyntheticVLADataset(episodes, tokenizer, cfg.chunk_len, action_norm, state_norm)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    print(f"[data] {len(ds)} 学習サンプル / vocab={tokenizer.vocab_size}")

    # --- モデル構築（ここが違い 1/2）---
    model_kwargs = dict(
        vocab_size=tokenizer.vocab_size, chunk_len=cfg.chunk_len,
        image_pool=cfg.image_pool, condition_vision=cfg.condition_vision,
    )
    model = (FlowVLA if cfg.model_type == "flow" else TinyVLA)(**model_kwargs).to(device)
    print(f"[model] {cfg.model_type} | パラメータ数 = {count_parameters(model):,}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # overfit-one-batch モード（課題用）: 1 バッチを繰り返し学習する
    fixed_batch = next(iter(loader)) if cfg.overfit_one_batch else None

    history: list[float] = []
    model.train()
    for epoch in range(cfg.epochs):
        running, nb = 0.0, 0
        iterator = (
            (fixed_batch for _ in range(len(loader))) if fixed_batch is not None else loader
        )
        for step, batch in enumerate(iterator):
            if cfg.limit_steps is not None and step >= cfg.limit_steps:
                break
            batch = _to_device(batch, device)
            loss = _compute_loss(model, cfg.model_type, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
            history.append(loss.item())
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            print(f"[train] epoch {epoch:3d}  loss={running / max(nb, 1):.5f}")

    # --- 保存 ---
    out_dir = Path(cfg.out_dir)
    save_policy(
        out_dir / "policy.pt", model, tokenizer, action_norm, state_norm,
        cfg.model_type, model_kwargs, cfg.chunk_len,
    )
    cfg.to_json(out_dir / "config.json")
    print(f"[save] {out_dir / 'policy.pt'}")

    # --- 閉ループ評価 ---
    metrics = {}
    if cfg.eval_episodes > 0:
        policy = PolicyWrapper(
            model, tokenizer, action_norm, state_norm, cfg.model_type, device, cfg.flow_steps
        )
        metrics = evaluate_policy(
            policy, n_episodes=cfg.eval_episodes, n_objects=cfg.n_objects,
            n_goals=cfg.n_goals, exec_horizon=cfg.exec_horizon,
        )
        print(f"[eval] success_rate={metrics['success_rate']:.3f}  "
              f"final_dist={metrics['mean_final_distance']:.3f}  "
              f"steps={metrics['mean_steps']:.1f}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "action_norm": action_norm,
        "state_norm": state_norm,
        "history": history,
        "metrics": metrics,
    }
