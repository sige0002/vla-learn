"""Flow Matching 版 FlowVLA を学習するスクリプト（M5）。

使い方:
  python scripts/train_flow.py
  python scripts/train_flow.py --config configs/m5_flow.json
  python scripts/train_flow.py --epochs 40 --flow-steps 10 --out-dir checkpoints/flow
"""
import argparse

import _bootstrap  # noqa: F401

from vla_learn.training.config import load_config
from vla_learn.training.trainer import run_training


def main() -> None:
    p = argparse.ArgumentParser(description="Train FlowVLA (flow matching)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--chunk-len", type=int, default=None)
    p.add_argument("--flow-steps", type=int, default=None, help="推論時の Euler 積分ステップ数")
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(
        args.config,
        model_type="flow",
        n_episodes=args.n_episodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        chunk_len=args.chunk_len,
        flow_steps=args.flow_steps,
        eval_episodes=args.eval_episodes,
        out_dir=args.out_dir,   # None なら config 値を尊重
        seed=args.seed,
        device=args.device,
    )
    # CLI も config も out_dir を指定しなかった場合のみ、flow 用の既定にする
    if args.out_dir is None and (args.config is None or cfg.out_dir == "checkpoints/mse"):
        cfg.out_dir = "checkpoints/flow"
    run_training(cfg)


if __name__ == "__main__":
    main()
