"""MSE（回帰）版 TinyVLA を学習するスクリプト（M4）。

使い方:
  python scripts/train_mse.py                       # 既定設定で学習
  python scripts/train_mse.py --config configs/m4_mse.json
  python scripts/train_mse.py --epochs 30 --n-episodes 1500 --out-dir checkpoints/mse
"""
import argparse

import _bootstrap  # noqa: F401  (src をパスに追加)

from vla_learn.training.config import load_config
from vla_learn.training.trainer import run_training


def main() -> None:
    p = argparse.ArgumentParser(description="Train TinyVLA (MSE)")
    p.add_argument("--config", type=str, default=None, help="JSON 設定ファイル（任意）")
    p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--chunk-len", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(
        args.config,
        model_type="mse",
        n_episodes=args.n_episodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        chunk_len=args.chunk_len,
        eval_episodes=args.eval_episodes,
        out_dir=args.out_dir,   # None なら config 値（無ければ TrainConfig 既定 checkpoints/mse）
        seed=args.seed,
        device=args.device,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
