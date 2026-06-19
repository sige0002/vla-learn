"""学習済み方策を読み込み、閉ループ評価するスクリプト。

使い方:
  python scripts/eval_policy.py --ckpt checkpoints/mse/policy.pt --n-episodes 100
  python scripts/eval_policy.py --ckpt checkpoints/flow/policy.pt --flow-steps 10
"""
import argparse

import _bootstrap  # noqa: F401

from vla_learn.evaluation.rollout import PolicyWrapper, evaluate_policy
from vla_learn.training.checkpoint import load_policy
from vla_learn.utils.device import get_device


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained VLA policy")
    p.add_argument("--ckpt", type=str, required=True, help="policy.pt のパス")
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--exec-horizon", type=int, default=4)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = get_device(args.device)
    bundle = load_policy(args.ckpt, map_location=str(device))
    policy = PolicyWrapper(
        bundle["model"], bundle["tokenizer"], bundle["action_norm"], bundle["state_norm"],
        model_type=bundle["model_type"], device=device, flow_steps=args.flow_steps,
    )
    metrics = evaluate_policy(
        policy, n_episodes=args.n_episodes, seed=args.seed, exec_horizon=args.exec_horizon
    )
    print("==== 評価結果 ====")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
