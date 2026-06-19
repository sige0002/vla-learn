"""学習済み方策のロールアウトを画像グリッドに保存して目で確認するデモ（任意・matplotlib 必要）。

使い方:
  python scripts/demo_rollout.py --ckpt checkpoints/mse/policy.pt --out assets/rollout.png
"""
import argparse

import _bootstrap  # noqa: F401

import numpy as np

from vla_learn.envs import Tabletop2DEnv
from vla_learn.evaluation.rollout import PolicyWrapper
from vla_learn.training.checkpoint import load_policy


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize a rollout as an image grid")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="assets/rollout.png")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--exec-horizon", type=int, default=4)
    p.add_argument("--flow-steps", type=int, default=10)
    args = p.parse_args()

    b = load_policy(args.ckpt)
    policy = PolicyWrapper(
        b["model"], b["tokenizer"], b["action_norm"], b["state_norm"],
        model_type=b["model_type"], flow_steps=args.flow_steps,
    )

    env = Tabletop2DEnv(max_steps=48)
    obs = env.reset(seed=args.seed)
    print(f"指示: {obs['instruction']}")
    frames = [obs["image"]]
    done = False
    while not done:
        chunk = policy.predict_chunk(obs)
        for k in range(min(args.exec_horizon, chunk.shape[0])):
            obs, _, done, info = env.step(chunk[k])
            frames.append(obs["image"])
            if done:
                break
    print(f"成功: {info['success']} / ステップ数: {info['t']}")

    from vla_learn.evaluation.visualize import save_image_grid
    # 最大 24 フレームを間引いて表示
    idx = np.linspace(0, len(frames) - 1, min(24, len(frames))).astype(int)
    save_image_grid(np.stack([frames[i] for i in idx]), args.out, ncol=8)
    print(f"保存しました: {args.out}")


if __name__ == "__main__":
    main()
