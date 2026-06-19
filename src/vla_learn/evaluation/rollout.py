"""閉ループ評価（rollout）。

学習中の損失が下がっても、それは「お手本との一致度」にすぎません。本当に知りたいのは
「実際に環境で動かしてタスクを成功できるか」です。そこで、学習した方策を環境に接続して
動かし、成功率などを測ります。

行動チャンクは「receding horizon（後退ホライズン）」で使います:
  chunk を予測 → 先頭 exec_horizon ステップだけ実行 → 観測し直して再予測…を繰り返す。
"""
from __future__ import annotations

import numpy as np
import torch

from ..datasets.normalization import Normalizer
from ..datasets.tokenizer import CharTokenizer
from ..envs.tabletop2d import Tabletop2DEnv


class PolicyWrapper:
    """学習済みモデルを「obs → 行動チャンク」に変換するラッパ（正規化・device 処理込み）。"""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: CharTokenizer,
        action_norm: Normalizer,
        state_norm: Normalizer,
        model_type: str = "mse",
        device: str | torch.device = "cpu",
        flow_steps: int = 10,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.action_norm = action_norm
        self.state_norm = state_norm
        self.model_type = model_type
        self.device = torch.device(device)
        self.flow_steps = flow_steps

    @torch.no_grad()
    def predict_chunk(self, obs: dict) -> np.ndarray:
        img = torch.from_numpy(np.ascontiguousarray(obs["image"]))[None].to(self.device)  # [1,3,H,W]
        state_np = self.state_norm.normalize(obs["state"].astype(np.float32))
        state = torch.from_numpy(np.ascontiguousarray(state_np))[None].to(self.device)    # [1,3]
        tokens = torch.tensor(
            [self.tokenizer.encode(obs["instruction"])], dtype=torch.long, device=self.device
        )  # [1, L]

        if self.model_type == "flow":
            a = self.model.sample(img, state, tokens, n_steps=self.flow_steps)  # [1,C,3]
        else:
            a = self.model(img, state, tokens)  # [1,C,3]
        a = self.action_norm.denormalize(a)[0].cpu().numpy()  # [C,3]（生の行動に戻す）
        return a


def rollout_episode(
    env: Tabletop2DEnv,
    policy: PolicyWrapper,
    exec_horizon: int = 4,
    seed: int | None = None,
) -> dict:
    """1 エピソードを閉ループで実行し、結果を返す。"""
    obs = env.reset(seed=seed)
    done, info = False, {"success": False, "t": 0}
    while not done:
        chunk = policy.predict_chunk(obs)  # [C, 3]
        for k in range(min(exec_horizon, chunk.shape[0])):
            obs, _, done, info = env.step(chunk[k])
            if done:
                break
    w = env.world
    final_dist = float(
        np.linalg.norm(w.objects_pos[w.target_obj] - w.goals_pos[w.target_goal])
    )
    return {"success": bool(info["success"]), "steps": int(info["t"]), "final_distance": final_dist}


def evaluate_policy(
    policy: PolicyWrapper,
    n_episodes: int = 50,
    seed: int = 10_000,
    n_objects: int = 3,
    n_goals: int = 2,
    max_steps: int = 48,
    exec_horizon: int = 4,
) -> dict:
    """複数エピソードで評価し、平均指標を返す。"""
    results = []
    for i in range(n_episodes):
        env = Tabletop2DEnv(n_objects=n_objects, n_goals=n_goals, max_steps=max_steps)
        results.append(rollout_episode(env, policy, exec_horizon=exec_horizon, seed=seed + i))
    succ = np.mean([r["success"] for r in results])
    dist = np.mean([r["final_distance"] for r in results])
    steps = np.mean([r["steps"] for r in results])
    return {
        "success_rate": float(succ),
        "mean_final_distance": float(dist),
        "mean_steps": float(steps),
        "n_episodes": n_episodes,
    }
