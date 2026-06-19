"""環境とエキスパートの健全性テスト。

データが「学習可能（=お手本が成功している）」ことを保証する土台のテスト。
"""
import numpy as np

from vla_learn.envs import Tabletop2DEnv, expert_action


def test_expert_solves_task():
    succ = 0
    N = 30
    for i in range(N):
        env = Tabletop2DEnv(seed=i)
        obs = env.reset()
        done, info = False, {}
        while not done:
            obs, _, done, info = env.step(expert_action(obs["world"]))
        succ += int(info["success"])
    assert succ / N >= 0.95, f"エキスパート成功率が低すぎます: {succ}/{N}"


def test_observation_shapes_and_range():
    env = Tabletop2DEnv(seed=0)
    obs = env.reset()
    assert obs["image"].shape == (3, 64, 64)
    assert obs["image"].dtype == np.float32
    assert 0.0 <= obs["image"].min() and obs["image"].max() <= 1.0
    assert obs["state"].shape == (3,)
    assert isinstance(obs["instruction"], str) and len(obs["instruction"]) > 0
