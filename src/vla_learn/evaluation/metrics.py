"""評価指標のまとめ（rollout 結果リスト → 集計）。"""
from __future__ import annotations

import numpy as np


def aggregate(results: list[dict]) -> dict:
    """rollout_episode の結果リストを平均指標に集計する。"""
    if not results:
        return {"success_rate": 0.0, "mean_final_distance": float("nan"), "mean_steps": float("nan")}
    return {
        "success_rate": float(np.mean([r["success"] for r in results])),
        "mean_final_distance": float(np.mean([r["final_distance"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "n_episodes": len(results),
    }
