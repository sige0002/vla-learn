"""解析的なエキスパート方策（人手のルールベース）。

模倣学習 (imitation learning) では「お手本（エキスパート）の行動」を集めて、
モデルにそれを真似させます。ここではそのお手本を、簡単な状態機械で作ります:

  1) 対象ブロックを持っていない → ブロックへ近づき、十分近ければグリッパを閉じて掴む
  2) 対象ブロックを持っている   → ゴールへ近づき、十分近ければグリッパを開いて置く

ニューラルネットは一切使いません。ワールド状態を直接読めるので毎回成功します。
"""
from __future__ import annotations

import numpy as np

from ..constants import ACTION_DIM, MAX_STEP
from .tabletop2d import WorldState

# 掴む/置くを判定する距離。env の GRASP/SUCCESS 半径より内側にしておけば、
# 多少ブレても掴める/置ける（粗い学習方策でも成功しやすい）。
PICK_THRESH = 0.10    # この距離以内まで近づいたら掴む
PLACE_THRESH = 0.08   # この距離以内まで近づいたら置く


def expert_action(world: WorldState) -> np.ndarray:
    """現在のワールド状態から、お手本の行動 [dx, dy, grip_cmd] を返す。"""
    agent = world.agent_xy
    obj = world.objects_pos[world.target_obj]
    goal = world.goals_pos[world.target_goal]
    holding = world.held == world.target_obj

    action = np.zeros(ACTION_DIM, dtype=np.float32)

    if not holding:
        d = obj - agent
        if np.linalg.norm(d) < PICK_THRESH:
            # 近いのでその場で掴む（移動なし・グリッパ閉）
            action[:] = (0.0, 0.0, 1.0)
        else:
            step = np.clip(d, -MAX_STEP, MAX_STEP)
            action[:] = (step[0], step[1], 0.0)  # 近づく・グリッパ開
    else:
        d = goal - agent
        if np.linalg.norm(d) < PLACE_THRESH:
            action[:] = (0.0, 0.0, 0.0)  # 置く（グリッパ開）
        else:
            step = np.clip(d, -MAX_STEP, MAX_STEP)
            action[:] = (step[0], step[1], 1.0)  # 運ぶ・グリッパ閉
    return action
