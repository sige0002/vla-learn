"""Tiny Tabletop 2D Pick-and-Place 環境（物理エンジン不要・実機不要）。

VLA を学ぶための最小タスク:
  「指定された色のブロックを、指定された色のゴールへ運ぶ」

観測 (observation):
  - image       : [3, 64, 64] float32   … 視覚入力（Vision）
  - state       : [3]         float32   … グリッパの固有受容状態 [ax, ay, gripper]
  - instruction : str                   … 言語指示（Language）

行動 (action): [dx, dy, grip_cmd]
  - dx, dy   : グリッパの移動量（ワールド座標差分, 各軸 ±MAX_STEP にクリップ）
  - grip_cmd : 0=開 / 1=閉（>=0.5 で閉と解釈）

「掴む」ルール（簡易）:
  グリッパを閉じた瞬間に GRASP_RADIUS 以内の物体があれば、その物体がグリッパに追従する。
  グリッパを開くと、その場に物体を離す。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import (
    ACTION_DIM,
    COLOR_NAMES,
    COLOR_JA,
    GRASP_RADIUS,
    IMG_SIZE,
    MAX_STEP,
    OBJ_RADIUS,
    SUCCESS_RADIUS,
)
from .render import render_world

# 言語指示テンプレート（日本語）。{o}=物体色, {g}=ゴール色 をあとで埋める。
INSTRUCTION_TEMPLATES: tuple[str, ...] = (
    "{o}のブロックを{g}のゴールに置いて",
    "{o}のブロックを{g}のエリアまで運んで",
    "{g}のゴールに{o}のブロックを動かして",
    "{o}ブロックをつかんで{g}ゴールへ",
)


@dataclass
class WorldState:
    """レンダリングと判定に必要な「世界の状態」をまとめた入れ物。"""

    objects_pos: np.ndarray   # [N, 2]
    objects_color: np.ndarray  # [N]
    goals_pos: np.ndarray      # [M, 2]
    goals_color: np.ndarray    # [M]
    agent_xy: np.ndarray       # [2]
    gripper: float             # 0 or 1
    target_obj: int
    target_goal: int
    held: int = -1             # 追従中の物体 index（-1 で無し）
    instruction: str = ""


def all_instruction_strings() -> list[str]:
    """語彙（vocabulary）構築用に、ありうる全指示文を列挙する。"""
    out = []
    for tmpl in INSTRUCTION_TEMPLATES:
        for o in COLOR_NAMES:
            for g in COLOR_NAMES:
                out.append(tmpl.format(o=COLOR_JA[o], g=COLOR_JA[g]))
    return out


class Tabletop2DEnv:
    """OpenAI Gym 風の最小 API（reset / step）を持つ環境。"""

    def __init__(
        self,
        n_objects: int = 3,
        n_goals: int = 2,
        max_steps: int = 48,
        size: int = IMG_SIZE,
        seed: int | None = None,
    ) -> None:
        assert 1 <= n_objects <= len(COLOR_NAMES)
        assert 1 <= n_goals <= len(COLOR_NAMES)
        self.n_objects = n_objects
        self.n_goals = n_goals
        self.max_steps = max_steps
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.world: WorldState | None = None

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng = self.rng

        # 物体・ゴールの色はそれぞれ重複なしで選ぶ（色で一意に指定できるように）
        obj_colors = rng.choice(len(COLOR_NAMES), size=self.n_objects, replace=False)
        goal_colors = rng.choice(len(COLOR_NAMES), size=self.n_goals, replace=False)

        objects_pos = self._sample_positions(self.n_objects, rng, margin=OBJ_RADIUS + 0.02)
        goals_pos = self._sample_positions(self.n_goals, rng, margin=0.12)
        agent_xy = rng.uniform(0.1, 0.9, size=2).astype(np.float32)

        target_obj = int(rng.integers(self.n_objects))
        target_goal = int(rng.integers(self.n_goals))

        tmpl = INSTRUCTION_TEMPLATES[int(rng.integers(len(INSTRUCTION_TEMPLATES)))]
        instruction = tmpl.format(
            o=COLOR_JA[COLOR_NAMES[int(obj_colors[target_obj])]],
            g=COLOR_JA[COLOR_NAMES[int(goal_colors[target_goal])]],
        )

        self.world = WorldState(
            objects_pos=objects_pos.astype(np.float32),
            objects_color=obj_colors.astype(np.int64),
            goals_pos=goals_pos.astype(np.float32),
            goals_color=goal_colors.astype(np.int64),
            agent_xy=agent_xy,
            gripper=0.0,
            target_obj=target_obj,
            target_goal=target_goal,
            held=-1,
            instruction=instruction,
        )
        self.t = 0
        return self._get_obs()

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        assert self.world is not None, "step の前に reset を呼んでください"
        w = self.world
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        dx = float(np.clip(action[0], -MAX_STEP, MAX_STEP))
        dy = float(np.clip(action[1], -MAX_STEP, MAX_STEP))
        grip_cmd = 1.0 if action[2] >= 0.5 else 0.0

        # 1) グリッパを動かす
        w.agent_xy = np.clip(w.agent_xy + np.array([dx, dy], np.float32), 0.0, 1.0)

        # 2) 掴む / 離す
        if grip_cmd >= 0.5:
            if w.held < 0:
                # いちばん近い物体が GRASP_RADIUS 以内なら掴む
                dists = np.linalg.norm(w.objects_pos - w.agent_xy[None, :], axis=1)
                nearest = int(np.argmin(dists))
                if dists[nearest] <= GRASP_RADIUS:
                    w.held = nearest
            if w.held >= 0:
                w.objects_pos[w.held] = w.agent_xy  # 追従
        else:
            w.held = -1  # 離す（その場に残る）
        w.gripper = grip_cmd

        self.t += 1
        success = self._is_success()
        done = bool(success or self.t >= self.max_steps)
        reward = 1.0 if success else 0.0
        info = {"success": bool(success), "t": self.t}
        return self._get_obs(), reward, done, info

    # ------------------------------------------------------------------
    def _is_success(self) -> bool:
        w = self.world
        d = float(np.linalg.norm(w.objects_pos[w.target_obj] - w.goals_pos[w.target_goal]))
        return d < SUCCESS_RADIUS

    def _get_obs(self) -> dict:
        w = self.world
        image = render_world(
            w.objects_pos, w.objects_color, w.goals_pos, w.goals_color,
            w.agent_xy, w.gripper, size=self.size,
        )
        state = np.array([w.agent_xy[0], w.agent_xy[1], w.gripper], dtype=np.float32)
        return {
            "image": image,
            "state": state,
            "instruction": w.instruction,
            # 以下はデータ生成・評価のための付加情報（推論時には使わない）
            "world": w,
        }

    def _sample_positions(self, n: int, rng: np.random.Generator, margin: float) -> np.ndarray:
        """互いに離れた n 個の位置をサンプル（重なりを避ける）。"""
        pts: list[np.ndarray] = []
        tries = 0
        while len(pts) < n and tries < 1000:
            tries += 1
            p = rng.uniform(margin, 1.0 - margin, size=2)
            if all(np.linalg.norm(p - q) > 0.22 for q in pts):
                pts.append(p)
        while len(pts) < n:  # 念のためのフォールバック
            pts.append(rng.uniform(margin, 1.0 - margin, size=2))
        return np.stack(pts)
