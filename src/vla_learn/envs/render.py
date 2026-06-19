"""ワールド状態を 64x64 の RGB 画像に描画する（NumPy のみ）。

VLA の「Vision（視覚）」入力を作る部分です。物体・ゴール・グリッパを色付きで描くので、
モデルは画像と言語指示を見て「どの色をどこへ運ぶか」を学べます。

座標系: ワールドは [0,1] x [0,1]。x は右方向、y は上方向。
画像配列は [3, H, W]（チャンネル先頭, row=上→下）。world(x,y) → pixel(col=x*W, row=(1-y)*H)。
"""
from __future__ import annotations

import numpy as np

from ..constants import (
    COLOR_RGB,
    GOAL_RADIUS,
    IMG_SIZE,
    OBJ_RADIUS,
    COLOR_NAMES,
)

_BG = 0.12  # 背景のグレー値


def _disk_mask(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    """中心 (cx,cy) 半径 r の円内を True にするマスク。cx,cy,r はピクセル単位。"""
    ys, xs = np.mgrid[0:h, 0:w]
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r


def _world_to_pixel(x: float, y: float, size: int) -> tuple[float, float]:
    """ワールド座標 (x,y)∈[0,1]^2 を画像のピクセル (col, row) に変換。"""
    col = x * (size - 1)
    row = (1.0 - y) * (size - 1)
    return col, row


def render_world(
    objects_pos: np.ndarray,      # [N, 2]
    objects_color: np.ndarray,    # [N]   各物体の色 index
    goals_pos: np.ndarray,        # [M, 2]
    goals_color: np.ndarray,      # [M]
    agent_xy: np.ndarray,         # [2]
    gripper: float,               # 0=開 / 1=閉
    size: int = IMG_SIZE,
) -> np.ndarray:
    """ワールド状態を [3, size, size] の float32 画像（値域 0..1）に描画する。"""
    img = np.full((3, size, size), _BG, dtype=np.float32)

    obj_r = OBJ_RADIUS * (size - 1)
    goal_r = GOAL_RADIUS * (size - 1)

    # --- ゴール（薄いリング）を先に描く ---
    for (gx, gy), c in zip(goals_pos, goals_color):
        col, row = _world_to_pixel(gx, gy, size)
        outer = _disk_mask(size, size, col, row, goal_r)
        inner = _disk_mask(size, size, col, row, goal_r * 0.6)
        ring = outer & ~inner
        rgb = COLOR_RGB[COLOR_NAMES[int(c)]]
        for ch in range(3):
            img[ch][ring] = rgb[ch] * 0.7  # ゴールは少し暗めのリング

    # --- 物体（塗りつぶしの円）を描く ---
    for (ox, oy), c in zip(objects_pos, objects_color):
        col, row = _world_to_pixel(ox, oy, size)
        mask = _disk_mask(size, size, col, row, obj_r)
        rgb = COLOR_RGB[COLOR_NAMES[int(c)]]
        for ch in range(3):
            img[ch][mask] = rgb[ch]

    # --- グリッパ（エンドエフェクタ）を白で描く ---
    col, row = _world_to_pixel(float(agent_xy[0]), float(agent_xy[1]), size)
    agent_r = 0.045 * (size - 1)
    if gripper >= 0.5:
        # 閉: 白い塗りつぶし円
        mask = _disk_mask(size, size, col, row, agent_r)
        for ch in range(3):
            img[ch][mask] = 1.0
    else:
        # 開: 白いリング（中は空き）
        outer = _disk_mask(size, size, col, row, agent_r)
        inner = _disk_mask(size, size, col, row, agent_r * 0.5)
        ring = outer & ~inner
        for ch in range(3):
            img[ch][ring] = 1.0

    return img
