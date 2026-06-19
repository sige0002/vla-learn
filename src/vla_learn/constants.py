"""教材全体で共有する定数。

ここを「唯一の真実の源 (single source of truth)」にしておくことで、
環境・データセット・モデル・学習・評価のあいだで shape や単位がズレないようにします。
"""
from __future__ import annotations

# 画像サイズ（正方形）。VLA の観測画像 [3, H, W] の H=W。
IMG_SIZE: int = 64

# 行動次元: [dx, dy, grip_cmd]
#   dx, dy : エンドエフェクタ（グリッパ）の移動量（ワールド座標の差分）
#   grip_cmd: グリッパ指令 0.0=開く / 1.0=閉じる（>=0.5 で閉と解釈）
ACTION_DIM: int = 3

# 固有受容感覚 (proprioception) の状態次元: [ax, ay, gripper]
STATE_DIM: int = 3

# 色の定義。物体（ブロック）とゴールはこの 4 色から選ばれる。
COLOR_NAMES: tuple[str, ...] = ("red", "green", "blue", "yellow")
COLOR_JA: dict[str, str] = {"red": "赤", "green": "緑", "blue": "青", "yellow": "黄"}
# 描画用 RGB（0..1）
COLOR_RGB: dict[str, tuple[float, float, float]] = {
    "red": (0.90, 0.20, 0.20),
    "green": (0.20, 0.80, 0.30),
    "blue": (0.25, 0.45, 0.95),
    "yellow": (0.95, 0.85, 0.20),
}

# 物理（簡易）パラメータ
# 注: GRASP/SUCCESS 半径はやや広めにしてあります。学習した方策は数手ごとに ±0.05〜0.08 程度
#     位置がブレるため、半径が狭すぎると「ほぼ届いているのに掴めない」状態が頻発します。
#     おもちゃ問題なので、粗い方策でも掴める寛容な設定にしてあります（解析エキスパートは 100% 成功）。
MAX_STEP: float = 0.08      # 1 ステップで動ける最大距離（各軸）
GRASP_RADIUS: float = 0.18  # この距離以内でグリッパを閉じると物体を掴める
SUCCESS_RADIUS: float = 0.12  # 対象物体がゴールにこの距離以内で成功
OBJ_RADIUS: float = 0.055   # 物体の半径（描画・配置用）
GOAL_RADIUS: float = 0.085  # ゴールマーカの半径（描画用）

# デフォルトの行動チャンク長（chunk_len）。一度に予測する未来の行動ステップ数。
DEFAULT_CHUNK_LEN: int = 8
