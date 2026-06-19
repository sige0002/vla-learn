"""時間方向の窓 (temporal window) と行動チャンク (action chunking)。

VLA は 1 ステップだけでなく「これから chunk_len ステップ分の行動」をまとめて予測することが多い
（pi0 や ACT, SmolVLA など）。これを action chunking と呼びます。利点:
  - 推論回数が減る（chunk をまとめて出して順に実行できる）
  - 行動が滑らかになりやすい

ここでは「ある時刻 t から chunk_len ステップ分の行動」を取り出し、
エピソード終端で足りない分はパディングして pad_mask で「ここは無効」と印を付けます。
"""
from __future__ import annotations

import numpy as np


def extract_action_chunk(
    actions: np.ndarray,  # [T, action_dim]
    t: int,
    chunk_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """actions[t : t+chunk_len] を取り出す。

    足りない分は最後の行動を繰り返してパディングし、pad_mask=0 を立てる。

    Returns:
        chunk:    [chunk_len, action_dim]
        pad_mask: [chunk_len]  1=有効ステップ / 0=パディング
    """
    T, action_dim = actions.shape
    chunk = np.zeros((chunk_len, action_dim), dtype=np.float32)
    pad_mask = np.zeros((chunk_len,), dtype=np.float32)

    n_valid = min(chunk_len, T - t)
    chunk[:n_valid] = actions[t : t + n_valid]
    pad_mask[:n_valid] = 1.0
    if n_valid < chunk_len:
        chunk[n_valid:] = actions[T - 1]  # 最後の行動で埋める（実行しても無害な「静止」に近い）
    return chunk, pad_mask
