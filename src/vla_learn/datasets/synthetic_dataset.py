"""合成 VLA データセット（エキスパートのデモから生成）。

データ生成の流れ:
  1. 環境を reset
  2. エキスパート方策で終端（成功）まで行動を集める
  3. 各時刻の「描画に必要な状態」と「行動」を記録

学習時はディスクには低次元の状態だけを保存し、画像は __getitem__ で都度レンダリングします。
（画像を全部保存するとギガ単位になるため。状態→画像のパイプライン自体も学びになります。）

1 サンプル = (image, state, instruction_tokens) → action_chunk
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import DEFAULT_CHUNK_LEN, IMG_SIZE
from ..envs.expert import expert_action
from ..envs.render import render_world
from ..envs.tabletop2d import Tabletop2DEnv
from .normalization import Normalizer
from .temporal import extract_action_chunk
from .tokenizer import CharTokenizer


# ----------------------------------------------------------------------
# 1) エピソード生成
# ----------------------------------------------------------------------
def generate_episodes(
    n_episodes: int,
    n_objects: int = 3,
    n_goals: int = 2,
    seed: int = 0,
    only_success: bool = True,
    action_noise: float = 0.0,
) -> list[dict]:
    """エキスパートのデモを n_episodes 本ぶん生成して返す。

    action_noise > 0 のとき「ノイズ注入（DART / DAgger 風）」を行う:
      実行する行動にだけ少量のガウスノイズを足して軌道を少しずらし、
      記録するラベルは“その（ずれた）状態における”クリーンなエキスパート行動にする。
      こうすると「軌道から少し外れた状態 → 戻すための正しい行動」のペアが集まり、
      閉ループ実行時の誤差蓄積（distribution shift）に強くなる。
      （素朴な模倣学習がなぜ崩れるか、の対策。M2/M4 の重要トピック）
    """
    base = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 7)
    episodes: list[dict] = []
    guard = 0
    while len(episodes) < n_episodes and guard < n_episodes * 20 + 50:
        guard += 1
        ep_seed = int(base.integers(1 << 31))
        env = Tabletop2DEnv(n_objects=n_objects, n_goals=n_goals, seed=ep_seed)
        obs = env.reset()

        agents, objs, acts = [], [], []
        done, info = False, {}
        while not done:
            w = obs["world"]
            a = expert_action(w)  # ← 記録するラベルは常にクリーンなエキスパート行動
            agents.append([float(w.agent_xy[0]), float(w.agent_xy[1]), float(w.gripper)])
            objs.append(w.objects_pos.copy())
            acts.append(a.copy())
            # 実行だけノイズを足して軌道をずらす（grip は乱さない）
            a_exec = a.copy()
            if action_noise > 0:
                a_exec[0] += noise_rng.normal(0.0, action_noise)
                a_exec[1] += noise_rng.normal(0.0, action_noise)
            obs, _, done, info = env.step(a_exec)

        if only_success and not info.get("success", False):
            continue

        w = env.world
        episodes.append(
            {
                "agent": np.asarray(agents, dtype=np.float32),       # [T, 3]
                "objects_pos": np.asarray(objs, dtype=np.float32),    # [T, N, 2]
                "objects_color": w.objects_color.copy(),              # [N]
                "goals_pos": w.goals_pos.copy(),                      # [M, 2]
                "goals_color": w.goals_color.copy(),                  # [M]
                "target_obj": int(w.target_obj),
                "target_goal": int(w.target_goal),
                "instruction": w.instruction,
                "actions": np.asarray(acts, dtype=np.float32),        # [T, 3]
            }
        )
    return episodes


# ----------------------------------------------------------------------
# 2) 正規化統計
# ----------------------------------------------------------------------
def build_normalizers(episodes: list[dict]) -> tuple[Normalizer, Normalizer]:
    """全エピソードから行動・状態の Normalizer を作る。"""
    actions = np.concatenate([ep["actions"] for ep in episodes], axis=0)  # [sumT, 3]
    states = np.concatenate([ep["agent"] for ep in episodes], axis=0)     # [sumT, 3]
    return Normalizer.fit(actions), Normalizer.fit(states)


# ----------------------------------------------------------------------
# 3) PyTorch Dataset
# ----------------------------------------------------------------------
class SyntheticVLADataset(Dataset):
    """(image, state, tokens) → action_chunk を返す Dataset。"""

    def __init__(
        self,
        episodes: list[dict],
        tokenizer: CharTokenizer,
        chunk_len: int = DEFAULT_CHUNK_LEN,
        action_normalizer: Normalizer | None = None,
        state_normalizer: Normalizer | None = None,
        img_size: int = IMG_SIZE,
    ) -> None:
        self.episodes = episodes
        self.tokenizer = tokenizer
        self.chunk_len = chunk_len
        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer
        self.img_size = img_size

        # (episode_idx, t) の平坦インデックスを作る
        self.index: list[tuple[int, int]] = []
        for ei, ep in enumerate(episodes):
            T = ep["actions"].shape[0]
            for t in range(T):
                self.index.append((ei, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ei, t = self.index[idx]
        ep = self.episodes[ei]

        # --- 画像をレンダリング（state → image） ---
        image = render_world(
            ep["objects_pos"][t], ep["objects_color"],
            ep["goals_pos"], ep["goals_color"],
            ep["agent"][t, :2], float(ep["agent"][t, 2]),
            size=self.img_size,
        )  # [3, H, W]

        # --- 固有受容状態 ---
        state = ep["agent"][t].astype(np.float32)  # [3]
        if self.state_normalizer is not None:
            state = self.state_normalizer.normalize(state)

        # --- 言語指示 → トークン ID ---
        tokens = np.asarray(self.tokenizer.encode(ep["instruction"]), dtype=np.int64)  # [L]

        # --- 行動チャンク ---
        chunk, pad_mask = extract_action_chunk(ep["actions"], t, self.chunk_len)  # [C,3],[C]
        if self.action_normalizer is not None:
            chunk = self.action_normalizer.normalize(chunk)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),  # float32 [3,H,W]
            "state": torch.from_numpy(np.ascontiguousarray(state)),  # float32 [3]
            "tokens": torch.from_numpy(tokens),                      # int64   [L]
            "action": torch.from_numpy(np.ascontiguousarray(chunk)),  # float32 [C,3]
            "pad_mask": torch.from_numpy(pad_mask),                  # float32 [C]
        }


# ----------------------------------------------------------------------
# 4) ディスク保存 / 読み込み
# ----------------------------------------------------------------------
def save_dataset(episodes: list[dict], root: str | Path) -> None:
    root = Path(root)
    (root / "episodes").mkdir(parents=True, exist_ok=True)
    for i, ep in enumerate(episodes):
        np.savez(
            root / "episodes" / f"ep_{i:05d}.npz",
            agent=ep["agent"],
            objects_pos=ep["objects_pos"],
            objects_color=ep["objects_color"],
            goals_pos=ep["goals_pos"],
            goals_color=ep["goals_color"],
            target_obj=ep["target_obj"],
            target_goal=ep["target_goal"],
            instruction=np.array(ep["instruction"]),
            actions=ep["actions"],
        )
    (root / "meta.json").write_text(
        json.dumps({"n_episodes": len(episodes)}, ensure_ascii=False), encoding="utf-8"
    )


def load_dataset(root: str | Path) -> list[dict]:
    root = Path(root)
    files = sorted((root / "episodes").glob("ep_*.npz"))
    episodes = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        episodes.append(
            {
                "agent": d["agent"],
                "objects_pos": d["objects_pos"],
                "objects_color": d["objects_color"],
                "goals_pos": d["goals_pos"],
                "goals_color": d["goals_color"],
                "target_obj": int(d["target_obj"]),
                "target_goal": int(d["target_goal"]),
                "instruction": str(d["instruction"]),
                "actions": d["actions"],
            }
        )
    return episodes
