"""合成データセットを LeRobotDataset 形式へエクスポートする（M6・任意）。

ねらい: 自作データを「業界標準のデータ規格」に載せ替える経験をする。
LeRobotDataset は (observation.images.*, observation.state, action, task) などの
特徴量を持つフレーム列としてエピソードを表現します。

注意:
  - LeRobot の書き込み API はバージョンで変わります。ここでは「自作 episode →
    フレーム辞書列」への変換（map_episode_to_frames）を中心に置き、これは lerobot 無しでも
    動いて単体テストできます。実際の書き込み部分はバージョン依存なので try で囲み、
    使い方を案内します（詳細は lessons/m6_lerobot_and_models.md）。

使い方:
  python scripts/make_dataset.py --out data/tabletop2d
  python scripts/export_lerobot.py --in data/tabletop2d --repo-id your-name/tabletop2d
"""
import argparse

import _bootstrap  # noqa: F401

import numpy as np

from vla_learn.datasets.synthetic_dataset import load_dataset
from vla_learn.envs.render import render_world


def map_episode_to_frames(ep: dict, img_size: int = 64) -> list[dict]:
    """自作 episode を LeRobot 風のフレーム辞書列へ変換する（lerobot 不要）。

    各フレーム:
      observation.image : [H, W, 3] uint8（LeRobot は HWC・uint8 画像が標準）
      observation.state : [3] float32
      action            : [3] float32
      task              : str（言語指示）
    """
    frames = []
    T = ep["actions"].shape[0]
    for t in range(T):
        img_chw = render_world(
            ep["objects_pos"][t], ep["objects_color"], ep["goals_pos"], ep["goals_color"],
            ep["agent"][t, :2], float(ep["agent"][t, 2]), size=img_size,
        )  # [3,H,W] float 0..1
        img_hwc = (np.transpose(img_chw, (1, 2, 0)) * 255).astype(np.uint8)  # [H,W,3] uint8
        frames.append(
            {
                "observation.image": img_hwc,
                "observation.state": ep["agent"][t].astype(np.float32),
                "action": ep["actions"][t].astype(np.float32),
                "task": ep["instruction"],
            }
        )
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description="Export synthetic dataset to LeRobotDataset")
    p.add_argument("--in", dest="inp", type=str, default="data/tabletop2d")
    p.add_argument("--repo-id", type=str, default="local/tabletop2d")
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    episodes = load_dataset(args.inp)
    print(f"[load] {len(episodes)} episodes from {args.inp}")
    # 変換ロジック自体はここで常に実行できる（lerobot 不要）
    total = sum(len(map_episode_to_frames(ep)) for ep in episodes[:1]) * len(episodes)
    print(f"[map] フレーム変換 OK（先頭エピソードで形を確認）。概算フレーム数 ~ {total}")

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:  # noqa: BLE001
        print("\n[info] lerobot が見つかりません（任意依存）。インストール:")
        print("       pip install lerobot")
        print(f"       （詳細: {type(e).__name__}）")
        print("[info] 変換ロジック map_episode_to_frames は lerobot 無しでも利用・テスト可能です。")
        return

    # --- 以下はバージョン依存。LeRobot v2 系の API を想定した一例 ---
    h = w = 64
    features = {
        "observation.image": {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (3,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (3,), "names": ["action"]},
    }
    try:
        ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, features=features)
        for ep in episodes:
            for fr in map_episode_to_frames(ep):
                ds.add_frame(fr)
            ds.save_episode()
        print(f"[done] LeRobotDataset を作成しました: {args.repo_id}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 書き込み API がこのバージョンと異なる可能性があります: {type(e).__name__}: {e}")
        print("       lessons/m6_lerobot_and_models.md の対応表と、お使いの lerobot のドキュメントを参照してください。")


if __name__ == "__main__":
    main()
