"""合成データセットを生成してディスクに保存するスクリプト（M3）。

学習スクリプトは内部で都度データ生成もできますが、データを固定して中身を観察したり、
LeRobot 形式へエクスポートしたりするにはファイルに保存しておくと便利です。

使い方:
  python scripts/make_dataset.py --n-episodes 1000 --out data/tabletop2d
"""
import argparse

import _bootstrap  # noqa: F401

from vla_learn.datasets.synthetic_dataset import generate_episodes, save_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic Tabletop2D dataset")
    p.add_argument("--n-episodes", type=int, default=1000)
    p.add_argument("--n-objects", type=int, default=3)
    p.add_argument("--n-goals", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="data/tabletop2d")
    args = p.parse_args()

    print(f"[gen] {args.n_episodes} エピソードを生成中…")
    episodes = generate_episodes(args.n_episodes, args.n_objects, args.n_goals, seed=args.seed)
    save_dataset(episodes, args.out)
    total = sum(ep["actions"].shape[0] for ep in episodes)
    print(f"[done] {len(episodes)} episodes / {total} steps を {args.out} に保存しました")


if __name__ == "__main__":
    main()
