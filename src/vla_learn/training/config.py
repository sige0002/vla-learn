"""学習設定（dataclass）と JSON からの読み込み。

設定をコードと分離しておくと、ハイパーパラメータ（学習率・エポック数など）を
configs/*.json で切り替えながら実験できます。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..constants import DEFAULT_CHUNK_LEN


@dataclass
class TrainConfig:
    model_type: str = "mse"        # "mse" | "flow"
    # --- データ ---
    n_episodes: int = 800
    val_episodes: int = 120
    n_objects: int = 3
    n_goals: int = 2
    chunk_len: int = DEFAULT_CHUNK_LEN
    action_noise: float = 0.03   # データ収集時のノイズ注入（DAgger 風, 閉ループ頑健性↑）
    # --- 最適化 ---
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    # --- flow 推論 ---
    flow_steps: int = 10
    # --- 評価 ---
    eval_episodes: int = 50
    exec_horizon: int = 4
    # --- その他 ---
    seed: int = 0
    device: str | None = None
    out_dir: str = "checkpoints/mse"
    log_every: int = 1
    overfit_one_batch: bool = False   # 課題用: 1 バッチだけで過学習できるか確認
    limit_steps: int | None = None    # 動作確認用に 1 エポックのステップ数を制限

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path: str | Path | None = None, **overrides) -> TrainConfig:
    """JSON 設定（任意）を読み、overrides で上書きして TrainConfig を返す。"""
    base: dict = {}
    if path is not None:
        base = json.loads(Path(path).read_text(encoding="utf-8"))
    base.update({k: v for k, v in overrides.items() if v is not None})
    return TrainConfig(**base)
