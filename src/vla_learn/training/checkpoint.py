"""学習済み方策の保存 / 読み込み。

モデルの重みだけでなく「再構築に必要なすべて」（モデル種別・コンストラクタ引数・
トークナイザ語彙・正規化統計）を 1 つの .pt にまとめます。こうすると評価スクリプトは
このファイル 1 個で方策を完全に復元できます。
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..datasets.normalization import Normalizer
from ..datasets.tokenizer import CharTokenizer
from ..models.flow_head import FlowVLA
from ..models.tiny_vla import TinyVLA


def save_policy(
    path: str | Path,
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    action_norm: Normalizer,
    state_norm: Normalizer,
    model_type: str,
    model_kwargs: dict,
    chunk_len: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": model_type,
            "model_kwargs": model_kwargs,
            "model_state": model.state_dict(),
            "tokenizer": {"vocab": tokenizer.vocab, "max_len": tokenizer.max_len},
            "action_norm": {"mean": action_norm.mean.tolist(), "std": action_norm.std.tolist()},
            "state_norm": {"mean": state_norm.mean.tolist(), "std": state_norm.std.tolist()},
            "chunk_len": chunk_len,
        },
        path,
    )


def load_policy(path: str | Path, map_location: str = "cpu") -> dict:
    """チェックポイントを読み、(model, tokenizer, action_norm, state_norm, ...) を組み立てて返す。"""
    import numpy as np

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    tok = CharTokenizer(ckpt["tokenizer"]["vocab"], ckpt["tokenizer"]["max_len"])
    action_norm = Normalizer(
        np.array(ckpt["action_norm"]["mean"], np.float32),
        np.array(ckpt["action_norm"]["std"], np.float32),
    )
    state_norm = Normalizer(
        np.array(ckpt["state_norm"]["mean"], np.float32),
        np.array(ckpt["state_norm"]["std"], np.float32),
    )
    cls = FlowVLA if ckpt["model_type"] == "flow" else TinyVLA
    model = cls(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return {
        "model": model,
        "model_type": ckpt["model_type"],
        "tokenizer": tok,
        "action_norm": action_norm,
        "state_norm": state_norm,
        "chunk_len": ckpt["chunk_len"],
    }
