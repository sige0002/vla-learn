"""vla_learn: 小さな VLA をスクラッチで作って学ぶための教材パッケージ。

主要モジュール:
  - envs       : Tiny Tabletop 2D 環境とエキスパート方策
  - datasets   : 合成データ生成・トークナイザ・正規化・行動チャンク
  - models     : 画像/言語/状態エンコーダ、TinyVLA(MSE)、FlowVLA(flow matching)
  - training   : 学習ループ・損失・設定・チェックポイント
  - evaluation : 閉ループ rollout と評価指標
  - utils      : シード固定・device 選択
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
