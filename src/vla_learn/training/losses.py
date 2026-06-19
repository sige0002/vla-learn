"""損失関数。

masked_mse は「パディングしたステップを損失から除外する MSE」です。
行動チャンクの末尾はエピソード終端でパディングされることがあるため、
pad_mask=0 の場所を平均から外します。

実体は循環 import 回避のため vla_learn.functional に置き、ここから再公開しています。
学習コードでは `from vla_learn.training.losses import masked_mse` で使えます。
"""
from __future__ import annotations

from ..functional import masked_mse

__all__ = ["masked_mse"]
