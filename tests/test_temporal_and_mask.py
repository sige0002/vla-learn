"""extract_action_chunk（時間窓）と masked_mse（マスク付き損失）の境界テスト。"""
import numpy as np
import torch

from vla_learn.datasets.temporal import extract_action_chunk
from vla_learn.functional import masked_mse


def _actions(T=10, A=3):
    return np.arange(T * A, dtype=np.float32).reshape(T, A)


def test_chunk_no_padding_at_start():
    acts = _actions(T=10)
    chunk, pad = extract_action_chunk(acts, t=0, chunk_len=8)
    assert chunk.shape == (8, 3) and pad.shape == (8,)
    assert np.array_equal(chunk, acts[0:8])
    assert np.array_equal(pad, np.ones(8, dtype=np.float32))


def test_chunk_max_padding_at_last_step():
    acts = _actions(T=10)
    chunk, pad = extract_action_chunk(acts, t=9, chunk_len=8)
    assert np.array_equal(chunk[0], acts[9])
    # 残り 7 ステップは最後の行動の繰り返しで埋まり pad_mask=0
    assert np.array_equal(chunk[1:], np.tile(acts[9], (7, 1)))
    assert np.array_equal(pad, np.array([1] + [0] * 7, dtype=np.float32))


def test_masked_mse_without_mask_equals_plain_mse():
    torch.manual_seed(0)
    pred, target = torch.randn(4, 8, 3), torch.randn(4, 8, 3)
    assert torch.allclose(masked_mse(pred, target, None), ((pred - target) ** 2).mean())


def test_masked_mse_ignores_padded_steps():
    pred = torch.zeros(1, 4, 2)
    target = torch.ones(1, 4, 2)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    # 有効 2 ステップの誤差は 1、pad 部分の誤差は無視されるので平均は 1
    assert float(masked_mse(pred, target, mask)) == 1.0
    # pad 部分の値をどれだけ壊しても損失は変わらない
    pred2 = pred.clone()
    pred2[:, 2:] = 100.0
    assert float(masked_mse(pred2, target, mask)) == 1.0


def test_masked_mse_all_padded_is_zero_not_nan():
    pred, target = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    mask = torch.zeros(2, 4)
    loss = masked_mse(pred, target, mask)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
