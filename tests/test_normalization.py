"""正規化の往復（normalize → denormalize）が元に戻ることのテスト。"""
import numpy as np
import torch

from vla_learn.datasets import Normalizer


def test_roundtrip_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 3)).astype(np.float32) * 5 + 2
    norm = Normalizer.fit(x)
    back = norm.denormalize(norm.normalize(x))
    assert np.allclose(x, back, atol=1e-4)


def test_roundtrip_torch():
    rng = np.random.default_rng(1)
    x_np = rng.normal(size=(50, 3)).astype(np.float32)
    norm = Normalizer.fit(x_np)
    x = torch.from_numpy(x_np)
    back = norm.denormalize(norm.normalize(x))
    assert torch.allclose(x, back, atol=1e-4)


def test_normalized_stats_are_standardized():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1000, 3)).astype(np.float32) * 3 + 1
    norm = Normalizer.fit(x)
    z = norm.normalize(x)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-1)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-1)


def test_zero_std_does_not_divide_by_zero():
    x = np.ones((10, 3), dtype=np.float32)  # 分散 0 の列
    norm = Normalizer.fit(x)
    z = norm.normalize(x)
    assert np.isfinite(z).all()
