"""モデルの forward / sample が正しい shape を返すことのテスト。"""
import torch

from vla_learn.constants import ACTION_DIM, IMG_SIZE
from vla_learn.models import FlowVLA, TinyVLA, count_parameters

VOCAB = 30
CHUNK = 8
B = 5
L = 16


def _dummy_inputs():
    image = torch.rand(B, 3, IMG_SIZE, IMG_SIZE)
    state = torch.rand(B, 3)
    tokens = torch.randint(0, VOCAB, (B, L))
    return image, state, tokens


def test_tiny_vla_forward_shape():
    model = TinyVLA(vocab_size=VOCAB, chunk_len=CHUNK)
    out = model(*_dummy_inputs())
    assert out.shape == (B, CHUNK, ACTION_DIM)
    n = count_parameters(model)
    assert 50_000 < n < 5_000_000, f"想定外のパラメータ数: {n}"


def test_flow_vla_loss_is_scalar_and_backprops():
    model = FlowVLA(vocab_size=VOCAB, chunk_len=CHUNK)
    image, state, tokens = _dummy_inputs()
    action = torch.rand(B, CHUNK, ACTION_DIM)
    pad_mask = torch.ones(B, CHUNK)
    loss = model.flow_loss(image, state, tokens, action, pad_mask)
    assert loss.ndim == 0
    loss.backward()  # 勾配が流れること
    assert any(p.grad is not None for p in model.parameters())


def test_flow_vla_sample_shape():
    model = FlowVLA(vocab_size=VOCAB, chunk_len=CHUNK)
    image, state, tokens = _dummy_inputs()
    a = model.sample(image, state, tokens, n_steps=5)
    assert a.shape == (B, CHUNK, ACTION_DIM)


def test_avg_pool_variant_runs():
    # 位置情報を捨てる版（比較用）も forward できること
    model = TinyVLA(vocab_size=VOCAB, chunk_len=CHUNK, image_pool="avg")
    out = model(*_dummy_inputs())
    assert out.shape == (B, CHUNK, ACTION_DIM)
