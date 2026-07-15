"""FlowVLA（M5）の健全性テスト。

- flow_loss が有限のスカラを返す
- sample の出力 shape が [B, C, A]
- 小さな固定バッチで flow_loss が下がる（学習が回る）
"""
import torch

from vla_learn.models import FlowVLA

VOCAB = 30
B, C, A = 4, 8, 3


def _batch():
    torch.manual_seed(0)
    img = torch.rand(B, 3, 64, 64)
    st = torch.randn(B, 3)
    tk = torch.randint(1, VOCAB, (B, 17))
    ac = torch.randn(B, C, A)
    pm = torch.ones(B, C)
    return img, st, tk, ac, pm


def test_flow_loss_is_finite_scalar():
    model = FlowVLA(vocab_size=VOCAB, chunk_len=C)
    img, st, tk, ac, pm = _batch()
    loss = model.flow_loss(img, st, tk, ac, pm)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_sample_shape():
    model = FlowVLA(vocab_size=VOCAB, chunk_len=C)
    img, st, tk, _, _ = _batch()
    out = model.sample(img, st, tk, n_steps=3)
    assert out.shape == (B, C, A)
    assert torch.isfinite(out).all()


def test_flow_loss_decreases_on_tiny_batch():
    torch.manual_seed(0)
    model = FlowVLA(vocab_size=VOCAB, chunk_len=C)
    img, st, tk, ac, pm = _batch()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def avg_loss(n=8):
        # flow_loss は τ とノイズを引くため確率的。複数回の平均で傾向を見る。
        with torch.no_grad():
            return sum(float(model.flow_loss(img, st, tk, ac, pm)) for _ in range(n)) / n

    before = avg_loss()
    model.train()
    for _ in range(150):
        loss = model.flow_loss(img, st, tk, ac, pm)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = avg_loss()
    assert after < before * 0.7, f"flow_loss が下がっていない: {before:.4f} -> {after:.4f}"
