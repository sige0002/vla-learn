"""「1 バッチに過学習できるか」テスト（学習機構の健全性チェック）。

機械学習デバッグの鉄則: モデルと学習ループが正しければ、小さな 1 バッチには必ず過学習できる。
できなければ、損失・shape・最適化・勾配のどこかにバグがある（M1〜M4 で繰り返し使う考え方）。
"""
import torch
from torch.utils.data import DataLoader

from vla_learn.datasets import (
    CharTokenizer,
    SyntheticVLADataset,
    build_normalizers,
    generate_episodes,
)
from vla_learn.envs import all_instruction_strings
from vla_learn.models import FlowVLA, TinyVLA
from vla_learn.training.losses import masked_mse
from vla_learn.utils import set_seed

CHUNK = 8


def _one_batch(bs=16):
    set_seed(0)
    eps = generate_episodes(n_episodes=8, seed=0)
    tok = CharTokenizer.from_corpus(all_instruction_strings())
    an, sn = build_normalizers(eps)
    ds = SyntheticVLADataset(eps, tok, CHUNK, an, sn)
    batch = next(iter(DataLoader(ds, batch_size=bs, shuffle=True)))
    return batch, tok


def test_mse_overfits_one_batch():
    batch, tok = _one_batch()
    model = TinyVLA(vocab_size=tok.vocab_size, chunk_len=CHUNK)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(200):
        pred = model(batch["image"], batch["state"], batch["tokens"])
        loss = masked_mse(pred, batch["action"], batch["pad_mask"])
        opt.zero_grad(); loss.backward(); opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.2 * first, f"1 バッチに過学習できていない: {first:.3f} -> {loss.item():.3f}"


def test_flow_loss_decreases_on_one_batch():
    batch, tok = _one_batch()
    set_seed(0)
    model = FlowVLA(vocab_size=tok.vocab_size, chunk_len=CHUNK)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = []
    last = []
    for i in range(300):
        loss = model.flow_loss(batch["image"], batch["state"], batch["tokens"],
                               batch["action"], batch["pad_mask"])
        opt.zero_grad(); loss.backward(); opt.step()
        if i < 10:
            first.append(loss.item())
        if i >= 290:
            last.append(loss.item())
    # flow はノイズ τ をサンプルするため損失は揺れる → 平均で比較
    assert sum(last) / len(last) < 0.7 * (sum(first) / len(first))
