"""Dataset / DataLoader が正しい shape のバッチを返すかのテスト。"""
import torch
from torch.utils.data import DataLoader

from vla_learn.constants import ACTION_DIM, IMG_SIZE, STATE_DIM
from vla_learn.datasets import (
    CharTokenizer,
    SyntheticVLADataset,
    build_normalizers,
    generate_episodes,
)
from vla_learn.envs import all_instruction_strings

CHUNK = 8


def _make_dataset():
    episodes = generate_episodes(n_episodes=10, seed=0)
    tok = CharTokenizer.from_corpus(all_instruction_strings())
    an, sn = build_normalizers(episodes)
    ds = SyntheticVLADataset(episodes, tok, chunk_len=CHUNK, action_normalizer=an, state_normalizer=sn)
    return ds, tok


def test_single_item_shapes():
    ds, tok = _make_dataset()
    item = ds[0]
    assert item["image"].shape == (3, IMG_SIZE, IMG_SIZE)
    assert item["state"].shape == (STATE_DIM,)
    assert item["tokens"].shape == (tok.max_len,)
    assert item["tokens"].dtype == torch.int64
    assert item["action"].shape == (CHUNK, ACTION_DIM)
    assert item["pad_mask"].shape == (CHUNK,)


def test_batch_shapes():
    ds, tok = _make_dataset()
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, IMG_SIZE, IMG_SIZE)
    assert batch["state"].shape == (4, STATE_DIM)
    assert batch["tokens"].shape == (4, tok.max_len)
    assert batch["action"].shape == (4, CHUNK, ACTION_DIM)
    assert batch["pad_mask"].shape == (4, CHUNK)


def test_pad_mask_marks_valid_steps():
    ds, _ = _make_dataset()
    # 末尾近くのサンプルではパディングが入る → pad_mask に 0 が含まれうる
    masks = [ds[i]["pad_mask"] for i in range(len(ds))]
    any_padded = any((m == 0).any().item() for m in masks)
    all_have_one_valid = all((m == 1).any().item() for m in masks)
    assert all_have_one_valid, "全サンプルに少なくとも 1 つの有効ステップが必要"
    assert any_padded, "終端付近ではパディングが発生するはず"
