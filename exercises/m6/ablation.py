"""課題②: ablation 用の雛形（穴埋め 3 か所）。

ねらい（本文 M4 の教訓 1・3 を「壊して」確かめる）:
  - image_pool='avg'        … 画像エンコーダで空間情報を捨てる → 「場所へ向かう」が苦手に
  - condition_vision=False  … FiLM を切って言語で視覚を変調しない → 「どの色か」の選択が壊れる

なぜ専用フラグで切り替えられないか:
  学習スクリプトの run_training（src/vla_learn/training/trainer.py）は、モデルへ
  vocab_size と chunk_len しか渡しません。一方 image_pool / condition_vision は
  TinyVLA / VLABackbone のコンストラクタ引数です。そこで「自分で短い学習ループを書く」のが正攻法。
  TinyVLA は **backbone_kwargs を受けるので、これらを直接渡せます。

使い方:
  python exercises/m6/ablation.py

注意: 重い学習はしません。傾向が出る最小設定（少データ・少エポック）で回します。
      success_rate は乱数でぶれるので、SEEDS を 2〜3 個にして平均を見ると傾向がはっきりします。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# src/ を import パスへ（scripts/_bootstrap と同じ役割）
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vla_learn.datasets import (
    CharTokenizer,
    SyntheticVLADataset,
    build_normalizers,
    generate_episodes,
)
from vla_learn.envs import all_instruction_strings
from vla_learn.evaluation.rollout import PolicyWrapper, evaluate_policy
from vla_learn.models import TinyVLA, count_parameters
from vla_learn.training.losses import masked_mse
from vla_learn.utils import set_seed

# --- 軽量設定（傾向が出る最小限。重くしない）---
N_EPISODES = 400
EPOCHS = 12
BATCH_SIZE = 128
LR = 1e-3
CHUNK_LEN = 8
EVAL_EPISODES = 50
SEEDS = [0]  # 余裕があれば [0, 1, 2] にして平均を見る


def train_one(condition: dict, seed: int) -> float:
    """1 条件・1 seed で学習し、success_rate を返す。"""
    set_seed(seed)
    device = torch.device("cpu")

    # データ・トークナイザ・正規化（trainer.py と同じ流れ）
    episodes = generate_episodes(N_EPISODES, seed=seed, action_noise=0.03)
    tok = CharTokenizer.from_corpus(all_instruction_strings())
    action_norm, state_norm = build_normalizers(episodes)
    ds = SyntheticVLADataset(episodes, tok, CHUNK_LEN, action_norm, state_norm)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # ----------------------------------------------------------------
    # 【穴埋め 1/3】condition（例 {"image_pool": "avg"}）を TinyVLA に渡す。
    #   ヒント: TinyVLA(vocab_size=..., chunk_len=..., **condition)
    # ----------------------------------------------------------------
    model = TinyVLA(vocab_size=tok.vocab_size, chunk_len=CHUNK_LEN, ____).to(device)
    print(f"  [{condition}] params={count_parameters(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch["image"], batch["state"], batch["tokens"])  # [B,C,A]
            # ------------------------------------------------------------
            # 【穴埋め 2/3】pad_mask を使った MSE 損失を計算する。
            #   ヒント: masked_mse(pred, target, mask)
            # ------------------------------------------------------------
            loss = ____
            opt.zero_grad()
            loss.backward()
            opt.step()

    # ----------------------------------------------------------------
    # 【穴埋め 3/3】学習済みモデルを PolicyWrapper に包み、評価する。
    #   ヒント: PolicyWrapper(model, tok, action_norm, state_norm, "mse", device)
    #          evaluate_policy(policy, n_episodes=EVAL_EPISODES)
    # ----------------------------------------------------------------
    policy = ____
    metrics = ____
    return metrics["success_rate"]


def main() -> None:
    conditions = {
        "baseline (flatten + FiLM)": {},                       # 既定。いちばん強いはず
        "avg pooling (空間情報を捨てる)": {"image_pool": "avg"},
        "no FiLM (言語で視覚を変調しない)": {"condition_vision": False},
    }
    print("=== ablation: success_rate を比較 ===")
    for name, cond in conditions.items():
        rates = [train_one(cond, s) for s in SEEDS]
        mean = sum(rates) / len(rates)
        print(f"{name:36s} success_rate={mean:.3f}  (seeds={rates})")
    print("\n期待される傾向: baseline > avg / no-FiLM。どちらが・なぜ落ちるか説明してみよう。")


if __name__ == "__main__":
    main()
