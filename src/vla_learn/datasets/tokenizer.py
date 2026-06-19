"""超シンプルな文字レベル・トークナイザ。

本物の VLA は BERT/Llama などのサブワード・トークナイザを使いますが、
ここでは「文字を ID に変換するだけ」の最小実装にします。日本語でも空白に依存せず動きます。

  PAD = 0（埋め草）。語彙は固定の指示文コーパスから作るので未知語 (OOV) は出ません。
"""
from __future__ import annotations

import json
from pathlib import Path

PAD_TOKEN = "<pad>"
PAD_ID = 0


class CharTokenizer:
    def __init__(self, vocab: list[str], max_len: int) -> None:
        self.vocab = vocab
        self.max_len = max_len
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def from_corpus(cls, corpus: list[str], max_len: int | None = None) -> "CharTokenizer":
        chars = sorted({ch for s in corpus for ch in s})
        vocab = [PAD_TOKEN] + chars  # index 0 を PAD に固定
        if max_len is None:
            max_len = max(len(s) for s in corpus)
        return cls(vocab, max_len)

    def encode(self, text: str) -> list[int]:
        """文字列 → 長さ max_len の ID 列（不足は PAD、超過は切り詰め）。"""
        ids = [self.stoi[ch] for ch in text if ch in self.stoi][: self.max_len]
        ids = ids + [PAD_ID] * (self.max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i != PAD_ID)

    # --- 保存 / 読み込み（JSON） ---
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"vocab": self.vocab, "max_len": self.max_len}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["vocab"], d["max_len"])
