"""Typst 教材ブックを PDF にビルドする（uv: システム TeX 不要）。

使い方:
  uv run --extra book python scripts/build_book.py            # book.pdf + 章別 mX.pdf
  uv run --extra book python scripts/build_book.py --only m4  # m4 章だけ
  uv run --extra book python scripts/build_book.py --book     # 統合 book.pdf だけ

画像パスは Typst の root 相対（例 `/figures/env_samples.png`）。root は book/ に固定。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import typst

BOOK = Path(__file__).resolve().parents[1] / "book"
BUILD = BOOK / "build"


def _compile(input_path: Path, output_path: Path) -> None:
    typst.compile(str(input_path), output=str(output_path), root=str(BOOK))
    print(f"[pdf] {output_path.relative_to(BOOK.parent)}  ({output_path.stat().st_size//1024} KB)")


def build_book() -> None:
    _compile(BOOK / "book.typ", BUILD / "book.pdf")


def build_chapter(stem: str) -> None:
    ch = BOOK / "chapters" / f"{stem}.typ"
    if not ch.exists():
        raise SystemExit(f"章が見つかりません: {ch}")
    # スタンドアロン用のラッパを book/ 直下に一時生成（相対 import/include のため）
    wrapper = BOOK / f"_standalone_{stem}.typ"
    wrapper.write_text(
        f'#import "lib/template.typ": *\n'
        f'#show: conf.with(title: "{stem.upper()}")\n'
        f'#include "chapters/{stem}.typ"\n',
        encoding="utf-8",
    )
    try:
        _compile(wrapper, BUILD / f"{stem}.pdf")
    finally:
        wrapper.unlink()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None, help="章 stem（例 m4）だけビルド")
    p.add_argument("--book", action="store_true", help="統合 book.pdf だけビルド")
    args = p.parse_args()

    BUILD.mkdir(parents=True, exist_ok=True)

    if args.only:
        build_chapter(args.only)
        return
    if args.book:
        build_book()
        return

    # 既定: 統合 + 章別すべて
    build_book()
    for ch in sorted((BOOK / "chapters").glob("m*.typ")):
        build_chapter(ch.stem)
    print("done.")


if __name__ == "__main__":
    main()
