#import "lib/template.typ": *

#show: conf.with(
  title: "小さな VLA を自作して学ぶ",
  subtitle: "PyTorch で作る Vision-Language-Action ── 図表と実装で学ぶ版",
  date: "2026 ・ github.com/sige0002/vla-learn",
)

// 前書き
#block(fill: rgb("#f3f4f6"), inset: 12pt, radius: 4pt)[
  本書は、小さな VLA（画像＋言語→行動）を PyTorch でスクラッチ自作するハンズオンの
  「図表＋実装を読む」版です。題材は実機・物理エンジン不要の *Tiny Tabletop 2D Pick-and-Place* で、
  ノート PC の CPU だけで最後まで動きます。各章の「実装を読む」枠は、次に読むべき
  ソースファイル（自作 `src/vla_learn/` と、有名 VLA の公式 repo）への地図です。
  対応する実行可能コード・演習・解答はリポジトリにあります。
]

#include "chapters/m0.typ"
#include "chapters/m1.typ"
#include "chapters/m2.typ"
#include "chapters/m3.typ"
#include "chapters/m4.typ"
#include "chapters/m5.typ"
#include "chapters/m6.typ"
