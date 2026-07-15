# book — 図表入り PDF 教科書（Typst）

`lessons/*.md`（ブラウザでさっと読む版）を拡張した、**図表と「実装を読む」案内つきの PDF 教科書**の
ソースです。組版には [Typst](https://typst.app/) を使い、**システムに TeX を入れず uv だけ**でビルドできます
（pip の `typst` パッケージがコンパイラを同梱）。

## ビルド

```bash
# 図を生成（初回・図を更新したいとき。matplotlib 必要）
uv run --extra viz python scripts/make_figures.py        # book/figures/*.png を生成
#   速く回したいときは --quick（学習を伴う図を小さめに）

# PDF をビルド
uv run --extra book python scripts/build_book.py         # book/build/book.pdf と章別 mX.pdf
uv run --extra book python scripts/build_book.py --only m4   # M4 章だけ
uv run --extra book python scripts/build_book.py --book      # 統合 book.pdf だけ
```

出力は `book/build/`（`book.pdf` が全6章＋前付の統合版、`m0.pdf`〜`m6.pdf` が章別）。

## 構成

```text
book/
├── book.typ           # 統合本（前付 + 各章を include）
├── lib/template.typ   # 体裁・コールアウト・図表ヘルパ（ハウススタイル）
├── chapters/mX.typ    # 各章の本文
├── figures/*.png      # scripts/make_figures.py が生成するデータ駆動の図
└── build/             # 生成物（PDF）
```

## 執筆のしかた（ハウススタイル）

各章は冒頭で `#import "../lib/template.typ": *` し、次のヘルパを使います。

- `#goal[...]` 章頭のゴール枠 / `#summary[...]` 章末まとめ
- `#theory[...]`（座学とのつながり）/ `#pitfall[...]`（つまずき・バグ）/ `#note[...]`（補足）
- `#readcode("src/vla_learn/....py", target: "Class.method")[なぜ読むか]` … **次に読む実装への地図**
- `#fig("/figures/xxx.png", caption: [...], width: 90%)` … 図（パスは root 相対 `/figures/...`）
- コード ```` ```python ... ``` ````、数式 `$...$`、表 `#table(...)`

`chapters/m0.typ` が見本です。図は「飾り」ではなく、対応する実装ファイルへの入口として置いています。

## Markdown 版との関係

- `lessons/*.md` … GitHub 上で読む Web 閲覧版（詳細度は PDF 版とほぼ同等。主要図も掲載）。リンクをたどりやすい。
- この PDF … 図表・数式・実装の読みどころを足した、腰を据えて読む版。

内容の正本は実装（`src/vla_learn/`）です。コードと文章がずれていたらコードが正です。
