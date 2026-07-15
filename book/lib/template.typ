// 教材ブック共通テンプレート（フォント・体裁・コールアウト・図表ヘルパ）
// 各章は先頭で  #import "../lib/template.typ": *  として使う。

#let MAIN_FONT = "Noto Sans CJK JP"
#let MONO_FONT = "Noto Sans Mono CJK JP"

#let ACCENT = rgb("#1f4e9c")

// ---- コールアウト（枠囲み）。絵文字は使わない（フォント欠落でtofuになるため）----
#let _callout(bg, bar, label, body) = block(
  width: 100%, fill: bg, stroke: (left: 3pt + bar), inset: 10pt, radius: 3pt,
  above: 0.9em, below: 0.9em,
)[
  #text(weight: "bold", fill: bar, size: 9.5pt)[#label] #v(0.25em) #body
]

#let goal(body)    = _callout(rgb("#fff7e6"), rgb("#c98a00"), "この章のゴール", body)
#let summary(body) = _callout(rgb("#eef7ef"), rgb("#2e7d4f"), "まとめ", body)
#let theory(body)  = _callout(rgb("#eef3ff"), rgb("#3a6ea5"), "座学とのつながり", body)
#let pitfall(body) = _callout(rgb("#fdeeee"), rgb("#c0392b"), "つまずき / バグの定番", body)
#let note(body)    = _callout(rgb("#f3f4f6"), rgb("#6b7280"), "補足", body)

// ---- 実装の読みどころ（パス + 対象 + 理由）----
#let readcode(path, target: none, body) = block(
  width: 100%, fill: rgb("#edf6ef"), stroke: (left: 3pt + rgb("#2e7d4f")), inset: 10pt, radius: 3pt,
  above: 0.9em, below: 0.9em,
)[
  #text(weight: "bold", fill: rgb("#2e7d4f"), size: 9.5pt)[実装を読む]
  #h(0.6em) #raw(path)#if target != none [ #text(fill: rgb("#555"))[→ #raw(target)]]
  #parbreak() #body
]

// ---- 図（パスは呼び出し元ファイルからの相対。章からは "../figures/..." ）----
#let fig(path, caption: none, width: 92%) = figure(
  image(path, width: width),
  caption: caption,
)

// ---- 文書設定。book.typ と章スタンドアロンの両方で #show: conf.with(...) する ----
#let conf(title: none, subtitle: none, date: none, body) = {
  set document(title: if title != none { title } else { "VLA 教材" })
  set page(paper: "a4", margin: (x: 2.5cm, top: 2.3cm, bottom: 2.2cm), numbering: "1")
  set text(font: MAIN_FONT, size: 10.5pt, lang: "ja")
  set par(justify: true, leading: 0.82em, spacing: 1.15em)
  set heading(numbering: "1.1")

  // 見出しスタイル
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    block(above: 0.4em, below: 0.9em)[
      #text(size: 9pt, fill: ACCENT, weight: "bold")[#counter(heading).display("1")] #v(-0.4em)
      #text(size: 19pt, weight: "bold", fill: ACCENT)[#it.body]
      #v(-0.2em) #line(length: 100%, stroke: 0.6pt + ACCENT.lighten(30%))
    ]
  }
  show heading.where(level: 2): it => block(above: 1.1em, below: 0.6em)[
    #text(size: 13.5pt, weight: "bold", fill: rgb("#222"))[#it]
  ]
  show heading.where(level: 3): it => block(above: 0.9em, below: 0.5em)[
    #text(size: 11.5pt, weight: "bold", fill: rgb("#333"))[#it]
  ]

  // コード（block raw）に薄い背景
  show raw.where(block: true): it => block(
    width: 100%, fill: rgb("#f6f8fa"), inset: 9pt, radius: 3pt, above: 0.8em, below: 0.8em,
  )[#set text(font: MONO_FONT, size: 8.7pt); #it]
  show raw.where(block: false): it => box(
    fill: rgb("#eef0f3"), inset: (x: 3pt, y: 0pt), outset: (y: 2pt), radius: 2pt,
  )[#set text(font: MONO_FONT, size: 9pt); #it]

  // 図キャプション
  show figure.caption: it => text(size: 8.8pt, fill: rgb("#555"))[#it]

  // 表
  set table(
    stroke: 0.5pt + rgb("#bbb"), inset: 6pt,
    fill: (_, y) => if y == 0 { rgb("#f0f4f8") },
  )
  show table.cell.where(y: 0): set text(weight: "bold")

  // タイトルページ
  if title != none {
    set page(numbering: none)
    v(8em)
    align(center)[
      #text(size: 26pt, weight: "bold", fill: ACCENT)[#title] \
      #if subtitle != none { v(0.4em); text(size: 13pt, fill: rgb("#444"))[#subtitle] } \
      #v(1.2em) #if date != none { text(size: 10pt, fill: rgb("#666"))[#date] }
    ]
    pagebreak()
    outline(title: [目次], indent: 1.2em, depth: 2)
  }

  body
}
