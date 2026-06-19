#import "../lib/template.typ": *

= LeRobot と有名 VLA <m6>

#goal[
  - 自作データを *`LeRobotDataset` 形式*（フレーム列・`observation.images.* / observation.state / action / task`）へ
    並べ替える流れと、その「変換ロジック」を *lerobot 無しでも動く純粋関数*として持つ設計眼を身につける。
  - *SmolVLA を精読*し、あなたが M4/M5 で自作した *TinyVLA / FlowVLA の部品と 1 対 1 で対応づけ*られるようになる。
  - π0 / π0.5・OpenVLA・GR00T・MolmoAct2 を「*入力 → 表現/トークン化 → 行動ヘッド（離散 or flow）→ 学習データ*」の
    軸で読み、*行動の出し方は「離散自己回帰」か「連続生成(flow/diffusion)」の二系統*に大別できると掴む。
  - 自作の最小 VLA が「どの部品を大規模・高機能に差し替えると、どの実在 VLA に化けるか」を地図として持つ。
]

この章は本書の到達点です。ここまでで、あなたは *画像 + 言語 + 状態 → 行動チャンク* を出す VLA を、
MSE 版（M4 の `TinyVLA`）と flow matching 版（M5 の `FlowVLA`）の 2 通りで*自作*しました。
本章はその経験を 2 方向に「外へ」つなげます。1 つは*データの規格*（LeRobot）に自作データを載せること、
もう 1 つは*モデルの地図*——SmolVLA を精読し、π0 / OpenVLA / GR00T / MolmoAct2 を「あなたが書いた部品の、
どこかを強化した版」として読むことです。

#note[
  この章には *2 つの正確さの約束*があります。読み進める前に頭に入れてください。
  *(1)* LeRobot の*書き込み API はバージョンで変わります*。後述のコードは「ある時点の一例」で、あなたの
  環境では動かないことがあります。だから本書は*変換ロジック（lerobot 不要・単体テスト可能）*を中心に据えます。
  *(2)* 有名 VLA の*内部実装の細部は、原典（公式 repo / arXiv）でしか確定できません*。本章は各事実に
  *出典（repo のファイルパス / arXiv 番号）と参照日（2026-06）*を添え、*憶測の数値は書きません*。
  公式 VLA は「*読む教材*（本書での実行は保証しません）」として扱います。
]

== LeRobotDataset とは何か（フレーム列という考え方）

LeRobot は Hugging Face のロボット学習ライブラリで、`LeRobotDataset` はその標準データ形式です。本書で重要なのは
*中身の考え方*だけなので、そこに絞ります。`LeRobotDataset` は、*1 つのエピソードを「フレーム列」として表現*します。
各フレーム（＝各時刻のレコード）は、*名前付きの特徴量 (features)* を持つ辞書です。VLA でよく使う特徴量を、本書の
自作データの何に当たるかと並べると次のようになります。

#table(
  columns: (auto, 1fr, auto),
  [LeRobot の特徴量名], [中身], [本書での対応],
  [`observation.images.<camera>`], [カメラ画像（複数台ぶん。`<camera>` はカメラ名）], [1 台なので `observation.image`],
  [`observation.state`], [固有受容状態（関節角・エンドエフェクタ位置など）], [`state`（`[ax, ay, grip]`）],
  [`action`], [その時刻の行動], [`action`（`[dx, dy, grip]`）],
  [`task`], [言語指示（*文字列*で持つ）], [`instruction`],
)

押さえるべき点は 3 つです。*(1)* 実機ロボットは複数視点を使うのが普通なので、画像のキー名にカメラ名を含めます
（例 `observation.images.top`, `observation.images.wrist`）。本書は 1 視点なので `observation.image` 1 つで十分です。
*(2)* *言語指示は `task`（文字列）として各フレームに付く*。トークン化は学習時にモデル側でやるので、データ側は文字列で
持ちます。*(3)* *エピソード境界・FPS・特徴量の dtype/shape・正規化統計はメタデータ（`meta/`）が管理*します。

#theory[
  M3 で `SyntheticVLADataset` を作ったとき張った伏線がこれです。本書の内部 dict（`image / state / tokens / action`）は、
  *名前が違うだけで `LeRobotDataset` の特徴量に対応*します。「実装の核心は自作、データ規格は LeRobot に合わせる」
  という本書の境界線（最初から `lerobot-train` に依存しない方針）の実演です。
]

=== ディスク上の実体とバージョン（v3.0 を例に）

`LeRobotDataset` は中身を 3 種類に分けて保存します。これは「画像は重い／低次元データは軽い／メタ情報は別」という
都合に素直に従った形です（以下は *lerobot v3.0* 系の構成。バージョンで変わります。出典: huggingface/lerobot
公式ドキュメント `LeRobotDataset v3.0`、参照 2026-06）。

#table(
  columns: (auto, auto, 1fr),
  [置き場所], [形式], [入るもの],
  [`data/.../*.parquet`], [Apache Parquet], [低次元の時系列（`observation.state` / `action` / タイムスタンプ等）],
  [`videos/.../*.mp4`], [MP4 動画], [カメラ画像（フレームを動画にまとめて格納）],
  [`meta/`], [JSON / JSONL / Parquet], [`info.json`（特徴量スキーマ・FPS）, `stats.json`（正規化統計）, `tasks.jsonl`（指示文↔ID）, エピソード情報],
)

#note[
  *バージョン依存の正直な注意*: `LeRobotDataset` の*コードベース版*は v2.1 → *v3.0* と変わってきました
  （v3.0 は概ね 1 ファイルに複数エピソードをまとめる方式で、v2.1 の「1 エピソード 1 ファイル」とは*構造が違う*）。
  書き込み API（後述の `create / add_frame / save_episode` や、v3.0 で必要になった `finalize()`）も版で変わります。
  *正確な使い方は、あなたが入れた lerobot の版のドキュメントを必ず参照*してください。
  出典: huggingface/lerobot `src/lerobot/datasets/`（`CODEBASE_VERSION`）, 公式ドキュメント、参照 2026-06。
]

=== 画像の格納形式（HWC・uint8）— ここが M4 と違う

VLA を自作したとき、画像は *`[3, 64, 64]`（CHW）の float（0..1）* でした（M3 の `render_world` の出力）。これは
*PyTorch の Conv が CHW を食べる*からです。一方、`LeRobotDataset` の画像特徴量はスキーマ上 *`[H, W, 3]`（HWC）* で
記述され、*保存・取り込みは uint8（0..255）* が基本です。これは画像/動画として保存・可視化する都合に合わせた、
*保存・交換のための形式*です。

```text
本書のモデル入力              LeRobot のスキーマ表記
  [3, H, W] float 0..1   ◀──▶   [H, W, 3]（uint8 で保存）
  （CHW, 学習で使う形）          （HWC, 保存で使う形）
```

#note[
  *もう一段の正直さ*: LeRobot の DataLoader は、学習時には画像を*再び CHW・float に直して*返す実装になっています
  （内部で uint8(HWC) → float32(CHW) へ変換し 0..1 へ正規化）。つまり「*保存は HWC・uint8、学習で使う形は CHW・float*」
  という往復があります。本書の export では *CHW float → HWC uint8* の向き（保存する向き）を作ります。
  出典: huggingface/lerobot `src/lerobot/datasets/`、参照 2026-06。
]

== 変換ロジックを読む（`map_episode_to_frames`・lerobot 不要）

export の心臓部は `scripts/export_lerobot.py` の `map_episode_to_frames` です。これは「*自作 episode →
LeRobot 風フレーム辞書の列*」へ変換する*純粋な関数*で、*lerobot をインストールしていなくても動き、単体テストできます*。

#readcode("scripts/export_lerobot.py", target: "map_episode_to_frames")[
  この関数 1 つが「自作データ → 規格の特徴量」への*対応づけ*の本体です。`render_world` で `[3,H,W] float` を作り、
  `np.transpose(..., (1,2,0)) * 255` で `[H,W,3] uint8` に並べ替え・型変換し、`task` には `instruction`（文字列）を
  そのまま入れます。*書き込み API ではなくこの変換*こそが学習目標である、と意識して読んでください。
]

```python
# scripts/export_lerobot.py より（抜粋）
def map_episode_to_frames(ep: dict, img_size: int = 64) -> list[dict]:
    frames = []
    T = ep["actions"].shape[0]
    for t in range(T):
        img_chw = render_world(...)                                  # [3,H,W] float 0..1
        img_hwc = (np.transpose(img_chw, (1, 2, 0)) * 255).astype(np.uint8)  # [H,W,3] uint8
        frames.append({
            "observation.image": img_hwc,                            # 保存は HWC・uint8
            "observation.state": ep["agent"][t].astype(np.float32),  # [3]
            "action":            ep["actions"][t].astype(np.float32), # [3]
            "task":              ep["instruction"],                   # str（トークン化しない）
        })
    return frames
```

読みどころは 3 つです。*(1)* 画像は M3 と同じく「都度レンダリング」——ディスクには低次元の状態だけ保存し、export 時に
`render_world` で作ります。*(2)* `img_chw → img_hwc` が、前節の *CHW float → HWC uint8* そのものです。
*(3)* `task` は文字列をそのまま入れます（トークン化はモデル側の仕事）。

#pitfall[
  ここで「画像が `[64,64,3]` の uint8、値域 0〜255」に変わることを*自分の目で確認*しておきましょう。M4 のモデル入力
  `[3,64,64]` float（0..1）と取り違えると、`permute` や `/255` の入れ忘れで「色が壊れる／学習が進まない」が起きます。
  `map_episode_to_frames` は lerobot 不要なので、`generate_episodes(n_episodes=1)[0]` を食わせれば手元で即 shape 確認できます。
]

=== なぜ「変換ロジックを独立させる」設計が良いか

ライブラリの*書き込み API は変わりやすい*が、「自分のデータ → 規格が要求する特徴量」への*対応づけ*は
*設計判断であって、変わりにくい*ものです。この 2 つを分離しておくと——API が変わっても直すのは薄い書き込み層だけで済み、
変換ロジックを*単体テストできる*（lerobot を入れずに CI で回せる）。`export_lerobot.py` の `main` は、変換を確認した後に
`lerobot` の import を `try` で囲み、無ければ案内して安全に終了し、書き込みは「v2 系を想定した一例」を `try` で囲って
*失敗しても落とさず warn を出す*設計にしてあります。

#note[
  これは VLA に限らず、外部規格に合わせるときの定石です。*「変わりやすい所」と「変わりにくい所」を分ける*という
  設計眼を、ここで養ってください。`features`（スキーマ）の `dtype` 名、`task` の渡し方、`finalize()` の要否などは
  版で変わります——*正確な使い方は、あなたが入れた lerobot のドキュメント*に従ってください。
]

== SmolVLA 精読 — 自作 VLA との対応表

ここからモデルです。*精読の対象に SmolVLA を選ぶ*のは、*LeRobot に直結*（同じエコシステム）し、*小型の OSS* で、
*VLM + 状態トークン + action expert + flow matching + action chunking* が一通り入っており、*あなたが M4/M5 で作った
部品とほぼ 1 対 1 で対応づけられる*からです。

#note[
  出典: SmolVLA 論文 *arXiv:2506.01844*「SmolVLA: A Vision-Language-Action Model for Affordable and Efficient
  Robotics」、公式実装 huggingface/lerobot（`src/lerobot/policies/smolvla/`）、参照 2026-06。
  以下はこれら一次情報に基づきますが、*内部実装の細部・最新の数値は必ず原典で確認*してください。
]

#fig("/figures/architecture.png", caption: [自作 VLA の forward（3 入力 → エンコーダ → FiLM 融合 → ヘッド）。
  SmolVLA は前半（融合）を*事前学習 VLM*に、後半（行動生成）を専用の *action expert(flow)* に置き換えた格好。
  対応する自作実装は `src/vla_learn/models/tiny_vla.py`（`VLABackbone`）と `flow_head.py`（`FlowVLA`）。], width: 88%)

=== SmolVLA の構成（設計思想レベル）

設計思想レベルで言うと、SmolVLA は「*事前学習済み VLM* が画像・言語・状態を統合し、その文脈を条件に
*action expert*（別の小さなネット）が *flow matching* で行動チャンクを生成する」流れです。確定している要点を並べます。

- *入力は「画像 + 言語(task) + 状態」*。画像は視覚エンコーダ（SigLIP 系）で視覚トークンに、言語は普通にトークン化、
  *状態（proprioception）は線形層で射影してトークンとして*入れます（本書と同じ 3 入力）。複数カメラは各画像を順に処理して
  並べます。出典: `configuration_smolvla.py`（状態は `nn.Linear` で 1 トークンへ射影）, 参照 2026-06。
- *VLM は事前学習済み*。既定のベースは *SmolVLM2*（`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`、SigLIP 視覚 +
  SmolLM2 言語）で、SmolVLA は*その LLM の先頭 16 層だけ*を使います。出典: `configuration_smolvla.py`
  （`vlm_model_name`）, 参照 2026-06。
- *action expert* が VLM の文脈を条件に*行動を生成*します。生成方式は *flow matching*（M5 で自作したもの）、出力は
  *連続値の行動チャンク*。既定の chunk は *50*（`chunk_size = 50`）です。出典: `configuration_smolvla.py`, 参照 2026-06。

#note[
  *規模の数値（不確実さの明記）*: 論文 arXiv:2506.01844 は「*約 4.5 億 (450M) パラメータ、うち約 1 億 (100M) が
  action expert*」と本文に明記しています。一方ベース checkpoint 名は「500M」を含みますが、SmolVLA は*その 16 層
  だけ*を使うため、*論文の 450M と checkpoint 名の 500M を混同しない*でください。正確な値は arXiv:2506.01844 と
  最新のモデルカードを参照。
]

=== 実装を読む — flow matching の学習と推論

SmolVLA の中核は 2 クラスです。*精読の中心*として、ここを必ず開いてください。

#readcode("src/lerobot/policies/smolvla/modeling_smolvla.py", target: "SmolVLAPolicy.select_action / forward")[
  `SmolVLAPolicy` は LeRobot の policy ラッパ。`forward`（学習）は `VLAFlowMatching` の flow matching 損失を計算し、
  `select_action`（推論）は*行動キュー*を持ち、空になったときだけ推論して*行動チャンクを生成*、1 手ずつ環境に返します。
  「学習＝損失、推論＝積分でチャンク生成、しかも毎ステップは推論しない」という*実運用の作法*が読めます。
]

#readcode("src/lerobot/policies/smolvla/modeling_smolvla.py", target: "VLAFlowMatching.forward / sample_actions")[
  flow matching の本体。学習は時刻 `tau`〜Beta(1.5,1) を引き、`x_t = tau*noise + (1-tau)*action` を作り、目標速度
  `u_t = noise - action` を*速度予測の MSE*で学習。推論は `t=1→0` を `dt = -1/num_steps`（既定 *10 ステップ*）で
  Euler 積分。*あなたが M5 で書いた `flow_loss` / `sample` と、設計が一致*しているのが分かります。
]

#theory[
  M5 の `FlowVLA.flow_loss` は `a_tau = (1-tau)*a0 + tau*a1`、`v_target = a1 - a0` でした（時刻は一様 `U(0,1)`、
  推論は `tau = 0→1`）。SmolVLA は「*ノイズと目標の置き方の向き*」と「*時刻分布*（Beta）」が違うだけで、
  *速度を予測して MSE で合わせ、積分で生成する*という骨格は完全に同じです。M5 を書けた時点で、この 2 ファイルは読めます。
]

=== 【中核】TinyVLA / FlowVLA との対応表

本章のいちばん大事な表です。*あなたが自作した部品が、SmolVLA では何に当たるか*を対応づけます。

#table(
  columns: (auto, 1fr, 1fr),
  [役割], [本書の自作部品（M4/M5）], [SmolVLA での対応（概念）],
  [視覚の符号化], [`ImageEncoder`（小さな CNN、flatten で空間保持）], [事前学習済み視覚エンコーダ（SigLIP 系）＋視覚トークン],
  [言語の符号化], [`TextEncoder`（文字埋め込み + 位置 + 1 層 Transformer）], [事前学習済み言語モデル（VLM の言語側）],
  [状態の符号化], [`StateEncoder`（小さな MLP）], [状態を線形射影した「状態トークン」],
  [視覚×言語の融合], [*FiLM*（言語で視覚を変調）＋ concat → 融合 MLP（`VLABackbone`）], [VLM の注意機構で全トークンを統合],
  [条件ベクトル], [`VLABackbone` の出力 `h [B, hidden]`], [VLM が出す文脈表現],
  [行動ヘッド（生成）], [`FlowVLA.velocity` + flow loss（M5）], [*action expert* による *flow matching*],
  [行動の形], [行動チャンク `[B, chunk_len, action_dim]`（chunk=8）], [行動チャンク（連続値・複数ステップ。chunk=50）],
  [学習則], [rectified flow の MSE（速度予測）], [flow matching（速度予測の MSE）],
)

この表の読み方は 2 つ。*(1)*「画像/言語エンコーダ → FiLM 融合 → flow ヘッド」（自作）が、SmolVLA では
「*事前学習 VLM → action expert(flow)*」に対応します。つまり*骨格は同じ*で、SmolVLA は前半を*巨大な事前学習 VLM*に、
後半を*専用の action expert*に置き換えた格好です。*(2)* あなたの `VLABackbone` が作る条件ベクトル `h` は、SmolVLA では
VLM の文脈表現に当たります。

=== 本書 TinyVLA との「距離」

正直に距離も書きます。主な違いは*事前学習の有無*（SmolVLA の VLM は大量の画像・言語で事前学習済み。本書はゼロから）、
*規模*（SmolVLA は約 450M、本書は約 0.42M の `TinyVLA` ／約 0.58M の `FlowVLA`）、*入力の豊かさ*（実機は複数カメラ・
高次元状態。本書は 1 視点・3 次元）、*データ*（実機データ vs 合成データ）です。逆に言えば、*設計の骨格（3 入力 → 融合 →
flow で行動チャンク）は共通*。だから「小さく作って大きく読む」が成立します。

#summary[
  *M4 + M5 を作り切った時点で、SmolVLA の「読み方」は手に入っています。* 残りの差は主に「VLM を事前学習で賢くして
  あるか」と「実機データで大規模に学習してあるか」です。
]

== π0 / π0.5 — flow matching と action expert（実機データ規模）

#note[
  出典: *π0 = arXiv:2410.24164*「π₀: A Vision-Language-Action Flow Model for General Robot Control」、
  *π0.5 = arXiv:2504.16054*、公式 repo Physical-Intelligence/openpi（`src/openpi/models/`）、参照 2026-06。
]

π0 は *flow matching で連続行動を生成*する VLA です（M5 と同じ系統）。*action expert* という*別の重みを持つネット*を
VLM に付け、*状態と行動をトークンとして*扱います。ベース VLM は *PaliGemma*（config では `gemma_2b` 相当 ＋ SigLIP 視覚）で、
*action expert は別の小さな Gemma（既定 `gemma_300m` 相当）*。論文も「ロボット用トークンに*別の重み*を使うと改善した」と
述べています。行動はチャンク化（config 既定の `action_horizon = 50`、タスク別 config で上書き）。
*π0 は、本書 `FlowVLA` の「事前学習 VLM + 専用 action expert + 実機大規模データ」版*と読めます。

#readcode("src/openpi/models/pi0.py", target: "embed_prefix / embed_suffix / compute_loss / sample_actions")[
  *自作との対応がいちばん綺麗に見えるファイル*。`embed_prefix` が*画像+言語（VLM 側）*を、`embed_suffix` が
  *状態+ノイズ行動+時刻（action expert 側）*を埋め込みます。`compute_loss` は時刻 `t`〜Beta(1.5,1) を引いて
  `x_t = t*noise + (1-t)*action`、目標 `u_t = noise - action` で*速度の MSE*。`sample_actions` は `t=1→0` を
  既定 *10 ステップ*で Euler 積分。M5 の `flow_loss` / `sample` と*同じ部品配置*です。
]

#readcode("src/openpi/models/pi0_config.py", target: "Pi0Config（pi05 フラグ）")[
  *pi0 と pi0.5 の差を確認する場所*。`paligemma_variant` / `action_expert_variant` でベース VLM と action expert の
  サイズが、`action_horizon` / `action_dim` で行動チャンクの形が決まります。*`pi05: bool` フラグ*が立つと挙動が変わる
  （後述）——「config の 1 フラグで派生を切り替える」設計が読めます。
]

=== π0.5 の差分（確定事項と解釈の分離）

π0.5 は π0 を基盤に、*オープンワールド汎化*（学習に無い新しい家庭での動作）を狙った発展版です。確定している差分を、
openpi の config と π0.5 論文から*事実として*挙げます。

- *config 上の差分*: `Pi0Config` の *`pi05=True`* で、(a) *状態入力を離散の言語トークン*として扱う、
  (b) action expert の時刻注入を *adaRMSNorm* で行う（π0 の sinusoidal 埋め込みに対して）、という違いが入ります。
  出典: openpi `src/openpi/models/pi0_config.py`（post-init の記述）, 参照 2026-06。
- *学習目標のハイブリッド*: π0.5 論文は「*FAST トークナイザによる自己回帰サンプリング*（離散）と
  *フロー場の反復積分*（連続）の*両方*で行動を予測するよう学習」と述べています。*高レベル＝離散トークン /
  低レベル＝連続生成*のハイブリッドです。出典: arXiv:2504.16054, 参照 2026-06。

#pitfall[
  *混同しやすい点（重要）*: 「*知識絶縁 (knowledge insulation)*」という定式化は、*π0.5 論文（arXiv:2504.16054）本文
  ではなく、別の後続論文 arXiv:2505.23705 で確立*されたものです。π0.5 論文自体はこの語を使っていません。
  *π0.5 に knowledge insulation を断定的に帰属させない*でください。また *π0.5 のベース VLM 名・パラメータ数・学習データの
  定量規模を一次資料で確証するのは難しい*——*断定せず原典参照*としてください。本書で確実に言えるのは
  「π0 系は flow matching + 専用 action expert で連続行動を出し、π0.5 は離散+連続のハイブリッドを使う」までです。
]

== OpenVLA — 離散トークン自己回帰 + 大規模 VLM

#note[
  出典: *arXiv:2406.09246*「OpenVLA: An Open-Source Vision-Language-Action Model」、公式 repo
  openvla/openvla（`prismatic/`, `vla-scripts/`）、参照 2026-06。
]

OpenVLA は *行動を「離散トークン」に変換し、VLM に自己回帰で生成させる*方式の代表です。各行動次元を *256 個のビンに
離散化*し、言語モデルの語彙のうち*使用頻度の低いトークン*を*行動トークンに置き換えて*扱います。入力は*画像 + 言語指示*で、
論文は*固有受容状態を入力しない（単一画像のみ）*と明記します。ベース VLM は *Prismatic VLM（DINOv2 + SigLIP の視覚 +
Llama-2 7B、計 7B）*、学習データは *Open X-Embodiment の約 97 万（970k）軌跡*です。

#readcode("prismatic/vla/action_tokenizer.py", target: "ActionTokenizer")[
  *連続行動を「単語」に量子化する核心*。既定 `n_bins = 256`、`-1..1` を `np.linspace` で等分してビン中心を作り、
  *ビン番号を `vocab_size - 番号` で語彙末尾のトークン ID に写す*（「使用頻度の低いトークンは語彙の末尾」という前提）。
  *あなたの `head`（連続値 `Linear`）を「256 クラス分類を action_dim×chunk 回」に置き換える発想*が、ここで具体になります。
]

#readcode("prismatic/models/vlas/openvla.py", target: "OpenVLA.predict_action")[
  `predict_action(image, instruction, ...)` は*行動トークンを自己回帰生成*（`generate` を `max_new_tokens=action_dim`）し、
  `decode_token_ids_to_actions` で連続値に戻して逆正規化します。*state 引数が無い*ことも読み取ってください（本書は state を使う）。
]

#note[
  *派生は分けて扱う*: 推論を高速化・多画像化する *OpenVLA-OFT* や、行動チャンクを少ないトークンへ圧縮する *FAST*
  トークナイザは、*いずれも基底 OpenVLA とは別の後続研究*です（FAST は Physical Intelligence 由来）。混同しないでください。
  fine-tune は `vla-scripts/finetune.py` の *LoRA*（既定 rank 32、対象は全 linear）で、データは *RLDS 形式*。出典は上記 repo、参照 2026-06。
]

== GR00T / MolmoAct2 — dual-system と action expert（LeRobot 統合）

=== GR00T — ヒューマノイド向け dual-system

#note[
  出典: *GR00T N1 = arXiv:2503.14734*「GR00T N1: An Open Foundation Model for Generalist Humanoid Robots」、
  公式 repo NVIDIA/Isaac-GR00T（`gr00t/`）、参照 2026-06。
]

GR00T は*ヒューマノイド向け*の基盤モデルで、特徴は *dual-system 構成*です。*System 2（遅い）= vision-language module* が
視覚と言語で状況を解釈し、*System 1（速い）= action module* が*リアルタイムで運動を生成*します。System 1 の実装は
*Diffusion Transformer (DiT) を flow matching で学習*したもの（連続行動）。両システムは*密結合して学習*し、状態は
*実体ごとの MLP*で共有埋め込みに射影、行動は*チャンク化*（論文 `H = 16`）して出します。「遅い思考 + 速い反射」という
*人間の二重過程*がそのまま設計になっているのが面白い点です。

#readcode("gr00t/policy/gr00t_policy.py", target: "Gr00tPolicy")[
  *推論の入口*。`video / state / language` の観測辞書を `AutoProcessor` で整形して `model.get_action(...)` を呼びます。
  *LeRobot 形式の規約に沿ったデータ*を扱う一方、policy 自体は「整形済み観測辞書」を食べる点に注意（生ファイルを直接ではない）。
  本書 `FlowVLA` の `sample` に当たる「観測 → 行動チャンク」の実機版がここに当たります。
]

#pitfall[
  *版差に注意（最重要の不確実さ）*: *ベース VLM は世代で変わります*。論文 *N1 は NVIDIA Eagle-2* と明記しますが、
  *2026-06 時点の repo は概ね N1.7 系*で、*Cosmos-Reason2-2B*（repo の設定では `nvidia/Eagle-Block2A-2B-v2` という
  表記も見られ、公式の整理は未確定）。世代（N1 / N1.5 / N1.6 / N1.7）で VLM 名も DiT 層数も異なります。
  *主張は必ず「どの版か」を明示*し、数値は原典で確認してください。また「end-to-end 学習」は*事前学習時に VLM の
  言語側を凍結*する記述があるため、「VLM の視覚側 + DiT を同時学習」と捉えるのが正確です。出典: arXiv:2503.14734、repo、参照 2026-06。
]

=== MolmoAct2 — 行動の「空間推論」と action expert（LeRobot policy）

#note[
  出典: 初代 *MolmoAct = arXiv:2508.07917*「Action Reasoning Models that can Reason in Space」（allenai/molmoact, Ai2 の
  *Molmo* を基盤）。*MolmoAct2*（後継・2026-06 時点の主役）= Ai2 が 2026-05 に公開、huggingface/lerobot に policy
  `src/lerobot/policies/molmoact2/` として統合（`MolmoAct2Policy`）。参照 2026-06。*MolmoAct2 は新しいため、
  細部は repo / 公式発表で確認*してください。
]

MolmoAct 系の独自性は *「行動する前に空間で推論する」(Action Reasoning Model)* 点です。初代は概ね 3 段階で、
(a) 観測を *depth-aware perception token（深度を意識した知覚トークン）* にして 3D 的に接地し、(b) 中間計画を
*画像上のウェイポイント列（visual reasoning trace）*として描き、(c) 最後に*低レベル行動*（初代は *256 ビンの離散
トークン*、`N=8` チャンク）を出します。観測 → 行動を*直接*写像する本書 `TinyVLA`/`FlowVLA` と違い、*「観測 → 空間推論
（中間トークン）→ 行動」*と*1 段はさむ*——chain-of-thought を行動に持ち込んだイメージです。

*MolmoAct2*（後継）は LeRobot policy として読めるのが利点です。確定している範囲で要点を挙げます。

- *action expert + flow matching の連続生成も持つ*: 行動表現は*ハイブリッド*で、`action_mode` を `discrete /
  continuous / both` で切り替え可能。離散側は *OpenFAST トークナイザ*、連続側は *flow matching* の denoiser。
  既定の推論は連続。出典: lerobot `src/lerobot/policies/molmoact2/`（`MolmoAct2Policy`, `hf_model/`）, 参照 2026-06。
- *VLM と action expert の橋渡しは KV-cache 経由*（VLM 各層の KV に action expert が cross-attend）。ベース VLM は
  *Molmo2-ER*（Molmo2 を身体性推論データで追加学習）。出典: 同 repo / Ai2 公式発表, 参照 2026-06。
- *depth トークン*（適応的な深度推論）は *MolmoAct2-Think 変種*の機能で、現状 LeRobot policy には未収録、と
  ドキュメントに明記。出典: lerobot 公式ドキュメント, 参照 2026-06。

#readcode("src/lerobot/policies/molmoact2/modeling_molmoact2.py", target: "MolmoAct2Policy")[
  *離散と連続を 1 つの policy で扱う*実例。`action_mode`（discrete/continuous/both）と、VLM 各層の KV に
  cross-attend する *action expert* の接続（KV-cache bridge）が読みどころ。`processor_molmoact2.py` は画像/言語/
  *状態（256 ビンに離散化して `<state_start>..<state_end>` で囲む）*の前処理を担います。
]

#pitfall[
  *世代の混同に注意*: *2026-06 時点の主役は MolmoAct2*、初代 MolmoAct は第 1 世代（歴史）です。初代の*状態
  （proprioception）入力*は一次資料に明確な記述が見当たらず（主入力は視覚 + 指示）、一方 *MolmoAct2 の LeRobot
  processor は状態を離散化して取り込む*——*世代で前提が違う*ので、必ず原典/repo で「どの版の話か」を確認してください。
]

== 比較表 — 行動表現で並べる

*設計思想レベルの地図*です。*数値や内部実装の断定は避け*、版で変わる点は前節までの注意に従ってください。正確な値・最新
情報は各 arXiv / 公式 repo（参照 2026-06）で確認します。

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  [モデル], [入力（状態）], [画像 enc / 言語 backbone], [行動表現・生成], [chunk（既定）], [fine-tune 単位],
  [*TinyVLA*（自作 M4）], [画+言+*状態*], [小 CNN / 1 層 Transformer], [連続・決定論的回帰(MSE)], [8], [全体（自作）],
  [*FlowVLA*（自作 M5）], [画+言+*状態*], [小 CNN / 1 層 Transformer], [連続・*flow matching*], [8], [全体（自作）],
  [*SmolVLA*], [画+言+*状態*], [SigLIP / SmolLM2（SmolVLM2）], [連続・*flow matching*（action expert）], [50], [policy 微調整],
  [*π0 / π0.5*], [画+言+*状態*], [SigLIP / Gemma（PaliGemma）], [π0:連続 flow ／ π0.5:離散+連続], [50（既定）], [openpi 微調整],
  [*OpenVLA*], [画+言（*状態なし*）], [DINOv2+SigLIP / Llama-2 7B], [*離散*256ビン・*自己回帰*], [単発（多くは1手）], [*LoRA*(rank32)],
  [*GR00T*], [画+言+*状態*], [Eagle 系→Cosmos 系（版依存）], [連続・*DiT + flow matching*], [16], [微調整（版依存）],
  [*MolmoAct2*], [画+言+*状態*（v2 で取込）], [SigLIP 系 / Molmo2-ER], [ハイブリッド（離散+*flow*）], [8（初代）], [LeRobot policy 微調整],
)

#note[
  *必要 GPU について*: 本書の自作（約 0.4〜0.6M）はノート PC / CPU でも回りますが、上記の実在 VLA はおおむね
  *数億〜70 億パラメータ級*で、微調整にも *GPU（多くは VRAM 16GB 以上、7B 級は LoRA でも相応のメモリ）* が要ります。
  正確な要件は各 repo の README を参照。本書はこれらを*実行しません*——「読む教材」として扱います。
]

この表で押さえるべきは「*行動の出し方は離散自己回帰か連続生成(flow/diffusion)かに大別され、あなたは両系統の最小版
（離散は卒業課題で少しだけ、連続は M5）を自分で書ける*」という地図です。個々のセルの厳密さより、*自作部品からの距離*を
読み取ってください。

=== 【総まとめ】自作部品を「どう差し替えると、どのモデルになるか」

「*あなたの自作部品の、どこを・どう差し替えると、そのモデルに化けるか*」を 1 枚にまとめます。これが本章の地図の最後の
ピースです。

#table(
  columns: (auto, 1fr, auto),
  [あなたの自作部品], [これを…に差し替えると], [→ このモデルに近づく],
  [`head`（連続値 `Linear`）], [行動を 256 ビンに離散化し、トークンとして*自己回帰生成*], [OpenVLA / MolmoAct（初代）],
  [flow ヘッド（`FlowVLA`）], [*action expert* + 事前学習 VLM + 実機大規模データ], [π0 / SmolVLA],
  [観測 → 行動を*直接*写像], [観測 → *空間推論トレース* → 行動 と 1 段はさむ], [MolmoAct 系],
  [単一の融合 MLP（`VLABackbone`）], [*System 2（VLM の熟考）+ System 1（高速 flow）* の二層], [GR00T],
  [*ゼロから学習*], [大規模*事前学習 VLM*（SigLIP 系 + 言語モデル）を土台にする], [実在 VLA すべてに共通],
)

#summary[
  腑に落ちてほしいのは——*自作の最小 VLA は、どの部品を大規模・高機能に差し替えるかで、各実在 VLA に化ける*という
  こと。逆に言えば、有名 VLA はどれも「あなたが書いた部品の、どこかを強化した版」として読めます。これが
  「小さく作って大きく読む」の到達点です。
]

== 卒業課題（capstone challenge）

ここまでで、*自作 VLA → データ規格（LeRobot）→ 有名 VLA の地図*がつながりました。最後は卒業課題で、自分の VLA を
*いじり倒して*理解を確かなものにします。各課題に*到達目標*と*評価方法*を付けます（重い学習は不要。多くは「設定を変えて
短く学習 → `success_rate` を見る」だけで観察できます。値は乱数でぶれるので、*絶対値でなく傾向*を読みます）。

+ *chunk_len / flow_steps を変えて比較する。*
  到達目標: 行動チャンク長と積分ステップ数が成功率・滑らかさに与える影響を、自分の言葉で説明できる。
  評価: `evaluation` の `success_rate` を 2〜3 設定で比較し、傾向（増やすと安定/飽和、等）を述べる。

+ *`image_pool='avg'` / `condition_vision=False` で ablation する。*
  到達目標: 「flatten で空間を保つ」「FiLM で言語接地する」が*なぜ効くか*を、壊した結果から説明できる。
  評価: 既定との `success_rate` 差を見て、部品を抜くと下がることを確認する。

+ *自作データを LeRobot 形式へ export してみる。*
  到達目標: `map_episode_to_frames` の出力 shape（`[H,W,3] uint8` / `[3]` float / 文字列 task）を説明でき、
  書き込み API が*なぜ版依存で、変換ロジックと分離すべきか*を言える。
  評価: lerobot 未導入でも、変換が通り「概算フレーム数」が出ることを確認（書き込みはスキップで可）。

+ *（発展）離散トークン化ヘッドを自作*して OpenVLA 流（256 ビン）と比べる。
  到達目標: 連続値ヘッド（`head`）を「`action_dim×chunk` 回の 256 クラス分類」に置き換え、*離散自己回帰の発想*を体得する。
  評価: 連続版（`TinyVLA`/`FlowVLA`）と離散版の成功率・学習挙動を比較し、長所短所を述べる。

+ *（発展）FlowVLA の action chunk を拡張*し、SmolVLA 風（長い chunk）に近づける。
  到達目標: `chunk_len` を増やしたときの学習安定性・受信ホライズン（receding horizon）実行への影響を観察できる。
  評価: chunk を伸ばして `success_rate` とロールアウトの滑らかさの変化を見る。

#summary[
  - *LeRobot*: `LeRobotDataset` は*フレーム列*で、`observation.images.* / observation.state / action / task` を持つ。
    *画像はスキーマ上 HWC・保存は uint8*（学習は CHW・float に戻る）。変換 `map_episode_to_frames` は *lerobot 無しで動く*。
    *書き込み API はバージョン依存*（v3.0 で `finalize()` 追加 等）なので分離してある。
  - *SmolVLA 精読*: 「VLM + 状態トークン + action expert + flow matching + action chunking」。*自作の
    『画像/言語エンコーダ → FiLM 融合 → flow ヘッド』が『事前学習 VLM → action expert(flow)』に対応*。M4+M5 を作れば読める。
  - *概観*: 行動の出し方で *離散自己回帰（OpenVLA / MolmoAct 初代）* と *連続生成（SmolVLA / π0 / GR00T）* に大別。
    π0.5・MolmoAct2 は*離散+連続のハイブリッド*、GR00T は *dual-system*。*数値・細部・版は原典参照*。
  - *卒業課題*で自作 VLA をいじり、理解を確定させる。これで「VLA を読める」から「VLA をいじれる」へ進めます。
]
