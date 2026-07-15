# M6: LeRobot と有名 VLA（規格理解 / SmolVLA 精読 / 概観）

> この章のゴール:
> - **(A) LeRobot 連携**: `LeRobotDataset` 形式（`observation.images.*` / `observation.state` /
>   `action` / `task`）の考え方を理解し、自作の合成データを **LeRobot 形式へ export** する流れを掴む。
>   変換ロジック（[`../scripts/export_lerobot.py`](../scripts/export_lerobot.py) の `map_episode_to_frames`）は
>   **lerobot 無しでも動く**。書き込み API は**バージョン依存**である点を正直に押さえる。
> - **(B) SmolVLA 精読**: 「VLM + 状態トークン + action expert + flow matching + action chunking」という構成を、
>   あなたが M4/M5 で自作した **TinyVLA / FlowVLA の部品と対応表**で結ぶ。
> - **(C) 有名 VLA 概観**: OpenVLA / π0・π0.5 / GR00T N1 / MolmoAct を「**入力 → 表現/トークン化 →
>   行動ヘッド（離散 or flow/diffusion）→ 学習データ**」の軸で読み、自作 VLA との距離感を掴む。
> - **(D) 卒業課題（capstone challenge）**への接続。
>
> 前提: [M3](m3_data_actions.md)（`SyntheticVLADataset` / LeRobot 風辞書の伏線）、
> [M4](m4_tiny_vla_mse.md)（`TinyVLA`・FiLM・語順）、[M5](m5_flow_matching.md)（`FlowVLA`・flow matching）。
> 所要時間: 90〜120 分（**読み物と実践が中心。重い学習はしません**）。

**この章の流れ**: A. LeRobot 連携（データ規格と export）→ B. SmolVLA 精読（自作部品との対応表）→
C. 有名 VLA 概観（OpenVLA / π0 / GR00T / MolmoAct を行動表現で比較）→ D. 卒業課題

---

## 0. この章の位置づけ

ここまでで、あなたは **画像 + 言語 + 状態 → 行動チャンク** を出す VLA を、MSE 版（[M4](m4_tiny_vla_mse.md) の
`TinyVLA`）と flow matching 版（[M5](m5_flow_matching.md) の `FlowVLA`）の 2 通りで**自作**しました。

この章では、その自作経験を 2 方向に「外へ」つなげます。

1. **データの規格（LeRobot）**: 自分のデータを、ロボット学習コミュニティの標準フォーマット
   **`LeRobotDataset`** に載せ替える経験をします。これで「自作データ ↔ 公開データ ↔ 公開モデル」が
   同じ土俵に乗ります。
2. **モデルの地図（有名 VLA）**: SmolVLA を**精読**し、OpenVLA / π0 / GR00T N1 / MolmoAct を**概観**します。
   どれも「あなたが作った部品の、大規模・高機能版」として読めます。

> 方針の再確認（[`../docs/curriculum.md`](../docs/curriculum.md)）: 本教材は**最初から `lerobot-train` に依存しません**。
> LeRobot は「**規格の理解・自作データの export・既存 policy の構成読解**」に使います。
> 巨大 policy の写経や、いきなり実機データでの大規模学習はしません。

この章には**正確さの注意**が 2 つあります。読み進める前に頭に入れてください。

- **LeRobot の書き込み API はバージョンで変わります。** 下のコード例は「ある時点の v2 系」を想定した一例で、
  あなたの環境では動かないことがあります。**変換ロジック自体（`map_episode_to_frames`）は lerobot 無しで動く**
  ので、まずそこを確実に理解します。
- **有名 VLA の内部実装の細部は、原典（論文 / 公式 repo）でしか確定できません。** この章は
  **設計思想の地図**に徹し、**憶測の数値は書きません**。各所に arXiv 番号と公式 repo を示すので、
  正確な値はそちらで確認してください。**「概要」と明記した箇所は、必ず原典を参照**してください。

---

## パート A: LeRobot 連携

### A-1. `LeRobotDataset` とは何か（考え方）

[LeRobot](https://github.com/huggingface/lerobot) は Hugging Face のロボット学習ライブラリで、
**`LeRobotDataset`** はその標準データ形式です。本教材で重要なのは「**中身の考え方**」だけなので、
そこに絞ります。

`LeRobotDataset` は、**1 つのエピソードを「フレーム列」として表現**します。各フレーム（＝各時刻のレコード）は、
**名前付きの特徴量 (features)** を持つ辞書です。VLA でよく使う特徴量は次のようなものです。

| 特徴量名 | 中身 | 本教材での対応 |
|---|---|---|
| `observation.images.<camera>` | カメラ画像（複数台ぶん。`<camera>` はカメラ名） | 本教材は 1 台なので `observation.image` |
| `observation.state` | 固有受容状態（関節角・エンドエフェクタ位置など） | `state`（`[ax, ay, gripper]`） |
| `action` | その時刻の行動 | `action`（`[dx, dy, grip_cmd]`） |
| `task` | 言語指示（**文字列**で持つことが多い） | `instruction` |

> [M3](m3_data_actions.md) の最後で張った伏線がこれです。本教材の内部 dict
> （`image / state / tokens / action`）は、名前が違うだけで `LeRobotDataset` の特徴量に**対応**します。
> 「実装の核心は自作、データ規格は LeRobot に合わせる」という[境界線](../docs/curriculum.md)の実演です。

ポイントを 3 つ。

1. **複数カメラを `observation.images.<name>` で持てる**。実機ロボットは複数視点を使うのが普通なので、
   キー名にカメラ名を含めます（例 `observation.images.top`, `observation.images.wrist`）。本教材は
   1 視点なので `observation.image` 1 つで十分です。
2. **言語指示は `task`（文字列）として各フレームに付く**。本教材は「色 → ゴール」を指示する 1 文を持ちます。
3. **エピソード境界・FPS・特徴量の dtype/shape はメタデータで管理される**。「どのフレームがどのエピソードか」
   「各特徴量の型」を、データセット側がまとめて持ちます。

### A-1.1 画像の格納形式（HWC・uint8）— ここが M4 と違う

VLA を自作したとき、画像は **`[3, 64, 64]`（CHW）の float（0..1）** でした
（[M3](m3_data_actions.md)、`render_world` の出力）。これは **PyTorch の Conv が CHW を食べる**からです。

一方、**`LeRobotDataset` の画像特徴量は `[H, W, 3]`（HWC）・`uint8`（0..255）が標準**です。これは画像
ファイル（PNG など）や動画として保存・可視化する都合に合わせた、**保存・交換のための形式**です。

```text
本教材のモデル入力           LeRobot のデータ格納
  [3, H, W] float 0..1   ◀──▶   [H, W, 3] uint8 0..255
  （CHW, 学習で使う形）          （HWC, 保存で使う形）
```

つまり **export では CHW float → HWC uint8 へ並べ替え・型変換**します。逆に、LeRobot からデータを読んで
学習に使うときは **HWC uint8 → CHW float に戻す**（`/255` して `permute`）のが定石です。**「学習で使う形」と
「保存で使う形」は別**、と覚えてください（[M1](m1_pytorch.md) のテンソル形状の話の応用です）。

### A-2. 変換ロジックを読む（`map_episode_to_frames`・lerobot 不要）

export の心臓部は [`../scripts/export_lerobot.py`](../scripts/export_lerobot.py) の **`map_episode_to_frames`** です。
これは「**自作 episode → LeRobot 風フレーム辞書の列**」へ変換する純粋な関数で、**lerobot をインストールして
いなくても動き、単体テストできます**。まず実物を読みましょう（抜粋）。

```python
# scripts/export_lerobot.py より（抜粋）
def map_episode_to_frames(ep: dict, img_size: int = 64) -> list[dict]:
    """自作 episode を LeRobot 風のフレーム辞書列へ変換する（lerobot 不要）。

    各フレーム:
      observation.image : [H, W, 3] uint8（LeRobot は HWC・uint8 画像が標準）
      observation.state : [3] float32
      action            : [3] float32
      task              : str（言語指示）
    """
    frames = []
    T = ep["actions"].shape[0]
    for t in range(T):
        img_chw = render_world(
            ep["objects_pos"][t], ep["objects_color"], ep["goals_pos"], ep["goals_color"],
            ep["agent"][t, :2], float(ep["agent"][t, 2]), size=img_size,
        )  # [3,H,W] float 0..1
        img_hwc = (np.transpose(img_chw, (1, 2, 0)) * 255).astype(np.uint8)  # [H,W,3] uint8
        frames.append(
            {
                "observation.image": img_hwc,
                "observation.state": ep["agent"][t].astype(np.float32),
                "action": ep["actions"][t].astype(np.float32),
                "task": ep["instruction"],
            }
        )
    return frames
```

読みどころは 3 つです。

- **画像は M3 と同じく「都度レンダリング」**: ディスクには低次元の状態だけ保存し
  （[M3](m3_data_actions.md) の `save_dataset`）、export 時に `render_world` で `[3,H,W] float` を作ります。
- **`img_chw` → `img_hwc` の変換が、A-1.1 で説明した CHW float → HWC uint8**:
  `np.transpose(img_chw, (1, 2, 0))` で軸を `(H, W, 3)` に並べ替え、`* 255` して `uint8` に丸めます。
- **`task` は `ep["instruction"]`（文字列）をそのまま入れる**: トークン化はしません。LeRobot 側は
  指示を**文字列**として持つのが基本だからです（トークン化は学習時にモデル側でやる）。

### A-2.1 自分で形を確かめる（重い処理なし）

`map_episode_to_frames` は lerobot 不要なので、**手元ですぐ shape を確認**できます。

```python
import numpy as np
from vla_learn.datasets import generate_episodes
# 関数は scripts 側にあるので、PYTHONPATH を通したうえで import する
import sys; sys.path.insert(0, "scripts")
from export_lerobot import map_episode_to_frames

ep = generate_episodes(n_episodes=1, seed=0)[0]
frames = map_episode_to_frames(ep)
print("フレーム数 T =", len(frames))
fr = frames[0]
print("image  :", fr["observation.image"].shape, fr["observation.image"].dtype)  # (64,64,3) uint8
print("state  :", fr["observation.state"].shape, fr["observation.state"].dtype)  # (3,) float32
print("action :", fr["action"].shape, fr["action"].dtype)                        # (3,) float32
print("task   :", repr(fr["task"]))                                              # 例 '青のブロックを…'
print("画像の値域:", fr["observation.image"].min(), "〜", fr["observation.image"].max())  # 0 〜 255
```

出力例（指示文・値はぶれます）:

```text
フレーム数 T = 23
image  : (64, 64, 3) uint8
state  : (3,) float32
action : (3,) float32
task   : '青のブロックを青のゴールに置いて'
画像の値域: 0 〜 255
```

> ここで「**画像が `[64,64,3]` の uint8、値域 0〜255**」になっていることが、A-1.1 の「保存で使う形」です。
> M4 のモデル入力 `[3,64,64]` float（0..1）との違いを、自分の目で確認しておきましょう。

### A-3. 実際に export する（書き込みはバージョン依存）

ここからが**バージョン依存**の領域です。`export_lerobot.py` の `main` は、おおむね次の構造です。

1. `load_dataset` で自作データを読む（lerobot 不要）。
2. `map_episode_to_frames` で**変換が通ることを確認**（lerobot 不要。ここまでは必ず動く）。
3. `lerobot` の import を **try で囲む**。無ければ案内を出して**安全に終了**。
4. ある時点の v2 系 API を想定して、`LeRobotDataset.create(...)` → `add_frame(...)` → `save_episode()` を呼ぶ。
   **API が違えば warn を出す**（落とさない）。

実行コマンドは次の通りです。

```bash
# 1) まず自作データを作る（無ければ）
uv run python scripts/make_dataset.py --n-episodes 1000 --out data/tabletop2d

# 2) LeRobot 形式へ export を試みる
uv run python scripts/export_lerobot.py --in data/tabletop2d --repo-id your-name/tabletop2d
```

**lerobot が入っていない場合**の出力例（変換は確認され、書き込みはスキップ）:

```text
[load] 1000 episodes from data/tabletop2d
[map] フレーム変換 OK（先頭エピソードで形を確認）。概算フレーム数 ~ 23000

[info] lerobot が見つかりません（任意依存）。インストール:
       uv pip install lerobot   （または uv sync --extra lerobot）
       （詳細: ModuleNotFoundError）
[info] 変換ロジック map_episode_to_frames は lerobot 無しでも利用・テスト可能です。
```

`features`（特徴量スキーマ）の定義は、概念としてはこうです（**スキーマの考え方を見るための例**であり、
実際のキー名・必須項目はバージョンに従ってください）。

```python
# export_lerobot.py より（抜粋・v2 系を想定した一例）
h = w = 64
features = {
    "observation.image": {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (3,), "names": ["state"]},
    "action":            {"dtype": "float32", "shape": (3,), "names": ["action"]},
}
# ds = LeRobotDataset.create(repo_id=..., fps=..., features=features)
# for ep in episodes:
#     for fr in map_episode_to_frames(ep):
#         ds.add_frame(fr)
#     ds.save_episode()
```

> **正直な注意（重要）**: 上の `create / add_frame / save_episode` という呼び方は**バージョンで変わります**。
> 引数名、`task` の渡し方（フレームごとか `add_frame` の引数か）、画像の dtype 名なども差があります。
> したがって `export_lerobot.py` はこの部分を **try で囲み、失敗しても落とさず warn を出す**設計にしています。
> **正確な使い方は、あなたが入れた lerobot のバージョンのドキュメント**を見てください。
> 一方で、**`map_episode_to_frames`（変換の本体）はバージョンに依存しません**。だからこの関数を理解・テスト
> できれば、「自作データを規格に載せる」という学習目標は達成です。

### A-3.1 なぜ「変換ロジックを独立させる」設計が良いか

ライブラリの**書き込み API は変わりやすい**が、**「自分のデータ → 規格が要求する特徴量」への対応づけ**は
**設計判断であって、変わりにくい**ものです。この 2 つを分離しておくと:

- API が変わっても、**直すのは薄い書き込み層だけ**で済む。
- **変換ロジックを単体テストできる**（lerobot を入れずに CI で回せる）。
- 「どの特徴量に何を入れるか」という**本質的な理解**が、コードとして残る。

これは VLA に限らず、外部規格に合わせるときの定石です。**「変わりやすい所」と「変わりにくい所」を分ける**、
という設計眼をここで養ってください。

---

## パート B: SmolVLA 精読 — 自作 VLA との対応表

ここからモデルの話です。**精読の対象に SmolVLA を選ぶ**のは、次の理由からです。

- **LeRobot に直結**（同じエコシステム。データ規格がパート A とつながる）。
- **小型の OSS**（VLM + 状態トークン + action expert + flow matching + action chunking が**一通り**入っている）。
- だから **あなたが M4/M5 で作った部品と、ほぼ 1 対 1 で対応づけられる**。

> 出典: SmolVLA 論文 **arXiv:2506.01844**「SmolVLA: A Vision-Language-Action Model for Affordable and
> Efficient Robotics」、公式実装 [huggingface/lerobot](https://github.com/huggingface/lerobot)、
> モデルカード `lerobot/smolvla_base`。**以下の構成説明はこれら一次情報に基づきますが、内部実装の細部は
> 必ず原典で確認してください。** 数値は原典に記載のある範囲だけを、幅をもって書きます。

### B-1. SmolVLA の構成（設計思想レベル）

設計思想レベルで言うと、SmolVLA は次の流れです。

```text
                ┌──────────────── VLM（事前学習済み）────────────────┐
  画像 ─▶ 視覚エンコーダ(SigLIP系) ─▶ 視覚トークン(圧縮) ─┐         │
  言語(task) ─▶ トークナイズ ──────────────────────────┼─▶ VLM デコーダ
  状態(state) ─▶ 線形射影 ─▶ 状態トークン ────────────┘   │ で文脈を統合
                └──────────────────────────────────┬───────┘
                                                    ▼
                                  ┌──── action expert（別の小さなネット）────┐
                                  │  flow matching で行動チャンクを生成        │
                                  └────────────────────────────────────────┘
                                                    ▼
                                          行動チャンク（連続値・複数ステップ）
```

要点（原典に基づく、設計思想レベルの要約）:

- **入力は「画像 + 言語(task) + 状態」**。画像は視覚エンコーダで視覚トークンに、言語は普通にトークン化、
  **状態（proprioception）は線形層で射影してトークンとして**入れます（本教材と同じ 3 入力）。
- **VLM は事前学習済み**（SigLIP 系の視覚 + 小型言語モデル）。**ゼロから学ばず、既に世界を知っている VLM を使う**
  のが、本物の VLA と本教材 TinyVLA の最大の違いです（後述の対応表）。
- **action expert** という**別の小さなネット**が、VLM の文脈ベクトルを条件に**行動を生成**します。
- 生成方式は **flow matching**（[M5](m5_flow_matching.md) で自作したもの）。出力は**連続値の行動チャンク**
  （action chunking）。
- **小型・安価**を狙った設計で、コンシューマ GPU や CPU でも動くことを主眼にしています（規模の正確な値は原典）。

> **不確実さの明記**: ベース VLM の正確な variant 名、各部のパラメータ数、コミュニティデータセットの正確な本数
> などは、論文本文とブログで表記がぶれることがあります。本教材では**断定しません**。値が必要なときは
> arXiv:2506.01844 と最新のモデルカードを参照してください。

### B-2. 【中核】TinyVLA / FlowVLA との対応表

本章のいちばん大事な表です。**あなたが自作した部品が、SmolVLA では何に当たるか**を対応づけます。

| 役割 | 本教材の自作部品（M4/M5） | SmolVLA での対応（概念） |
|---|---|---|
| 視覚の符号化 | `ImageEncoder`（小さな CNN、flatten で空間保持） | **事前学習済み視覚エンコーダ（SigLIP 系）** + 視覚トークン圧縮 |
| 言語の符号化 | `TextEncoder`（文字埋め込み + 位置 + 1 層 Transformer） | **事前学習済み言語モデル**（VLM の言語側） |
| 状態の符号化 | `StateEncoder`（小さな MLP） | **状態を線形射影した「状態トークン」** |
| 視覚×言語の融合 | **FiLM**（言語で視覚を変調）＋ concat → 融合 MLP（`VLABackbone`） | **VLM デコーダ内の注意機構**で全トークンを統合 |
| 条件ベクトル | `VLABackbone` の出力 `h [B, hidden]` | VLM が出す**文脈表現** |
| 行動ヘッド（生成） | `FlowVLA` の `velocity` + flow loss（[M5](m5_flow_matching.md)） | **action expert** による **flow matching** |
| 行動の形 | 行動チャンク `[B, chunk_len, action_dim]` | 行動チャンク（連続値・複数ステップ） |
| 学習則 | rectified flow の MSE（速度予測） | flow matching |

この表の読み方を 2 つ補足します。

- **「画像/言語エンコーダ → FiLM 融合 → flow ヘッド」（自作）が、SmolVLA では「事前学習 VLM →
  action expert（flow）」に対応**します。つまり**骨格は同じ**で、SmolVLA は前半（融合）を
  **巨大な事前学習 VLM**に、後半（行動生成）を **専用の action expert**に置き換えた格好です。
- あなたの `VLABackbone` が作る条件ベクトル `h` は、SmolVLA では VLM の文脈表現に当たります。そして
  [M5](m5_flow_matching.md) で書いた **`flow_loss` / `sample`（Euler 積分）の発想は、SmolVLA の action expert と
  同じ系統**です（規模と事前学習の有無が違うだけ）。

> つまり **M4 + M5 を作り切った時点で、SmolVLA の「読み方」は手に入っています**。残りの差は主に
> 「**VLM を事前学習で賢くしてあるか**」と「**実機データで大規模に学習してあるか**」です。

### B-3. 本教材 TinyVLA との「距離」

正直に距離も書きます。本教材の `TinyVLA`/`FlowVLA` と SmolVLA の主な違いは次の通りです。

- **事前学習の有無**: SmolVLA の VLM は**大量の画像・言語で事前学習済み**。本教材は**ゼロから**学ぶので、
  言語理解も視覚理解も「このおもちゃ世界の範囲」だけです。
- **規模**: SmolVLA は数億パラメータ級、本教材は約 0.42M（[M4](m4_tiny_vla_mse.md) の `TinyVLA`）／約 0.58M（[M5](m5_flow_matching.md) の `FlowVLA`）。
- **入力の豊かさ**: 実機は複数カメラ・高次元の関節状態。本教材は 1 視点・3 次元状態。
- **データ**: SmolVLA は LeRobot コミュニティの実機データ（規模は原典参照）。本教材は合成データ。

逆に言えば、**設計の骨格（3 入力 → 融合 → flow で行動チャンク）は共通**です。だから「小さく作って大きく読む」
が成立します。これが本教材の[基本方針](../docs/curriculum.md)でした。

---

## パート C: 有名 VLA 概観 — 行動表現で地図を描く

ここからは**概観**です。**設計思想の地図**に徹し、**憶測の数値は書きません**。各モデルを
**「入力 → 表現/トークン化 → 行動ヘッド（離散 or flow/diffusion）→ 学習データ」**の軸で 1〜2 段落にまとめ、
自作 VLA との距離感を述べます。**正確な数値・内部実装は、必ず原典（arXiv / 公式 repo）で確認**してください。

> **共通の読み筋**: VLA は行動の出し方で大きく **2 系統**に分かれます。
> - **離散トークン自己回帰**（行動を「単語」に量子化して言語モデルのように 1 トークンずつ生成）
> - **連続生成（flow matching / diffusion）**（[M5](m5_flow_matching.md) で自作した系統）
> 本教材の `TinyVLA` は「決定論的回帰」、`FlowVLA` は「連続生成」。下の各モデルを、この軸の上に置いて読みます。

### C-1. OpenVLA — 離散トークン自己回帰 + 大規模 VLM

> 出典: **arXiv:2406.09246**「OpenVLA: An Open-Source Vision-Language-Action Model」、
> 公式 repo [openvla/openvla](https://github.com/openvla/openvla)。

OpenVLA は **行動を「離散トークン」に変換し、VLM に自己回帰で生成させる**方式の代表です。各行動次元を
**256 個のビンに離散化**し、言語モデルの語彙の一部（使用頻度の低いトークン）を**行動トークンに置き換えて**
扱います。入力は**画像 + 言語指示**で、論文には「**現状は単一画像のみ対応**（固有受容状態は入力しない）」と
明記されています。ベース VLM は **Prismatic VLM（DINOv2 + SigLIP の視覚エンコーダ + Llama 2 7B）**、規模は
**7B**（表記は出典で確認）。学習データは **Open X-Embodiment の約 97 万（970k）軌跡**です。

**自作 VLA との距離**: 行動の出し方が本教材と**最も違う**例です。本教材は行動を連続値（回帰 or flow）で
出しましたが、OpenVLA は**行動を「単語」に量子化して言語モデルのように吐く**。あなたの言葉でいえば、
`TinyVLA` の `head`（連続値を出す `Linear`）を「**256 クラス分類を action_dim×chunk 回**」に置き換えた発想です
（卒業課題 ⑤ でこの離散化ヘッドを少しだけ自作します）。状態を使わない点も本教材（state を使う）と対照的です。

### C-2. π0 / π0.5 — flow matching と action expert（実機データ規模）

> 出典: **π0 = arXiv:2410.24164**「π₀: A Vision-Language-Action Flow Model for General Robot Control」、
> **π0.5 = arXiv:2504.16054**、公式 repo [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)、
> 公式ブログ physicalintelligence.company。

**π0** は **flow matching で連続行動を生成**する VLA です（[M5](m5_flow_matching.md) と同じ系統）。**action expert**
という**ロボット用の追加ネット**を VLM に付け、**状態（関節角）と行動をトークンとして**扱います。ベース VLM は
**PaliGemma（約 3B）**。行動は**チャンク化**して出します。学習データは自社収集 + 公開データで、**規模は大きい**
（時間数・ロボット構成数などは原典参照）。**つまり π0 は、本教材 `FlowVLA` の「事前学習 VLM + action expert +
実機大規模データ」版**と読めます。**M5 を作ったあなたには、いちばん地続きの本格 VLA**です。

**π0.5** は π0 を基盤に、**オープンワールド汎化**（学習に無い新しい家庭での動作）を狙った発展版です。Web データ
（QA・物体位置特定）や高レベルのサブタスク予測を**異種データで co-training** し、**高レベルは離散トークン予測 /
低レベルは連続生成**という**ハイブリッド**を使います（事前学習は離散トークン、ポストトレーニングで flow matching の
action expert を追加）。

> **不確実さの明記（重要）**: π0.5 で語られる「知識絶縁 (knowledge insulation)」という**定式化は、π0.5 論文本文
> ではなく後続の arXiv:2505.23705 で確立**され、公式 repo がそれを紐づけています。また **π0.5 のベース VLM 名・
> パラメータ数・学習データの定量規模は、一次資料に明記が見当たりません**。これらは**断定せず**、原典を参照して
> ください。本教材で確実に言えるのは「π0 系は flow matching + action expert で連続行動を出す」までです。

### C-3. GR00T N1 — ヒューマノイド向け dual-system

> 出典: **arXiv:2503.14734**「GR00T N1: An Open Foundation Model for Generalist Humanoid Robots」、
> 公式 repo [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)。

GR00T N1 は **ヒューマノイド向け**の基盤モデルで、特徴は **dual-system 構成**です。

- **System 2（遅い）= vision-language module**: 視覚と言語で状況を解釈する、VLM 系の**遅い熟考**。
- **System 1（速い）= action module**: **リアルタイムで運動を生成**する**速い行動**。実装は
  **Diffusion Transformer (DiT) を action flow-matching で学習**したものです（連続行動）。

両システムを **end-to-end で密結合**して同時に学習します。ベース VLM は **Eagle-2**、状態は**実体ごとの MLP**で
共有埋め込みに射影し、行動は**チャンク化**して出します。「遅い思考 + 速い反射」という**人間の二重過程**の比喩が
そのまま設計になっているのが面白い点です。

**自作 VLA との距離**: 本教材 `FlowVLA` は **System 1 に近い**（速い行動生成、しかも flow/diffusion 系）。GR00T は
その上に **System 2（VLM の熟考）** を載せ、**ヒューマノイドの高次元・多実体**へスケールさせた格好です。
「行動生成の核は M5 で作ったものと同系統、その上に思考層が乗る」と捉えると見通しが良いです。

> **不確実さの明記**: 要旨では System 1 を「diffusion transformer」までしか書いておらず、**flow-matching の語は
> 本文記載**です。後継（N1.5 など）の VLM 名は出典間で表記が割れます。**世代・VLM 名・数値は原典で確認**して
> ください。

### C-4. MolmoAct — 行動の「推論（トレース）」

> 出典: **arXiv:2508.07917**「MolmoAct: Action Reasoning Models that can Reason in Space」、
> 公式 repo [allenai/molmoact](https://github.com/allenai/molmoact)、Ai2 の VLM **Molmo** を基盤。

MolmoAct は **「行動推論モデル (Action Reasoning Model)」** という位置づけで、**行動する前に空間で推論する**
ことを前面に出します。おおむね **3 段階**で、(a) 観測・指示を **depth-aware perception token（深度を意識した
知覚トークン）** にして 3D 的に接地し、(b) 中間的な空間計画を**画像上のウェイポイント列（visual reasoning
trace）**として描き、(c) 最後に**低レベル行動**を出します。行動は **256 ビンの離散トークン**で、**チャンク**して
出します。ベース VLM は Ai2 の **Molmo**（言語側 Qwen2.5-7B + 視覚 SigLIP2）。**重み・コード・データまで公開**
（Apache 2.0）です。

**自作 VLA との距離**: 行動の出し方は **OpenVLA に近い離散トークン**ですが、**「行動の前に推論の痕跡（trace）を
作る」**点が独特です。本教材 `TinyVLA`/`FlowVLA` は観測から行動へ**直接**写像しましたが、MolmoAct は
**「観測 → 空間推論（中間トークン）→ 行動」**と**1 段はさむ**。chain-of-thought（思考の連鎖）を**行動に持ち込んだ**
イメージです。

> **不確実さの明記**: MolmoAct の**状態（proprioception）入力の扱い**は一次資料に明確な記述が見当たりません。
> 主入力は視覚観測 + 指示です。後継（MolmoAct2 など）もあるため、初代との混同に注意し、**詳細は原典参照**と
> してください。

### C-5. 比較表（行動表現で並べる）

**設計思想レベルの地図**です。**数値や内部実装の断定は避けています**。正確な値・最新情報は各 arXiv / 公式 repo で
確認してください。

| モデル | 行動表現 | 生成方式 | 状態(state)入力 | 特徴（要点） |
|---|---|---|---|---|
| **Diffusion Policy**（前史） | 連続 | **diffusion（DDPM）** | あり（言語なし） | 生成的行動ヘッドの原典。M5 の flow の先祖 |
| **ACT / ALOHA**（前史） | 連続 | 決定論（CVAE+L1）+ **temporal ensembling** | あり（言語なし） | 行動チャンクの代表作。M4 6.3 節の出所 |
| **TinyVLA（自作・M4）** | 連続 | 決定論的回帰（MSE） | あり | 最小構成。flatten で空間保持 + FiLM で言語接地 |
| **FlowVLA（自作・M5）** | 連続 | **flow matching** | あり | 行動ヘッドを生成モデルに差し替え |
| **SmolVLA** | 連続 | **flow matching** | あり（状態トークン） | 小型 OSS・LeRobot 直結・action expert。**精読対象** |
| **OpenVLA** | **離散**（256 ビン） | **自己回帰**（トークン） | なし（単一画像のみ） | 大規模 VLM（7B）。行動を「単語」化 |
| **π0** | 連続 | **flow matching** | あり | action expert + 実機大規模データ。**M5 の本格版** |
| **π0.5** | ハイブリッド | 離散 + **flow matching** | あり | オープンワールド汎化。co-training |
| **GR00T N1** | 連続 | **DiT + flow matching** | あり | ヒューマノイド・**dual-system**（System2 VLM + System1 行動） |
| **MolmoAct** | **離散**（256 ビン） | 自己回帰 + **空間推論トレース** | 不明（原典参照） | 「行動の前に推論」。visual reasoning trace |

> この表で押さえるべきは「**行動の出し方は離散自己回帰か連続生成（flow/diffusion）かに大別され、
> あなたは両系統の最小版（離散は卒業課題 ⑤、連続は M5）を自分で書ける**」という地図です。
> 個々のセルの厳密さより、**自作部品からの距離**を読み取ってください。

VLA 以前の系譜も 2 つだけ押さえます（表の「前史」行）。**Diffusion Policy**（Chi et al. 2023）は
「行動チャンクを diffusion で生成する」ことを確立した原典で、**M5 の flow ヘッドの直接の先祖**です
（言語なし・視覚+状態のみ。あなたの diffusion 座学に最も近い実装）。**ACT**（Zhao et al. 2023, ALOHA）は
「行動チャンク + temporal ensembling + CVAE」でテレオペ模倣を実用にした代表作で、本教材の `chunk_len` や
[M4 6.3 節](m4_tiny_vla_mse.md)の ensembling の出所です。また OpenVLA の 2025 改良版 **OpenVLA-OFT** は
離散自己回帰を「並列デコード + 連続値の L1 回帰」に置き換えて推論を大幅高速化しました — 離散 vs 連続の
議論が今も動いている好例です。

3 系統のトレードオフを一言ずつ:
**連続回帰（M4）** は最速・最簡だが多峰を平均で潰す。
**離散トークン（OpenVLA 系）** は行動を「単語」にするので**言語モデルの機構（語彙・自己回帰・尤度）をそのまま使える**
のが最大の利点だが、自己回帰のぶん推論が遅く、ビン離散化の量子化誤差が乗る。
**flow / diffusion（M5, π0 系）** は多峰を保ったまま連続値を生成できるが、積分ステップのぶん推論が重い。
この 3 択の感触は [`../exercises/m6/discrete_head.py`](../exercises/m6/discrete_head.py)
（256 ビン離散化ヘッドを自作して MSE 版・flow 版と比べる演習）で手を動かして確かめられます。

### C-6. 【総まとめ】自作部品を「どう差し替えると、どのモデルになるか」

B-2 では自作部品と SmolVLA を 1 対 1 で結びました。ここでは**他の有名 VLA も同じ目線**で、
「**あなたの自作部品の、どこを・どう差し替えると、そのモデルに化けるか**」を 1 枚にまとめます。
これが本章の地図の最後のピースです。

| あなたの自作部品 | これを…に差し替えると | → このモデルになる |
|---|---|---|
| `head`（連続値 `Linear`） | 行動を 256 ビンに離散化し、トークンとして**自己回帰生成** | OpenVLA / MolmoAct |
| flow ヘッド（`FlowVLA`） | **action expert** + 事前学習 VLM + 実機大規模データ | π0 / SmolVLA |
| 観測 → 行動を**直接**写像 | 観測 → **空間推論トレース** → 行動 と 1 段はさむ | MolmoAct |
| 単一の融合 MLP（`VLABackbone`） | **System 2（VLM の熟考）+ System 1（高速 flow）** の二層 | GR00T N1 |
| **ゼロから学習** | 大規模**事前学習 VLM**（SigLIP 系 + 言語モデル）を土台にする | 実在 VLA すべてに共通 |

> ここで腑に落ちてほしいのは——**自作の最小 VLA は、どの部品を大規模・高機能に差し替えるかで、
> 各実在 VLA に化ける**ということ。逆に言えば、有名 VLA はどれも「あなたが書いた部品の、
> どこかを強化した版」として読めます。これが「小さく作って大きく読む」の到達点です。

### C-7. 実 VLA の使い方は「fine-tune」— 本教材との最大の運用差

本教材は終始スクラッチ学習でしたが、実務で OpenVLA や π0 系を新しいロボット・新タスクに使うときは、
**事前学習済みチェックポイントを fine-tune する**のが標準です。押さえる概念は 3 つだけ:

- **どこを凍結し、どこを学習するか**: 定番は「VLM（視覚・言語）側は凍結または低学習率、action head
  （expert）は付け替えて学習」。行動空間（関節数・単位系）はロボットごとに違うので head の付け替えは
  ほぼ必須です — あなたが M4→M5 でやった「**head 差し替え**」の実務版がこれです。
- **LoRA（低ランク適応）**: 7B 級 VLM の全重みを更新する代わりに、attention 層などに小さな低ランク行列を
  挿して**そこだけ学習**します。OpenVLA の公式 fine-tune も LoRA（rank 32）が既定。座学で知る LoRA の
  適用先が「VLM の attention 層 + action head」だと分かっていれば、実 repo の設定ファイルが読めます。
- **データ効率と co-training**: 新タスクのデモは数十〜数百本のことが多く、事前学習で使った広い分布の
  データと混ぜて学習（co-training）して忘却を防ぎます（π0.5 の主要テーマ）。

> **推論効率の話も 1 分だけ**: 実機の制御は 10〜50Hz で回るのに、7B 級 VLA の 1 推論は数十〜数百 ms
> かかります。対策の軸は (1) **action chunking** で推論回数を減らす（[M3](m3_data_actions.md) で学んだ理由の実務版）、
> (2) 自己回帰系は **KV キャッシュ**（過去トークンの attention 計算を使い回す）や**並列デコード**（OpenVLA-OFT）、
> (3) **重みの量子化**（8/4bit 化。※本教材で出た「行動の離散化」とは別物）や蒸留です。

> **卒業課題の思考実験**: 本教材の `VLABackbone` を凍結して `head` だけ再学習したら、成功率はどこまで戻るか？
> `for p in model.backbone.parameters(): p.requires_grad = False` の 1 行で試せます（M1 の autograd の回収。
> 「事前学習済み表現がどれだけ仕事をしているか」を最小構成で体感できます）。

---

## パート D: 卒業課題へ

### D-0. 実機に行くと何が増えるか（正直リスト）

本教材は「実機なし」で完結しますが、その先（LeRobot の実機チュートリアル等）へ進むときに
**新たに増えるもの**を正直に列挙しておきます。逆に言えば、これ以外（モデル・損失・データ形式・評価の考え方）は
本教材で学んだままです:

- **観測**: カメラのキャリブレーション・複数視点・遅延やフレーム落ち。「完璧な観測」は無い。
- **制御**: 制御周波数（例 10〜50Hz）と行動の単位系（関節角 or エンドエフェクタ座標）。チャンク長は周波数に合わせて再設計。
- **データ**: テレオペでの人手収集コスト（数十〜数百エピソード）。LeRobot のテレオペ機能が定番。
- **分布シフトの強化版**: 照明・背景・物体の個体差。画像 augmentation やドメインランダム化が効いてくる。
- **安全**: 非常停止・作業領域制限・速度制限が**最優先**。学習方策は平気で変な行動を出す前提で設計する。

ここまでで、**自作 VLA → データ規格（LeRobot）→ 有名 VLA の地図**がつながりました。最後は
**卒業課題（capstone challenge）**で、自分の VLA を**いじり倒して**理解を確かなものにします。

卒業課題（一例）:

1. **chunk_len / flow_steps を変えて成功率を比較**する。
2. **`image_pool='avg'` / `condition_vision=False` で ablation**（部品を抜くと何が壊れるか）。
3. **新しい命令テンプレや色を足して般化**を見る。
4. **自作データを LeRobot 形式へ export** してみる（パート A の実践）。
5. **（発展）離散トークン化ヘッドを自作**して OpenVLA 流（C-1）と比べる。
6. **（発展）temporal ensembling を実装**する（[M4 6.3 節](m4_tiny_vla_mse.md)）。毎ステップ予測 +
   過去チャンクの指数重み平均で境界の段差を消し、素朴な receding horizon と成功率・軌道の滑らかさを比べる。

→ 課題は [`../exercises/m6/README.md`](../exercises/m6/README.md)、取り組みの指針は
[`../solutions/m6/README.md`](../solutions/m6/README.md) にあります。

> **重い学習は不要**です。多くの課題は「設定を変えて短く学習 → `success_rate` を見る」だけで観察できます。
> 値は環境・乱数でぶれるので、**絶対値でなく傾向**（部品を抜くと下がる、設定で変わる）を読みましょう。

---

## 理解度チェック（即答できるか）

卒業課題に進む前に、次に**自分の言葉で即答**できるか確かめてください。詰まったら、示した節に戻りましょう。

1. LeRobot では画像が `[H, W, 3] uint8` なのに、学習時は `[3, H, W] float` に戻すのはなぜ?
2. `map_episode_to_frames`（変換ロジック）を lerobot 無しでもテストできるよう独立させる利点を 2 つ。
3. SmolVLA の「**action expert**」は、あなたの自作のどの部品に対応するか?
4. **OpenVLA と π0 の最大の違い**を「行動表現」の語で一言。
5. GR00T N1 の **System 1 / System 2** は、それぞれ自作 VLA の何に近いか?

> 答えに詰まったら: 1 → A-1.1 / 2 → A-3.1 / 3 → B-2 / 4 → C-1・C-2・C-5 / 5 → C-3。
> すべて即答できれば、あなたは「VLA を読む地図」を手にしています。卒業課題で「いじれる」へ進みましょう。

---

## まとめ

- **(A) LeRobot**: `LeRobotDataset` は**フレーム列**で、`observation.images.* / observation.state / action / task`
  を持つ。**画像は HWC・uint8 が保存標準**（学習は CHW・float）。変換 `map_episode_to_frames` は **lerobot 無しで
  動く**。**書き込み API はバージョン依存**なので分離してある。**`lerobot-train` には最初から依存しない**方針。
- **(B) SmolVLA 精読**: 「VLM + 状態トークン + action expert + flow matching + action chunking」。**自作の
  『画像/言語エンコーダ → FiLM 融合 → flow ヘッド』が『事前学習 VLM → action expert(flow)』に対応**。M4+M5 を
  作れば読める。差は主に「事前学習の有無」と「データ規模」。
- **(C) 概観**: 行動の出し方で **離散自己回帰（OpenVLA / MolmoAct）** と **連続生成（SmolVLA / π0 / GR00T）** に
  大別。GR00T は **dual-system**、MolmoAct は **行動前の空間推論**が特徴。**数値・細部は原典参照**。
- **(D)**: 卒業課題で自作 VLA をいじり、理解を確定させる。

## 次の章へ

これで本教材の本編は終わりです。あなたは **小さな VLA を 2 通り（MSE / flow）自作**し、**データを規格化**でき、
**有名 VLA を『同じ部品の大規模版』として読める**地図を手に入れました。

最後の仕上げは [`../exercises/m6/README.md`](../exercises/m6/README.md) の**卒業課題**です。ひとつでも自分の手で
やり切ると、「VLA を読める」から「VLA をいじれる」へ進めます。さらに先へ行きたくなったら、本章で挙げた各
arXiv / 公式 repo を、**自作部品との対応表を片手に**読んでみてください。
