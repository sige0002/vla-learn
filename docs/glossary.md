# 用語集（Glossary）

本教材で使う用語を、初心者向けに 1〜3 行で説明します。可能な範囲で**本教材での具体例**（実装上の対応物）を併記しました。
コード上の実体は `../src/vla_learn/` にあります。各章の本文は [`../lessons/`](../lessons/) を参照してください。

並びは五十音（日本語読み）→ アルファベットの順です。まず日本語の用語、続いて英字始まりの用語を載せます。

---

## 日本語（五十音順）

### 位置埋め込み（positional embedding）
トークンの「並び順」を表すベクトル。Transformer は単語をまとめて見るため、何もしないと語順が消えます。そこで各位置に固有のベクトルを足して「何番目か」を区別させます。
本教材では `TextEncoder`（`../src/vla_learn/models/text_encoder.py`）が token 埋め込みに位置埋め込みを加算しています。これがないと「赤を青ゴール」と「青を赤ゴール」が同じ表現になり、grounding できません。flow head の時刻埋め込み（[SinusoidalTimeEmbedding](#正弦波時刻埋め込みsinusoidaltimeembedding)）も同じ発想です。

### 埋め込み（embedding）
記号（文字・トークンなど）を、意味を表す密ベクトルに変換すること、またはその変換表。離散の ID を連続ベクトルにして、ニューラルネットが扱えるようにします。
本教材では `TextEncoder` の token 埋め込みが該当します（文字 ID → ベクトル）。→ [トークナイザ](#トークナイザtokenizer)

### エキスパート（expert）
お手本となる行動を生成する存在。模倣学習では、このエキスパートの (観測, 行動) を集めて学習データにします。
本教材のエキスパートは**解析的に計算する関数** `expert_action(world)`（`../src/vla_learn/envs/expert.py`）で、対象ブロックへ近づき掴み、ゴールへ運ぶ手続きを直接書いたものです。成功率はおよそ 100% で、これが学習データの品質の上限になります。→ [模倣学習](#模倣学習imitation-learning--behavior-cloning)

### エンコーダ（encoder、画像／言語／状態）
入力をモデル内部の特徴ベクトルに変換する部品。VLA では「画像・言語・状態」をそれぞれ別のエンコーダで特徴に変えてから融合します。本教材の 3 つのエンコーダ:

- **画像エンコーダ** `ImageEncoder`（`../src/vla_learn/models/image_encoder.py`）: 画像 `[3,64,64]` を特徴ベクトルにする小さな CNN。
- **言語エンコーダ** `TextEncoder`（`../src/vla_learn/models/text_encoder.py`）: token 列を特徴ベクトルにする（埋め込み + 位置埋め込み + 1 層 Transformer + PAD 除外平均）。
- **状態エンコーダ** `StateEncoder`（`../src/vla_learn/models/state_encoder.py`）: 状態 `[3]` を特徴ベクトルにする MLP。

→ [融合](#融合fusion)

### 回帰（regression）／ MSE
連続値を直接予測する学習方式。予測と正解の差の二乗（平均二乗誤差 = MSE; Mean Squared Error）を小さくします。1 つの正解に向かって縮むので、出力は平均的な 1 つの値に決まります（決定論的）。
本教材では M4 の `TinyVLA`（`../src/vla_learn/models/tiny_vla.py`）が MSE 回帰版です。多峰性のあるデータでは「平均」をとってしまう弱点があり、その対処が M5 の [flow matching](#flow-matching--rectified-flow) です。→ [多峰性](#多峰性multimodality) / [masked_mse](#masked_mse) 

### 行動チャンク（action chunk / action chunking）
1 ステップだけでなく、これから `chunk_len` ステップ分の行動をまとめて予測する手法。推論回数が減り、行動が滑らかになりやすい利点があります。
本教材では `extract_action_chunk(actions, t, chunk_len)`（`../src/vla_learn/datasets/temporal.py`）が時刻 t から `chunk_len` 個を切り出し、`pad_mask`（1=有効 / 0=パディング）を返します。モデル出力の形は `[B, C, A]`（C=chunk_len, A=action_dim=3）です。ACT・π0・SmolVLA など実用 VLA でも使われます。→ [chunk_len](#chunk_len) / [後退ホライズン](#後退ホライズンreceding-horizon)

### 後退ホライズン（receding horizon）
行動チャンクの使い方の 1 つ。「チャンクを予測 → 先頭の数ステップだけ実行 → もう一度観測して再予測」を繰り返す方式です。先を全部実行せず、こまめに観測し直すので誤差がたまりにくくなります。
本教材では `rollout_episode(..., exec_horizon=4)`（`../src/vla_learn/evaluation/rollout.py`）が該当し、8 ステップ予測して先頭 4 ステップだけ実行します。→ [distribution shift / 誤差蓄積](#distribution-shift--誤差蓄積) / [rollout](#rollout成功率success-rate)

### 固有受容状態（proprioception / state）
自分の身体の状態（関節角・手先位置・グリッパ開閉など）を表す情報。外界の画像とは別に、ロボット自身の内部状態を与えるものです。
本教材の状態は `state[3] = (ax, ay, gripper)`（エージェントの xy 位置とグリッパ開閉、`STATE_DIM=3`、`../src/vla_learn/constants.py`）。`StateEncoder` で特徴に変換します。→ [エンコーダ](#エンコーダencoder画像言語状態)

### 正規化（normalization）
入力・出力の各次元を「平均 0・標準偏差 1」付近にそろえる前処理。ニューラルネットはこのスケールで最も安定して学習できます。
本教材の `Normalizer`（`../src/vla_learn/datasets/normalization.py`）は `(x - mean) / std` で標準化し、`denormalize` で元に戻します。**学習時は行動を正規化して損失を計算し、推論時は出力を逆正規化して環境に渡す**のが定石です。正規化忘れ・逆正規化忘れは典型的なバグです。

### 正弦波時刻埋め込み（SinusoidalTimeEmbedding）
連続時刻 τ∈[0,1] を、複数周波数の sin / cos を並べたベクトルに変換するもの。位置埋め込みと同じ発想で、連続値を滑らかにベクトル化します。
本教材では flow head（`../src/vla_learn/models/flow_head.py`）が τ をこの方式でベクトル化し、速度予測ネットへ渡します。→ [flow matching](#flow-matching--rectified-flow) / [位置埋め込み](#位置埋め込みpositional-embedding)

### 成功率（success rate）
評価エピソードのうちタスクを達成できた割合。学習中の損失（お手本との一致度）とは別に、「実際に環境で動かして成功するか」を測る最終指標です。
本教材では対象ブロックがゴールから `SUCCESS_RADIUS=0.12`（`../src/vla_learn/constants.py`）以内に入れば成功とし、`evaluate_policy(...)`（`../src/vla_learn/evaluation/rollout.py`）が `success_rate` を返します。→ [rollout](#rollout成功率success-rate)

### 速度場（velocity field）
flow matching で、各時刻 τ と各位置で「どの向きにどれだけ進むか」を表すベクトル場。これを τ=0→1 まで積分するとサンプルが生成されます。
本教材の rectified flow では真の速度は一定で v* = a1 − a0。ネットワーク `velocity(a, τ, h)`（`../src/vla_learn/models/flow_head.py`）がこれを予測し、推論時は Euler 法 `a ← a + v·dt` で積分します。→ [flow matching](#flow-matching--rectified-flow)

### 多峰性（multimodality）
正解が 1 つに定まらず、「右に避ける／左に避ける」のように複数のもっともらしい答えが共存する性質。
MSE 回帰は答えの平均をとるため、多峰なデータではどちらでもない中間の（しばしば不正解の）行動を出してしまいます。これを避けるのが [flow matching](#flow-matching--rectified-flow)（生成モデル）で、複数の峰からサンプリングできます。→ [回帰／MSE](#回帰regression-mse)

### 融合（fusion）
画像・言語・状態など複数のエンコーダ出力を 1 つの条件ベクトルにまとめる処理。
本教材の `VLABackbone`（`../src/vla_learn/models/tiny_vla.py`）は 3 つの特徴を連結（concat）し、`fusion` という MLP で混ぜて条件ベクトル `h [B,256]` を作ります。この `h` を行動ヘッド（M4 は回帰、M5 は flow）に渡します。→ [エンコーダ](#エンコーダencoder画像言語状態) / [FiLM](#film任意--発展)

### 方策（policy）
観測（画像・言語・状態など）を入力として行動を出力する関数。VLA 本体はこの方策にあたります。
本教材では `TinyVLA`（M4, 回帰）と `FlowVLA`（M5, flow）が方策です。評価時は `PolicyWrapper`（`../src/vla_learn/evaluation/rollout.py`）で正規化・device 処理を包み、`obs → 行動チャンク` に変換します。

### 埋め込み・位置・トークンの関係（補足）
言語側の流れはおおむね次の通りです。文字列 → [トークナイザ](#トークナイザtokenizer)で ID 列 → [token 埋め込み](#埋め込みembedding) → [位置埋め込み](#位置埋め込みpositional-embedding)を加算 → Transformer → [PAD](#padパディング) を除いて平均、で言語特徴ができます。

### トークナイザ（tokenizer）
文字列を、モデルが扱える ID の列に変換する部品。本物の VLA はサブワード分割（BERT / Llama など）を使いますが、本教材では実装を最小化するため**文字レベル**にしています。
`CharTokenizer`（`../src/vla_learn/datasets/tokenizer.py`）は固定コーパスから語彙を作り、`encode(str)` で長さ `max_len` の ID 列にそろえます（不足は [PAD](#padパディング)=0、超過は切り詰め）。この ID 列を [token 埋め込み](#埋め込みembedding)に通すと言語処理が始まります。

### 模倣学習（imitation learning / behavior cloning）
エキスパートのお手本 (観測, 行動) を集め、「同じ観測なら同じ行動を出す」よう教師あり学習する手法。観測→行動の教師あり回帰として解く最も単純な形を **behavior cloning（BC）** と呼びます。
本教材の学習は基本これです。ただし素朴な BC は閉ループで崩れやすく（→ [distribution shift](#distribution-shift--誤差蓄積)）、M2 でその理由と対策（ノイズ注入など）を扱います。→ [エキスパート](#エキスパートexpert)

---

## アルファベット

### action_dim
行動ベクトルの次元数。本教材では `ACTION_DIM=3`（`../src/vla_learn/constants.py`）で、`[dx, dy, grip_cmd]`（手先の xy 移動量とグリッパ指令）を表します。モデル出力の形 `[B, C, action_dim]` の最後の軸です。→ [行動チャンク](#行動チャンクaction-chunk--action-chunking)

### behavior cloning
→ [模倣学習](#模倣学習imitation-learning--behavior-cloning) を参照。

### chunk_len
一度に予測する未来の行動ステップ数（行動チャンクの長さ）。本教材の既定値は `DEFAULT_CHUNK_LEN=8`（`../src/vla_learn/constants.py`）。モデル出力 `[B, C, A]` の `C` がこれにあたります。→ [行動チャンク](#行動チャンクaction-chunk--action-chunking)

### diffusion との関係
diffusion はノイズから少しずつデータへ「ノイズ除去」していく生成モデル。flow matching は同じゴール（ノイズ→データの生成）を、確率フローの**速度場の積分**として定式化したものです。本教材の rectified flow は「まっすぐな経路」を選んだ特に単純な形で、座学の diffusion 知識をそのまま実装に橋渡しできます。→ [flow matching](#flow-matching--rectified-flow) / [速度場](#速度場velocity-field)

### distribution shift / 誤差蓄積
学習時に見た状態の分布と、実際に方策を動かしたとき訪れる状態の分布がズレる現象。1 手の小さな誤差が次の観測をお手本から外し、その外れた観測でさらに誤り…と**誤差が雪だるま式に蓄積**します。素朴な模倣学習が閉ループで崩れる主因です。
本教材では M2 で体験し、対策として学習データへのノイズ注入（`action_noise`、`../src/vla_learn/training/config.py` の既定 0.03）や [後退ホライズン](#後退ホライズンreceding-horizon)を使います。

### embedding
→ [埋め込み](#埋め込みembedding) を参照。

### FiLM（言語による視覚の条件付け）
Feature-wise Linear Modulation の略。条件ベクトル（ここでは言語特徴）から倍率 γ とバイアス β を作り、別の特徴を `(1+γ)·x + β` でチャンネルごとに変調（条件付け）する手法です。
**本教材では `ImageEncoder`（`../src/vla_learn/models/image_encoder.py`）が FiLM を使い、言語特徴で画像の中間特徴を変調します**。これにより「指示で名指しされた色」に視覚を向けられます。FiLM を切る（`condition_vision=False`）と言語をほぼ無視して grounding が崩れることを M4 の演習で実験できます。融合（連結 + MLP）と併用しています。→ [融合](#融合fusion) / [エンコーダ](#エンコーダencoder画像言語状態)

### flow matching / rectified flow
ノイズ a0〜N(0, I) から目標 a1 へ向かう経路上の**速度場**を学習し、推論時にその速度を積分してサンプルを生成する生成モデル。**rectified（linear）flow** は経路を直線 `a_τ = (1−τ)a0 + τ a1` にとる単純版で、真の速度は一定 `v* = a1 − a0` になります。
本教材の `FlowVLA`（`../src/vla_learn/models/flow_head.py`）はこの v を予測するよう学習し（`flow_loss`）、推論は τ=0→1 を Euler 積分（`sample`、既定 `flow_steps=10`）。M4 の MSE 版と差し替えるだけで、[多峰性](#多峰性multimodality)を扱える生成的な行動ヘッドになります。π0 / SmolVLA と同系統です。→ [速度場](#速度場velocity-field) / [diffusion との関係](#diffusion-との関係)

### LeRobotDataset
Hugging Face の LeRobot が定める、ロボット学習用データセットの標準形式。エピソード・フレーム・各種特徴（画像・状態・行動・指示）を統一的に扱えます。
本教材では M6 で、自作の合成データをこの形式に export し（`../scripts/export_lerobot.py`）、SmolVLA など既存 policy の構成を読むのに使います。最初から LeRobot に依存はしません（[カリキュラム方針](../docs/curriculum.md) 参照）。

### masked_mse
パディングを除いた平均二乗誤差。`masked_mse(pred, target, mask=None)`（`../src/vla_learn/functional.py`）で、`pad_mask=0` のステップを損失計算から外します。
行動チャンクはエピソード終端で末尾がパディングされる（[行動チャンク](#行動チャンクaction-chunk--action-chunking)参照）ため、無効ステップを学習に混ぜないようマスクします。M4 の回帰損失でも M5 の flow 損失でも使われます。

### MSE
→ [回帰／MSE](#回帰regression-mse) を参照。

### PAD（パディング）
長さをそろえるための「埋め草」。系列を固定長にするとき、足りない分を特別な値で埋めます。
本教材のトークナイザは `PAD_ID=0`（`<pad>`、`../src/vla_learn/datasets/tokenizer.py`）で文字列を `max_len` にそろえ、言語エンコーダは PAD を除いて平均します。行動チャンク側の `pad_mask` も同じく「無効ステップ」を示し、[masked_mse](#masked_mse) で損失から除外します。

### policy
→ [方策](#方策policy) を参照。

### proprioception
→ [固有受容状態](#固有受容状態proprioception--state) を参照。

### rectified flow
→ [flow matching](#flow-matching--rectified-flow) を参照。

### rollout（成功率／success rate）
学習済み方策を環境に接続して 1 エピソード動かすこと（閉ループ実行）。複数エピソードの rollout を集計して [成功率](#成功率success-rate)などを測ります。
本教材では `rollout_episode(...)` が 1 エピソード、`evaluate_policy(...)` が複数エピソードを実行します（`../src/vla_learn/evaluation/rollout.py`）。損失が下がっても成功率が上がるとは限らないため、rollout 評価が重要です。→ [後退ホライズン](#後退ホライズンreceding-horizon)

### state
→ [固有受容状態](#固有受容状態proprioception--state) を参照。

### velocity field
→ [速度場](#速度場velocity-field) を参照。

### VLA（Vision-Language-Action）
画像（Vision）と言語指示（Language）を入力に、ロボットの行動（Action）を出力するモデルの総称。VLM（視覚言語モデル）に行動出力を足したもの、と捉えると分かりやすいです。
本教材の到達点は、画像 + 言語 + 状態 → 行動チャンクを出す小さな VLA（M4 `TinyVLA` / M5 `FlowVLA`）を自作することです。有名 VLA（SmolVLA・π0・OpenVLA・GR00T N1・MolmoAct）は「同じ部品の大規模版」として M6 で読みます。

---

> 用語の定義はリポジトリの実装（`../src/vla_learn/`）に合わせています。本文での詳しい説明や図は各章 [`../lessons/`](../lessons/) を参照してください。
