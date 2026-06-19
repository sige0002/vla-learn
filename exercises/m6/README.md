# 演習 M6: 卒業課題（capstone challenge）

> 対応する本文: [`../../lessons/m6_lerobot_and_models.md`](../../lessons/m6_lerobot_and_models.md)
> 取り組みの指針（解答ではありません）: [`../../solutions/m6/README.md`](../../solutions/m6/README.md)

ここは**卒業課題**です。これまでの章（M1〜M5）と違い、**1 問 1 概念の小問ではなく**、自作した VLA を
**自分でいじり倒して理解を確かめる**実験課題を並べます。型でいえば、ほぼ**型 5（自由実験）と型 4（小実装）**です。

> **重い学習はしません。** 各課題は「設定を変えて**短く**学習 → `success_rate` を見る」か、
> 「**1 バッチ過学習**で配線を確かめる」で観察できます。フル設定（1500 ep × 30〜40 epoch）を毎回回す必要は
> ありません。CPU で数十秒〜数分に収まる範囲で回し、**絶対値でなく傾向**を読んでください。

## 共通の評価指標

すべての課題は、基本的に**閉ループ評価の指標**で判断します（[M4](../../lessons/m4_tiny_vla_mse.md) 由来）。

- **`success_rate`**: 対象ブロックを対象ゴールに運べたエピソードの割合（**主指標**）。
- **`mean_final_distance`**: 終了時の「対象ブロック ↔ 対象ゴール」距離（小さいほど良い）。
- **`mean_steps`**: 平均ステップ数（参考）。

評価は `evaluate_policy`（[`../../src/vla_learn/evaluation/rollout.py`](../../src/vla_learn/evaluation/rollout.py)）で取れます。
`success_rate` は乱数・環境でぶれるので、**同じ評価 seed・同じ `n_episodes`** で比べ、差が小さいときは
**複数 seed** で見てください。

> **軽量化のコツ**: まずは `n_episodes`（学習データ）を 300〜500、`epochs` を 8〜12、評価 `n_episodes` を 50 程度に
> 落として「**傾向が出る最小設定**」で回すのがおすすめです。傾向が見えてから、必要なら増やします。

---

## 課題①: chunk_len と flow_steps を変えて成功率を比較する

**ねらい**: 行動チャンク長 (`chunk_len`) と flow の積分ステップ数 (`flow_steps`) が、成功率にどう効くかを体感する。

### 到達目標
- `chunk_len` を変えると「滑らかさ／追従性」のトレードオフがあることを、`success_rate` の変化として観察できる。
- flow 版で `flow_steps` を変えると、生成の質と計算コストのトレードオフがあることを言葉で説明できる。

### やること
1. **`chunk_len` 比較（MSE 版）**: `chunk_len ∈ {4, 8, 16}` で学習し、評価する。`train_mse.py` は
   `--chunk-len` を受け付けます。

   ```bash
   python scripts/train_mse.py --n-episodes 400 --epochs 12 --chunk-len 4  --out-dir checkpoints/c4
   python scripts/train_mse.py --n-episodes 400 --epochs 12 --chunk-len 8  --out-dir checkpoints/c8
   python scripts/train_mse.py --n-episodes 400 --epochs 12 --chunk-len 16 --out-dir checkpoints/c16

   for d in c4 c8 c16; do
     echo "== $d =="; python scripts/eval_policy.py --ckpt checkpoints/$d/policy.pt --n-episodes 50
   done
   ```

2. **`flow_steps` 比較（flow 版）**: flow 版を 1 つ学習し、**評価時の** `--flow-steps` だけを変えて比べる
   （学習をやり直す必要はありません。`flow_steps` は**推論時**の Euler 積分ステップ数です）。

   ```bash
   python scripts/train_flow.py --config configs/m5_flow.json   # 1 回学習（または軽量設定で）
   for s in 2 5 10 20; do
     echo "== flow_steps=$s =="
     python scripts/eval_policy.py --ckpt checkpoints/flow/policy.pt --n-episodes 50 --flow-steps $s
   done
   ```

### 評価方法
- 表を作る: `chunk_len`（または `flow_steps`）→ `success_rate` / `mean_final_distance`。
- **問いに答える**: chunk_len を大きくすると何が起きたか？ `flow_steps` を 2 まで減らすと成功率はどうなり、
  なぜそうなると考えられるか？（ヒント: [M5](../../lessons/m5_flow_matching.md) の「積分ステップが粗いと生成が雑になる」）

> 注意: `exec_horizon`（チャンクのうち実際に実行するステップ数）は `eval_policy.py --exec-horizon` で変えられます。
> `chunk_len` と `exec_horizon` の関係（`exec_horizon ≤ chunk_len`）も意識すると、観察が深まります。

---

## 課題②: ablation — `image_pool='avg'` / `condition_vision=False`

**ねらい**: [M4](../../lessons/m4_tiny_vla_mse.md) の**3 つの設計教訓**のうち 2 つ（空間情報の保持・言語による視覚の変調）を、
**部品を抜いて壊して**確かめる。これが本課題群の山場です。

### 到達目標
- `image_pool='avg'`（空間情報を捨てる）にすると「場所へ向かう」能力が落ち、`success_rate` が下がることを観察できる。
- `condition_vision=False`（FiLM を切って言語で視覚を変調しない）にすると「どの色を運ぶか」の選択が壊れ、
  `success_rate` が下がることを観察できる。
- なぜ落ちるのかを、自分の言葉で説明できる。

### 重要: これは「フラグ」では切り替えられません（小実装課題）
`image_pool` と `condition_vision` は **`TinyVLA` / `VLABackbone` のコンストラクタ引数**ですが、
学習スクリプトの `run_training`（[`../../src/vla_learn/training/trainer.py`](../../src/vla_learn/training/trainer.py)）は
モデルへ `vocab_size` と `chunk_len` **しか渡しません**。したがって ablation は、**自分で短い学習ループを書く**のが
正攻法です（`TinyVLA` は `**backbone_kwargs` を受けるので、`image_pool=...` / `condition_vision=...` を**直接**渡せます）。

雛形は [`ablation.py`](ablation.py) に置きました。`____` を 3 か所だけ埋めて、3 条件を比較してください。

```bash
python exercises/m6/ablation.py
```

### 評価方法
- 3 条件（`flatten+FiLM`（既定）/ `avg`+FiLM / `flatten`+`condition_vision=False`）の `success_rate` を並べる。
- **期待される傾向**: 既定がいちばん高く、`avg` と `condition_vision=False` はそれぞれ落ちる。
  どちらが・なぜ落ちるかを説明する（[M4](../../lessons/m4_tiny_vla_mse.md) の教訓 1・3 に対応）。

> 観察のコツ: 1 回だと差が乱数に埋もれることがあります。**seed を 2〜3 個**変えて平均を見ると、傾向がはっきりします。

---

## 課題③: 新しい命令の言い換え・色を足して般化を見る

**ねらい**: 学習に出なかった指示・設定に対して、方策がどれだけ般化するかを観察する。

### 到達目標
- 「学習データに無い状況」での `success_rate` は、学習時より下がりうることを体感する。
- VLA の般化が「データの多様性」に強く依存することを、実験で確認する。

### やること（難易度の低い順に 1 つ以上）
1. **評価 seed をずらすだけ**（最も簡単な般化チェック）: 学習はそのまま、`eval_policy.py --seed` を
   学習と別の値にして、**未知の初期配置**での成功率を見る（既定の評価 seed は 10000）。
2. **`n_objects` / `n_goals` を変える**: 評価環境の物体数・ゴール数を学習時と変えて、難易度の般化を見る。
   これは `evaluate_policy(..., n_objects=..., n_goals=...)` で指定できます（`eval_policy.py` には専用フラグが
   無いので、短い評価スクリプトを書くか、解答の例を参照）。
3. **（発展）新しい色・命令テンプレを足す**: 色・指示文は**環境側で定義**されています
   （[`../../src/vla_learn/constants.py`](../../src/vla_learn/constants.py) の `COLOR_NAMES` / `COLOR_JA` / `COLOR_RGB`、
   指示テンプレは [`../../src/vla_learn/envs/tabletop2d.py`](../../src/vla_learn/envs/tabletop2d.py)）。
   ここに新色や新テンプレを足すと、`all_instruction_strings()` も自動で増え、トークナイザ語彙も広がります。
   **足したうえで学習し直し**、新しい指示に従えるかを見ます（語彙が変わるので再学習が必要）。

### 評価方法
- 「学習時の条件」と「般化条件」で `success_rate` を比較する表を作る。
- **問いに答える**: 般化条件で成功率はどう変わったか？ 下がった場合、データの何を増やせば改善しそうか？

---

## 課題④: 自作データを LeRobot 形式へ export してみる

**ねらい**: 本文パート A の実践。**自作データを業界標準フォーマットに載せる**経験をする。

### 到達目標
- `map_episode_to_frames`（lerobot 不要）で、自作 episode が **`observation.image[H,W,3] uint8` /
  `observation.state[3] / action[3] / task(str)`** に変換できることを、**自分の目で**確認できる。
- LeRobot の**書き込み API はバージョン依存**である、という現実を理解する。

### やること
1. **変換ロジックを単体で確認（lerobot 不要・必ずできる）**: 本文 A-2.1 のスニペットを実行し、
   `observation.image` が `(64, 64, 3) uint8`（値域 0〜255）、`task` が指示文字列であることを確認する。
2. **export スクリプトを実行**: 

   ```bash
   python scripts/make_dataset.py --n-episodes 200 --out data/tabletop2d
   python scripts/export_lerobot.py --in data/tabletop2d --repo-id your-name/tabletop2d
   ```

   lerobot が無ければ「変換 OK・書き込みスキップ」の案内が出ます（**それで正解**）。
3. **（任意）lerobot を入れて書き込みを試す**: `pip install lerobot` して再実行。**API がバージョンで違い**、
   warn が出ることもあります。**落ちなければ学びとしては十分**です（書き込みの正確な使い方は、入れた
   バージョンのドキュメントを参照）。

### 評価方法
- 変換後フレームの **shape・dtype・値域**を 1 つ書き出して、本文 A-1.1 の「保存で使う形（HWC uint8）」と
  「学習で使う形（CHW float）」の違いを説明できれば達成。

> ねらいは「書き込みを成功させること」ではなく、「**変換ロジックを理解し、規格の特徴量に正しく対応づけられる**」
> ことです。`map_episode_to_frames` が読めて書けるなら合格です。

---

## 課題⑤（発展）: 離散トークン化ヘッドを自作して OpenVLA 流と比較する

**ねらい**: 本文 C-1（OpenVLA）の**離散トークン自己回帰**の発想を、最小限だけ自作して体験する。
本教材の `TinyVLA`（連続回帰）/ `FlowVLA`（連続生成）に対し、**第 3 の行動表現＝離散**を手で作ります。

### 到達目標
- 連続値の行動を **K 個のビンに離散化** → **クラス分類**として解く、という発想を実装できる。
- 「回帰（MSE）」「生成（flow）」「離散分類」の 3 つが、**同じ条件ベクトル `h` の上に乗る別ヘッド**だと理解する。
- **1 バッチ過学習**で「離散ヘッドの学習配線が正しい」ことを確認できる（重い学習はしない）。

### やること（小実装）
1. **行動の離散化**: 正規化済み行動 `[-c, c]` を `K`（例 21）ビンに量子化する関数 `quantize` と、
   逆変換 `dequantize` を書く（各次元独立、ビン中心へ戻す）。
2. **離散ヘッド付き VLA**: `VLABackbone`（[`../../src/vla_learn/models/tiny_vla.py`](../../src/vla_learn/models/tiny_vla.py)）を
   そのまま使い、ヘッドを `nn.Linear(hidden, chunk_len * action_dim * K)` にする。出力を
   `[B, chunk_len, action_dim, K]` に reshape すれば、**各 (ステップ, 次元) ごとに K クラス分類**になります。
   （OpenVLA のように 1 トークンずつ自己回帰させるのは大変なので、**ここでは全ステップ同時に分類**する簡易版で十分です。）
3. **損失**: `CrossEntropyLoss` を `[B*chunk_len*action_dim, K]` 対 `[B*chunk_len*action_dim]` で計算
   （`pad_mask=0` の位置は除外）。
4. **1 バッチ過学習**: [`test_overfit_tiny_batch.py`](../../tests/test_overfit_tiny_batch.py) と同じ要領で、
   小さな 1 バッチに対し loss がしっかり下がることを確認する。

雛形は [`discrete_head.py`](discrete_head.py)（穴埋め）に用意しました。

### 評価方法
- **1 バッチ過学習**: cross-entropy が初期の 2 割以下まで下がれば、離散ヘッドの配線は正しい（**主評価**）。
- **（任意）比較**: 余力があれば、同じ軽量設定で「連続回帰版」と「離散版」の `success_rate` を比べ、
  本教材のおもちゃ問題で**どちらが扱いやすいか**を考察する（`dequantize` → 環境へ、の逆変換に注意）。
- **問いに答える**: ビン数 `K` を小さく（例 5）すると何が起きるか？ 行動の分解能と分類のしやすさの
  トレードオフを説明する。

> これは「OpenVLA を再現する」課題ではありません。**「行動を単語に量子化して分類で解く」という発想を、
> 自分の手で 1 回触る**のが目的です。自己回帰・大規模 VLM・実機データは原典（arXiv:2406.09246）に譲ります。

---

## 仕上げ: あなたの「VLA 地図」を一言で

最後に、ノートに次を**自分の言葉で**書いてみてください（提出物ではなく、理解の確認です）。

1. 自作 `TinyVLA` / `FlowVLA` の各部品は、SmolVLA の何に対応するか（本文 B-2 の対応表を見ずに）。
2. OpenVLA / π0 / GR00T N1 / MolmoAct を「行動表現（離散 or 連続）」で分類すると、それぞれどこに入るか。
3. 自分の VLA に**次に足すなら**何か（事前学習 VLM？ 複数カメラ？ 離散ヘッド？ 推論層？）と、その理由。

これがスラスラ書ければ、**「VLA を読める」から「VLA をいじれる」へ**到達しています。お疲れさまでした。

---

## 関連リンク

- 本文 → [`../../lessons/m6_lerobot_and_models.md`](../../lessons/m6_lerobot_and_models.md)
- 取り組みの指針 → [`../../solutions/m6/README.md`](../../solutions/m6/README.md)
- 演習の歩き方 → [`../README.md`](../README.md)
- 実装本体 → [`../../src/vla_learn/`](../../src/vla_learn/)
