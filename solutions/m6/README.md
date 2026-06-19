# 解答（指針）M6: 卒業課題の取り組み方・観点・期待される傾向

> 対応する課題: [`../../exercises/m6/README.md`](../../exercises/m6/README.md)
> 本文: [`../../lessons/m6_lerobot_and_models.md`](../../lessons/m6_lerobot_and_models.md)

卒業課題は**自由実験が中心**なので、ここは「正解コード集」ではなく**取り組み方の指針・見るべき観点・
期待される結果の傾向**をまとめます。穴埋め雛形（`ablation.py` / `discrete_head.py`）については、配線が
正しくなる**埋め方の要点**だけ示します（写経用ではなく、自分の答えと突き合わせる用です）。

> **数値は環境・乱数で大きくぶれます。** 以下の「期待される傾向」は**向き（上がる/下がる/効く/壊れる）**の話で、
> 絶対値の予言ではありません。差が小さいときは seed を複数振って平均で判断してください。

---

## 課題①: chunk_len / flow_steps の比較

### 見るべき観点
- **`chunk_len`**: 「1 回で何ステップ先まで設計するか」。大きいほど軌道は滑らかになりやすい一方、
  予測が**遠い未来まで**当てる必要が出て難しくもなります。`exec_horizon`（実際に実行する数）との関係も重要で、
  `exec_horizon` が `chunk_len` に対して大きすぎると「古い計画を実行しすぎて」観測の更新が遅れます。
- **`flow_steps`**: [M5](../../lessons/m5_flow_matching.md) の **Euler 積分のステップ数**。**学習ではなく推論**の
  パラメータです（だから学習し直さず評価だけ変えれば比較できる）。粗いほど速いが、生成が雑になります。

### 期待される傾向
- `chunk_len` を極端に小さく（例 4）すると、本教材のおもちゃ問題では大きく崩れないことも多いですが、
  滑らかさや一貫性で不利になりがちです。極端に大きく（例 16）すると、当てる範囲が広がり学習が難しくなる方向。
  **「中庸（8 付近）が無難」**という、デフォルト値の意味が体感できれば十分です。
- `flow_steps` を **2 のように粗く**すると、`success_rate` は下がる傾向。**10 と 20 ではあまり変わらない**ことも多く、
  「ある程度から先は増やしても頭打ち（コストだけ増える）」という飽和を観察できます。

### つまずきやすい点
- 別 `--out-dir` を指定しないと**チェックポイントを上書き**してしまいます。条件ごとにディレクトリを分けること。
- 評価 seed・`n_episodes` を条件間で揃えること（揃えないと比較になりません）。

---

## 課題②: ablation（`image_pool='avg'` / `condition_vision=False`）

ここが山場です。雛形 [`../../exercises/m6/ablation.py`](../../exercises/m6/ablation.py) の**3 つの穴**は、要点だけ言うと:

- **穴 1（モデル構築）**: `TinyVLA(vocab_size=tok.vocab_size, chunk_len=CHUNK_LEN, **condition)`。
  `TinyVLA` は `**backbone_kwargs` を受けるので、`image_pool` / `condition_vision` を**そのまま渡せます**
  （[`../../src/vla_learn/models/tiny_vla.py`](../../src/vla_learn/models/tiny_vla.py)）。
- **穴 2（損失）**: `masked_mse(pred, batch["action"], batch["pad_mask"])`。
  パディングを除外して有効ステップだけで MSE（[M3](../../lessons/m3_data_actions.md)）。
- **穴 3（評価）**: `PolicyWrapper(model, tok, action_norm, state_norm, "mse", device)` で包み、
  `evaluate_policy(policy, n_episodes=EVAL_EPISODES)` で `success_rate` を取る。

### なぜ「フラグで切り替えられない」のか（理解の核心）
[`../../src/vla_learn/training/trainer.py`](../../src/vla_learn/training/trainer.py) の `run_training` は、
モデルへ `vocab_size` と `chunk_len` **しか渡しません**（`model_kwargs` を見てください）。`TrainConfig` にも
`image_pool` / `condition_vision` は**ありません**。だから ablation は**自分で短い学習ループを書く**のが正攻法で、
雛形はそれを最小限にしたものです。「学習スクリプトが渡す引数」と「モデルが受ける引数」は別物だ、という
気づきも、この課題の収穫です。

### 期待される傾向（[M4](../../lessons/m4_tiny_vla_mse.md) の教訓と対応）
- **`image_pool='avg'`**（教訓 1・空間情報を捨てる）: 「**場所へ向かう**」が苦手になり、`success_rate` が下がる方向。
  avg は画像を 1 点に潰すので、「対象やゴールが画面の**どこ**にあるか」が消えます。
- **`condition_vision=False`**（教訓 3・FiLM を切る）: 「**どの色を運ぶか**」の選択が壊れ、`success_rate` が
  下がる方向。言語が視覚を変調しないので、複数候補から対象を選ぶ steering（誘導）がほぼ効かなくなります
  （本文・[M4](../../lessons/m4_tiny_vla_mse.md) の「対象選択がほぼ 0」という実測の追体験）。
- **baseline（flatten + FiLM）** がいちばん高い、が期待される並びです。

### 観点
- どちらの ablation も**学習 loss はそれなりに下がる**のに、**閉ループ成功率が落ちる**ことがあります。
  「loss は下がるのに動かすと失敗」という [M2](../../lessons/m2_imitation.md) のテーマがここでも効きます。
  **損失の良さと、タスク成功は別**だと再確認してください。
- 1 回だと乱数に埋もれることがあるので、`SEEDS = [0, 1, 2]` にして平均で見ると傾向が安定します。

---

## 課題③: 般化（言い換え・色追加）

### 見るべき観点
- **般化の段階**: (1) 評価 seed をずらすだけ（未知の初期配置）→ (2) `n_objects`/`n_goals` を変える（難易度）→
  (3) 新色・新テンプレを足す（語彙・意味の拡張、**要再学習**）。下に行くほど難しく、効果も大きい。
- **般化は「データの多様性」次第**: 学習データに似た状況が無いほど落ちます。

### 期待される傾向
- **評価 seed をずらすだけ**なら、初期配置は学習分布と同種なので、成功率は大きくは変わらないはず
  （ここが大崩れするなら過学習を疑う）。
- **`n_objects` を増やす**と、紛らわしい候補が増えて成功率は下がる方向。
- **新色を足して再学習**すると、データ量が同じなら 1 色あたりの例が減るので、**全体の成功率が下がる**ことも。
  「新しい概念を足すなら、それに見合うデータが要る」という VLA の現実を体感できれば狙い通りです。

### 実装の注意
- 色・指示テンプレは[環境側](../../src/vla_learn/envs/tabletop2d.py)と
  [`constants.py`](../../src/vla_learn/constants.py)で定義され、`all_instruction_strings()` が
  全パターンを列挙 → トークナイザ語彙が決まります。**新色/新テンプレを足すと語彙が変わるので、
  学習し直しが必要**です（既存チェックポイントは使い回せません）。
- `n_objects`/`n_goals` を評価で変えるには、`evaluate_policy(..., n_objects=..., n_goals=...)` を呼ぶ短い
  スクリプトを書きます（`eval_policy.py` には専用フラグがありません）。例:

  ```python
  import sys; sys.path.insert(0, "src")
  from vla_learn.evaluation.rollout import PolicyWrapper, evaluate_policy
  from vla_learn.training.checkpoint import load_policy
  b = load_policy("checkpoints/mse/policy.pt")
  pol = PolicyWrapper(b["model"], b["tokenizer"], b["action_norm"], b["state_norm"], b["model_type"])
  print(evaluate_policy(pol, n_episodes=50, n_objects=4, n_goals=3))  # 学習より難しい設定
  ```

---

## 課題④: LeRobot export

### 観点
- **ねらいは書き込み成功ではなく、変換の理解**。本文 A-3 の通り、`map_episode_to_frames`（lerobot 不要）が
  読めて、特徴量（`observation.image`/`observation.state`/`action`/`task`）への対応づけを説明できれば達成です。
- **HWC uint8（保存）と CHW float（学習）の違い**を、自分が書き出した shape/dtype/値域で言えること（本文 A-1.1）。

### 期待される結果
- lerobot 未インストール: 「`[map] フレーム変換 OK …`」と出て、書き込みは案内を出してスキップ（**これで正解**）。
- lerobot インストール済み: バージョンが合えば作成成功、合わなければ warn（落ちない）。**どちらでも学びは成立**。

### つまずきやすい点
- `from export_lerobot import map_episode_to_frames` する前に `sys.path` に `scripts` を足すこと（本文 A-2.1）。
- 「書き込みが warn になった＝失敗」ではありません。**API がバージョン依存**だからで、想定どおりの挙動です。

---

## 課題⑤（発展）: 離散トークン化ヘッド

雛形 [`../../exercises/m6/discrete_head.py`](../../exercises/m6/discrete_head.py) の**3 つの穴**の要点:

- **`quantize`**: `a` を `[-A_RANGE, A_RANGE]` に `clamp` → `(a + A_RANGE) / (2*A_RANGE)` で `[0,1]` → `* (K-1)` して
  `round().long()` → `clamp(0, K-1)`。各次元独立に「どのビンか」を整数で返します。
- **ヘッド**: `nn.Linear(hidden, chunk_len * action_dim * n_bins)`。forward で `[B, C, A, K]` に reshape し、
  **各 (ステップ, 次元) ごとに K クラス分類**にします。
- **損失**: `nn.functional.cross_entropy(logits_flat[valid], target_flat[valid])`。`pad_mask=0` を除外。

### 観点（いちばん大事なところ）
- **同じ `h`（`VLABackbone` の出力）の上に、ヘッドだけ差し替える**——これが本課題の核心です。
  `TinyVLA`=回帰ヘッド、`FlowVLA`=flow ヘッド、本課題=分類ヘッド。**条件付けの骨格は共通**で、
  **行動の表現方法だけが違う**。OpenVLA（離散）と π0/SmolVLA（連続）の分岐が、まさにこの「ヘッドの選択」だと
  実感できます（本文 C の比較表）。
- 本教材は OpenVLA の**自己回帰**までは作りません（全ステップ同時分類の簡易版）。それでも「**行動を単語に
  量子化して分類で解く**」という発想は十分に体験できます。

### 期待される傾向
- **1 バッチ過学習（主評価）**: cross-entropy が初期の 2 割以下まで下がれば配線 OK
  （[`test_overfit_tiny_batch.py`](../../tests/test_overfit_tiny_batch.py) と同じ判定基準）。下がらなければ、
  `quantize` の範囲・`reshape` の順序・`valid` マスクの取り違えを疑う。
- **ビン数 `K` を小さく（例 5）**すると、分類は簡単になりますが**行動の分解能が粗く**なり、`dequantize` 後の
  行動がガタつきます。逆に大きすぎると 1 ビンあたりの例が減って学習が難しくなる方向。**分解能 vs 学習しやすさ**の
  トレードオフが見えれば狙い通りです。
- **（任意）連続版との比較**: 本教材のおもちゃ問題（行動が滑らかで連続的）では、**連続回帰/flow の方が素直**な
  ことが多いです。離散化は「言語モデルの語彙に乗せられる」「マルチモーダルな行動分布を表現しやすい」といった
  別の利点があり、**大規模 VLM と相性が良い**（OpenVLA の選択）——という設計上の意味合いを説明できれば上出来です。

---

## 仕上げ（自己評価）の観点

本文・課題を終えたら、次がスラスラ言えるかで理解を測れます。

1. **対応表**: 自作 `ImageEncoder`/`TextEncoder`/`StateEncoder`/`FiLM`/`VLABackbone`/`FlowVLA` が、SmolVLA の
   どの部品に当たるか（本文 B-2）。
2. **行動表現での分類**: OpenVLA/MolmoAct（離散自己回帰）vs SmolVLA/π0/GR00T（連続生成）、GR00T の dual-system、
   MolmoAct の「行動前の空間推論」（本文 C・比較表）。**数値や細部は原典参照**、という姿勢も含めて。
3. **次の一手**: 自分の VLA に足すなら何か（事前学習 VLM / 複数カメラ / 離散ヘッド / 推論層）と理由。

ここまで言えれば、**「VLA を読める」から「VLA をいじれる」**へ到達しています。

---

## 関連リンク

- 課題 → [`../../exercises/m6/README.md`](../../exercises/m6/README.md)
- 本文 → [`../../lessons/m6_lerobot_and_models.md`](../../lessons/m6_lerobot_and_models.md)
- 解答の使い方 → [`../README.md`](../README.md)
