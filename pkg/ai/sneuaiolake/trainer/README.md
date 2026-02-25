# sneuaiolake trainer

このディレクトリは `sneuaiolake` の学習環境です。  
現在は **PPO + GAE(λ) (Actor-Critic) で学習し、提出用には Value head のみをONNXで出力** します。

## セットアップ (uv)

```bash
uv sync
```

## 単体学習

```bash
uv run python train.py \
  --save models_ac/v0 \
  --result-dir play_results/models_ac \
  --prefix bootstrap_
```

出力:

- `models_ac/v0_ac.keras` : Actor-Critic継続学習用
- `models_ac/v0_ac.onnx` : Actor-Critic model (Rust学習対局用)
- `models_ac/v0.keras` : Value model (Keras)
- `models_ac/v0.onnx` : Value model (Rust推論用)

## バッチ学習 (v0から)

```bash
uv run python batch_train.py --start-version 0 --end-version 5
```

`batch_train.py` は以下を実行します。

1. v0 が無ければ初期重みモデル (`v0_init`) を作成し、`v0_init` 同士の自己対局データを生成
2. v0 を学習
3. v1以降を自己対戦データで順次学習

デフォルト対局本数は (`bootstrap-games=120`, `games-vs-prev=100`, `games-self=100`, `games-vs-baseline=100`) です。  
v1以降の過去世代対戦は、`v{n-1}` より古い世代から毎回ランダムに最大5世代を選び、各100試合実行します。加えて `random` と100試合実行します。
学習時は `v{n-1}_ac.onnx` を名前に含むプレイヤーの手だけを使うため、opponent 側の手は policy 学習に入りません。  
デフォルトの最適化パラメータは `entropy-coef=0.001`, `ppo-clip-eps=0.2`, `value-clip-eps=0.2`, `gae-lambda=0.95` です。

## 対戦データ生成 (`play` バイナリ)

自己対戦:

```bash
cd game
cargo run --release --bin play -- \
  --p0-model ../models_ac/v0_ac.onnx \
  --p1-model ../models_ac/v0_ac.onnx \
  --games 200 \
  --result-dir ../play_results/models_ac \
  --prefix v0_self
```

## ELOリーグ評価

既存モデルの総当たりリーグを実行し、ELOを出します（デフォルトは各カード5試合、先後入れ替えあり）。
デフォルトで `random` baseline も参加します。

```bash
uv run python elo_league.py \
  --model-dir models_ac \
  --result-dir elo_viewer/data \
  --games-per-match 5
```

例: 最新10世代 + random を評価

```bash
uv run python elo_league.py \
  --latest 10 \
  --games-per-match 5
```

`elo_league.py` は `elo_viewer/data` に以下を出力します。

- `manifest.json` (実行履歴)
- `{run_id}_summary.json` (ELO集計)
- `{run_id}_*_*.json` (個別対局)

## ELOビジュアライザ (http.server)

`python -m http.server` 前提で、クリックだけでリーグ結果と個別試合を見られます。

```bash
python -m http.server
```

ブラウザで `http://localhost:8000/elo_viewer/` を開いてください。
