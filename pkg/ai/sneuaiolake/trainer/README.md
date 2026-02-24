# sneuaiolake trainer

このディレクトリは `sneuaiolake` の学習環境です。  
現在は **PPO (Actor-Critic) で学習し、提出用には Value head のみをONNXで出力** します。

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

デフォルトは世代更新を速くするため、対局本数を小さめ (`bootstrap-games=12`, `games-vs-prev=10`, `games-self=10`, `games-vs-baseline=10`) にしています。  
v1以降の過去世代対戦は、`v{n-1}` より古い世代から毎回ランダムに最大5世代を選び、各10試合実行します。加えて `random` と10試合実行します。
学習時は `v{n-1}_ac.onnx` を名前に含むプレイヤーの手だけを使うため、opponent 側の手は policy 学習に入りません。

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
