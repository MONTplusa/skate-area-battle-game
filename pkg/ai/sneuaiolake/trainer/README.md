# ニューラルネットワーク評価関数

このディレクトリには、スケートエリアバトルゲームのためのニューラルネットワークによる評価関数の実装が含まれています。

## モデルの概要

このモデルは、盤面情報を入力として受け取り、その状態の評価値（スカラー値）を出力する回帰モデルです。評価値は、盤面の有利不利を表す数値で、高いほど有利な状態を示します。

### モデル構造

- 入力：盤面情報（20x20x5の形状を想定）
- 特徴抽出：畳み込み層 + バッチ正規化
- 特徴集約：グローバルプーリング
- 特徴変換：全結合層 + ドロップアウト
- 出力：評価値（スカラー値）

### 学習設定

- 損失関数：平均二乗誤差（MSE）
- メトリクス：平均絶対誤差（MAE）
- オプティマイザ：Adam（学習率0.001）

## 依存関係

必要なPythonパッケージをインストールするには、以下のコマンドを実行してください：

```bash
pip install -r requirements.txt
```

## トレーニングスクリプト

`train.py`スクリプトは、KerasとTensorFlowを使用して回帰モデルをトレーニングします。

### 使用方法

```bash
python train.py [--base BASE_MODEL_PATH] [--save SAVE_MODEL_PATH] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
```

#### オプション

- `--base`: ベースとなるモデルのパス（ONNXフォーマット）。指定しない場合は新しいモデルから開始します。
- `--save`: 保存先モデルのパス（ONNXフォーマット）。デフォルトは`model.onnx`です。
- `--epochs`: トレーニングのエポック数。デフォルトは10です。
- `--batch-size`: バッチサイズ。デフォルトは32です。

### 例

新しいモデルからトレーニングを開始し、結果を`my_model.onnx`として保存：

```bash
python train.py --save my_model.onnx
```

既存のモデル`base_model.onnx`からトレーニングを継続し、結果を`improved_model.onnx`として保存：

```bash
python train.py --base base_model.onnx --save improved_model.onnx --epochs 20
```

## 注意事項

現在の実装では、トレーニングにダミーデータを使用しています。実際のゲームデータを使用するには、`prepare_dummy_data`関数を適切に修正する必要があります。
