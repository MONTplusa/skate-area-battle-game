import argparse
import os

import numpy as np
import tensorflow as tf
import tf2onnx
from tensorflow import keras
from tensorflow.keras import layers

# 定数
BOARD_SIZE = 20  # 盤面のサイズ（仮の値、実際の値に合わせて調整）

def create_model():
    """
    盤面情報から評価値を予測する回帰モデルを作成
    入力: 盤面情報
    出力: 評価値（スカラー値）
    """
    # 入力層 - 盤面情報の形状は後で調整可能
    # 現時点では仮に (BOARD_SIZE, BOARD_SIZE, 5) とする
    # 5チャンネルは例えば [ボード状態, プレイヤー1位置, プレイヤー2位置, 色情報, 岩情報] など
    inputs = keras.Input(shape=(BOARD_SIZE, BOARD_SIZE, 5))

    # 特徴抽出部分（畳み込みネットワーク）
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # グローバル特徴量の抽出
    x = layers.GlobalAveragePooling2D()(x)

    # 全結合層
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)  # 過学習防止
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)  # 過学習防止

    # 出力層 - 評価値（スカラー値）
    # 回帰問題なので活性化関数なし（線形）
    output = layers.Dense(1)(x)

    # モデル作成
    model = keras.Model(inputs=inputs, outputs=output)

    # モデルのコンパイル
    # 回帰問題なので損失関数はMSE（平均二乗誤差）
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']  # 平均絶対誤差も計測
    )

    return model

def load_onnx_model(model_path):
    """
    ONNXモデルを読み込み、TensorFlowモデルに変換
    """
    try:
        print(f"ONNXモデルを読み込みます: {model_path}")

        # ONNXモデルをTensorFlowモデルに変換
        # 注意: 実際のプロジェクトでは、モデルの構造に応じて適切な変換方法を選択する必要がある
        import onnx
        from onnx_tf.backend import prepare

        # ONNXモデルを読み込む
        onnx_model = onnx.load(model_path)

        # ONNXモデルをTensorFlowモデルに変換
        tf_rep = prepare(onnx_model)

        # TensorFlowモデルをKerasモデルに変換
        # 注意: この部分は実際のモデル構造に応じて調整が必要
        # 簡易的な実装として、新しいモデルを作成し、重みをコピーする方法もある

        print(f"ONNXモデルを正常に読み込みました: {model_path}")
        return tf_rep.tensor_dict
    except Exception as e:
        print(f"ONNXモデルの読み込みに失敗しました: {e}")
        print("エラーの詳細: ", str(e))
        print("新しいモデルを作成します。")
        return create_model()

def save_onnx_model(model, save_path):
    """
    KerasモデルをONNX形式で保存
    """
    # 入力と出力の名前を指定
    input_signature = [tf.TensorSpec((None, BOARD_SIZE, BOARD_SIZE, 5), tf.float32, name="input")]

    # KerasモデルをONNX形式に変換
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)

    # ONNXモデルを保存
    import onnx
    onnx.save(onnx_model, save_path)
    print(f"モデルをONNX形式で保存しました: {save_path}")

def prepare_dummy_data(num_samples=1000):
    """
    テスト用のダミーデータを生成
    実際のトレーニングでは、ゲームプレイから生成されたデータを使用する

    Returns:
        x_train: 入力データ (盤面情報)
        y_train: 教師データ (評価値)
    """
    # 入力データ: (サンプル数, ボードサイズ, ボードサイズ, チャンネル数)
    x_train = np.random.random((num_samples, BOARD_SIZE, BOARD_SIZE, 5)).astype(np.float32)

    # 教師データ: (サンプル数, 1) - 評価値（-1.0〜1.0の範囲）
    # 実際のゲームでは、勝敗や有利不利の度合いを表す値になる
    y_train = np.random.uniform(-1.0, 1.0, (num_samples, 1)).astype(np.float32)

    return x_train, y_train

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='ニューラルネットワーク評価関数のトレーニング')
    parser.add_argument('--base', type=str, help='ベースとなるモデルのパス（ONNXフォーマット）')
    parser.add_argument('--save', type=str, default='model.onnx', help='保存先モデルのパス（ONNXフォーマット）')
    parser.add_argument('--epochs', type=int, default=10, help='トレーニングのエポック数')
    parser.add_argument('--batch-size', type=int, default=32, help='バッチサイズ')
    args = parser.parse_args()

    # モデルの読み込みまたは作成
    if args.base and os.path.exists(args.base):
        model = load_onnx_model(args.base)
    else:
        model = create_model()
        print("新しいモデルを作成しました。")

    # モデルの概要を表示
    model.summary()

    # ダミーデータの準備（実際のトレーニングでは実データを使用）
    x_train, y_train = prepare_dummy_data()

    # モデルのトレーニング
    print("モデルのトレーニングを開始します...")
    history = model.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        verbose=1
    )

    # トレーニング結果の表示
    print("トレーニング完了")
    print(f"最終損失 (MSE): {history.history['loss'][-1]:.4f}")
    print(f"最終平均絶対誤差 (MAE): {history.history['mae'][-1]:.4f}")

    # モデルをONNX形式で保存
    save_onnx_model(model, args.save)

if __name__ == '__main__':
    main()
    main()
