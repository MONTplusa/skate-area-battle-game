import argparse
import glob
import json
import os

import numpy as np
import tensorflow as tf
import tf2onnx
import tqdm
from tensorflow import keras
from tensorflow.keras import layers

# 定数
BOARD_SIZE = 20  # 盤面のサイズ
NUM_CHANNELS = 6  # 入力チャンネル数

def create_model():
    """
    盤面情報から評価値を予測する回帰モデルを作成
    入力: 盤面情報 (BOARD_SIZE x BOARD_SIZE x NUM_CHANNELS)
    出力: 評価値（スカラー値）

    チャンネル構成:
    - チャンネル0: 各セルのボードの数値を0-1に正規化したもの
    - チャンネル1: プレイヤー0の色がついているマスに1がたっているチャンネル
    - チャンネル2: プレイヤー1の色がついているマスに1がたっているチャンネル
    - チャンネル3: 岩があるマスに1がたっているチャンネル
    - チャンネル4: プレイヤー0の現在位置だけに1がたっているチャンネル
    - チャンネル5: プレイヤー1の現在位置だけに1がたっているチャンネル
    """
    # 入力層
    inputs = keras.Input(shape=(BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS))

    # 特徴抽出部分（畳み込みネットワーク）
    # ResNetスタイルのブロックを使用
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 複数の残差ブロック
    for _ in range(4):
        x = residual_block(x, 64)

    # 特徴マップのサイズを縮小
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    for _ in range(2):
        x = residual_block(x, 128)

    # グローバル特徴量の抽出
    x = layers.GlobalAveragePooling2D()(x)

    # 全結合層
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)  # 過学習防止
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)  # 過学習防止

    # 出力層 - 評価値（スカラー値）
    # tanh活性化関数を使用して出力を-1から1の範囲に制限
    output = layers.Dense(1, activation='tanh')(x)

    # モデル作成
    model = keras.Model(inputs=inputs, outputs=output)

    # モデルのコンパイル
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # 平均二乗誤差
        metrics=['mae']  # 平均絶対誤差も計測
    )

    return model

def residual_block(x, filters):
    """
    残差ブロックの実装
    """
    shortcut = x

    # 1つ目の畳み込み層
    y = layers.Conv2D(filters, (3, 3), padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    # 2つ目の畳み込み層
    y = layers.Conv2D(filters, (3, 3), padding='same')(y)
    y = layers.BatchNormalization()(y)

    # ショートカット接続を追加
    y = layers.add([shortcut, y])
    y = layers.Activation('relu')(y)

    return y

def load_onnx_model(model_path):
    """
    ONNXモデルを読み込み、TensorFlowモデルに変換
    """
    try:
        print(f"ONNXモデルを読み込みます: {model_path}")

        # ONNXモデルをTensorFlowモデルに変換
        import onnx
        from onnx_tf.backend import prepare

        # ONNXモデルを読み込む
        onnx_model = onnx.load(model_path)

        # ONNXモデルをTensorFlowモデルに変換
        tf_rep = prepare(onnx_model)

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
    input_signature = [tf.TensorSpec((None, BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), tf.float32, name="input")]

    # KerasモデルをONNX形式に変換
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)

    # ONNXモデルを保存
    import onnx
    onnx.save(onnx_model, save_path)
    print(f"モデルをONNX形式で保存しました: {save_path}")

def load_battle_data(data_dir):
    """
    バトルデータをJSONファイルから読み込む

    Args:
        data_dir: JSONファイルが格納されているディレクトリパス

    Returns:
        x_data: 入力データ (盤面情報)
        y_data: 教師データ (評価値)
    """
    print(f"{data_dir} からバトルデータを読み込みます...")

    # JSONファイルのリストを取得
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        raise ValueError(f"{data_dir} にJSONファイルが見つかりません")

    print(f"{len(json_files)} 件のバトルデータを読み込みます")

    x_data = []
    y_data = []

    # 各JSONファイルを処理
    for json_file in tqdm.tqdm(json_files):
        try:
            with open(json_file, 'r') as f:
                battle_result = json.load(f)

            # 最終状態から勝者を判定
            final_state = battle_result["finalState"]

            # 色の数をカウント
            scores = [0, 0]
            for y in range(BOARD_SIZE):
                for x in range(BOARD_SIZE):
                    color = final_state["colors"][y][x]
                    if color == -1:
                        continue
                    scores[color] += final_state["board"][y][x]

            # 勝者の判定（色が多い方が勝ち）
            if scores[0] > scores[1]:
                value = 1.0
            elif scores[0] < scores[1]:
                value = -1.0
            else:
                value = 0.0

            # 各状態を入力データに変換
            for move in battle_result["moves"]:
                state = move["state"]

                # 入力データの作成
                # move["turn"]に合わせて必要な読み替えをやってくれるので、必ずplayer0からの視点になっている
                input_data = create_input_data(state, move["turn"])
                x_data.append(input_data)

                # 教師データの作成（勝者なら1.0、敗者なら-1.0）
                y_data.append([value])

        except Exception as e:
            print(f"ファイル {json_file} の処理中にエラーが発生しました: {e}")
            continue

    if not x_data:
        raise ValueError("有効なデータが見つかりませんでした")

    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32)

def create_input_data(state, next_player):
    """
    GameState型のデータから入力データを作成

    Args:
        state: GameState型のデータ
        next_player: 次のプレイヤーのインデックス（0または1）

    Returns:
        input_data: (BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS) の形状の入力データ
    """
    # 入力データの初期化
    input_data = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), dtype=np.float32)

    # 次のプレイヤーが1となるように入力を作成する
    # つまり、next_player == 0の場合はプレイヤー0とプレイヤー1を逆に読み替える必要がある
    player0 = 1 - next_player
    player1 = next_player

    # ボードの最大値を取得して正規化
    board_max = 0
    for row in state["board"]:
        row_max = max(row)
        if row_max > board_max:
            board_max = row_max

    # チャンネル0: ボードの数値を0-1に正規化
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board_max > 0:
                input_data[y, x, 0] = state["board"][y][x] / board_max

    # チャンネル1: プレイヤー0の色
    # チャンネル2: プレイヤー1の色
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            color = state["colors"][y][x]
            if color == player0:
                input_data[y, x, 1] = 1.0
            elif color == player1:
                input_data[y, x, 2] = 1.0

    # チャンネル3: 岩の位置
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if state["rocks"][y][x]:
                input_data[y, x, 3] = 1.0

    # チャンネル4: プレイヤー0の位置
    player0_x = state[f"player{player0}"]["x"]
    player0_y = state[f"player{player0}"]["y"]
    input_data[player0_y, player0_x, 4] = 1.0

    # チャンネル5: プレイヤー1の位置
    player1_x = state[f"player{player1}"]["x"]
    player1_y = state[f"player{player1}"]["y"]
    input_data[player1_y, player1_x, 5] = 1.0

    return input_data

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='ニューラルネットワーク評価関数のトレーニング')
    parser.add_argument('--base', type=str, help='ベースとなるモデルのパス（ONNXフォーマット）')
    parser.add_argument('--save', type=str, default='model.onnx', help='保存先モデルのパス（ONNXフォーマット）')
    parser.add_argument('--epochs', type=int, default=10, help='トレーニングのエポック数')
    parser.add_argument('--batch-size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--data-dir', type=str, default='output', help='バトルデータが格納されているディレクトリ')
    args = parser.parse_args()

    # モデルの読み込みまたは作成
    if args.base and os.path.exists(args.base):
        model = load_onnx_model(args.base)
    else:
        model = create_model()
        print("新しいモデルを作成しました。")

    # モデルの概要を表示
    model.summary()

    try:
        # バトルデータの読み込み
        data_dir = os.path.join("trainer", args.data_dir)
        x_train, y_train = load_battle_data(data_dir)

        print(f"読み込んだデータ: {len(x_train)} サンプル")

        # データをトレーニングセットと検証セットに分割
        split_idx = int(len(x_train) * 0.8)
        x_val = x_train[split_idx:]
        y_val = y_train[split_idx:]
        x_train = x_train[:split_idx]
        y_train = y_train[:split_idx]

        # モデルのトレーニング
        print("モデルのトレーニングを開始します...")
        history = model.fit(
            x_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_val, y_val),
            verbose=1
        )

        # トレーニング結果の表示
        print("トレーニング完了")
        print(f"最終損失 (MSE): {history.history['loss'][-1]:.4f}")
        print(f"最終平均絶対誤差 (MAE): {history.history['mae'][-1]:.4f}")
        print(f"検証損失 (MSE): {history.history['val_loss'][-1]:.4f}")
        print(f"検証平均絶対誤差 (MAE): {history.history['val_mae'][-1]:.4f}")

    except Exception as e:
        print(f"データの読み込みまたはトレーニング中にエラーが発生しました: {e}")
        print("ダミーデータを使用してモデルをテストします...")

        # ダミーデータの生成
        num_samples = 1000
        x_dummy = np.random.random((num_samples, BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS)).astype(np.float32)
        y_dummy = np.random.choice([-1.0, 1.0], size=(num_samples, 1)).astype(np.float32)

        # ダミーデータでのトレーニング
        model.fit(
            x_dummy, y_dummy,
            epochs=2,  # ダミーデータなので少ないエポック数
            batch_size=args.batch_size,
            validation_split=0.2,
            verbose=1
        )

    # モデルをONNX形式で保存
    save_onnx_model(model, args.save)

if __name__ == '__main__':
    main()
