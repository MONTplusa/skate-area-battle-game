import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx
import tqdm
from tensorflow import keras
from tensorflow.keras import layers

# 定数
BOARD_SIZE = 20  # 盤面のサイズ
NUM_CHANNELS = 6  # 入力チャンネル数

def residual_block(x, filters):
    """残差ブロック：階層的特徴学習"""
    fx = layers.Conv2D(filters, 3, padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Activation('relu')(fx)
    fx = layers.Conv2D(filters, 3, padding='same')(fx)
    fx = layers.BatchNormalization()(fx)

    # 残差接続
    out = layers.Add()([x, fx])
    out = layers.Activation('relu')(out)
    return out

def spatial_attention_block(x):
    """空間注意機構：重要な領域に注目"""
    # チャンネル注意
    channel_attention = layers.Conv2D(x.shape[-1] // 8, 1, activation='relu')(x)
    channel_attention = layers.Conv2D(x.shape[-1], 1, activation='sigmoid')(channel_attention)
    x = layers.Multiply()([x, channel_attention])

    # 空間注意
    spatial_avg = tf.reduce_mean(x, axis=-1, keepdims=True)
    spatial_max = tf.reduce_max(x, axis=-1, keepdims=True)
    spatial_concat = layers.Concatenate()([spatial_avg, spatial_max])
    spatial_attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(spatial_concat)
    x = layers.Multiply()([x, spatial_attention])

    return x

def create_advanced_model():
    """
    深層学習による自動特徴抽出を重視したモデル

    特徴：
    - 残差接続による深い特徴学習
    - マルチスケール畳み込み
    - 空間・チャンネル注意機構
    - グローバル・ローカル特徴の統合
    """
    inputs = keras.Input(shape=(BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS))

    # 初期特徴抽出（局所パターン）
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 残差ブロックによる階層的特徴学習
    for i in range(6):  # 複数の残差ブロック
        x = residual_block(x, filters=64)

    # マルチスケール畳み込み（異なる範囲の影響力）
    multi_scale = []
    for kernel_size in [3, 5, 7]:
        branch = layers.Conv2D(32, kernel_size, padding='same')(x)
        branch = layers.BatchNormalization()(branch)
        branch = layers.Activation('relu')(branch)
        multi_scale.append(branch)

    x = layers.Concatenate()(multi_scale)

    # 空間注意機構（重要な位置に注目）
    x = spatial_attention_block(x)

    # グローバル特徴抽出
    global_avg = layers.GlobalAveragePooling2D()(x)
    global_max = layers.GlobalMaxPooling2D()(x)
    global_features = layers.Concatenate()([global_avg, global_max])

    # 局所情報も保持するための追加パス
    local_features = layers.Conv2D(64, 1)(x)
    local_features = layers.GlobalAveragePooling2D()(local_features)

    # 特徴統合
    combined = layers.Concatenate()([global_features, local_features])

    # 最終的な評価値予測
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='tanh')(x)

    model = keras.Model(inputs=inputs, outputs=output)
    return model

def save_models(model, save_path):
    """
    Kerasモデルを保存し、さらにONNX形式でも保存
    """
    # 拡張子を取り除いたベース名を取得
    base_path = os.path.splitext(save_path)[0]

    # Kerasモデルを保存
    keras_path = f"{base_path}.keras"
    model.save(keras_path)
    print(f"モデルをKeras形式で保存しました: {keras_path}")

    # ONNX形式でも保存
    onnx_path = f"{base_path}.onnx"

    # 入力と出力の名前を指定
    input_signature = [tf.TensorSpec((None, BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), tf.float32, name="input")]

    # KerasモデルをONNX形式に変換
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=11)

    # ONNXモデルを保存
    import onnx
    onnx.save(onnx_model, onnx_path)
    print(f"モデルをONNX形式で保存しました: {onnx_path}")

def load_battle_data(data_dir, prefix=None):
    """
    バトルデータをJSONファイルから読み込む

    Args:
        data_dir: JSONファイルが格納されているディレクトリパス
        prefix: ファイル名のプレフィックス（指定された場合はそのプレフィックスを持つファイルのみを読み込む）

    Returns:
        x_data: 入力データ (盤面情報)
        y_data: 教師データ (評価値)
    """
    print(f"{data_dir} からバトルデータを読み込みます...")

    # JSONファイルのリストを取得
    if prefix:
        json_files = glob.glob(os.path.join(data_dir, f"{prefix}*.json"))
        print(f"プレフィックス '{prefix}' を持つファイルのみを読み込みます")
    else:
        json_files = glob.glob(os.path.join(data_dir, "*.json"))

    if not json_files:
        if prefix:
            raise ValueError(f"{data_dir} にプレフィックス '{prefix}' を持つJSONファイルが見つかりません")
        else:
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

            # 最終スコアを確認
            scores = [0, 0]
            for y in range(BOARD_SIZE):
                for x in range(BOARD_SIZE):
                    color = final_state["colors"][y][x]
                    if color == -1:
                        continue
                    scores[color] += final_state["board"][y][x]

            # 各状態を入力データに変換
            moves = battle_result["moves"]
            num_moves = [
                sum(1 for move in moves if move["player"] == 0),
                sum(1 for move in moves if move["player"] == 1)
            ]
            counts = [0, 0]
            for move in moves:
                player = move["player"]
                state = move["state"]

                # 入力データの作成
                input_data = create_input_data(player, state)
                x_data.append(input_data)

                # 教師データの作成（勝者なら1.0、敗者なら-1.0）
                if player == 0 and scores[0] > scores[1]:
                    final_value = 1.0
                elif player == 0 and scores[0] < scores[1]:
                    final_value = -1.0
                elif player == 1 and scores[0] > scores[1]:
                    final_value = -1.0
                elif player == 1 and scores[0] < scores[1]:
                    final_value = 1.0
                else:
                    final_value = 0.0

                # 非線形的な進歩を考慮した教師信号
                progress = counts[player] / num_moves[player]
                # 終盤に向けて重みを非線形に増加
                value = final_value * (1 - np.exp(-3 * progress))
                y_data.append([value])

                counts[player] += 1

        except Exception as e:
            print(f"ファイル {json_file} の処理中にエラーが発生しました: {e}")
            continue

    if not x_data:
        raise ValueError("有効なデータが見つかりませんでした")

    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32)

def create_input_data(player, state):
    """
    GameState型のデータから入力データを作成

    Args:
        player: プレイヤー（0または1）
        state: GameState型のデータ

    Returns:
        input_data: (BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS) の形状の入力データ
    """
    # 入力データの初期化
    input_data = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), dtype=np.float32)

    # 自分が0となるように必要なら反転する
    player0 = player
    player1 = 1 - player

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

def create_dataset_with_augmentation(x_data, y_data, batch_size, shuffle=True):
    """データ拡張を含むデータセット作成"""
    @tf.function
    def augment(x, y):
        # 回転・反転による拡張
        if tf.random.uniform([]) > 0.5:
            x = tf.image.flip_left_right(x)

        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        x = tf.image.rot90(x, k=k)

        return x, y

    print(f"データセット作成開始: x_data shape: {x_data.shape}, dtype: {x_data.dtype}")
    print(f"y_data shape: {y_data.shape}, dtype: {y_data.dtype}")

    # データセットの作成（CPUで処理）
    with tf.device('/CPU:0'):
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    # データ拡張適用
    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def create_callbacks(model_path):
    """コールバック設定"""
    # ベストモデルのパスを設定
    base_path = os.path.splitext(model_path)[0]
    best_model_path = f"{base_path}_best.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_loss',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks

def main():
    # GPUの設定
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPUメモリの動的割り当てを有効化しました")
        except RuntimeError as e:
            print(f"GPUの設定中にエラーが発生しました: {e}")

    # GPUの設定状態を確認
    print("TensorFlow GPUの設定状態:")
    print(f"GPUデバイス: {gpus}")
    print(f"メモリ増長を許可: {tf.config.experimental.get_memory_growth(gpus[0]) if gpus else 'No GPU'}")

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='改善されたニューラルネットワーク評価関数のトレーニング')
    parser.add_argument('--base', type=str, help='ベースとなるモデルのパス（Kerasモデル、.keras形式のみ対応）')
    parser.add_argument('--save', type=str, default='model', help='保存先モデルのパス（拡張子は自動的に.kerasと.onnxの両方で保存されます）')
    parser.add_argument('--epochs', type=int, default=20, help='トレーニングのエポック数')
    parser.add_argument('--batch-size', type=int, default=128, help='バッチサイズ')
    parser.add_argument('--result-dir', type=str, required=True, help='バトルデータが格納されているディレクトリ')
    parser.add_argument('--prefix', type=str, default=None, help='読み込むJSONファイルのプレフィックス（例: random_random）')
    args = parser.parse_args()

    # 保存先のパスをチェック
    base_path = os.path.splitext(args.save)[0]
    keras_path = f"{base_path}.keras"
    onnx_path = f"{base_path}.onnx"

    if Path(keras_path).exists() or Path(onnx_path).exists():
        print(f"エラー: 保存先のモデル {keras_path} または {onnx_path} はすでに存在します")
        return

    # モデルの読み込みまたは作成
    if args.base:
        # 拡張子が.kerasでない場合は追加
        base_path = args.base
        if not base_path.endswith('.keras'):
            base_path = f"{base_path}.keras"

        if not os.path.exists(base_path):
            print(f"エラー: 指定されたモデルファイル {base_path} が見つかりません")
            return

        try:
            # Kerasモデルとして読み込む
            model = keras.models.load_model(base_path)
            print(f"Kerasモデルを読み込みました: {base_path}")
        except Exception as e:
            print(f"エラー: モデルの読み込みに失敗しました: {e}")
            print("モデルはKeras形式(.keras)である必要があります。")
            return
    else:
        # 改善されたモデルを作成
        model = create_advanced_model()
        print("新しい改善されたモデルを作成しました。")

    # モデルの概要を表示
    model.summary()

    try:
        # バトルデータの読み込み
        x_train, y_train = load_battle_data(args.data_dir, args.prefix)
        print(f"読み込んだデータ: {len(x_train)} サンプル")

        # データをトレーニングセットと検証セットに分割
        split_idx = int(len(x_train) * 0.8)
        x_val = x_train[split_idx:]
        y_val = y_train[split_idx:]
        x_train = x_train[:split_idx]
        y_train = y_train[:split_idx]

        # 学習率スケジューリング
        initial_learning_rate = 1e-3
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True
        )

        # オプティマイザ設定（AdamW with weight decay）
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.01
        )

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        # データ拡張付きデータセットの作成
        train_dataset = create_dataset_with_augmentation(x_train, y_train, args.batch_size)
        val_dataset = create_dataset_with_augmentation(x_val, y_val, args.batch_size, shuffle=False)

        # コールバック設定
        callbacks = create_callbacks(args.save)

        # モデルのトレーニング
        print("モデルのトレーニングを開始します...")
        history = model.fit(
            train_dataset,
            epochs=args.epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
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
        print("トレーニングを中止します。")
        print("ヒント: --prefix オプションを使用して特定のプレフィックスを持つファイルのみを読み込むことができます。")
        print("例: python train.py --prefix random_random")
        return

    # モデルをKeras形式とONNX形式の両方で保存
    save_models(model, args.save)

if __name__ == '__main__':
    main()
