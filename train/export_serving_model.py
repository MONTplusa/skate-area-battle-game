import os

import tensorflow as tf


def export_model():
    # Kerasモデルの読み込み
    model = tf.keras.models.load_model("model.keras")

    # SavedModel形式でエクスポート
    export_path = "../pkg/ai/sneuaiolake/saved_model"
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    model.save(export_path, save_format="tf")
    print(f"Model exported to: {export_path}")


if __name__ == "__main__":
    export_model()
