import subprocess as sp

prev_version = 27
curr_version = 28


def train(version, base_version):
    sp.check_call(
        [
            "python",
            "train.py",
            "--save",
            f"models_01/v{version}",
            *(["--base", f"models_01/v{base_version}"] if base_version > 0 else []),
            "--result-dir",
            "play_results/models_01",
            "--prefix",
            f"v{base_version}",
        ]
    )


def self_play(p0_version, p1_version, games):
    p1_model_flag = (
        ["--p1-model", f"../models_01/v{p1_version}.onnx"] if p1_version > 0 else []
    )
    sp.check_call(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "play",
            "--",
            "--p0-model",
            f"../models_01/v{p0_version}.onnx",
            *p1_model_flag,
            "--games",
            str(games),
            "--result-dir",
            "../play_results/models_01",
            "--prefix",
            f"v{p0_version}_v{p1_version}",
        ],
        cwd="game",
    )


while True:
    # 学習
    if curr_version > 28:
        train(curr_version, prev_version)

    # 自己対局
    # 今までの bot との結果も混ぜておく
    for op_version in range(prev_version + 1):
        self_play(curr_version, op_version, 200)
    self_play(curr_version, curr_version, 1000)

    print(f"{curr_version} の自己対戦が完了しました。")

    prev_version += 1
    curr_version += 1
