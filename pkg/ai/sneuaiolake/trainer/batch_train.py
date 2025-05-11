import subprocess as sp

prev_version = 2
curr_version = 3

while True:
    if prev_version > 2:
        # 学習
        sp.check_call([
            "python", "train.py",
            "--save", f"models_01/v{curr_version}",
            *(["--base", f"models_01/v{prev_version}"] if prev_version > 0 else []),
            "--prefix", f"model_01_v{prev_version}"
        ])

    # 自己対局 (4並列)
    children = []
    for i in range(4):
        # 1つはGPUを使い、もう2つはCPUを使う
        env = {}
        if i > 0:
            env["CUDA_VISIBLE_DEVICES"] = ""
        children.append(sp.Popen([
            "python", "self_play.py",
            "--model", f"models_01/v{curr_version}",
            "--games", "250",
            "--output", "output",
            "--prefix", f"model_01_v{curr_version}_b{i}",
        ], env=env))
    for child in children:
        child.wait()


    print(f"Version {curr_version} の自己対戦が完了しました。")

    prev_version += 1
    curr_version += 1
