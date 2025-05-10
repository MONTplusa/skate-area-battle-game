import subprocess as sp

prev_version = 0
curr_version = 1

while True:
    # 学習
    sp.check_call([
        "python", "train.py",
        "--save", f"v{curr_version}",
        *(["--base", f"v{prev_version}"] if prev_version > 0 else []),
        "--prefix", f"v{prev_version}"
    ])
    # 自己対局
    sp.check_call([
        "python", "self_play.py",
        "--model", f"v{curr_version}",
        "--games", "2000",
        "--output", "output",
        "--prefix", f"v{curr_version}",
    ])

    print(f"Version {curr_version} の自己対戦が完了しました。")

    prev_version += 1
    curr_version += 1
