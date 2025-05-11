import subprocess as sp

curr_version = 1

children = []
for i in range(4):
    # 1つはGPUを使い、もう2つはCPUを使う
    env = {}
    if i > 0:
        env["CUDA_VISIBLE_DEVICES"] = ""
    children.append(sp.Popen([
        "python", "self_play.py",
        "--model", f"v{curr_version}",
        "--games", "5",
        "--output", ".",
        "--prefix", f"v{curr_version}_b{i}",
    ], env=env))

for child in children:
    child.wait()
