import argparse
import os
import random
import subprocess as sp


def run_play(
    result_dir: str,
    prefix: str,
    p0_model: str,
    p1_model: str,
    games: int,
    jobs: int | None,
) -> None:
    cmd = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "play",
        "--",
        "--p0-model",
        p0_model,
        "--p1-model",
        p1_model,
        "--games",
        str(games),
        "--result-dir",
        f"../{result_dir}",
        "--prefix",
        prefix,
    ]
    if jobs is not None:
        cmd.extend(["--jobs", str(jobs)])

    sp.check_call(cmd, cwd="game")


def init_model(model_prefix: str, result_dir: str) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        "train.py",
        "--save",
        model_prefix,
        "--result-dir",
        result_dir,
        "--init-only",
    ]
    sp.check_call(cmd)


def train_model(
    version: int,
    model_dir: str,
    result_dir: str,
    prefix: str,
    base_model: str | None,
    on_policy_model_substr: str | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    value_loss_coef: float,
    entropy_coef: float,
    ppo_clip_eps: float,
    value_clip_eps: float,
    gamma: float,
    gae_lambda: float,
    target_kl: float,
) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        "train.py",
        "--save",
        f"{model_dir}/v{version}",
        "--result-dir",
        result_dir,
        "--prefix",
        prefix,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--value-loss-coef",
        str(value_loss_coef),
        "--entropy-coef",
        str(entropy_coef),
        "--ppo-clip-eps",
        str(ppo_clip_eps),
        "--value-clip-eps",
        str(value_clip_eps),
        "--gamma",
        str(gamma),
        "--gae-lambda",
        str(gae_lambda),
        "--target-kl",
        str(target_kl),
    ]

    if base_model is not None:
        cmd.extend(["--base", f"{model_dir}/{base_model}"])
    if on_policy_model_substr is not None:
        cmd.extend(["--on-policy-model-substr", on_policy_model_substr])

    sp.check_call(cmd)


def model_path_for_play(model_dir: str, version: int | str) -> str:
    if isinstance(version, int):
        return f"../{model_dir}/v{version}_ac.onnx"
    name = version if version.endswith(".onnx") else f"{version}_ac.onnx"
    return f"../{model_dir}/{name}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Actor-Critic self-play batch trainer")
    parser.add_argument("--start-version", type=int, default=0)
    parser.add_argument("--end-version", type=int, default=5)
    parser.add_argument("--model-dir", type=str, default="models_ac")
    parser.add_argument("--result-dir", type=str, default="play_results/models_ac")
    parser.add_argument("--jobs", type=int, default=None)

    parser.add_argument("--bootstrap-games", type=int, default=500)
    parser.add_argument("--games-vs-prev", type=int, default=20)
    parser.add_argument("--games-self", type=int, default=500)
    parser.add_argument("--games-vs-baseline", type=int, default=20)

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--ppo-clip-eps", type=float, default=0.2)
    parser.add_argument("--value-clip-eps", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--target-kl", type=float, default=0.0)

    args = parser.parse_args()

    if args.start_version < 0 or args.end_version < args.start_version:
        raise SystemExit("invalid version range")

    ensure_dir(args.model_dir)
    ensure_dir(args.result_dir)

    v0_onnx = f"{args.model_dir}/v0.onnx"
    v0_ac_onnx = f"{args.model_dir}/v0_ac.onnx"
    v0_init_prefix = f"{args.model_dir}/v0_init"
    v0_init_onnx = f"{v0_init_prefix}_ac.onnx"

    if args.start_version == 0 and (not os.path.exists(v0_onnx) or not os.path.exists(v0_ac_onnx)):
        print("[bootstrap] generating v0_init model")
        if not os.path.exists(v0_init_onnx):
            init_model(v0_init_prefix, args.result_dir)

        print("[bootstrap] generating on-policy data for v0")
        run_play(
            args.result_dir,
            "bootstrap_v0init_self",
            model_path_for_play(args.model_dir, "v0_init"),
            model_path_for_play(args.model_dir, "v0_init"),
            args.bootstrap_games,
            args.jobs,
        )

        print("[train] v0")
        train_model(
            version=0,
            model_dir=args.model_dir,
            result_dir=args.result_dir,
            prefix="bootstrap_v0init_",
            base_model="v0_init",
            on_policy_model_substr="v0_init_ac.onnx",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            ppo_clip_eps=args.ppo_clip_eps,
            value_clip_eps=args.value_clip_eps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            target_kl=args.target_kl,
        )

    loop_start = max(args.start_version, 1)
    for version in range(loop_start, args.end_version + 1):
        prev = version - 1

        if not os.path.exists(f"{args.model_dir}/v{prev}_ac.onnx"):
            raise SystemExit(f"missing previous model: {args.model_dir}/v{prev}_ac.onnx")

        print(f"[self-play] generating data from v{prev}")

        past_versions = list(range(prev))
        if len(past_versions) > 5:
            opponents = random.sample(past_versions, 5)
        else:
            opponents = past_versions

        for opponent in opponents:
            run_play(
                args.result_dir,
                f"v{prev}_v{opponent}",
                model_path_for_play(args.model_dir, prev),
                model_path_for_play(args.model_dir, opponent),
                args.games_vs_prev,
                args.jobs,
            )

        run_play(
            args.result_dir,
            f"v{prev}_self",
            model_path_for_play(args.model_dir, prev),
            model_path_for_play(args.model_dir, prev),
            args.games_self,
            args.jobs,
        )

        run_play(
            args.result_dir,
            f"v{prev}_random",
            model_path_for_play(args.model_dir, prev),
            "random",
            args.games_vs_baseline,
            args.jobs,
        )

        print(f"[train] v{version} (base=v{prev})")
        train_model(
            version=version,
            model_dir=args.model_dir,
            result_dir=args.result_dir,
            prefix=f"v{prev}_",
            base_model=f"v{prev}",
            on_policy_model_substr=f"v{prev}_ac.onnx",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            ppo_clip_eps=args.ppo_clip_eps,
            value_clip_eps=args.value_clip_eps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            target_kl=args.target_kl,
        )

    print("batch training completed")


if __name__ == "__main__":
    main()
