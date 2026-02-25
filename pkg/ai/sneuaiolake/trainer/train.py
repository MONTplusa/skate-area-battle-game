import argparse
import glob
import json
import os
import platform
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx
from tensorflow import keras
from tensorflow.keras import layers
import tqdm

BOARD_SIZE = 20
NUM_CHANNELS = 8
MAX_DISTANCE = BOARD_SIZE - 1
NUM_ACTIONS = 4 * MAX_DISTANCE
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
SCORE_NORM = 10000.0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def residual_block(x, filters):
    fx = layers.Conv2D(filters, 3, padding="same")(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Activation("relu")(fx)
    fx = layers.Conv2D(filters, 3, padding="same")(fx)
    fx = layers.BatchNormalization()(fx)
    out = layers.Add()([x, fx])
    out = layers.Activation("relu")(out)
    return out


def create_actor_critic_model() -> keras.Model:
    inputs = keras.Input(shape=(BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), name="input")

    x = layers.Conv2D(64, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for _ in range(6):
        x = residual_block(x, 64)

    g_avg = layers.GlobalAveragePooling2D()(x)
    g_max = layers.GlobalMaxPooling2D()(x)
    feat = layers.Concatenate()([g_avg, g_max])

    feat = layers.Dense(256, activation="relu")(feat)
    feat = layers.Dropout(0.2)(feat)

    policy_logits = layers.Dense(NUM_ACTIONS, name="policy_logits")(feat)
    value = layers.Dense(1, activation="tanh", name="value")(feat)

    return keras.Model(inputs=inputs, outputs=[policy_logits, value], name="actor_critic")


def resolve_base_model_path(base: str) -> str | None:
    candidates: list[str]
    if base.endswith(".keras"):
        candidates = [base]
    else:
        candidates = [f"{base}_ac.keras", f"{base}.keras"]

    for path in candidates:
        if Path(path).exists():
            return path
    return None


def create_input_data(player: int, state: dict) -> np.ndarray:
    input_data = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), dtype=np.float32)

    player0 = player
    player1 = 1 - player

    board_max = 0
    for row in state["board"]:
        row_max = max(row)
        if row_max > board_max:
            board_max = row_max

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board_max > 0:
                input_data[y, x, 0] = state["board"][y][x] / board_max

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            color = state["colors"][y][x]
            if color == player0:
                input_data[y, x, 1] = 1.0
            elif color == player1:
                input_data[y, x, 2] = 1.0

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if state["rocks"][y][x]:
                input_data[y, x, 3] = 1.0

    p0_x = state[f"player{player0}"]["x"]
    p0_y = state[f"player{player0}"]["y"]
    input_data[p0_y, p0_x, 4] = 1.0

    p1_x = state[f"player{player1}"]["x"]
    p1_y = state[f"player{player1}"]["y"]
    input_data[p1_y, p1_x, 5] = 1.0

    p0_score = 0
    p1_score = 0
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            color = state["colors"][y][x]
            if color == player0:
                p0_score += state["board"][y][x]
            elif color == player1:
                p1_score += state["board"][y][x]

    input_data[:, :, 6] = p0_score / SCORE_NORM
    input_data[:, :, 7] = p1_score / SCORE_NORM

    return input_data


def encode_action(move: dict) -> int:
    dx = move["toX"] - move["fromX"]
    dy = move["toY"] - move["fromY"]

    if dx != 0 and dy != 0:
        raise ValueError("diagonal move is invalid")

    if dy > 0:
        direction = 0
        dist = dy
    elif dx > 0:
        direction = 1
        dist = dx
    elif dy < 0:
        direction = 2
        dist = -dy
    elif dx < 0:
        direction = 3
        dist = -dx
    else:
        raise ValueError("zero-length move is invalid")

    if dist < 1 or dist > MAX_DISTANCE:
        raise ValueError(f"move distance out of range: {dist}")

    return direction * MAX_DISTANCE + (dist - 1)


def legal_action_mask(state: dict, player: int) -> np.ndarray:
    mask = np.zeros((NUM_ACTIONS,), dtype=np.float32)

    px = state[f"player{player}"]["x"]
    py = state[f"player{player}"]["y"]
    opx = state[f"player{1 - player}"]["x"]
    opy = state[f"player{1 - player}"]["y"]

    for direction, (dx, dy) in enumerate(DIRS):
        for dist in range(1, BOARD_SIZE):
            x = px + dx * dist
            y = py + dy * dist

            if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
                break

            if state["rocks"][y][x]:
                break

            if x == opx and y == opy:
                break

            action_id = direction * MAX_DISTANCE + (dist - 1)
            mask[action_id] = 1.0

    return mask


def compute_final_outcome(final_state: dict) -> tuple[float, float]:
    scores = [0, 0]
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            color = final_state["colors"][y][x]
            if color == -1:
                continue
            scores[color] += final_state["board"][y][x]

    if scores[0] > scores[1]:
        return 1.0, -1.0
    if scores[0] < scores[1]:
        return -1.0, 1.0
    return 0.0, 0.0


def load_actor_critic_data(
    result_dir: str,
    prefix: str | None,
    on_policy_model_substr: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    if prefix:
        files = sorted(glob.glob(os.path.join(result_dir, f"{prefix}*.json")))
        print(f"Loading files with prefix '{prefix}' from {result_dir}")
    else:
        files = sorted(glob.glob(os.path.join(result_dir, "*.json")))
        print(f"Loading all JSON files from {result_dir}")

    if not files:
        raise ValueError("No training data files found")

    print(f"Found {len(files)} battle files")

    states: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[float] = []
    trajectories: list[list[int]] = []

    invalid_action_count = 0

    for path in tqdm.tqdm(files):
        try:
            with open(path, "r", encoding="utf-8") as f:
                result = json.load(f)

            moves = result["moves"]
            if not moves:
                continue

            outcome0, outcome1 = compute_final_outcome(result["finalState"])
            outcomes = [outcome0, outcome1]
            player_names = [
                result["initialState"]["player0Name"],
                result["initialState"]["player1Name"],
            ]

            current_state = result["initialState"]
            per_player_indices: list[list[int]] = [[], []]

            for move in moves:
                player = int(move["player"])
                if (
                    on_policy_model_substr is not None
                    and on_policy_model_substr not in player_names[player]
                ):
                    current_state = move["state"]
                    continue

                action_id = encode_action(move)
                mask = legal_action_mask(current_state, player)
                if mask.sum() <= 0:
                    current_state = move["state"]
                    continue

                if mask[action_id] <= 0:
                    invalid_action_count += 1
                    current_state = move["state"]
                    continue

                states.append(create_input_data(player, current_state))
                actions.append(action_id)
                masks.append(mask)
                rewards.append(0.0)
                dones.append(0.0)

                idx = len(states) - 1
                per_player_indices[player].append(idx)

                current_state = move["state"]

            for player in (0, 1):
                step_indices = per_player_indices[player]
                if not step_indices:
                    continue
                rewards[step_indices[-1]] = outcomes[player]
                dones[step_indices[-1]] = 1.0
                trajectories.append(step_indices)

        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not states:
        raise ValueError("No valid transitions collected")

    if invalid_action_count > 0:
        print(f"Skipped {invalid_action_count} transitions due to invalid action encoding")

    x = np.asarray(states, dtype=np.float32)
    a = np.asarray(actions, dtype=np.int32)
    m = np.asarray(masks, dtype=np.float32)
    r = np.asarray(rewards, dtype=np.float32)
    d = np.asarray(dones, dtype=np.float32)

    return x, a, m, r, d, trajectories


def make_dataset(x, a, m, r, adv, old_logp, old_values, batch_size: int, shuffle: bool):
    ds = tf.data.Dataset.from_tensor_slices((x, a, m, r, adv, old_logp, old_values))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 10000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def compute_old_policy_info(
    model: keras.Model,
    x: np.ndarray,
    a: np.ndarray,
    m: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    ds = tf.data.Dataset.from_tensor_slices((x, a, m)).batch(batch_size)
    old_log_probs: list[np.ndarray] = []
    old_values: list[np.ndarray] = []
    very_neg = -1e9

    for states, actions, masks in ds:
        logits, values = model(states, training=False)
        values = tf.squeeze(values, axis=-1)
        masked_logits = tf.where(masks > 0.0, logits, very_neg)
        log_probs = tf.nn.log_softmax(masked_logits, axis=-1)
        action_one_hot = tf.one_hot(actions, NUM_ACTIONS, dtype=tf.float32)
        selected_log_prob = tf.reduce_sum(log_probs * action_one_hot, axis=-1)

        old_log_probs.append(selected_log_prob.numpy().astype(np.float32))
        old_values.append(values.numpy().astype(np.float32))

    return np.concatenate(old_log_probs), np.concatenate(old_values)


def compute_gae_and_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    trajectories: list[list[int]],
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards, dtype=np.float32)

    for trajectory in trajectories:
        next_advantage = 0.0
        for pos in range(len(trajectory) - 1, -1, -1):
            idx = trajectory[pos]

            if pos + 1 < len(trajectory):
                next_value = float(values[trajectory[pos + 1]])
            else:
                next_value = 0.0

            non_terminal = 1.0 - float(dones[idx])
            delta = float(rewards[idx]) + gamma * non_terminal * next_value - float(values[idx])
            next_advantage = delta + gamma * gae_lambda * non_terminal * next_advantage

            advantages[idx] = next_advantage
            returns[idx] = next_advantage + float(values[idx])

    return returns.astype(np.float32), advantages.astype(np.float32)


def compute_batch_losses(
    model,
    states,
    actions,
    masks,
    returns,
    advantages,
    old_log_probs,
    old_values,
    value_coef: float,
    entropy_coef: float,
    ppo_clip_eps: float,
    value_clip_eps: float,
):
    # PPO ratio must compare deterministic old/new policies.
    # Keep model in inference mode during loss evaluation to avoid
    # Dropout/BatchNorm stochasticity corrupting the ratio.
    logits, values = model(states, training=False)
    values = tf.squeeze(values, axis=-1)

    very_neg = tf.constant(-1e9, dtype=logits.dtype)
    masked_logits = tf.where(masks > 0.0, logits, very_neg)

    log_probs = tf.nn.log_softmax(masked_logits, axis=-1)
    probs = tf.nn.softmax(masked_logits, axis=-1)

    action_one_hot = tf.one_hot(actions, NUM_ACTIONS, dtype=tf.float32)
    selected_log_prob = tf.reduce_sum(log_probs * action_one_hot, axis=-1)

    ratio = tf.exp(selected_log_prob - old_log_probs)
    clipped_ratio = tf.clip_by_value(ratio, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps)
    policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

    if value_clip_eps > 0.0:
        clipped_values = old_values + tf.clip_by_value(
            values - old_values,
            -value_clip_eps,
            value_clip_eps,
        )
        value_loss_unclipped = tf.square(returns - values)
        value_loss_clipped = tf.square(returns - clipped_values)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_unclipped, value_loss_clipped))
    else:
        value_loss = 0.5 * tf.reduce_mean(tf.square(returns - values))

    entropy = -tf.reduce_sum(probs * log_probs, axis=-1)
    entropy_bonus = tf.reduce_mean(entropy)

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
    clip_fraction = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > ppo_clip_eps, tf.float32))
    approx_kl = tf.reduce_mean(old_log_probs - selected_log_prob)

    return total_loss, policy_loss, value_loss, entropy_bonus, clip_fraction, approx_kl


def save_models(model: keras.Model, save_path: str) -> None:
    base_path = os.path.splitext(save_path)[0]

    actor_critic_path = f"{base_path}_ac.keras"
    model.save(actor_critic_path)
    print(f"Saved actor-critic model to {actor_critic_path}")

    actor_critic_onnx_path = f"{base_path}_ac.onnx"
    input_signature = [
        tf.TensorSpec((None, BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), tf.float32, name="input")
    ]
    ac_onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=11)

    import onnx

    onnx.save(ac_onnx_model, actor_critic_onnx_path)
    print(f"Saved actor-critic ONNX model to {actor_critic_onnx_path}")

    value_model = keras.Model(inputs=model.input, outputs=model.get_layer("value").output)

    value_keras_path = f"{base_path}.keras"
    value_model.save(value_keras_path)
    print(f"Saved value model to {value_keras_path}")

    onnx_path = f"{base_path}.onnx"
    onnx_model, _ = tf2onnx.convert.from_keras(value_model, input_signature=input_signature, opset=11)

    onnx.save(onnx_model, onnx_path)
    print(f"Saved value ONNX model to {onnx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO + GAE actor-critic training for sneuaiolake")
    parser.add_argument("--base", type=str, default=None, help="base actor-critic model path")
    parser.add_argument("--save", type=str, default="model", help="output model prefix")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--ppo-clip-eps", type=float, default=0.2)
    parser.add_argument("--value-clip-eps", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--target-kl", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--on-policy-model-substr", type=str, default=None)
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPUs: {gpus}")

    base_path = os.path.splitext(args.save)[0]
    output_paths = [
        f"{base_path}_ac.keras",
        f"{base_path}_ac.onnx",
        f"{base_path}.keras",
        f"{base_path}.onnx",
    ]
    for p in output_paths:
        if Path(p).exists():
            raise SystemExit(f"output already exists: {p}")

    if args.base:
        resolved = resolve_base_model_path(args.base)
        if resolved is None:
            raise SystemExit(f"base model not found: {args.base}")
        model = keras.models.load_model(resolved, compile=False)
        if not isinstance(model.outputs, list) or len(model.outputs) != 2:
            raise SystemExit(
                f"base model must be actor-critic with 2 outputs, got: {resolved}"
            )
        print(f"Loaded base actor-critic model: {resolved}")
    else:
        model = create_actor_critic_model()
        print("Created new actor-critic model")

    model.summary()

    if args.init_only:
        save_models(model, args.save)
        return

    x, a, m, rewards, dones, trajectories = load_actor_critic_data(
        args.result_dir,
        args.prefix,
        args.on_policy_model_substr,
    )
    print(f"Collected transitions: {len(x)}")
    print(f"Collected trajectories: {len(trajectories)}")

    old_policy_model = keras.models.clone_model(model)
    old_policy_model.set_weights(model.get_weights())
    old_logp, old_values = compute_old_policy_info(old_policy_model, x, a, m, args.batch_size)
    del old_policy_model

    returns, advantages = compute_gae_and_returns(
        rewards=rewards,
        dones=dones,
        values=old_values,
        trajectories=trajectories,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )

    adv_mean = float(np.mean(advantages))
    adv_std = float(np.std(advantages))
    if adv_std > 1e-8:
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
    else:
        advantages = advantages - adv_mean

    advantages = advantages.astype(np.float32)
    returns = returns.astype(np.float32)
    old_logp = old_logp.astype(np.float32)
    old_values = old_values.astype(np.float32)

    print(
        "Reward stats: mean={:.5f} std={:.5f} min={:.5f} max={:.5f}".format(
            float(np.mean(rewards)),
            float(np.std(rewards)),
            float(np.min(rewards)),
            float(np.max(rewards)),
        )
    )
    print(
        "Advantage stats (normalized): mean={:.5f} std={:.5f} min={:.5f} max={:.5f}".format(
            float(np.mean(advantages)),
            float(np.std(advantages)),
            float(np.min(advantages)),
            float(np.max(advantages)),
        )
    )

    train_ds = make_dataset(
        x,
        a,
        m,
        returns,
        advantages,
        old_logp,
        old_values,
        args.batch_size,
        shuffle=True,
    )

    if platform.system() == "Darwin" and hasattr(keras.optimizers, "legacy"):
        optimizer = keras.optimizers.legacy.Adam(learning_rate=args.learning_rate, clipnorm=1.0)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)

    for epoch in range(1, args.epochs + 1):
        train_sums = {
            "total": 0.0,
            "policy": 0.0,
            "value": 0.0,
            "entropy": 0.0,
            "clip": 0.0,
            "kl": 0.0,
            "count": 0,
        }

        for states, actions, masks, batch_returns, batch_advantages, old_log_probs, batch_old_values in train_ds:
            with tf.GradientTape() as tape:
                (
                    total_loss,
                    policy_loss,
                    value_loss,
                    entropy_bonus,
                    clip_fraction,
                    approx_kl,
                ) = compute_batch_losses(
                    model,
                    states,
                    actions,
                    masks,
                    batch_returns,
                    batch_advantages,
                    old_log_probs,
                    batch_old_values,
                    args.value_loss_coef,
                    args.entropy_coef,
                    args.ppo_clip_eps,
                    args.value_clip_eps,
                )

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            batch_size = int(states.shape[0])
            train_sums["total"] += float(total_loss) * batch_size
            train_sums["policy"] += float(policy_loss) * batch_size
            train_sums["value"] += float(value_loss) * batch_size
            train_sums["entropy"] += float(entropy_bonus) * batch_size
            train_sums["clip"] += float(clip_fraction) * batch_size
            train_sums["kl"] += float(approx_kl) * batch_size
            train_sums["count"] += batch_size

        count = max(train_sums["count"], 1)
        train_metrics = {
            "total": train_sums["total"] / count,
            "policy": train_sums["policy"] / count,
            "value": train_sums["value"] / count,
            "entropy": train_sums["entropy"] / count,
            "clip": train_sums["clip"] / count,
            "kl": train_sums["kl"] / count,
        }

        print(
            "Epoch {}/{} | train total={:.5f} policy={:.5f} value={:.5f} entropy={:.5f} clip={:.3f} kl={:.5f}".format(
                epoch,
                args.epochs,
                train_metrics["total"],
                train_metrics["policy"],
                train_metrics["value"],
                train_metrics["entropy"],
                train_metrics["clip"],
                train_metrics["kl"],
            )
        )

        if args.target_kl > 0.0 and train_metrics["kl"] > args.target_kl:
            print(
                "Early stopping due to KL threshold: {:.5f} > {:.5f}".format(
                    train_metrics["kl"],
                    args.target_kl,
                )
            )
            break

    save_models(model, args.save)


if __name__ == "__main__":
    main()
