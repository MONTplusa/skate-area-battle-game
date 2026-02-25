import argparse
import itertools
import json
import re
import subprocess as sp
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


VERSION_RE = re.compile(r"^v(\d+)_ac\.onnx$")
SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
BASELINES = {"random", "trivial", "statiolake", "montplusa"}


@dataclass(frozen=True)
class Participant:
    name: str
    play_arg: str


def parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def sanitize(s: str) -> str:
    return SANITIZE_RE.sub("_", s)


def discover_model_versions(model_dir: Path) -> list[int]:
    versions: list[int] = []
    for path in model_dir.glob("v*_ac.onnx"):
        m = VERSION_RE.match(path.name)
        if m is None:
            continue
        versions.append(int(m.group(1)))
    return sorted(set(versions))


def resolve_participants(
    model_dir: Path,
    start_version: int | None,
    end_version: int | None,
    latest: int | None,
    models_raw: list[str],
    include_baselines: list[str],
) -> list[Participant]:
    participants: list[Participant] = []
    versions = discover_model_versions(model_dir)

    if start_version is not None:
        versions = [v for v in versions if v >= start_version]
    if end_version is not None:
        versions = [v for v in versions if v <= end_version]
    if latest is not None:
        versions = versions[-latest:]

    if models_raw:
        selected_versions: list[int] = []
        selected_baselines: list[str] = []
        for item in models_raw:
            if item in BASELINES:
                selected_baselines.append(item)
                continue
            if item.startswith("v"):
                item = item[1:]
            selected_versions.append(int(item))
        versions = [v for v in versions if v in set(selected_versions)]
        include_baselines = unique_keep_order(include_baselines + selected_baselines)

    for v in versions:
        model_path = model_dir / f"v{v}_ac.onnx"
        if not model_path.exists():
            continue
        participants.append(Participant(name=f"v{v}", play_arg=str(model_path)))

    for b in include_baselines:
        if b not in BASELINES:
            raise SystemExit(f"unknown baseline: {b}")
        participants.append(Participant(name=b, play_arg=b))

    if len(participants) < 2:
        raise SystemExit("need at least 2 participants")

    return participants


def compute_scores(final_state: dict) -> tuple[int, int]:
    board = final_state["board"]
    colors = final_state["colors"]
    score0 = 0
    score1 = 0
    for y, row in enumerate(colors):
        for x, color in enumerate(row):
            if color == 0:
                score0 += board[y][x]
            elif color == 1:
                score1 += board[y][x]
    return score0, score1


def load_match_results(result_dir: Path, prefix: str) -> tuple[list[tuple[float, float]], list[str]]:
    files = sorted(result_dir.glob(f"{prefix}_*.json"))
    if not files:
        raise RuntimeError(f"no result files found for prefix: {prefix}")

    outcomes: list[tuple[float, float]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            battle = json.load(f)
        score0, score1 = compute_scores(battle["finalState"])
        if score0 > score1:
            outcomes.append((1.0, 0.0))
        elif score0 < score1:
            outcomes.append((0.0, 1.0))
        else:
            outcomes.append((0.5, 0.5))
    return outcomes, [p.name for p in files]


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin ELO league for existing AC models")
    parser.add_argument("--model-dir", type=str, default="models_ac")
    parser.add_argument("--result-dir", type=str, default="elo_viewer/data")
    parser.add_argument("--games-per-match", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--k-factor", type=float, default=32.0)
    parser.add_argument("--initial-rating", type=float, default=1500.0)
    parser.add_argument("--start-version", type=int, default=None)
    parser.add_argument("--end-version", type=int, default=None)
    parser.add_argument("--latest", type=int, default=None, help="use only latest N model versions")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma separated explicit targets (e.g. v20,v21,v22 or random)",
    )
    parser.add_argument(
        "--include-baselines",
        type=str,
        default="random",
        help="comma separated baselines from {random,trivial,statiolake,montplusa}",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--no-swap-sides", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trainer_dir = Path(__file__).resolve().parent
    game_dir = trainer_dir / "game"
    model_dir = (trainer_dir / args.model_dir).resolve()
    result_dir = (trainer_dir / args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    include_baselines = parse_csv(args.include_baselines)
    models_raw = parse_csv(args.models)
    participants = resolve_participants(
        model_dir=model_dir,
        start_version=args.start_version,
        end_version=args.end_version,
        latest=args.latest,
        models_raw=models_raw,
        include_baselines=include_baselines,
    )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    swap_sides = not args.no_swap_sides

    pairings = list(itertools.combinations(participants, 2))
    total_series = len(pairings) * (2 if swap_sides else 1)
    total_games = total_series * args.games_per_match

    print("participants:", ", ".join(p.name for p in participants))
    print(f"pairings={len(pairings)}, series={total_series}, scheduled_games={total_games}")
    print(f"result_dir={result_dir}")
    print(f"run_id={run_id}")

    ratings = {p.name: args.initial_rating for p in participants}
    totals = {p.name: {"W": 0, "L": 0, "D": 0, "G": 0} for p in participants}
    pair_results: dict[tuple[str, str], dict[str, int]] = {}
    series_results: list[dict] = []

    series_idx = 0
    for p_a, p_b in pairings:
        sides = [(p_a, p_b)]
        if swap_sides:
            sides.append((p_b, p_a))

        for p0, p1 in sides:
            series_idx += 1
            prefix = sanitize(f"{run_id}_{p0.name}_vs_{p1.name}")

            cmd = [
                "cargo",
                "run",
                "--release",
                "--bin",
                "play",
                "--",
                "--p0-model",
                p0.play_arg,
                "--p1-model",
                p1.play_arg,
                "--games",
                str(args.games_per_match),
                "--result-dir",
                str(result_dir),
                "--prefix",
                prefix,
            ]
            if args.jobs is not None:
                cmd.extend(["--jobs", str(args.jobs)])

            print(f"[{series_idx}/{total_series}] {p0.name} vs {p1.name} ({args.games_per_match} games)")
            if args.dry_run:
                print("  dry-run:", " ".join(cmd))
                continue

            sp.check_call(cmd, cwd=game_dir)
            outcomes, files = load_match_results(result_dir, prefix)

            wins0 = sum(1 for s0, s1 in outcomes if s0 > s1)
            wins1 = sum(1 for s0, s1 in outcomes if s1 > s0)
            draws = len(outcomes) - wins0 - wins1
            print(f"  result {p0.name}:{wins0} {p1.name}:{wins1} draw:{draws}")
            series_results.append(
                {
                    "prefix": prefix,
                    "p0": p0.name,
                    "p1": p1.name,
                    "games": len(outcomes),
                    "p0_wins": wins0,
                    "p1_wins": wins1,
                    "draws": draws,
                    "files": files,
                }
            )

            key = tuple(sorted((p0.name, p1.name)))
            if key not in pair_results:
                pair_results[key] = {"A": 0, "B": 0, "D": 0}

            for s0, s1 in outcomes:
                ra = ratings[p0.name]
                rb = ratings[p1.name]
                ea = expected_score(ra, rb)
                delta = args.k_factor * (s0 - ea)
                ratings[p0.name] = ra + delta
                ratings[p1.name] = rb - delta

                totals[p0.name]["G"] += 1
                totals[p1.name]["G"] += 1

                if s0 == 1.0:
                    totals[p0.name]["W"] += 1
                    totals[p1.name]["L"] += 1
                    if key[0] == p0.name:
                        pair_results[key]["A"] += 1
                    else:
                        pair_results[key]["B"] += 1
                elif s1 == 1.0:
                    totals[p1.name]["W"] += 1
                    totals[p0.name]["L"] += 1
                    if key[0] == p1.name:
                        pair_results[key]["A"] += 1
                    else:
                        pair_results[key]["B"] += 1
                else:
                    totals[p0.name]["D"] += 1
                    totals[p1.name]["D"] += 1
                    pair_results[key]["D"] += 1

    if args.dry_run:
        return

    print("\nPair Results")
    print("============")
    for (a, b), r in sorted(pair_results.items()):
        print(f"{a} vs {b}: {r['A']}-{r['B']}-{r['D']} (W-L-D for {a})")

    print("\nELO Ratings")
    print("===========")
    ranking = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for i, (name, rating) in enumerate(ranking, start=1):
        t = totals[name]
        print(
            f"{i:>2}. {name:<16} ELO={rating:7.2f}  "
            f"W-L-D={t['W']}-{t['L']}-{t['D']}  G={t['G']}"
        )

    summary = {
        "schema": "elo_league_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "config": {
            "model_dir": str(model_dir),
            "result_dir": str(result_dir),
            "games_per_match": args.games_per_match,
            "swap_sides": swap_sides,
            "k_factor": args.k_factor,
            "initial_rating": args.initial_rating,
            "jobs": args.jobs,
        },
        "participants": [p.name for p in participants],
        "ratings": ratings,
        "totals": totals,
        "pair_results": [
            {"a": a, "b": b, "a_wins": r["A"], "b_wins": r["B"], "draws": r["D"]}
            for (a, b), r in sorted(pair_results.items())
        ],
        "series_results": series_results,
    }
    summary_filename = f"{run_id}_summary.json"
    summary_path = result_dir / summary_filename
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary: {summary_path}")

    manifest_path = result_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"schema": "elo_manifest_v1", "runs": []}
    runs = [r for r in manifest.get("runs", []) if r.get("run_id") != run_id]
    runs.append(
        {
            "run_id": run_id,
            "created_at": summary["created_at"],
            "summary_file": summary_filename,
            "participants": len(summary["participants"]),
            "series": len(series_results),
            "games": sum(int(s["games"]) for s in series_results),
        }
    )
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    manifest["runs"] = runs
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Updated manifest: {manifest_path}")


if __name__ == "__main__":
    main()
