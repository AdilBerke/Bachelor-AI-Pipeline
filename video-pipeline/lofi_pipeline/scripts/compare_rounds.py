"""
Compare training rounds for a scenario.

Usage:
  python compare_rounds.py --scenario lofi_girl_desk
  python compare_rounds.py --scenario lofi_girl_desk --rounds 1,3,5
  python compare_rounds.py --all
"""
import argparse
import json
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent.parent
SCENARIOS_DIR = PIPELINE_ROOT / "scenarios"


def load_json(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {}


def get_round_summary(round_dir):
    notes = load_json(round_dir / "notes.json")
    feedback = load_json(round_dir / "feedback.json")
    samples = sorted((round_dir / "samples").glob("*.gif")) if (round_dir / "samples").exists() else []

    return {
        "round": notes.get("round_id", round_dir.name),
        "steps": notes.get("training_steps", "?"),
        "duration_min": notes.get("duration_minutes", "?"),
        "checkpoint": Path(notes.get("checkpoint_loaded", "?")).name if notes.get("checkpoint_loaded") else "?",
        "quality": feedback.get("quality_score", "-"),
        "animation": feedback.get("animation_score", "-"),
        "detail": feedback.get("detail_score", "-"),
        "samples": [s.name for s in samples],
        "notes": notes.get("start_time", "?")[:10] if notes.get("start_time") else "?",
        "feedback_summary": feedback.get("what_changed_since_previous", ""),
    }


def print_table(rows):
    headers = ["Round", "Steps", "Min", "Quality", "Animation", "Detail", "Samples", "Date"]
    col_w = [7, 6, 6, 9, 10, 7, 9, 12]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        row = [
            str(r["round"]).ljust(col_w[0]),
            str(r["steps"]).ljust(col_w[1]),
            str(r["duration_min"]).ljust(col_w[2]),
            str(r["quality"]).ljust(col_w[3]),
            str(r["animation"]).ljust(col_w[4]),
            str(r["detail"]).ljust(col_w[5]),
            str(len(r["samples"])).ljust(col_w[6]),
            str(r["notes"]).ljust(col_w[7]),
        ]
        print(" | ".join(row))


def compare_scenario(scenario_id, round_filter=None):
    scenario_dir = SCENARIOS_DIR / scenario_id
    rounds_dir = scenario_dir / "rounds"

    if not rounds_dir.exists():
        print(f"No rounds found for scenario '{scenario_id}'")
        return

    round_dirs = sorted(rounds_dir.iterdir())
    if round_filter:
        allowed = {f"round_{int(r):02d}" for r in round_filter.split(",")}
        round_dirs = [d for d in round_dirs if d.name in allowed]

    summaries = [get_round_summary(d) for d in round_dirs if d.is_dir()]

    if not summaries:
        print(f"No round data found for '{scenario_id}'")
        return

    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario_id} — {len(summaries)} rounds")
    print(f"{'='*60}")
    print_table(summaries)

    print(f"\nSamples per round:")
    for s in summaries:
        if s["samples"]:
            print(f"  Round {s['round']}: {', '.join(s['samples'])}")

    print(f"\nKey changes:")
    for s in summaries:
        if s["feedback_summary"]:
            print(f"  Round {s['round']}: {s['feedback_summary'][:80]}")


def main():
    parser = argparse.ArgumentParser(description="Compare Lo-Fi training rounds")
    parser.add_argument("--scenario", type=str, help="Scenario ID")
    parser.add_argument("--rounds", type=str, help="Comma-separated round numbers, e.g. 1,3,5")
    parser.add_argument("--all", action="store_true", help="Compare all scenarios")
    args = parser.parse_args()

    if args.all:
        for scenario_dir in sorted(SCENARIOS_DIR.iterdir()):
            if scenario_dir.is_dir() and (scenario_dir / "scenario.yaml").exists():
                compare_scenario(scenario_dir.name)
    elif args.scenario:
        compare_scenario(args.scenario, args.rounds)
    else:
        print("ERROR: provide --scenario or --all")


if __name__ == "__main__":
    main()
