"""
Build training overview report (CSV + Markdown).

Usage:
  python build_report.py
  python build_report.py --scenario lofi_girl_desk
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent.parent
SCENARIOS_DIR = PIPELINE_ROOT / "scenarios"
REPORTS_DIR = PIPELINE_ROOT / "reports"


def load_json(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {}


def collect_all_rounds():
    rows = []
    for scenario_dir in sorted(SCENARIOS_DIR.iterdir()):
        if not scenario_dir.is_dir():
            continue
        rounds_dir = scenario_dir / "rounds"
        if not rounds_dir.exists():
            continue
        for round_dir in sorted(rounds_dir.iterdir()):
            if not round_dir.is_dir():
                continue
            notes = load_json(round_dir / "notes.json")
            feedback = load_json(round_dir / "feedback.json")
            if not notes:
                continue
            rows.append({
                "scenario_id": notes.get("scenario_id", scenario_dir.name),
                "round_id": notes.get("round_id", round_dir.name),
                "start_time": notes.get("start_time", ""),
                "end_time": notes.get("end_time", ""),
                "duration_minutes": notes.get("duration_minutes", ""),
                "training_steps": notes.get("training_steps", ""),
                "checkpoint_loaded": notes.get("checkpoint_loaded", ""),
                "lora_path": notes.get("checkpoints_saved", [""])[-1] if notes.get("checkpoints_saved") else "",
                "sample_output_path": ";".join(notes.get("samples_generated", [])),
                "peak_gpu_gb": notes.get("peak_gpu_memory_gb", ""),
                "training_speed": notes.get("training_speed_steps_per_sec", ""),
                "quality_score": feedback.get("quality_score", ""),
                "animation_score": feedback.get("animation_score", ""),
                "detail_score": feedback.get("detail_score", ""),
                "realism_score": feedback.get("realism_score", ""),
                "what_changed": feedback.get("what_changed_since_previous", ""),
                "improvements": feedback.get("improvements_visible", ""),
                "regressions": feedback.get("regressions_visible", ""),
                "issues": feedback.get("remaining_issues", ""),
                "recommendation": feedback.get("recommendation_next", ""),
            })
    return rows


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path):
    lines = [
        "# Lo-Fi LoRA Training — Übersicht",
        f"\n_Generiert: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
        "## Alle Trainingsrunden\n",
        "| Szenario | Runde | Steps | Dauer | Quality | Animation | Detail | Geändert |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        changed = (r["what_changed"] or "")[:60]
        lines.append(
            f"| {r['scenario_id']} | {r['round_id']} | {r['training_steps']} "
            f"| {r['duration_minutes']} min | {r['quality_score']} | {r['animation_score']} "
            f"| {r['detail_score']} | {changed} |"
        )

    scenarios = {}
    for r in rows:
        scenarios.setdefault(r["scenario_id"], []).append(r)

    lines += ["\n## Fortschritt pro Szenario\n"]
    for scenario_id, scenario_rows in scenarios.items():
        lines.append(f"### {scenario_id}\n")
        qs = [r["quality_score"] for r in scenario_rows if isinstance(r["quality_score"], (int, float))]
        if qs:
            lines.append(f"- Runden: {len(scenario_rows)}")
            lines.append(f"- Bester Quality-Score: {max(qs)}/10")
            lines.append(f"- Letzte Runde: {scenario_rows[-1]['round_id']}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build Lo-Fi training report")
    parser.add_argument("--scenario", type=str, default=None, help="Filter to one scenario")
    args = parser.parse_args()

    rows = collect_all_rounds()
    if args.scenario:
        rows = [r for r in rows if r["scenario_id"] == args.scenario]

    if not rows:
        print("No training data found yet.")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "training_overview.csv"
    md_path = REPORTS_DIR / "model_evolution.md"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(f"Report written:")
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {md_path}")
    print(f"  Rows:     {len(rows)}")


if __name__ == "__main__":
    main()
