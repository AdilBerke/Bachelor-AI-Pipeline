"""
Metric Watcher — automatische Qualitaetsbewertung pro Training-Checkpoint.

Laeuft parallel zu train_lora.py in einem zweiten Terminal.
Ueberwacht den Checkpoints-Ordner, generiert bei jedem neuen Checkpoint
ein kurzes Test-Video und bewertet es mit evaluate_video.py.

Verwendung:
  # Terminal 1:
  python scripts/train_lora.py --scenario rabbit_lake_v2 --round 1 --steps 200 --fresh

  # Terminal 2 (sofort starten):
  python scripts/metric_watcher.py --scenario rabbit_lake_v2 --round 1

  # Mit Zielwerten (stoppt Training automatisch bei Erreichen):
  python scripts/metric_watcher.py --scenario rabbit_lake_v2 --round 1 \\
    --target-overall 0.82 --target-motion 0.50 --target-sharpness 0.22

Ausgabe:
  - Echtzeit-ASCII-Graph im Terminal
  - round_dir/metrics_log.json  (alle Ergebnisse)
  - round_dir/STOP_TRAINING     (wenn Ziel erreicht → train_lora.py stoppt)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

PIPELINE_ROOT = Path(__file__).parent.parent
CONFIGS_DIR   = PIPELINE_ROOT / "configs"
SCENARIOS_DIR = PIPELINE_ROOT / "scenarios"
EVALUATE_SCRIPT = Path(__file__).parent.parent.parent / "pipeline" / "realistic_rabbit" / "evaluate_video.py"
GENERATE_SCRIPT = Path(__file__).parent.parent.parent / "pipeline" / "v003_manual" / "generate.py"


def clr(c, t): return f"\033[{c}m{t}\033[0m"
def cyan(t):   return clr("36", t)
def green(t):  return clr("32", t)
def yellow(t): return clr("33", t)
def red(t):    return clr("31", t)
def dim(t):    return clr("2",  t)
def bold(t):   return clr("1",  t)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_metrics_log(log_path):
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return {"entries": []}


def save_metrics_log(log_path, data):
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)


def generate_eval_video(scenario_cfg, paths_cfg, ckpt_path, out_path, eval_frames, eval_steps):
    """Generiert ein kurzes Eval-Video mit dem aktuellen Checkpoint."""
    env = os.environ.copy()
    env.update(paths_cfg.get("env", {}))

    cmd = [
        paths_cfg["python"], str(GENERATE_SCRIPT),
        "--lora", str(ckpt_path),
        "--steps", str(eval_steps),
        "--frames", str(eval_frames),
        "--seed", "42",
        "--guidance-scale", str(scenario_cfg.get("guidance_scale", 9.0)),
        "--prompt", scenario_cfg["prompt"].strip(),
        "--negative-prompt", scenario_cfg["negative_prompt"].strip(),
        "--output", str(out_path),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return result.returncode == 0


def evaluate_video(video_path, python_bin):
    """Fuehrt evaluate_video.py aus und gibt Metriken als dict zurueck."""
    result = subprocess.run(
        [python_bin, str(EVALUATE_SCRIPT), str(video_path), "--save-report"],
        capture_output=True, text=True,
    )
    metrics_file = Path(str(video_path).replace(".mp4", ".metrics.json"))
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
        return {
            "auto_ssim":     m["temporal_ssim"]["mean"],
            "auto_sharpness": m["sharpness"]["normalized"],
            "auto_flicker":  m["flicker"]["mean"],
            "auto_motion":   m["motion"]["mean"],
            "auto_color":    m["color_consistency"]["consistency"],
            "auto_overall":  m["overall_score"],
        }
    return None


def ascii_graph(entries, key, width=40, label=""):
    """Zeichnet einen Mini-ASCII-Graph fuer eine Metrik ueber Steps."""
    if not entries:
        return ""
    values = [e["metrics"].get(key, 0) for e in entries]
    steps  = [e["step"] for e in entries]
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        vmax = vmin + 0.001
    bars = []
    bar_width = max(1, width // max(len(values), 1))
    for v in values:
        h = int((v - vmin) / (vmax - vmin) * 8)
        bars.append("█" * h or "▁")
    line = " ".join(bars)
    return f"  {label:<18} [{line[:width]}]  {values[-1]:.4f}"


def check_targets(metrics, targets):
    """Gibt True zurueck wenn alle Ziele erreicht sind."""
    if not targets:
        return False
    checks = []
    if "overall" in targets:
        checks.append(metrics.get("auto_overall", 0) >= targets["overall"])
    if "motion" in targets:
        checks.append(metrics.get("auto_motion", 0) >= targets["motion"])
    if "sharpness" in targets:
        checks.append(metrics.get("auto_sharpness", 0) >= targets["sharpness"])
    if "flicker" in targets:
        checks.append(metrics.get("auto_flicker", 1) <= targets["flicker"])
    return all(checks) if checks else False


def print_progress(entries, targets, step, total_known):
    """Gibt aktuellen Status mit ASCII-Grafik aus."""
    os.system("clear")
    print(bold(f"\n  Metric Watcher — Step {step}" + (f"/{total_known}" if total_known else "")))
    print(dim(f"  {datetime.now().strftime('%H:%M:%S')}"))
    print()

    if entries:
        last = entries[-1]["metrics"]
        print(f"  {'Metrik':<20} {'Aktuell':>8}   {'Ziel':>8}   Status")
        print(f"  {'-'*56}")

        def row(label, key, target_key=None, good_high=True, target_val=None):
            val = last.get(key, 0)
            tgt = target_val or (targets.get(target_key or key.replace("auto_", ""), None))
            if tgt is not None:
                ok = (val >= tgt) if good_high else (val <= tgt)
                status = green("✓") if ok else yellow("…")
                tgt_str = f"{tgt:.4f}"
            else:
                status = dim("—")
                tgt_str = "—"
            col = green if (good_high and val >= 0.8) or (not good_high and val <= 0.02) else (
                  yellow if (good_high and val >= 0.6) or (not good_high and val <= 0.06) else red)
            print(f"  {label:<20} {col(f'{val:.4f}'):>8}   {tgt_str:>8}   {status}")

        row("SSIM (Stabilitaet)", "auto_ssim")
        row("Schaerfe",           "auto_sharpness", "sharpness")
        row("Flicker (↓ gut)",    "auto_flicker",   "flicker", good_high=False)
        row("Motion px/Frame",    "auto_motion",     "motion")
        row("Farbkonstanz",       "auto_color")
        print(f"  {'-'*56}")
        row("GESAMT-SCORE",       "auto_overall",    "overall")
        print()

        if len(entries) >= 2:
            print(dim("  Verlauf über Steps:"))
            print(ascii_graph(entries, "auto_overall",  label="Gesamt"))
            print(ascii_graph(entries, "auto_sharpness", label="Schaerfe"))
            print(ascii_graph(entries, "auto_motion",    label="Motion"))
            print()

        if targets and check_targets(last, targets):
            print(green(bold("  ✓✓✓ ALLE ZIELE ERREICHT — STOP_TRAINING gesetzt")))
        else:
            missing = []
            if targets.get("overall") and last.get("auto_overall", 0) < targets["overall"]:
                missing.append(f"Gesamt {last.get('auto_overall',0):.3f} < {targets['overall']:.3f}")
            if targets.get("motion") and last.get("auto_motion", 0) < targets["motion"]:
                missing.append(f"Motion {last.get('auto_motion',0):.3f} < {targets['motion']:.3f}")
            if targets.get("sharpness") and last.get("auto_sharpness", 0) < targets["sharpness"]:
                missing.append(f"Schaerfe {last.get('auto_sharpness',0):.3f} < {targets['sharpness']:.3f}")
            if missing:
                print(yellow(f"  Noch nicht erreicht: {', '.join(missing)}"))
    else:
        print(dim("  Warte auf ersten Checkpoint…"))
    print()


def main():
    parser = argparse.ArgumentParser(description="Metric Watcher fuer LoRA Training")
    parser.add_argument("--scenario", required=True, help="Scenario ID (z.B. rabbit_lake_v2)")
    parser.add_argument("--round", type=int, required=True, help="Round Nummer")
    parser.add_argument("--eval-frames", type=int, default=49,
                        help="Frames fuer Eval-Video (Standard: 49 ≈ 6s, schneller)")
    parser.add_argument("--eval-steps", type=int, default=40,
                        help="Inference-Steps fuer Eval-Video (Standard: 40, schneller als volle 60)")
    parser.add_argument("--target-overall",   type=float, default=None, help="Ziel Gesamt-Score (0-1)")
    parser.add_argument("--target-motion",    type=float, default=None, help="Ziel Motion px/Frame")
    parser.add_argument("--target-sharpness", type=float, default=None, help="Ziel Schaerfe (0-1)")
    parser.add_argument("--target-flicker",   type=float, default=None, help="Max. erlaubtes Flicker (0-1)")
    parser.add_argument("--poll-interval", type=int, default=20,
                        help="Sekunden zwischen Checkpoint-Checks (Standard: 20)")
    args = parser.parse_args()

    scenario_dir = SCENARIOS_DIR / args.scenario
    if not scenario_dir.exists():
        print(red(f"Szenario nicht gefunden: {scenario_dir}"))
        sys.exit(1)

    scenario_cfg = load_yaml(scenario_dir / "scenario.yaml")
    paths_cfg    = load_yaml(CONFIGS_DIR / "model_paths.yaml")
    python_bin   = paths_cfg["python"]

    round_dir  = scenario_dir / "rounds" / f"round_{args.round:02d}"
    ckpt_dir   = round_dir / "checkpoints"
    eval_dir   = round_dir / "metrics_eval"
    log_path   = round_dir / "metrics_log.json"
    stop_file  = round_dir / "STOP_TRAINING"
    eval_dir.mkdir(parents=True, exist_ok=True)

    scenario_targets = scenario_cfg.get("metric_targets", {})
    targets = {}
    targets["overall"]   = args.target_overall  or scenario_targets.get("overall")
    targets["motion"]    = args.target_motion    or scenario_targets.get("motion")
    targets["sharpness"] = args.target_sharpness or scenario_targets.get("sharpness")
    targets["flicker"]   = args.target_flicker   or scenario_targets.get("flicker")
    targets = {k: v for k, v in targets.items() if v is not None}

    log_data = load_metrics_log(log_path)
    if "entries" not in log_data:
        log_data["entries"] = []
    log_data["scenario"] = args.scenario
    log_data["round"]    = args.round
    log_data["target"]   = targets

    processed_steps = {e["step"] for e in log_data["entries"]}

    print(bold(f"\n  Metric Watcher gestartet"))
    print(f"  Szenario: {args.scenario}  Runde: {args.round}")
    print(f"  Beobachte: {ckpt_dir}")
    if targets:
        print(f"  Ziele: " + "  ".join(f"{k}≥{v}" for k, v in targets.items()))
    print(dim(f"  Eval: {args.eval_frames} Frames, {args.eval_steps} Steps"))
    print(dim(f"  Poll: alle {args.poll_interval}s"))
    print()

    save_metrics_log(log_path, log_data)

    while True:
        ckpts = []
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("lora_weights_step_*.safetensors"))

        new_ckpts = [c for c in ckpts if int(c.stem.split("_")[-1]) not in processed_steps]

        for ckpt in new_ckpts:
            step = int(ckpt.stem.split("_")[-1])
            print(cyan(f"\n  [Step {step:05d}] Neuer Checkpoint — generiere Eval-Video..."))

            eval_video = eval_dir / f"step_{step:05d}.mp4"
            t0 = time.time()
            ok = generate_eval_video(scenario_cfg, paths_cfg, ckpt, eval_video,
                                      args.eval_frames, args.eval_steps)
            gen_time = time.time() - t0

            if not ok or not eval_video.exists():
                print(yellow(f"  Generierung fehlgeschlagen (step {step}) — übersprungen"))
                processed_steps.add(step)
                continue

            print(dim(f"  Generierung: {gen_time:.0f}s — analysiere..."))
            metrics = evaluate_video(eval_video, python_bin)

            if metrics is None:
                print(yellow(f"  Auswertung fehlgeschlagen (step {step})"))
                processed_steps.add(step)
                continue

            entry = {
                "step": step,
                "checkpoint": ckpt.name,
                "eval_video": str(eval_video),
                "timestamp": datetime.now().isoformat(),
                "gen_time_s": round(gen_time, 1),
                "metrics": metrics,
                "target_reached": check_targets(metrics, targets),
            }
            log_data["entries"].append(entry)
            save_metrics_log(log_path, log_data)
            processed_steps.add(step)

            print_progress(log_data["entries"], targets, step, None)

            if entry["target_reached"]:
                stop_file.write_text(f"Targets reached at step {step}\n")
                print(green(bold(f"\n  ✓ Ziele erreicht bei Step {step} — STOP_TRAINING gesetzt")))
                print(dim(f"  {stop_file}"))
                print(dim("  train_lora.py wird beim nächsten Check gestoppt."))
                return

        if not new_ckpts and log_data["entries"]:
            print_progress(log_data["entries"], targets,
                          log_data["entries"][-1]["step"], None)

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
