"""
Lo-Fi LoRA Training CLI

Usage:
  python train_lora.py --scenario lofi_girl_desk --round 1
  python train_lora.py --scenario lofi_girl_desk --round 2 --resume
  python train_lora.py --scenario rainy_window --round 1 --steps 100
  python train_lora.py --scenario lofi_girl_desk --round 3 --from-checkpoint /path/to/lora.safetensors
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

PIPELINE_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PIPELINE_ROOT / "configs"
SCENARIOS_DIR = PIPELINE_ROOT / "scenarios"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def find_last_checkpoint(scenario_dir, round_id):
    """Find the highest-step checkpoint from a given round."""
    round_dir = scenario_dir / "rounds" / f"round_{round_id:02d}"
    ckpt_dir = round_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("lora_weights_step_*.safetensors"))
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def find_auto_checkpoint(scenario_dir, round_id):
    """Auto-detect checkpoint: use previous round's last checkpoint, or base."""
    paths_cfg = load_yaml(CONFIGS_DIR / "model_paths.yaml")

    if round_id > 1:
        ckpt = find_last_checkpoint(scenario_dir, round_id - 1)
        if ckpt:
            print(f"  Auto-detected checkpoint from round {round_id - 1}: {Path(ckpt).name}")
            return ckpt

    base = paths_cfg["global_base_checkpoint"]
    print(f"  Using global base checkpoint: {Path(base).name}")
    return base


def build_training_config(scenario_cfg, base_cfg, checkpoint, output_dir, steps):
    """Merge base config with scenario-specific overrides."""
    cfg = {}

    cfg["model"] = {
        "model_source": "LTXV_13B_097_DEV",
        "training_mode": "lora",
    }
    if checkpoint is not None:
        cfg["model"]["load_checkpoint"] = checkpoint
    cfg["lora"] = base_cfg["lora"]
    cfg["conditioning"] = base_cfg["conditioning"]
    cfg["optimization"] = dict(base_cfg["optimization"])
    cfg["optimization"]["steps"] = steps
    cfg["acceleration"] = base_cfg["acceleration"]
    cfg["data"] = {
        "preprocessed_data_root": scenario_cfg["precomputed_dir"],
        "num_dataloader_workers": base_cfg["data"]["num_dataloader_workers"],
    }
    cfg["validation"] = dict(base_cfg["validation"])
    cfg["validation"]["video_dims"] = scenario_cfg["resolution"] + [scenario_cfg["frames"]]
    cfg["validation"]["seed"] = scenario_cfg["seed"]
    cfg["validation"]["skip_initial_validation"] = True
    cfg["checkpoints"] = base_cfg["checkpoints"]
    cfg["output_dir"] = str(output_dir)
    cfg["flow_matching"] = base_cfg["flow_matching"]
    cfg["hub"] = base_cfg["hub"]
    cfg["wandb"] = base_cfg["wandb"]

    return cfg


def save_notes(round_dir, scenario_id, round_id, checkpoint, start_time, end_time, steps, log_path):
    """Auto-generate notes.json after training."""
    duration = (end_time - start_time).total_seconds() / 60

    ckpt_dir = round_dir / "checkpoints"
    saved_checkpoints = sorted(ckpt_dir.glob("lora_weights_step_*.safetensors")) if ckpt_dir.exists() else []

    peak_gpu = None
    training_speed = None
    if log_path.exists():
        log_text = log_path.read_text()
        for line in log_text.splitlines():
            if "Peak GPU memory" in line:
                try:
                    peak_gpu = float(line.split(":")[1].strip().split()[0])
                except Exception:
                    pass
            if "Training speed" in line:
                try:
                    training_speed = float(line.split(":")[1].strip().split()[0])
                except Exception:
                    pass

    notes = {
        "scenario_id": scenario_id,
        "round_id": round_id,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_minutes": round(duration, 1),
        "training_steps": steps,
        "checkpoint_loaded": checkpoint,
        "checkpoints_saved": [str(c) for c in saved_checkpoints],
        "peak_gpu_memory_gb": peak_gpu,
        "training_speed_steps_per_sec": training_speed,
        "samples_generated": [],
    }

    with open(round_dir / "notes.json", "w") as f:
        json.dump(notes, f, indent=2)

    return notes


def main():
    parser = argparse.ArgumentParser(description="Lo-Fi LoRA Training CLI")
    parser.add_argument("--scenario", required=True, help="Scenario ID (e.g. lofi_girl_desk)")
    parser.add_argument("--round", type=int, required=True, help="Round number (1, 2, 3, ...)")
    parser.add_argument("--steps", type=int, default=100, help="Training steps (default: 100)")
    parser.add_argument("--resume", action="store_true", help="Auto-load checkpoint from previous round")
    parser.add_argument("--from-checkpoint", type=str, default=None, help="Explicit checkpoint path")
    parser.add_argument("--fresh", action="store_true", help="Start with no LoRA checkpoint (fresh LoRA init)")
    parser.add_argument("--no-generate", action="store_true", help="Skip sample generation after training")
    parser.add_argument("--target-score", type=float, default=None,
                        help="Ziel-Gesamt-Score (0-1). Training stoppt automatisch wenn metric_watcher.py "
                             "STOP_TRAINING setzt (Standard: kein automatischer Stopp)")
    parser.add_argument("--config", type=str, default="base_lora.yaml",
                        help="Basis-Config-Datei (Standard: base_lora.yaml, fuer v2: base_lora_v2.yaml)")
    args = parser.parse_args()

    scenario_dir = SCENARIOS_DIR / args.scenario
    if not scenario_dir.exists():
        print(f"ERROR: Scenario '{args.scenario}' not found at {scenario_dir}")
        sys.exit(1)

    scenario_cfg_path = scenario_dir / "scenario.yaml"
    if not scenario_cfg_path.exists():
        print(f"ERROR: scenario.yaml missing at {scenario_cfg_path}")
        sys.exit(1)

    scenario_cfg = load_yaml(scenario_cfg_path)
    base_cfg = load_yaml(CONFIGS_DIR / args.config)
    paths_cfg = load_yaml(CONFIGS_DIR / "model_paths.yaml")

    round_dir = scenario_dir / "rounds" / f"round_{args.round:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, 'fresh', False):
        checkpoint = None
        print(f"Starting fresh — no LoRA checkpoint loaded (new LoRA from scratch)")
    elif args.from_checkpoint:
        checkpoint = args.from_checkpoint
        print(f"Using explicit checkpoint: {checkpoint}")
    else:
        checkpoint = find_auto_checkpoint(scenario_dir, args.round)

    print(f"\n{'='*60}")
    print(f"  Lo-Fi LoRA Training")
    print(f"  Scenario : {args.scenario}")
    print(f"  Round    : {args.round}")
    print(f"  Steps    : {args.steps}")
    print(f"  Checkpoint: {Path(checkpoint).name if checkpoint else 'none (fresh LoRA)'}")
    print(f"  Output   : {round_dir}")
    print(f"{'='*60}\n")

    training_cfg = build_training_config(scenario_cfg, base_cfg, checkpoint, round_dir, args.steps)
    config_path = round_dir / "config.yaml"
    save_yaml(config_path, training_cfg)
    print(f"Config written: {config_path}")

    log_path = round_dir / "log.txt"
    env = os.environ.copy()
    env.update(paths_cfg.get("env", {}))

    gpu_fraction = float(os.environ.get("LOFI_GPU_FRACTION", "0.8"))
    cmd = [paths_cfg["python"], paths_cfg["trainer_script"], str(config_path),
           "--gpu-fraction", str(gpu_fraction)]
    print(f"Starting training...\n")

    stop_file = round_dir / "STOP_TRAINING"
    if stop_file.exists():
        stop_file.unlink()

    start_time = datetime.now()
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        stop_event = threading.Event()

        def watch_stop_file():
            while not stop_event.is_set():
                if stop_file.exists():
                    print(f"\n  [metric_watcher] STOP_TRAINING erkannt — beende Training...")
                    process.terminate()
                    return
                time.sleep(15)

        if args.target_score is not None:
            watcher_thread = threading.Thread(target=watch_stop_file, daemon=True)
            watcher_thread.start()

        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        process.wait()
        stop_event.set()
    end_time = datetime.now()

    if process.returncode != 0:
        print(f"\nERROR: Training exited with code {process.returncode}")
        print(f"Log: {log_path}")
        sys.exit(process.returncode)

    notes = save_notes(round_dir, args.scenario, args.round, checkpoint, start_time, end_time, args.steps, log_path)
    print(f"\nTraining complete in {notes['duration_minutes']:.1f} min")
    print(f"Notes saved: {round_dir / 'notes.json'}")

    if not args.no_generate:
        print(f"\nGenerating samples...")
        generate_script = PIPELINE_ROOT / "scripts" / "generate_samples.py"
        subprocess.run([paths_cfg["python"], str(generate_script),
                        "--scenario", args.scenario, "--round", str(args.round)],
                       env=env, check=True)


if __name__ == "__main__":
    main()
