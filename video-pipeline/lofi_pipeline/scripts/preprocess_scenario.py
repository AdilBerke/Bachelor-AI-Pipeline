"""
Preprocess a new scenario: compute VAE latents + T5 embeddings.

Usage:
  python preprocess_scenario.py --scenario rainy_window
  python preprocess_scenario.py --scenario cafe_scene --dataset my_dataset.jsonl
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

PIPELINE_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PIPELINE_ROOT / "configs"
SCENARIOS_DIR = PIPELINE_ROOT / "scenarios"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Preprocess scenario clips for LoRA training")
    parser.add_argument("--scenario", required=True, help="Scenario ID")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset.jsonl (default: scenarios/{scenario}/dataset.jsonl)")
    parser.add_argument("--text-only", action="store_true",
                        help="Nur T5-Embeddings neu berechnen, VAE-Latents beibehalten. "
                             "Nutzen wenn nur der Prompt geändert wurde (spart ~80%% Zeit).")
    args = parser.parse_args()

    scenario_dir = SCENARIOS_DIR / args.scenario
    paths_cfg = load_yaml(CONFIGS_DIR / "model_paths.yaml")

    dataset_jsonl = args.dataset or str(scenario_dir / "dataset.jsonl")
    if not Path(dataset_jsonl).exists():
        print(f"ERROR: dataset.jsonl not found at {dataset_jsonl}")
        print(f"Create it with one JSON per line: {{\"media_path\": \"assets/clip.mp4\", \"caption\": \"...\"}}")
        sys.exit(1)

    precomputed_dir = scenario_dir / "precomputed"
    precomputed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Preprocessing scenario: {args.scenario}")
    print(f"  Dataset: {dataset_jsonl}")
    print(f"  Output:  {precomputed_dir}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env.update(paths_cfg.get("env", {}))

    scenario_cfg = load_yaml(scenario_dir / "scenario.yaml")
    res = scenario_cfg["resolution"]
    frames = scenario_cfg["frames"]
    bucket = f"{res[0]}x{res[1]}x{frames}"

    cmd = [
        paths_cfg["python"], paths_cfg["preprocess_script"],
        str(dataset_jsonl),
        "--output-dir", str(precomputed_dir),
        "--resolution-buckets", bucket,
    ]
    if args.text_only:
        cmd.append("--text-only")
        print("  Modus: Nur T5-Text-Embeddings (VAE-Latents werden beibehalten)")

    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    process.wait()

    if process.returncode == 0:
        print(f"\nPreprocessing complete. Output: {precomputed_dir}")
    else:
        print(f"\nERROR: Preprocessing failed (exit code {process.returncode})")
        sys.exit(process.returncode)


if __name__ == "__main__":
    main()
