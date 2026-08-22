#!/usr/bin/env python3
"""
LTX LoRA v003 Overnight Pipeline
=================================
Step 1 — Download 8-second clips at 480p from YouTube
Step 2 — Quality filter (black bars, motion) + resize to 832x480, trim to 65 frames @ 8fps
Step 3 — Build dataset.jsonl / train.jsonl / validation.jsonl
Step 4 — LTX-Video preprocess (latents + condition embeddings)
Step 5 — LoRA training with checkpoint every 200 steps

Run with: python3 run_overnight.py [--from-step N]
All progress is logged to logs/v003_overnight.log
State persists in daten/processed/ltx_lora_v003/pipeline_state.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


BASE = Path("/home/BA_Musikproduktion/Documents/Bachelor_VisiualStudio")
BACH = BASE

YTDLP        = BASE / ".venv/bin/yt-dlp"
FFMPEG       = Path("/usr/bin/ffmpeg")
FFPROBE      = Path("/usr/bin/ffprobe")
LTX_PYTHON   = BASE / "code/code/tools/ComfyUI/.venv/bin/python"
LTX_PREPROC  = BASE / "code/code/tools/LTX-Video-Trainer/scripts/preprocess_dataset.py"
LTX_TRAIN    = BASE / "code/code/tools/LTX-Video-Trainer/scripts/train.py"
LTX_TRAINER_SRC = BASE / "code/code/tools/LTX-Video-Trainer/src"

DATASET_DIR  = BACH / "daten/processed/ltx_lora_v003"
CLIPS_RAW    = DATASET_DIR / "clips_raw"
ASSETS_DIR   = DATASET_DIR / "assets"
PRECOMPUTED  = DATASET_DIR / "precomputed"
TRAIN_OUT    = BACH / "training/video/ltx_lora_v003"
LOG_FILE     = BACH / "logs/v003_overnight.log"
STATE_FILE   = DATASET_DIR / "pipeline_state.json"
CONFIG_FILE  = DATASET_DIR / "ltx_lora_v003.yaml"
DATASET_JSONL = DATASET_DIR / "dataset.jsonl"
TRAIN_JSONL  = DATASET_DIR / "train.jsonl"
VAL_JSONL    = DATASET_DIR / "validation.jsonl"


TARGET_W     = 832
TARGET_H     = 480
TARGET_FPS   = 8
TARGET_FRAMES = 65
TARGET_SECS  = TARGET_FRAMES / TARGET_FPS
DOWNLOAD_SECS = 10
BUCKET       = f"{TARGET_W}x{TARGET_H}x{TARGET_FRAMES}"

TRIGGER_TOKEN = "lofi_loop"
MODEL_SOURCE  = "LTXV_2B_0.9.6_DEV"


YOUTUBE_SOURCES = [
    ("rIFDqCYMOAQ", "Japanese Street Night — rainy japanese night street, neon sign reflections on wet cobblestones, dark moody blue tones, soft lantern light, peaceful quiet atmosphere"),
    ("8EM7btM7-XQ", "Rainy Night by the Fireplace — cozy indoor scene with fireplace glow, sleeping cat on wooden floor, warm amber lighting, rain on window"),
    ("oFm6rbA5BD8", "night at the bookstore — anime illustrated bookstore interior, warm reading lamp, bookshelves, cozy night atmosphere, subtle ambient motion"),
    ("lTRiuFIWV54", "1 A.M Study Session — anime lofi girl sitting at wooden desk studying late at night, warm amber lamp glowing softly, books stacked"),
    ("TURbeWK2wwg", "4 A.M Study Session — anime lofi girl studying at desk before dawn, soft lamp light, quiet cozy room, pencil and notebook"),
    ("wAPCSnAhhC8", "2 A.M Study Session — cozy bedroom desk scene, lofi girl writing, warm indoor lighting, night outside window"),
    ("l98w9OSKVNA", "12 A.M Study Session — midnight study room, warm lamp on desk, anime girl focused on books, calm peaceful atmosphere"),
    ("odhh4sInQW0", "Tokyo Japanese Lofi — tokyo cityscape at night, illustrated japanese city lights, warm glowing windows, peaceful urban scene"),
    ("FZS0CVd6XD8", "Kamui Japanese Lofi — japanese style illustrated night scene, cherry blossom, soft glowing lanterns, serene atmosphere"),
    ("STeUOTUC0-k", "Midnight Serenity Japanese Village — japanese village at midnight, soft moonlight, traditional wooden houses, quiet ambient scene"),
    ("MOl4s-VIuLQ", "Rainy Kyoto Night — illustrated Kyoto street at night, rain on cobblestones, neon-lit shop signs, warm cozy ambiance"),
    ("CIgeJPv5wLI", "Cowboy Bebop Lofi — anime illustrated jazz cafe interior, warm sepia tones, vintage instruments, nostalgic atmosphere"),
    ("N3ur5Ey21zg", "Studio Ghibli Lofi — ghibli-inspired illustrated countryside scene, soft pastel colors, gentle wind through tall grass"),
    ("D8KDOutFVOs", "cant sleep lofi jazzhop — lofi bedroom at night, soft blue moonlight through window, cozy sheets, calm reflective mood"),
    ("NDfF_XwNtIw", "4am cant sleep lofi — quiet dark room at 4am, dim nightstand lamp, anime character gazing at ceiling, peaceful night"),
    ("hzpt3fQjY9U", "3am why so sad lofi — lofi girl at rainy window at 3am, neon reflections on glass, melancholy cozy atmosphere"),
    ("TtdaSQDc5XU", "Retro Rooftop Tokyo — retro illustrated Tokyo rooftop at night, city lights skyline, warm lamp, relaxing city view"),
    ("rjYm7L9B7R0", "Night Balcony Lofi — lofi character on balcony at night, city lights below, gentle breeze, peaceful urban evening"),
    ("dusfmL_bm1o", "ZETSU Japanese Lofi — japanese illustrated shrine path at dusk, stone lanterns glowing, misty atmospheric scene"),
    ("goxmvGJkoi0", "Kami Japanese Lofi — japanese spirit forest illustration, soft glowing particles, misty bamboo forest, serene mystical atmosphere"),
    ("y0XC_Q0sjzg", "nights at the record shop — vintage record shop interior at night, warm lamp, vinyl records, cozy nostalgic atmosphere"),
    ("SMfHacO_o38", "Rainy Jazz Lofi Cozy Study — cozy library with rain on window, warm reading lamp, open books, soft jazz atmosphere"),
    ("YMZHqONealk", "Night Chill Lofi Jazz Hip Hop Cafe — cozy japanese cafe interior, warm pendant lights, steam from coffee cup, rainy window"),
    ("WJQ4Z2tq1VY", "Night Chill Lofi Jazzy Beats — late night cafe scene, warm indoor glow against dark city outside, relaxed cozy vibe"),
    ("AMcVJmb5mvk", "90s Japanese Lofi Hiphop — 90s japanese city at night, neon signs, retro aesthetics, warm glowing shops, nostalgic chill mood"),
    ("Sdty5B8BP4k", "Tokyo Lofi Hiphop Night — tokyo night skyline illustration, glowing city windows, peaceful urban nightscape, chill vibe"),
    ("Q74YUAxqs00", "Tokyo Lofi Hiphop — illustrated tokyo daytime cafe, cherry blossoms outside window, peaceful study atmosphere"),
    ("FJY2Mn4vu_M", "Snowy Japanese Lofi — japanese village under snowfall, warm glowing windows, peaceful winter night scene"),
    ("rt1mRnRp79A", "coffee beats jazzy japan lofi — japanese coffee shop interior, warm soft lighting, steam from cup, cozy morning scene"),
    ("qUnsNVJUJbM", "Tokyo After Midnight Lofi Rain — tokyo street after midnight with rain, reflective wet pavement, peaceful empty streets"),
    ("Ll9fp4RnElo", "Sleepy rainy nights Lofi cat — cozy room with sleeping cat, rain sounds outside, warm blanket scene, soft indoor light"),
    ("N7JCXQJyBE4", "SLEEPY Lofi Cat — lofi cat sleeping on windowsill, moonlight, rain drops on glass, cozy peaceful night"),
    ("tIMtzkZ93gg", "cozy reading fort — pillow fort with fairy lights, open book, warm glow, cozy safe corner, soft ambient light"),
    ("SuzmYTFGBuQ", "Lofi Zen Kyoto Sake Bar Rain — kyoto sake bar at rainy night, warm paper lanterns, wooden interior, serene japanese atmosphere"),
    ("9F0Kzbni5R0", "Lofi Zen Nestled Above the City Rain — cozy apartment above city lights, rain on window, warm indoor scene"),
    ("34c0xTj8N5o", "Radio rainy days Lofi — rainy day radio scene, vintage radio on wooden desk, soft lamp light, peaceful indoor atmosphere"),
    ("KxpViLLzlrc", "Chillhop Hideaway cozy beats — illustrated cozy hideaway with warm lantern light, nature outside, serene retreat"),
    ("4BYuBc3-MLs", "Pumpkin Companions Lofi Autumn — cozy autumn room with pumpkins, warm orange candlelight, halloween lofi aesthetic"),
    ("MtT5_PgLJlY", "CHILL VIBES Simpson Lofi — simpson-style illustrated living room, cozy couch, warm tv glow, nostalgic domestic scene"),
    ("3Rl4btmmkf0", "Chill And Sleep Lofi cat — cozy bed with cat, soft moonlight, peaceful sleeping atmosphere, gentle night scene"),
]

TIMESTAMPS = [30, 300, 900, 1800]


def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

log = logging.getLogger(__name__)


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"completed_steps": [], "downloaded_clips": [], "accepted_clips": []}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))



def make_clip_filename(video_id: str, start_sec: int) -> str:
    return f"{video_id}_s{start_sec:04d}_d{DOWNLOAD_SECS}.mp4"


def download_clips(state: dict):
    log.info("=" * 60)
    log.info("STEP 1 — Download 8-second clips from YouTube")
    log.info("=" * 60)

    already = set(state.get("downloaded_clips", []))
    CLIPS_RAW.mkdir(parents=True, exist_ok=True)

    total_attempts = len(YOUTUBE_SOURCES) * len(TIMESTAMPS)
    done = 0
    new = 0

    for video_id, _desc in YOUTUBE_SOURCES:
        for start in TIMESTAMPS:
            clip_name = make_clip_filename(video_id, start)
            clip_path = CLIPS_RAW / clip_name

            if clip_name in already or clip_path.exists():
                done += 1
                continue

            end = start + DOWNLOAD_SECS
            url = f"https://www.youtube.com/watch?v={video_id}"
            out_tmpl = str(CLIPS_RAW / f"{video_id}_s{start:04d}_d{DOWNLOAD_SECS}.%(ext)s")

            cmd = [
                str(YTDLP),
                "--format", "bestvideo[height>=360][height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height>=360][height<=720]+bestaudio/best[height>=360][height<=720]/best",
                "--match-filter", "!is_live",
                "--download-sections", f"*{start}-{end}",
                "--force-keyframes-at-cuts",
                "--no-playlist",
                "--output", out_tmpl,
                "--merge-output-format", "mp4",
                "--postprocessor-args", "ffmpeg:-an",
                "--quiet",
                "--no-warnings",
                url,
            ]

            log.info(f"Downloading {video_id} @ {start}s … ({done+1}/{total_attempts})")
            try:
                result = subprocess.run(cmd, timeout=120, capture_output=True, text=True)
                if result.returncode == 0 and clip_path.exists():
                    log.info(f"  ✓ {clip_name}")
                    state["downloaded_clips"].append(clip_name)
                    save_state(state)
                    new += 1
                else:
                    webm_path = CLIPS_RAW / f"{video_id}_s{start:04d}_d{DOWNLOAD_SECS}.webm"
                    mkv_path  = CLIPS_RAW / f"{video_id}_s{start:04d}_d{DOWNLOAD_SECS}.mkv"
                    found = next((p for p in [webm_path, mkv_path] if p.exists()), None)
                    if found:
                        log.info(f"  Converting {found.name} → mp4")
                        conv = subprocess.run(
                            [str(FFMPEG), "-y", "-i", str(found), "-c:v", "libx264", "-an", "-t", str(DOWNLOAD_SECS), str(clip_path)],
                            capture_output=True, timeout=60
                        )
                        found.unlink(missing_ok=True)
                        if clip_path.exists():
                            state["downloaded_clips"].append(clip_name)
                            save_state(state)
                            new += 1
                        else:
                            log.warning(f"  ✗ conversion failed for {video_id}@{start}s")
                    else:
                        err = result.stderr[:200] if result.stderr else "(no stderr)"
                        log.warning(f"  ✗ {video_id}@{start}s: {err}")
            except subprocess.TimeoutExpired:
                log.warning(f"  ✗ timeout {video_id}@{start}s")
            except Exception as e:
                log.warning(f"  ✗ error {video_id}@{start}s: {e}")

            done += 1

    log.info(f"Download step complete — {new} new clips, {done} total attempts")
    log.info(f"Clips in {CLIPS_RAW}: {len(list(CLIPS_RAW.glob('*.mp4')))}")
    state["completed_steps"] = list(set(state.get("completed_steps", [])) | {"download"})
    save_state(state)



def probe_video(path: Path) -> Optional[dict]:
    try:
        r = subprocess.run(
            [str(FFPROBE), "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode != 0:
            return None
        d = json.loads(r.stdout)
        vs = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
        if not vs:
            return None
        w = int(vs.get("width", 0))
        h = int(vs.get("height", 0))
        fps_str = vs.get("r_frame_rate", "25/1")
        num, den = (int(x) for x in fps_str.split("/"))
        fps = num / den if den else 25
        dur = float(d.get("format", {}).get("duration", 0))
        nb_frames = vs.get("nb_frames")
        frames = int(nb_frames) if nb_frames and nb_frames != "N/A" else int(dur * fps)
        return {"width": w, "height": h, "fps": fps, "duration": dur, "frames": frames}
    except Exception:
        return None


def detect_black_bars(path: Path) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) of detected content area via cropdetect."""
    try:
        r = subprocess.run(
            [str(FFMPEG), "-i", str(path), "-vf", "cropdetect=24:16:0", "-frames:v", "80", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30
        )
        matches = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
        if not matches:
            return (0, 0, 0, 0)
        counts = {}
        for m in matches:
            key = tuple(int(x) for x in m)
            counts[key] = counts.get(key, 0) + 1
        best = max(counts, key=counts.__getitem__)
        return best
    except Exception:
        return (0, 0, 0, 0)


def compute_motion_score(path: Path) -> float:
    """Average per-frame pixel difference (0 = static, 1 = fully dynamic)."""
    try:
        r = subprocess.run(
            [str(FFMPEG), "-i", str(path),
             "-vf", "scale=128:72,select='not(mod(n\\,4))',setpts=N/FRAME_RATE/TB,signalstats",
             "-frames:v", "20", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30
        )
        r2 = subprocess.run(
            [str(FFMPEG), "-i", str(path),
             "-vf", "scale=160:90,select=gt(scene\\,0.005),metadata=print:file=-",
             "-frames:v", "60", "-vsync", "vfr", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30
        )
        motion_frames = r2.stdout.count("lavfi.scene_score")
        return min(1.0, motion_frames / 10.0)
    except Exception:
        return 0.5


def quality_score(path: Path, info: dict) -> tuple[float, str]:
    """Return (score 0-5, reason) for a raw clip."""
    w, h = info["width"], info["height"]

    if h < 240 or w < 320:
        return 0.0, "resolution too low"

    if info["frames"] < TARGET_FRAMES:
        return 0.0, f"too short: {info['frames']} frames < {TARGET_FRAMES}"

    crop_w, crop_h, crop_x, crop_y = detect_black_bars(path)
    if crop_w > 0 and crop_h > 0:
        bar_fraction_w = 1 - crop_w / w
        bar_fraction_h = 1 - crop_h / h
        if bar_fraction_w > 0.25 or bar_fraction_h > 0.25:
            return 1.0, f"heavy black bars (crop={crop_w}x{crop_h} from {w}x{h})"
    else:
        bar_fraction_w = bar_fraction_h = 0.0

    motion = compute_motion_score(path)
    if motion < 0.05:
        return 0.5, "nearly static"

    res_score = min(1.0, min(w, h * 832 / 480) / 832)
    score = 2.0 + res_score * 1.5 + motion * 1.5
    score -= bar_fraction_w * 2 + bar_fraction_h * 2
    return min(5.0, max(0.0, score)), "ok"


def resize_clip(src: Path, dst: Path) -> bool:
    """Resize to 832x480, trim to TARGET_FRAMES at TARGET_FPS, no audio."""
    cmd = [
        str(FFMPEG), "-y",
        "-i", str(src),
        "-vf", (
            f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
            f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:black,"
            f"fps={TARGET_FPS}"
        ),
        "-frames:v", str(TARGET_FRAMES),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-an",
        str(dst),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        return r.returncode == 0 and dst.exists() and dst.stat().st_size > 10_000
    except Exception:
        return False


def make_caption(video_id: str, desc: str) -> str:
    """Build a lofi training caption from the source description."""
    if " — " in desc:
        scene = desc.split(" — ", 1)[1]
    else:
        scene = desc
    return f"{TRIGGER_TOKEN}, {scene}, calm camera, stable composition, subtle loop motion, 16:9"


def prepare_assets(state: dict):
    log.info("=" * 60)
    log.info("STEP 2 — Quality filter + resize clips")
    log.info("=" * 60)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    source_map = {vid: desc for vid, desc in YOUTUBE_SOURCES}
    accepted = set(state.get("accepted_clips", []))
    rows = []

    raw_clips = sorted(CLIPS_RAW.glob("*.mp4"))
    log.info(f"Raw clips to evaluate: {len(raw_clips)}")

    for raw in raw_clips:
        stem = raw.stem
        asset_name = stem + ".mp4"
        asset_path = ASSETS_DIR / asset_name

        parts = stem.rsplit("_s", 1)
        video_id = parts[0] if len(parts) == 2 else stem

        if asset_name in accepted and asset_path.exists():
            cap = make_caption(video_id, source_map.get(video_id, video_id))
            rows.append({"media_path": f"assets/{asset_name}", "caption": cap,
                         "video_id": video_id, "source_file": str(raw)})
            continue

        info = probe_video(raw)
        if not info:
            log.warning(f"  ✗ probe failed: {raw.name}")
            continue

        score, reason = quality_score(raw, info)
        log.info(f"  {raw.name}: score={score:.1f} ({reason})")

        if score < 1.5:
            log.info(f"    → rejected")
            continue

        ok = resize_clip(raw, asset_path)
        if not ok:
            log.warning(f"  ✗ resize failed: {raw.name}")
            continue

        cap = make_caption(video_id, source_map.get(video_id, video_id))
        rows.append({"media_path": f"assets/{asset_name}", "caption": cap,
                     "video_id": video_id, "source_file": str(raw)})
        accepted.add(asset_name)
        state["accepted_clips"] = list(accepted)
        save_state(state)
        log.info(f"    ✓ accepted → {asset_name}")

    log.info(f"Assets accepted: {len(rows)}")
    state["accepted_count"] = len(rows)
    state["completed_steps"] = list(set(state.get("completed_steps", [])) | {"prepare"})
    save_state(state)
    return rows



def build_dataset(rows: list[dict], state: dict):
    log.info("=" * 60)
    log.info("STEP 3 — Build dataset.jsonl / train.jsonl / validation.jsonl")
    log.info("=" * 60)

    if not rows:
        log.error("No accepted clips — cannot build dataset")
        sys.exit(1)

    import random
    random.seed(42)
    random.shuffle(rows)

    n_val = max(2, len(rows) // 6)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    def write_jsonl(path: Path, data: list[dict]):
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data) + "\n",
                        encoding="utf-8")

    write_jsonl(DATASET_JSONL, rows)
    write_jsonl(TRAIN_JSONL, train_rows)
    write_jsonl(VAL_JSONL, val_rows)

    log.info(f"Dataset: {len(rows)} total, {len(train_rows)} train, {len(val_rows)} val")
    state["dataset_counts"] = {"total": len(rows), "train": len(train_rows), "val": len(val_rows)}
    state["completed_steps"] = list(set(state.get("completed_steps", [])) | {"dataset"})
    save_state(state)



def run_preprocess(state: dict):
    log.info("=" * 60)
    log.info("STEP 4 — LTX preprocess (latents + condition embeddings)")
    log.info("=" * 60)

    PRECOMPUTED.mkdir(parents=True, exist_ok=True)

    latents_dir = PRECOMPUTED / "latents" / "assets"
    if latents_dir.exists():
        n_done = len(list(latents_dir.glob("*.pt")))
        n_total = sum(1 for _ in DATASET_JSONL.read_text().splitlines() if _.strip())
        if n_done >= n_total:
            log.info(f"Precompute already done ({n_done}/{n_total}) — skipping")
            state["completed_steps"] = list(set(state.get("completed_steps", [])) | {"preprocess"})
            save_state(state)
            return

    env = {**os.environ,
           "PYTHONPATH": str(LTX_TRAINER_SRC),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    cmd = [
        str(LTX_PYTHON),
        str(LTX_PREPROC),
        str(DATASET_JSONL),
        "--resolution-buckets", BUCKET,
        "--caption-column", "caption",
        "--video-column", "media_path",
        "--output-dir", str(PRECOMPUTED),
        "--model-source", MODEL_SOURCE,
        "--id-token", TRIGGER_TOKEN,
        "--vae-tiling",
    ]

    log.info("Running: " + " ".join(cmd))
    log.info(f"Working directory: {DATASET_DIR}")

    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(DATASET_DIR),
        timeout=7200,
    )

    if result.returncode != 0:
        log.error(f"Preprocess failed with code {result.returncode}")
        _fix_caption_embeddings()
        sys.exit(1)

    _fix_caption_embeddings()

    state["completed_steps"] = list(set(state.get("completed_steps", [])) | {"preprocess"})
    save_state(state)
    log.info("Preprocess complete")


def _fix_caption_embeddings():
    """Move any misplaced caption .pt files to the correct conditions/assets/ dir."""
    conditions_assets = PRECOMPUTED / "conditions" / "assets"
    conditions_assets.mkdir(parents=True, exist_ok=True)

    for pt in PRECOMPUTED.rglob("*.pt"):
        if conditions_assets in pt.parents:
            continue
        target = conditions_assets / pt.name
        if not target.exists():
            log.info(f"  Moving condition embedding: {pt.name} → conditions/assets/")
            shutil.move(str(pt), str(target))



def find_latest_checkpoint() -> Optional[Path]:
    ckpt_dir = TRAIN_OUT / "checkpoints"
    if not ckpt_dir.exists():
        return None
    files = sorted(ckpt_dir.glob("lora_weights_step_*.safetensors"))
    return files[-1] if files else None


def update_config_checkpoint(checkpoint: Optional[Path]):
    """Patch load_checkpoint in the yaml config."""
    if not CONFIG_FILE.exists():
        return
    text = CONFIG_FILE.read_text()
    val = f'"{checkpoint}"' if checkpoint else "null"
    text = re.sub(r"load_checkpoint:.*", f"load_checkpoint: {val}", text)
    CONFIG_FILE.write_text(text)


def run_training(state: dict):
    log.info("=" * 60)
    log.info("STEP 5 — LoRA training (2000 steps, checkpoint every 200)")
    log.info("=" * 60)

    TRAIN_OUT.mkdir(parents=True, exist_ok=True)

    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt:
        log.info(f"Resuming from checkpoint: {latest_ckpt.name}")
        update_config_checkpoint(latest_ckpt)
    else:
        log.info("Starting training from scratch")
        update_config_checkpoint(None)

    env = {**os.environ,
           "PYTHONPATH": str(LTX_TRAINER_SRC),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    cmd = [str(LTX_PYTHON), str(LTX_TRAIN), str(CONFIG_FILE)]

    log.info("Running: " + " ".join(cmd))
    result = subprocess.run(cmd, env=env, timeout=28800)

    if result.returncode != 0:
        log.error(f"Training failed with code {result.returncode}")
        sys.exit(1)

    state["completed_steps"] = list(set(state.get("completed_steps", [])) | {"train"})
    save_state(state)
    log.info("Training complete!")

    ckpts = sorted((TRAIN_OUT / "checkpoints").glob("*.safetensors"))
    log.info(f"Saved {len(ckpts)} checkpoints:")
    for c in ckpts:
        log.info(f"  {c.name}")



def main():
    parser = argparse.ArgumentParser(description="LTX LoRA v003 overnight pipeline")
    parser.add_argument("--from-step", type=int, default=1,
                        help="Start from step N (1=download, 2=prepare, 3=dataset, 4=preprocess, 5=train)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download step (use existing clips_raw)")
    args = parser.parse_args()

    setup_logging()
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║    LTX LoRA v003 Overnight Pipeline              ║")
    log.info("║    832x480 @ 8fps, 65 frames, 2000 steps         ║")
    log.info("╚══════════════════════════════════════════════════╝")

    state = load_state()
    completed = set(state.get("completed_steps", []))

    from_step = args.from_step

    if from_step <= 1 and "download" not in completed and not args.skip_download:
        download_clips(state)
    else:
        log.info("Step 1 (download) — skipped")

    if from_step <= 2:
        rows = prepare_assets(state)
    else:
        rows = []
        source_map = {vid: desc for vid, desc in YOUTUBE_SOURCES}
        for asset_name in state.get("accepted_clips", []):
            asset_path = ASSETS_DIR / asset_name
            if asset_path.exists():
                stem = Path(asset_name).stem
                parts = stem.rsplit("_s", 1)
                video_id = parts[0] if len(parts) == 2 else stem
                cap = make_caption(video_id, source_map.get(video_id, video_id))
                rows.append({"media_path": f"assets/{asset_name}", "caption": cap,
                             "video_id": video_id, "source_file": str(CLIPS_RAW / asset_name)})
        log.info(f"Step 2 (prepare) — skipped, using {len(rows)} existing assets")

    if from_step <= 3 and "dataset" not in completed:
        build_dataset(rows, state)
    else:
        log.info("Step 3 (dataset) — skipped")

    if from_step <= 4 and "preprocess" not in completed:
        run_preprocess(state)
    else:
        log.info("Step 4 (preprocess) — skipped")

    if from_step <= 5 and "train" not in completed:
        run_training(state)
    else:
        log.info("Step 5 (train) — skipped")

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║    Pipeline complete!                            ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info(f"LoRA weights: {TRAIN_OUT}/checkpoints/")
    log.info(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
