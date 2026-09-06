"""
Post-process generated MP4s: upscale + frame interpolation for smoother HD output.

Methods:
  minterpolate (default): ffmpeg motion-compensated interpolation + lanczos upscale
  rife: GPU optical flow interpolation (much smoother for anime) + Real-ESRGAN upscale

Usage:
  python enhance_video.py --scenario rainy_window --round 8 --checkpoint 100
  python enhance_video.py --scenario rainy_window --round 8 --target-fps 60
  python enhance_video.py --scenario rainy_window --round 8 --method rife
  python enhance_video.py --input step_00100_0.mp4 --method rife --target-fps 60
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent.parent
SCENARIOS_DIR = PIPELINE_ROOT / "scenarios"
TOOLS_ROOT = PIPELINE_ROOT.parent.parent / "tools"
_RIFE_DIR = TOOLS_ROOT / "rife-ncnn-vulkan" / "rife-ncnn-vulkan-20221029-ubuntu"
RIFE_BIN = _RIFE_DIR / "rife-ncnn-vulkan"
RIFE_MODEL = _RIFE_DIR / "rife-v4"   # supports custom frame count (-n)
ESRGAN_MODEL = TOOLS_ROOT / "realesrgan-models" / "RealESRGAN_x4plus_anime_6B.pth"


def enhance_minterpolate(input_mp4: Path, output_mp4: Path, scale: int, target_fps: int) -> bool:
    """ffmpeg minterpolate + lanczos upscale."""
    w = 832 * scale
    h = 480 * scale
    vf = (
        f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        f":me=umh:mb_size=8:search_param=64:scd=none,"
        f"scale={w}:{h}:flags=lanczos+accurate_rnd"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(input_mp4),
        "-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p", str(output_mp4),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-400:]}")
        return False
    return True


def enhance_rife(input_mp4: Path, output_mp4: Path, scale: int, target_fps: int) -> bool:
    """RIFE optical flow interpolation + Real-ESRGAN anime upscale."""
    if not RIFE_BIN.exists():
        print(f"  ERROR: RIFE binary not found at {RIFE_BIN}")
        print(f"  Fallback: using minterpolate")
        return enhance_minterpolate(input_mp4, output_mp4, scale, target_fps)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames_in = tmp / "frames_in"
        frames_rife = tmp / "frames_rife"
        frames_esrgan = tmp / "frames_esrgan"
        frames_in.mkdir()
        frames_rife.mkdir()
        frames_esrgan.mkdir()

        # Step 1: Extract frames from input (8fps native)
        print(f"  [1/4] Extracting frames...")
        r = subprocess.run([
            "ffmpeg", "-y", "-i", str(input_mp4),
            "-vf", "fps=8", str(frames_in / "%08d.png")
        ], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ERROR extracting frames: {r.stderr[-200:]}")
            return False

        n_frames = len(list(frames_in.glob("*.png")))
        # RIFE doubles frames per pass; run multiple passes to reach target_fps
        multiplier = target_fps // 8  # e.g. 60fps → multiplier=7 → use -n
        target_n = n_frames * multiplier

        # Step 2: RIFE interpolation
        print(f"  [2/4] RIFE interpolation ({n_frames} → {target_n} frames)...")
        rife_model = RIFE_MODEL if RIFE_MODEL.exists() else (_RIFE_DIR / "rife-v4")
        r = subprocess.run([
            str(RIFE_BIN),
            "-i", str(frames_in),
            "-o", str(frames_rife),
            "-m", str(rife_model),
            "-n", str(target_n),
            "-f", "%08d.png",
        ], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  RIFE failed ({r.stderr[-300:]}), trying minterpolate fallback...")
            return enhance_minterpolate(input_mp4, output_mp4, scale, target_fps)

        # Step 3: Real-ESRGAN upscale (anime model)
        if ESRGAN_MODEL.exists() and scale >= 2:
            print(f"  [3/4] Real-ESRGAN upscale (x{scale})...")
            try:
                import torch
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer

                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_block=6, num_grow_ch=32, scale=4)
                upsampler = RealESRGANer(
                    scale=4, model_path=str(ESRGAN_MODEL), model=model,
                    tile=512, tile_pad=10, pre_pad=0, half=True,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                )
                import cv2
                import numpy as np
                for frame_path in sorted(frames_rife.glob("*.png")):
                    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                    out, _ = upsampler.enhance(img, outscale=scale)
                    cv2.imwrite(str(frames_esrgan / frame_path.name), out)
                upscale_frames = frames_esrgan
            except Exception as e:
                print(f"  ESRGAN failed ({e}), using lanczos fallback...")
                upscale_frames = frames_rife
        else:
            upscale_frames = frames_rife

        # Step 4: Reassemble video
        w = 832 * scale
        h = 480 * scale
        print(f"  [4/4] Assembling {w}x{h} @ {target_fps}fps...")
        scale_filter = "" if ESRGAN_MODEL.exists() else f",scale={w}:{h}:flags=lanczos"
        r = subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(target_fps),
            "-i", str(upscale_frames / "%08d.png"),
            "-vf", f"fps={target_fps}{scale_filter}",
            "-c:v", "libx264", "-preset", "slow", "-crf", "16",
            "-pix_fmt", "yuv420p", str(output_mp4),
        ], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ERROR assembling: {r.stderr[-200:]}")
            return False

    return True


def enhance(input_mp4: Path, output_mp4: Path, scale: int, target_fps: int, method: str) -> bool:
    w = 832 * scale
    h = 480 * scale
    print(f"  Enhancing: {input_mp4.name}  [{method}]")
    print(f"  → {output_mp4.name}  ({w}x{h} @ {target_fps}fps)")

    if method == "rife":
        ok = enhance_rife(input_mp4, output_mp4, scale, target_fps)
    else:
        ok = enhance_minterpolate(input_mp4, output_mp4, scale, target_fps)

    if ok and output_mp4.exists():
        size_mb = output_mp4.stat().st_size / 1024 / 1024
        print(f"  Done — {size_mb:.1f} MB")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Upscale + interpolate LoRA output videos")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--scale", type=int, default=2,
                        help="Upscale factor (default: 2 → 1664x960)")
    parser.add_argument("--target-fps", type=int, default=24,
                        help="Target FPS (default: 24)")
    parser.add_argument("--method", type=str, default="minterpolate",
                        choices=["minterpolate", "rife"],
                        help="Interpolation method: minterpolate (default) or rife (GPU, better for anime)")
    args = parser.parse_args()

    if args.input:
        inp = Path(args.input)
        if not inp.exists():
            print(f"ERROR: {inp} not found")
            sys.exit(1)
        suffix = f"_hd{args.scale}x_{args.target_fps}fps_{args.method}"
        out = inp.with_name(inp.stem + suffix + ".mp4")
        ok = enhance(inp, out, args.scale, args.target_fps, args.method)
        sys.exit(0 if ok else 1)

    if not args.scenario or args.round is None:
        print("ERROR: provide --input OR --scenario + --round")
        sys.exit(1)

    samples_dir = SCENARIOS_DIR / args.scenario / "rounds" / f"round_{args.round:02d}" / "samples"
    if not samples_dir.exists():
        print(f"ERROR: {samples_dir} not found")
        sys.exit(1)

    if args.checkpoint:
        mp4s = [samples_dir / f"step_{args.checkpoint:05d}_0.mp4"]
    else:
        mp4s = sorted(p for p in samples_dir.glob("step_*_0.mp4")
                      if "_hd" not in p.stem)

    if not mp4s:
        print(f"ERROR: No MP4s found in {samples_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Enhancing {len(mp4s)} video(s)  [{args.method}]")
    print(f"  832x480@8fps → {832*args.scale}x{480*args.scale}@{args.target_fps}fps")
    print(f"{'='*60}\n")

    ok_count = 0
    for mp4 in mp4s:
        suffix = f"_hd{args.scale}x_{args.target_fps}fps_{args.method}"
        out = mp4.with_name(mp4.stem + suffix + ".mp4")
        if enhance(mp4, out, args.scale, args.target_fps, args.method):
            ok_count += 1

    print(f"\nDone. {ok_count}/{len(mp4s)} enhanced.")
    print(f"Output in: {samples_dir}")


if __name__ == "__main__":
    main()
