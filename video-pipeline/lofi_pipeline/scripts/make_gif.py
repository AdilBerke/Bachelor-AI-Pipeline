"""
Vollständige GIF + MP4 Pipeline.

MP4 (8fps, 960×544)
  → minterpolate (flüssige Bewegung)
  → Real-ESRGAN 4x Anime (schärfer, mehr Detail)
  → GIF  (960×544, 15fps, palette-optimiert)
  → MP4  (1920×1088, 24fps, H.264)

Verwendung:
  python scripts/make_gif.py rounds/round_08/samples/best.mp4
  python scripts/make_gif.py rounds/round_08/samples/best.mp4 --no-upscale
  python scripts/make_gif.py rounds/round_08/samples/best.mp4 --gif-only
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PIPELINE_ROOT  = Path(__file__).parent.parent
MODELS_DIR     = PIPELINE_ROOT / "models"
ESRGAN_MODEL   = Path("/home/BA_Musikproduktion/Documents/Bachelor_VisiualStudio"
                      "/tools/realesrgan-models/RealESRGAN_x4plus_anime_6B.pth")



def clr(c, t): return f"\033[{c}m{t}\033[0m"
def cyan(t):   return clr("36", t)
def green(t):  return clr("32", t)
def dim(t):    return clr("2",  t)
def bold(t):   return clr("1",  t)



def interpolate(src: Path, dst: Path, target_fps: int = 24):
    print(cyan(f"\n[1/3] Frame-Interpolation {src.name} → {target_fps}fps"))
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(r.stderr.decode()[-500:])
        raise RuntimeError("minterpolate fehlgeschlagen")
    print(green(f"  ✓ {dst.name}"))



def upscale_video(src: Path, dst: Path):
    print(cyan(f"\n[2/3] Real-ESRGAN 4x Anime Upscaling"))

    if not ESRGAN_MODEL.exists():
        print(f"  Modell nicht gefunden: {ESRGAN_MODEL}")
        raise FileNotFoundError(f"Real-ESRGAN model missing: {ESRGAN_MODEL}")

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=6, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(ESRGAN_MODEL),
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
        device=device,
    )

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w, out_h = w * 2, h * 2
    print(dim(f"  {w}×{h} → {w*4}×{h*4} → resize {out_w}×{out_h}  |  {total} frames  |  {device}"))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (out_w, out_h))

    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sr, _ = upsampler.enhance(rgb, outscale=2)
        out.write(cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))
        if (i + 1) % 10 == 0 or i == total - 1:
            print(dim(f"  Frame {i+1}/{total}"), end="\r")

    cap.release()
    out.release()
    print()

    cmd = [
        "ffmpeg", "-y", "-i", tmp_path,
        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        str(dst),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    os.unlink(tmp_path)
    print(green(f"  ✓ {dst.name}  ({out_w}×{out_h})"))



def make_gif(src: Path, dst: Path, fps: int = 15, width: int = 960):
    print(cyan(f"\n[3/3] GIF erstellen ({width}px, {fps}fps)"))
    palette_tmp = dst.with_suffix(".palette.png")

    subprocess.run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen=max_colors=256:stats_mode=diff",
        str(palette_tmp),
    ], capture_output=True, check=True)

    subprocess.run([
        "ffmpeg", "-y", "-i", str(src), "-i", str(palette_tmp),
        "-filter_complex",
        f"fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
        "-loop", "0",
        str(dst),
    ], capture_output=True, check=True)

    palette_tmp.unlink(missing_ok=True)

    size_mb = dst.stat().st_size / 1024 / 1024
    print(green(f"  ✓ {dst.name}  ({size_mb:.1f} MB)"))
    if size_mb > 15:
        print(dim(f"  Hinweis: GIF > 15MB — mit --gif-width 640 verkleinern"))



def finalize_mp4(src: Path, dst: Path):
    """Kopiert nur Metadaten neu — kein Re-Encode."""
    subprocess.run([
        "ffmpeg", "-y", "-i", str(src),
        "-c", "copy",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        str(dst),
    ], capture_output=True, check=True)
    size_mb = dst.stat().st_size / 1024 / 1024
    print(green(f"  ✓ {dst.name}  ({size_mb:.1f} MB)"))



def main():
    parser = argparse.ArgumentParser(description="GIF + MP4 Pipeline")
    parser.add_argument("input", help="Eingabe MP4 (aus generate.py)")
    parser.add_argument("--output", help="Ausgabe-Prefix (Standard: input ohne .mp4)")
    parser.add_argument("--no-upscale",  action="store_true", help="Real-ESRGAN überspringen")
    parser.add_argument("--gif-only",    action="store_true", help="Nur GIF, kein MP4")
    parser.add_argument("--gif-fps",     type=int, default=15, help="GIF Framerate (Standard: 15)")
    parser.add_argument("--gif-width",   type=int, default=960, help="GIF Breite in px (Standard: 960)")
    parser.add_argument("--interp-fps",  type=int, default=24, help="Interpolations-FPS (Standard: 24)")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"Datei nicht gefunden: {src}")
        sys.exit(1)

    prefix = Path(args.output) if args.output else src.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    print(bold(f"\n  Lo-Fi GIF Pipeline"))
    print(dim(f"  Eingabe: {src.name}"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        interp_mp4 = tmp / "interp.mp4"
        interpolate(src, interp_mp4, args.interp_fps)

        if args.no_upscale:
            work_mp4 = interp_mp4
        else:
            upscaled_mp4 = tmp / "upscaled.mp4"
            upscale_video(interp_mp4, upscaled_mp4)
            work_mp4 = upscaled_mp4

        gif_out = Path(str(prefix) + "_final.gif")
        make_gif(work_mp4, gif_out, args.gif_fps, args.gif_width)

        if not args.gif_only:
            mp4_out = Path(str(prefix) + "_final.mp4")
            print(cyan(f"\n[3b/3] MP4 finalisieren"))
            finalize_mp4(work_mp4, mp4_out)

    print(bold(f"\n  Fertig!"))
    print(f"  GIF: {gif_out}")
    if not args.gif_only:
        print(f"  MP4: {mp4_out}")


if __name__ == "__main__":
    main()
