#!/bin/bash
# Testmatrix: Realistischer Hase am See
# Startet Tests A, B, B2 sequentiell (Tests C/D/G brauchen ein Referenzbild — separate Anleitung)
#
# Verwendung:
#   bash run_tests.sh          → alle automatischen Tests (A, B, B2)
#   bash run_tests.sh B        → nur Test B
#   bash run_tests.sh C ref.jpg → Test C mit Referenzbild

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/BA_Musikproduktion/Documents/Bachelor_VisiualStudio/.venv/bin/python"
GEN="$SCRIPT_DIR/generate_realistic.py"
MAKE_GIF="/home/BA_Musikproduktion/Documents/Bachelor_VisiualStudio/Bachelorarbeit/lofi_pipeline/scripts/make_gif.py"
RIFE_BIN="/home/BA_Musikproduktion/Documents/Bachelor_VisiualStudio/tools/rife-ncnn-vulkan/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan"
RIFE_MODEL_DIR="/home/BA_Musikproduktion/Documents/Bachelor_VisiualStudio/tools/rife-ncnn-vulkan/rife-ncnn-vulkan-20221029-ubuntu"
LORA_R8="/home/BA_Musikproduktion/Documents/Bachelor_VisiualStudio/Bachelorarbeit/lofi_pipeline/scenarios/rabbit_lake/rounds/round_08/checkpoints/lora_weights_step_00150.safetensors"
EVAL="$SCRIPT_DIR/evaluate_video.py"
RESULTS="$SCRIPT_DIR/results.json"

cd "$SCRIPT_DIR"

# Automatische Metriken nach jeder Generierung
evaluate_after() {
    local mp4="$1"
    local test_id="$2"
    echo ""
    echo "  [Metriken] $mp4"
    $PYTHON "$EVAL" "$mp4" --save-report --update-results "$RESULTS" --test-id "$test_id" 2>/dev/null
}

run_test_A() {
    echo ""
    echo "=== TEST A: Baseline mit Lo-Fi LoRA R8 (~15 min, 960×544) ==="
    mkdir -p outputs/test_A
    $PYTHON "$GEN" \
        --lora "$LORA_R8" \
        --width 960 --height 544 --frames 97 \
        --steps 80 --guidance 9.0 --seed 777 \
        --output outputs/test_A/rabbit_lofi_r8.mp4
    evaluate_after outputs/test_A/rabbit_lofi_r8.mp4 A
    echo "Test A fertig → outputs/test_A/rabbit_lofi_r8.mp4"
}

run_test_B() {
    echo ""
    echo "=== TEST B: Text-zu-Video 768×432 (~7 min) ==="
    mkdir -p outputs/test_B
    $PYTHON "$GEN" \
        --width 768 --height 432 --frames 49 \
        --steps 50 --guidance 7.0 --seed 42 \
        --output outputs/test_B/rabbit_t2v_768x432.mp4
    evaluate_after outputs/test_B/rabbit_t2v_768x432.mp4 B
    echo "Test B fertig → outputs/test_B/rabbit_t2v_768x432.mp4"
    echo ""
    echo "Extrahiere Frame 1 als Referenzbild für Tests C und D..."
    ffmpeg -y -i outputs/test_B/rabbit_t2v_768x432.mp4 -vframes 1 -q:v 2 ref_rabbit_from_B.jpg 2>/dev/null
    echo "  → ref_rabbit_from_B.jpg (nutze dieses oder eigenes Foto als ref_rabbit.jpg)"
}

run_test_B2() {
    echo ""
    echo "=== TEST B2: Text-zu-Video 1280×720 (~22 min) ==="
    mkdir -p outputs/test_B2
    $PYTHON "$GEN" \
        --width 1280 --height 720 --frames 49 \
        --steps 60 --guidance 7.0 --seed 42 \
        --output outputs/test_B2/rabbit_t2v_1280x720.mp4
    evaluate_after outputs/test_B2/rabbit_t2v_1280x720.mp4 B2
    echo "Test B2 fertig → outputs/test_B2/rabbit_t2v_1280x720.mp4"
}

run_test_C() {
    local ref="${1:-ref_rabbit.jpg}"
    echo ""
    echo "=== TEST C: Bild-zu-Video 960×544 Standard (~11 min) ==="
    echo "  Referenzbild: $ref"
    mkdir -p outputs/test_C
    $PYTHON "$GEN" \
        --image "$ref" --image-noise-scale 0.15 \
        --width 960 --height 544 --frames 49 \
        --steps 50 --guidance 6.0 --seed 42 \
        --output outputs/test_C/rabbit_i2v_standard.mp4
    evaluate_after outputs/test_C/rabbit_i2v_standard.mp4 C
    echo "Test C fertig → outputs/test_C/rabbit_i2v_standard.mp4"
}

run_test_D() {
    local ref="${1:-ref_rabbit.jpg}"
    echo ""
    echo "=== TEST D: Bild-zu-Video starke Konditionierung (~11 min) ==="
    echo "  Referenzbild: $ref"
    mkdir -p outputs/test_D
    $PYTHON "$GEN" \
        --image "$ref" --image-noise-scale 0.05 \
        --width 960 --height 544 --frames 49 \
        --steps 50 --guidance 5.0 --seed 42 \
        --output outputs/test_D/rabbit_i2v_strong.mp4
    evaluate_after outputs/test_D/rabbit_i2v_strong.mp4 D
    echo "Test D fertig → outputs/test_D/rabbit_i2v_strong.mp4"
}

run_test_E() {
    local input="${1:?Bitte Eingabe-MP4 angeben, z.B. outputs/test_C/rabbit_i2v_standard.mp4}"
    echo ""
    echo "=== TEST E: Real-ESRGAN Upscaling → 1920×1088 ==="
    mkdir -p outputs/test_E
    local out="outputs/test_E/$(basename "${input%.mp4}")_esrgan.mp4"
    $PYTHON "$MAKE_GIF" "$input" --output "outputs/test_E/$(basename "${input%.mp4}")" --gif-only --no-interpolation 2>/dev/null || \
    $PYTHON "$MAKE_GIF" "$input" --output "outputs/test_E/$(basename "${input%.mp4}")" --gif-only
    echo "Test E fertig (Upscaled MP4 + GIF in outputs/test_E/)"
}

run_test_F() {
    local input="${1:?Bitte Eingabe-MP4 angeben}"
    echo ""
    echo "=== TEST F: RIFE v4.6 Frame-Interpolation (8fps → 30fps) ==="
    mkdir -p outputs/test_F/frames_in outputs/test_F/frames_out

    # Frames extrahieren
    ffmpeg -y -i "$input" outputs/test_F/frames_in/%08d.png 2>/dev/null
    frame_count=$(ls outputs/test_F/frames_in/*.png 2>/dev/null | wc -l)
    echo "  $frame_count Frames extrahiert"

    # RIFE interpolation (4× = 8fps → 32fps≈30fps)
    "$RIFE_BIN" \
        -i outputs/test_F/frames_in \
        -o outputs/test_F/frames_out \
        -m "$RIFE_MODEL_DIR/rife-v4.6" \
        -n 4 2>/dev/null || \
    "$RIFE_BIN" \
        -i outputs/test_F/frames_in \
        -o outputs/test_F/frames_out \
        -m rife-v4.6

    # Frames zusammensetzen
    local out="outputs/test_F/$(basename "${input%.mp4}")_rife30fps.mp4"
    ffmpeg -y -r 30 -i "outputs/test_F/frames_out/%08d.png" \
        -c:v libx264 -crf 18 -pix_fmt yuv420p \
        -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
        "$out" 2>/dev/null
    echo "Test F fertig → $out"
    rm -rf outputs/test_F/frames_in outputs/test_F/frames_out
}

# ── Einstiegspunkt ──────────────────────────────────────────────────────────

case "${1:-all}" in
    A)   run_test_A ;;
    B)   run_test_B ;;
    B2)  run_test_B2 ;;
    C)   run_test_C "${2}" ;;
    D)   run_test_D "${2}" ;;
    E)   run_test_E "${2}" ;;
    F)   run_test_F "${2}" ;;
    all)
        run_test_A
        run_test_B
        run_test_B2
        echo ""
        echo "=== Automatische Tests (A, B, B2) abgeschlossen ==="
        echo ""
        echo "Nächste Schritte:"
        echo "  1. Videos ansehen und bestes auswählen"
        echo "  2. Referenzbild für Tests C/D bereitstellen:"
        echo "     cp ref_rabbit_from_B.jpg ref_rabbit.jpg   (Frame aus Test B)"
        echo "     oder: eigenes Foto einkopieren als ref_rabbit.jpg"
        echo "  3. bash run_tests.sh C ref_rabbit.jpg"
        echo "  4. bash run_tests.sh D ref_rabbit.jpg"
        echo "  5. Bestes I2V-Ergebnis upscalen + interpolieren:"
        echo "     bash run_tests.sh E outputs/test_C/rabbit_i2v_standard.mp4"
        echo "     bash run_tests.sh F outputs/test_C/rabbit_i2v_standard.mp4"
        ;;
    compare)
        echo ""
        echo "=== VERGLEICH aller bisherigen Testergebnisse ==="
        $PYTHON "$EVAL" --compare \
            outputs/test_A/ outputs/test_B/ outputs/test_B2/ \
            outputs/test_C/ outputs/test_D/ outputs/test_E/ outputs/test_G/ 2>/dev/null
        ;;
    *)
        echo "Unbekannter Test: $1"
        echo "Verwendung: bash run_tests.sh [A|B|B2|C|D|E|F|all|compare] [optionales Argument]"
        exit 1
        ;;
esac
