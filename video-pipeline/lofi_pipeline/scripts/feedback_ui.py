"""
Lokale Feedback-UI für Lo-Fi LoRA Training.
Zeigt alle generierten Samples, ermöglicht Bewertung und startet Training/Generierung.
Feedback wird automatisch in Prompt-Korrekturen übersetzt.

Starten: python scripts/feedback_ui.py
Öffnen:  http://localhost:7860
"""
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, request, send_from_directory

PIPELINE_ROOT = Path(__file__).parent.parent
SCENARIOS_DIR = PIPELINE_ROOT / "scenarios"
CONFIGS_DIR = PIPELINE_ROOT / "configs"

app = Flask(__name__)
_running = {}
_running_lock = threading.Lock()


CORRECTIONS = {
    "ohren_fehlen": {
        "label": "Ohren fehlen / nicht sichtbar",
        "prompt_add": "both rabbits with long clearly visible upright ears, prominent tall ears",
        "negative_add": "missing ears, no ears, hidden ears, floppy ears, short ears",
        "caption_hint": "long clearly visible upright ears",
        "needs_training": False,
    },
    "dritter_hase": {
        "label": "Drei Hasen statt zwei",
        "prompt_add": "exactly two rabbits, only two rabbits, a pair of two rabbits",
        "negative_add": "three rabbits, four rabbits, trio, extra rabbit, third rabbit",
        "caption_hint": None,
        "needs_training": False,
    },
    "falsche_tierart": {
        "label": "Kein Hase (Vogel / falsches Tier)",
        "prompt_add": "rabbit with round body and upright ears, clearly a rabbit not a bird",
        "negative_add": "bird, chick, beak, duck, cat, dog, wrong animal",
        "caption_hint": "anime rabbit character, clearly a rabbit with round body and long ears",
        "needs_training": True,
    },
    "mund_offen": {
        "label": "Mund offen / überrascht",
        "prompt_add": "calm closed mouths, serene peaceful expression",
        "negative_add": "open mouth, surprised expression, anxious, agitated",
        "caption_hint": None,
        "needs_training": False,
    },
    "zu_3d": {
        "label": "Stil zu 3D / photorealistisch",
        "prompt_add": "flat anime illustration, hand-drawn 2D anime art, cel shading",
        "negative_add": "3D render, CGI, photorealistic, volumetric lighting, realistic fur, plastic texture",
        "caption_hint": None,
        "needs_training": True,
    },
    "keine_animation": {
        "label": "Keine / zu wenig Animation",
        "prompt_add": "animated loop, visible motion, gently swaying leaves, twinkling stars, rippling water",
        "negative_add": "completely static, frozen image, no movement, still frame",
        "caption_hint": "visible gentle motion and animation",
        "needs_training": True,
    },
    "hasen_zu_weit": {
        "label": "Hasen zu weit auseinander",
        "prompt_add": "two rabbits sitting very close together, shoulders touching, cuddling pose",
        "negative_add": "rabbits far apart, separated rabbits",
        "caption_hint": None,
        "needs_training": False,
    },
    "kein_see": {
        "label": "See fehlt / nicht sichtbar",
        "prompt_add": "calm dark lake clearly visible in middle ground with reflections",
        "negative_add": "no water, no lake",
        "caption_hint": None,
        "needs_training": False,
    },
    "schlechte_spiegelung": {
        "label": "Wasser-Spiegelungen fehlen",
        "prompt_add": "softly rippling lake water with shimmering cottage and moon reflections, mirror-like water surface",
        "negative_add": "flat water, no reflections",
        "caption_hint": None,
        "needs_training": False,
    },
    "zu_hell": {
        "label": "Szene zu hell / nicht dunkel genug",
        "prompt_add": "deep dark indigo night sky, dark nocturnal atmosphere, dimly lit scene",
        "negative_add": "bright daylight, overexposed, washed out, bright green",
        "caption_hint": None,
        "needs_training": False,
    },
    "gesichter_fehlen": {
        "label": "Gesichter nicht erkennbar",
        "prompt_add": "both rabbits facing forward, clearly visible faces, visible round eyes",
        "negative_add": "no faces, empty faces, broken faces, turned away",
        "caption_hint": "clear visible face with round eyes",
        "needs_training": True,
    },
    "keine_ohren_bewegung": {
        "label": "Ohren bewegen sich nicht",
        "prompt_add": "rabbit ears gently swaying in soft breeze, ear tips moving like leaves in wind",
        "negative_add": None,
        "caption_hint": "ears gently swaying in soft breeze",
        "needs_training": True,
    },
    "kein_baum": {
        "label": "Baum fehlt",
        "prompt_add": "large framing tree on right side with gently swaying leaves",
        "negative_add": None,
        "caption_hint": None,
        "needs_training": False,
    },
    "laterne_fehlt": {
        "label": "Laterne fehlt",
        "prompt_add": "warm glowing amber lantern between the two rabbits, lantern light glowing",
        "negative_add": "no lantern, no light source",
        "caption_hint": None,
        "needs_training": False,
    },
}



def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def get_all_samples(scenario_id: str):
    scenario_dir = SCENARIOS_DIR / scenario_id
    rounds_dir = scenario_dir / "rounds"
    samples = []
    if not rounds_dir.exists():
        return samples
    ratings = load_ratings(scenario_id)
    for round_dir in sorted(rounds_dir.iterdir()):
        if not round_dir.is_dir():
            continue
        samples_dir = round_dir / "samples"
        if not samples_dir.exists():
            continue
        for f in sorted(samples_dir.iterdir()):
            if f.suffix not in (".mp4", ".gif"):
                continue
            key = f"{round_dir.name}/{f.name}"
            samples.append({
                "key": key,
                "round": round_dir.name,
                "filename": f.name,
                "rel_path": f"media/{scenario_id}/{round_dir.name}/{f.name}",
                "is_gif": f.suffix == ".gif",
                "size_kb": round(f.stat().st_size / 1024),
                "mtime": f.stat().st_mtime,
                "rating": ratings.get(key, {}).get("rating", None),
                "note": ratings.get(key, {}).get("note", ""),
                "checkpoint": _find_checkpoint(round_dir, f.name),
            })
    samples.sort(key=lambda x: x["mtime"], reverse=True)
    return samples


def _find_checkpoint(round_dir: Path, filename: str):
    m = re.match(r"step_(\d+)", filename)
    if m:
        ckpt = round_dir / "checkpoints" / f"lora_weights_step_{m.group(1)}.safetensors"
        return str(ckpt) if ckpt.exists() else None
    ckpt_dir = round_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("lora_weights_step_*.safetensors"))
        return str(ckpts[-1]) if ckpts else None
    return None


def ratings_path(scenario_id: str):
    return SCENARIOS_DIR / scenario_id / "ratings.json"


def load_ratings(scenario_id: str):
    p = ratings_path(scenario_id)
    return json.loads(p.read_text()) if p.exists() else {}


def save_rating(scenario_id, key, rating, note=""):
    data = load_ratings(scenario_id)
    data[key] = {"rating": rating, "note": note, "ts": time.time()}
    ratings_path(scenario_id).write_text(json.dumps(data, indent=2))


def next_round_number(scenario_id: str):
    rounds_dir = SCENARIOS_DIR / scenario_id / "rounds"
    if not rounds_dir.exists():
        return 1
    existing = [d for d in rounds_dir.iterdir() if d.is_dir() and d.name.startswith("round_")]
    nums = [int(d.name.split("_")[1]) for d in existing] if existing else [0]
    return max(nums) + 1


def apply_corrections_to_scenario(scenario_id: str, correction_ids: list):
    """Wendet Feedback-Korrekturen auf scenario.yaml an und gibt Zusammenfassung zurück."""
    scenario_path = SCENARIOS_DIR / scenario_id / "scenario.yaml"
    cfg = load_yaml(scenario_path)

    prompt = cfg.get("prompt", "").strip().rstrip(",")
    negative = cfg.get("negative_prompt", "").strip().rstrip(",")
    needs_training = False
    caption_hints = []
    applied = []

    for cid in correction_ids:
        c = CORRECTIONS.get(cid)
        if not c:
            continue
        applied.append(c["label"])
        if c.get("prompt_add") and c["prompt_add"] not in prompt:
            prompt = c["prompt_add"] + ",\n  " + prompt
        if c.get("negative_add") and c["negative_add"] not in negative:
            negative = c["negative_add"] + ",\n  " + negative
        if c.get("needs_training"):
            needs_training = True
        if c.get("caption_hint"):
            caption_hints.append(c["caption_hint"])

    cfg["prompt"] = prompt
    cfg["negative_prompt"] = negative

    backup = scenario_path.with_suffix(".yaml.bak")
    backup.write_text(scenario_path.read_text())

    with open(scenario_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, width=120)

    return {
        "applied": applied,
        "needs_training": needs_training,
        "caption_hints": caption_hints,
    }


def run_background(cmd, env, job_id, job_meta):
    log_path = PIPELINE_ROOT / "scripts" / f"_job_{job_id}.log"
    job_meta["log"] = str(log_path)
    with _running_lock:
        _running[job_id] = job_meta
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=str(PIPELINE_ROOT.parent.parent.parent))
        with _running_lock:
            _running[job_id]["pid"] = proc.pid
        proc.wait()
        with _running_lock:
            _running[job_id]["done"] = True
            _running[job_id]["returncode"] = proc.returncode



@app.route("/api/scenarios")
def api_scenarios():
    return jsonify([d.name for d in SCENARIOS_DIR.iterdir() if d.is_dir()])


@app.route("/api/samples/<scenario_id>")
def api_samples(scenario_id):
    return jsonify(get_all_samples(scenario_id))


@app.route("/api/corrections")
def api_corrections():
    return jsonify({k: {"label": v["label"], "needs_training": v["needs_training"]}
                    for k, v in CORRECTIONS.items()})


@app.route("/api/rate", methods=["POST"])
def api_rate():
    d = request.json
    save_rating(d["scenario_id"], d["key"], int(d["rating"]), d.get("note", ""))
    return jsonify({"ok": True})


@app.route("/api/apply_feedback", methods=["POST"])
def api_apply_feedback():
    """Korrekturen auf scenario.yaml anwenden."""
    d = request.json
    result = apply_corrections_to_scenario(d["scenario_id"], d["corrections"])
    return jsonify(result)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    d = request.json
    scenario_id = d["scenario_id"]
    checkpoint = d["checkpoint"]
    seed = int(d.get("seed", 777))
    guidance = float(d.get("guidance", 9.0))
    steps = int(d.get("steps", 60))
    job_id = f"gen_{scenario_id}_{int(time.time())}"

    paths_cfg = load_yaml(CONFIGS_DIR / "model_paths.yaml")
    scenario_cfg = load_yaml(SCENARIOS_DIR / scenario_id / "scenario.yaml")

    ckpt_path = Path(checkpoint)
    samples_dir = ckpt_path.parent.parent / "samples"
    samples_dir.mkdir(exist_ok=True)
    out_path = samples_dir / f"manual_s{seed}_g{guidance}_{int(time.time())}.mp4"

    cmd = [paths_cfg["python"], paths_cfg["generate_script"],
           "--lora", checkpoint, "--steps", str(steps), "--seed", str(seed),
           "--guidance-scale", str(guidance), "--output", str(out_path),
           "--prompt", scenario_cfg["prompt"].strip(),
           "--negative-prompt", scenario_cfg["negative_prompt"].strip()]

    env = os.environ.copy()
    env.update(paths_cfg.get("env", {}))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    meta = {"type": "generate", "scenario": scenario_id, "done": False, "output": str(out_path)}
    threading.Thread(target=run_background, args=(cmd, env, job_id, meta), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/train", methods=["POST"])
def api_train():
    d = request.json
    scenario_id = d["scenario_id"]
    checkpoint = d.get("checkpoint")
    steps = int(d.get("steps", 50))
    job_id = f"train_{scenario_id}_{int(time.time())}"

    paths_cfg = load_yaml(CONFIGS_DIR / "model_paths.yaml")
    next_round = next_round_number(scenario_id)

    cmd = [paths_cfg["python"], str(PIPELINE_ROOT / "scripts" / "train_lora.py"),
           "--scenario", scenario_id, "--round", str(next_round), "--steps", str(steps)]
    cmd += ["--from-checkpoint", checkpoint] if checkpoint else ["--resume"]

    env = os.environ.copy()
    env.update(paths_cfg.get("env", {}))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    meta = {"type": "train", "scenario": scenario_id, "round": next_round, "done": False}
    threading.Thread(target=run_background, args=(cmd, env, job_id, meta), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "round": next_round})


@app.route("/api/jobs")
def api_jobs():
    with _running_lock:
        return jsonify(dict(_running))


@app.route("/api/make_gif", methods=["POST"])
def api_make_gif():
    d = request.json
    samples_dir = SCENARIOS_DIR / d["scenario_id"] / "rounds" / d["round"] / "samples"
    mp4 = samples_dir / d["filename"]
    gif = mp4.with_suffix(".gif")
    cmd = ["ffmpeg", "-y", "-i", str(mp4),
           "-vf", "fps=8,scale=624:-1:flags=lanczos,split[s0][s1];"
                  "[s0]palettegen=max_colors=256:stats_mode=diff[p];"
                  "[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
           "-loop", "0", str(gif)]
    r = subprocess.run(cmd, capture_output=True)
    return jsonify({"ok": r.returncode == 0})


@app.route("/media/<scenario_id>/<round_name>/<filename>")
def serve_media(scenario_id, round_name, filename):
    return send_from_directory(
        SCENARIOS_DIR / scenario_id / "rounds" / round_name / "samples", filename)



HTML = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lo-Fi Training Feedback</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0f14;color:#e0e0e0;font-family:system-ui,sans-serif;padding-bottom:60px}
header{background:#1a1d24;padding:14px 20px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #2a2d36;position:sticky;top:0;z-index:10}
h1{font-size:1.1rem;font-weight:700;color:#f0c060}
select{background:#2a2d36;color:#e0e0e0;border:1px solid #3a3d46;padding:5px 10px;border-radius:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:18px;padding:18px}
.card{background:#1a1d24;border-radius:12px;overflow:hidden;border:2px solid #2a2d36;transition:border-color .2s}
.card:hover{border-color:#3a3d46}
.card.rated-high{border-color:#4a7}
.card.rated-low{border-color:#a44}
video,img{width:100%;display:block;max-height:220px;object-fit:cover;background:#000;cursor:pointer}
.card-body{padding:12px}
.meta{font-size:.72rem;color:#777;margin-bottom:8px;display:flex;flex-wrap:wrap;gap:6px}
.tag{background:#2a2d36;padding:2px 7px;border-radius:8px}
.tag.good{background:#1a3a1a;color:#4d4}
/* Stars */
.stars{display:flex;gap:3px;margin:6px 0;cursor:pointer}
.star{font-size:1.3rem;color:#333;transition:color .1s;user-select:none}
.star.on{color:#f0c060}
/* Feedback panel */
.fb-toggle{background:#23263a;border:1px solid #3a3d46;color:#aaa;font-size:.78rem;padding:4px 10px;border-radius:6px;cursor:pointer;width:100%;text-align:left;margin-top:8px}
.fb-panel{display:none;margin-top:8px;background:#12151c;border-radius:8px;padding:10px}
.fb-panel.open{display:block}
.fb-title{font-size:.75rem;color:#888;margin-bottom:6px;font-weight:600}
.checks{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:8px}
.checks label{font-size:.72rem;color:#bbb;display:flex;align-items:center;gap:5px;cursor:pointer}
.checks input[type=checkbox]{accent-color:#f0c060}
.training-warn{font-size:.68rem;color:#f0c060;margin-bottom:6px}
textarea{width:100%;background:#0d0f14;border:1px solid #2a2d36;border-radius:6px;color:#ccc;padding:6px 8px;font-size:.78rem;resize:vertical;min-height:40px;margin-top:6px}
.btn-row{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
button{padding:5px 12px;border-radius:6px;border:none;cursor:pointer;font-size:.78rem;font-weight:600;transition:opacity .15s}
button:hover{opacity:.8}
button:disabled{opacity:.4;cursor:default}
.btn-apply{background:#e07030;color:#fff}
.btn-train{background:#f0c060;color:#1a1d24}
.btn-gen{background:#4a7fd4;color:#fff}
.btn-gif{background:#2a5d34;color:#fff}
.btn-sm{padding:3px 9px;font-size:.72rem}
.status{font-size:.72rem;color:#6af;margin-top:5px;min-height:14px}
/* Jobs bar */
.jobs-bar{position:fixed;bottom:0;left:0;right:0;background:#1a1d24;border-top:1px solid #2a2d36;padding:7px 18px;display:flex;gap:10px;align-items:center;font-size:.75rem;color:#888}
.jpill{padding:2px 10px;border-radius:10px;background:#2a2d36}
.jpill.run{border:1px solid #f0c060;color:#f0c060}
.jpill.done{border:1px solid #4a7;color:#4a7}
/* Modal */
.modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:50;align-items:center;justify-content:center}
.modal.open{display:flex}
.mbox{background:#1a1d24;border-radius:12px;padding:22px;width:340px;border:1px solid #3a3d46}
.mbox h3{color:#f0c060;margin-bottom:14px}
.mbox label{display:block;font-size:.8rem;color:#aaa;margin-bottom:3px}
.mbox input{width:100%;background:#12151c;border:1px solid #2a2d36;color:#e0e0e0;padding:6px 9px;border-radius:6px;margin-bottom:10px}
</style>
</head>
<body>
<header>
  <h1>🐰 Lo-Fi Feedback</h1>
  <select id="sel" onchange="load()"></select>
  <button class="btn-train btn-sm" onclick="openModal('train',null)">+ Neue Runde</button>
  <span style="margin-left:auto;font-size:.75rem;color:#555" id="count"></span>
</header>

<div class="grid" id="grid"></div>

<!-- Generate modal -->
<div class="modal" id="genModal">
  <div class="mbox">
    <h3>🎬 Neu generieren</h3>
    <label>Seed</label><input id="gSeed" type="number" value="777">
    <label>Guidance Scale</label><input id="gGuid" type="number" value="9.0" step="0.5">
    <label>Inference Steps</label><input id="gSteps" type="number" value="60">
    <input type="hidden" id="gCkpt">
    <div class="btn-row">
      <button class="btn-gen" onclick="doGen()">Generieren</button>
      <button onclick="closeModal()" style="background:#333;color:#ccc">Abbrechen</button>
    </div>
    <div class="status" id="genSt"></div>
  </div>
</div>

<!-- Train modal -->
<div class="modal" id="trainModal">
  <div class="mbox">
    <h3>⚡ Training starten</h3>
    <label>Steps</label><input id="tSteps" type="number" value="50">
    <label>Checkpoint (leer = auto)</label><input id="tCkpt" placeholder="Automatisch">
    <div class="btn-row">
      <button class="btn-train" onclick="doTrain()">Starten</button>
      <button onclick="closeModal()" style="background:#333;color:#ccc">Abbrechen</button>
    </div>
    <div class="status" id="trainSt"></div>
  </div>
</div>

<div class="jobs-bar" id="jbar">Keine Jobs</div>

<script>
let scenario='', samples=[], corrections={};

async function init(){
  const [sc,cr]=await Promise.all([fetch('/api/scenarios').then(r=>r.json()),fetch('/api/corrections').then(r=>r.json())]);
  corrections=cr;
  const sel=document.getElementById('sel');
  sc.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;sel.appendChild(o)});
  if(sc.length){scenario=sc[0];sel.value=scenario;load();}
  setInterval(pollJobs,4000);
}

async function load(){
  scenario=document.getElementById('sel').value;
  const r=await fetch(`/api/samples/${scenario}`);
  samples=await r.json();
  document.getElementById('count').textContent=`${samples.length} Samples`;
  render();
}

function render(){
  document.getElementById('grid').innerHTML=samples.map((s,i)=>{
    const ratingClass=s.rating>=8?'rated-high':s.rating<=4&&s.rating!==null?'rated-low':'';
    const checks=Object.entries(corrections).map(([id,c])=>
      `<label><input type="checkbox" id="c_${i}_${id}" data-train="${c.needs_training}"> ${c.label}</label>`
    ).join('');
    return `<div class="card ${ratingClass}" id="card${i}">
      ${s.is_gif?`<img src="/${s.rel_path}" loading="lazy">`:`<video src="/${s.rel_path}" autoplay loop muted playsinline></video>`}
      <div class="card-body">
        <div class="meta">
          <span class="tag">${s.round}</span>
          <span class="tag">${s.filename}</span>
          <span class="tag">${s.size_kb} KB</span>
          ${s.rating!==null?`<span class="tag good">${s.rating}/10</span>`:''}
        </div>

        <!-- Sterne-Bewertung -->
        <div class="stars" id="stars${i}">
          ${[1,2,3,4,5,6,7,8,9,10].map(n=>`<span class="star ${s.rating>=n?'on':''}" onclick="rate(${i},${n})">★</span>`).join('')}
        </div>

        <!-- Korrekturen -->
        <button class="fb-toggle" onclick="toggleFb(${i})">🔧 Was stimmt nicht? (Korrekturen)</button>
        <div class="fb-panel" id="fb${i}">
          <div class="fb-title">Problem auswählen — wird automatisch in Prompt-Korrekturen übersetzt:</div>
          <div class="checks">${checks}</div>
          <div class="training-warn" id="twarn${i}" style="display:none">⚠ Einige Korrekturen brauchen eine neue Trainingsrunde</div>
          <textarea id="note${i}" placeholder="Eigene Anmerkungen..." onblur="saveNote(${i},this.value)">${s.note||''}</textarea>
          <div class="btn-row">
            <button class="btn-apply" onclick="applyAndGen(${i})">✓ Anwenden + Neu generieren</button>
            <button class="btn-apply btn-sm" style="background:#805020" onclick="applyAndTrain(${i})">✓ Anwenden + Training</button>
          </div>
          <div class="status" id="st${i}"></div>
        </div>

        <!-- Aktions-Buttons -->
        <div class="btn-row" style="margin-top:8px">
          ${s.checkpoint?`<button class="btn-train btn-sm" onclick="openModal('train','${s.checkpoint}')">⚡ Weiter trainieren</button>`:''}
          ${s.checkpoint?`<button class="btn-gen btn-sm" onclick="openModal('gen','${s.checkpoint}')">🎬 Neu generieren</button>`:''}
          ${!s.is_gif?`<button class="btn-gif btn-sm" onclick="makeGif(${i})">GIF erstellen</button>`:''}
        </div>
      </div>
    </div>`;
  }).join('');

  // Checkbox-Change Listener für Training-Warnung
  samples.forEach((_,i)=>{
    document.querySelectorAll(`[id^="c_${i}_"]`).forEach(cb=>{
      cb.addEventListener('change',()=>updateTrainWarn(i));
    });
  });
}

function updateTrainWarn(i){
  const any=Array.from(document.querySelectorAll(`[id^="c_${i}_"]`))
    .some(cb=>cb.checked&&cb.dataset.train==='true');
  document.getElementById(`twarn${i}`).style.display=any?'block':'none';
}

function toggleFb(i){document.getElementById(`fb${i}`).classList.toggle('open')}

function getSelectedCorrections(i){
  return Array.from(document.querySelectorAll(`[id^="c_${i}_"]`))
    .filter(cb=>cb.checked).map(cb=>cb.id.replace(`c_${i}_`,''));
}

async function applyAndGen(i){
  const corrs=getSelectedCorrections(i);
  if(!corrs.length){setStatus(i,'Kein Problem ausgewählt.');return;}
  setStatus(i,'Wende Korrekturen an...');
  const r=await fetch('/api/apply_feedback',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({scenario_id:scenario,corrections:corrs})});
  const res=await r.json();
  setStatus(i,`Angewendet: ${res.applied.join(', ')}. Starte Generierung...`);
  const s=samples[i];
  if(s.checkpoint){
    await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({scenario_id:scenario,checkpoint:s.checkpoint,seed:777,guidance:9.0,steps:60})});
    setStatus(i,'Generierung gestartet! Refresh in ~10 min.');
  }else{setStatus(i,'Kein Checkpoint — bitte manuell generieren.');}
}

async function applyAndTrain(i){
  const corrs=getSelectedCorrections(i);
  if(!corrs.length){setStatus(i,'Kein Problem ausgewählt.');return;}
  setStatus(i,'Wende Korrekturen an...');
  await fetch('/api/apply_feedback',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({scenario_id:scenario,corrections:corrs})});
  const s=samples[i];
  const r=await fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({scenario_id:scenario,checkpoint:s.checkpoint,steps:50})});
  const d=await r.json();
  setStatus(i,`Training Round ${d.round} gestartet.`);
}

function setStatus(i,msg){document.getElementById(`st${i}`).textContent=msg;}

async function rate(i,val){
  samples[i].rating=val;
  const note=document.getElementById(`note${i}`)?.value||'';
  await fetch('/api/rate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({scenario_id:scenario,key:samples[i].key,rating:val,note})});
  render();
}

async function saveNote(i,note){
  if(samples[i].rating!==null)
    await fetch('/api/rate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({scenario_id:scenario,key:samples[i].key,rating:samples[i].rating,note})});
}

function openModal(type,ckpt){
  if(type==='gen'){document.getElementById('gCkpt').value=ckpt||'';document.getElementById('genModal').classList.add('open');}
  else{document.getElementById('tCkpt').value=ckpt||'';document.getElementById('trainModal').classList.add('open');}
}
function closeModal(){document.querySelectorAll('.modal').forEach(m=>m.classList.remove('open'));}

async function doGen(){
  const ckpt=document.getElementById('gCkpt').value;
  if(!ckpt){document.getElementById('genSt').textContent='Kein Checkpoint!';return;}
  document.getElementById('genSt').textContent='Startet...';
  await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({scenario_id:scenario,checkpoint:ckpt,
      seed:+document.getElementById('gSeed').value,
      guidance:+document.getElementById('gGuid').value,
      steps:+document.getElementById('gSteps').value})});
  document.getElementById('genSt').textContent='Generierung gestartet!';
  setTimeout(closeModal,1500);
}

async function doTrain(){
  document.getElementById('trainSt').textContent='Startet...';
  const r=await fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({scenario_id:scenario,
      checkpoint:document.getElementById('tCkpt').value||null,
      steps:+document.getElementById('tSteps').value})});
  const d=await r.json();
  document.getElementById('trainSt').textContent=`Round ${d.round} gestartet!`;
  setTimeout(closeModal,1500);
}

async function makeGif(i){
  const s=samples[i];
  const[round,filename]=s.key.split('/');
  await fetch('/api/make_gif',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({scenario_id:scenario,round,filename})});
  setTimeout(load,3000);
}

async function pollJobs(){
  const r=await fetch('/api/jobs');
  const jobs=await r.json();
  const bar=document.getElementById('jbar');
  const entries=Object.entries(jobs);
  if(!entries.length){bar.textContent='Keine Jobs';return;}
  bar.innerHTML=entries.map(([id,j])=>
    `<span class="jpill ${j.done?'done':'run'}">${j.type==='train'?'⚡':'🎬'} ${j.scenario} ${j.done?'✓':'⏳'}</span>`
  ).join('');
  if(entries.some(([,j])=>j.done&&j.type==='generate'))setTimeout(load,1000);
}

init();
</script>
</body>
</html>"""


@app.route("/")
def root():
    return Response(HTML, mimetype="text/html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\n{'='*50}")
    print(f"  Lo-Fi Feedback UI")
    print(f"  http://localhost:{port}")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
