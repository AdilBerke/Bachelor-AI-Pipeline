import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  Film,
  Loader2,
  Play,
  Plus,
  Search,
  Sparkles,
  Star,
  Trash2,
  UploadCloud,
  Wand2,
} from "lucide-react";

export const Route = createFileRoute("/studio/video")({
  head: () => ({
    meta: [
      { title: "LOFI.GEN · Video" },
      { name: "description", content: "GIF-Generierung und Training für Lo-Fi Szenarien." },
    ],
  }),
  component: VideoPage,
});

// ── Video Backend — gleiche web_api.py wie Audio (Port 8000) ───────────────
import { getSettings } from "../lib/settings";

export function videoApiBase(): string {
  return getSettings().apiUrl.replace(/\/+$/, "");
}

export async function videoRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${videoApiBase()}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`Video-API ${res.status} auf ${path}`);
  return res.json() as Promise<T>;
}

function videoMediaUrl(relPath: string): string {
  return `${videoApiBase()}/api/video/media/${relPath}`;
}

// ── Stil-Optionen ────────────────────────────────────────────────────────────

const VISUAL_STYLES = [
  { id: "lofi_anime", label: "Lo-Fi Anime", hint: "2D Cel-Shading, handgezeichnet", guidance: 9.0 },
  { id: "lofi_illustrated", label: "Lo-Fi Illustration", hint: "Aquarell / weiche Illustration", guidance: 5.0 },
] as const;

type VisualStyle = typeof VISUAL_STYLES[number]["id"];

// ── Standard-Negativ-Prompt (bewährt aus rabbit_lake + rainy_window) ─────────

const STANDARD_NEGATIVE_PROMPT =
  "completely static, frozen image, no movement, still frame, " +
  "3D render, CGI, photorealistic, volumetric lighting, subsurface scattering, " +
  "plastic texture, 3D model, game engine render, realistic fur, ray tracing, " +
  "text, watermark, signature, UI elements, player controls, overlays, " +
  "blurry, low quality, grain, compression artifacts, " +
  "bright green, washed out, grayscale, black canvas, " +
  "open mouth, surprised expression, anxious expression";

// ── Prompt-Templates (basierend auf rabbit_lake / rainy_window Prompts) ──────

function buildPromptTemplate(style: VisualStyle, subject: string): string {
  const base =
    style === "lofi_anime"
      ? "lofi_girl, flat anime illustration, hand-drawn anime style, 2D anime art, cel shading, " +
        "lo-fi anime illustration, animated loop, visible ambient motion, "
      : "lofi_girl, lo-fi illustration, watercolor aesthetic, soft artistic style, " +
        "dreamy illustrated scene, animated loop, visible gentle motion, ";

  const subjectBlock = subject.trim()
    ? subject.trim() + ", "
    : "[Beschreibe hier das Hauptmotiv detailliert: Figuren, Hintergrund, Licht, Atmosphäre], ";

  return (
    base +
    subjectBlock +
    "animated smooth loop, cozy lo-fi atmosphere, polished illustration quality, " +
    "16:9 stable composition, no text, no UI, no overlays, no watermarks"
  );
}

// ── Feedback-Dimensionen (quantifiziertes subjektives Feedback) ─────────────
// Jede Dimension ist unabhängig und mappt auf einen konkreten Trainingsparameter.
// Basis: RLHF-Prinzip — dimensionale Scores sind konsistenter als absolute Gesamtbewertungen.

const FEEDBACK_DIMS = [
  {
    key: "stil_treue",
    label: "Lo-Fi Stil-Treue",
    low: "Falsch (3D / photorealistisch)",
    high: "Perfekt (2D Anime / Illustration)",
    param: "LoRA-Gewichtung & Stil-Keywords",
    anchors: {
      1: "Kein Anime-Stil — photorealistisch oder 3D-Optik dominiert",
      2: "Anime-Elemente erkennbar, aber Stil-Brüche dominieren das Bild",
      3: "Erkennbarer Lo-Fi-Stil mit vereinzelten Inkonsistenzen",
      4: "Klarer Lo-Fi-Anime-Stil, konsistentes Hand-gezeichnetes Erscheinungsbild",
      5: "Perfekter Lo-Fi-Stil — vollständig kohärent, kein Stil-Bruch",
    } as Record<number, string>,
  },
  {
    key: "bewegung",
    label: "Bewegungsintensität",
    low: "Eingefroren / statisch",
    high: "Genau die richtige Animation",
    param: "Motion-Zielwert & Bewegungs-Keywords",
    anchors: {
      1: "Komplett statisch — kein Wasser, keine Pflanzen, kein Pixel verändert sich",
      2: "Minimalste Variation — wie leicht komprimiertes Standbild, kaum merklich",
      3: "Erkennbare Bewegung, aber schwach — Wasser hat leichte Variation",
      4: "Klare Bewegung — Wasserripple, Pflanzen schwingen, Ohren bewegen sich",
      5: "Lebhafte, realistische Bewegung — alle Elemente animiert wie echter Wind",
    } as Record<number, string>,
  },
  {
    key: "schaerfe",
    label: "Schärfe & Detailgrad",
    low: "Unscharf / matschig",
    high: "Klar, knackig, details sichtbar",
    param: "Inference-Steps & Sharpness-Ziel",
    anchors: {
      1: "Stark verschwommen — Kanten nicht erkennbar, extreme Weichzeichnung",
      2: "Weich — Details ansatzweise erkennbar, Textur und Linien nicht trennbar",
      3: "Akzeptabel — Objekte klar, aber keine feinen Details oder Cel-Linien",
      4: "Klar — gute Kantendefinition, Cel-Shading-Linien erkennbar",
      5: "Sehr scharf — präzise Kantenlinien, maximale Details, keine Artefakte",
    } as Record<number, string>,
  },
  {
    key: "stimmung",
    label: "Atmosphäre / Stimmung",
    low: "Passt nicht zur Szene",
    high: "Trifft Stimmung genau",
    param: "Farb-Keywords & Guidance Scale",
    anchors: {
      1: "Falsche oder keine Stimmung — passt nicht zum beschriebenen Szenario",
      2: "Ansatzweise — Stimmung vorhanden aber widersprüchliche Elemente",
      3: "Spürbar — Stimmung erkennbar, einzelne Elemente passen nicht rein",
      4: "Klare, konsistente Stimmung über alle Bildelemente hinweg",
      5: "Vollständig immersiv — erzeugt exakt die beschriebene Atmosphäre",
    } as Record<number, string>,
  },
  {
    key: "motiv_genauigkeit",
    label: "Motiv-Genauigkeit",
    low: "Falscher Inhalt",
    high: "Genau wie gewünscht",
    param: "Guidance Scale & Prompt-Fokus",
    anchors: {
      1: "Motiv fehlt oder komplett falsche Anatomie / falsches Tier",
      2: "Erkennbar mit starken Fehlern (Ohren fehlen, falsche Körperhaltung)",
      3: "Klar erkennbar, kleinere anatomische Fehler oder Platzierungsprobleme",
      4: "Korrekt positioniert, Anatomie stimmt, Proportionen passen",
      5: "Exakt wie beschrieben — alle Details und Positionen korrekt",
    } as Record<number, string>,
  },
] as const;

type FeedbackDimKey = typeof FEEDBACK_DIMS[number]["key"];

const ISSUE_PRESETS = [
  { id: "keine_animation", label: "Keine / zu wenig Animation" },
  { id: "pflanzen_statisch", label: "Pflanzen / Bäume statisch" },
  { id: "zu_3d", label: "Stil zu 3D / photorealistisch" },
  { id: "zu_hell", label: "Szene zu hell / nicht dunkel genug" },
  { id: "ueberbelichtung", label: "Überbelichtet / ausgebrannte Highlights" },
  { id: "schlechte_spiegelung", label: "Wasser-Spiegelungen fehlen" },
  { id: "kein_see", label: "See / Wasser fehlt" },
  { id: "ohren_fehlen", label: "Ohren fehlen / nicht sichtbar" },
  { id: "dritter_hase", label: "Falsche Anzahl Figuren" },
  { id: "falsche_tierart", label: "Falsches Motiv / Tier" },
  { id: "gesichter_fehlen", label: "Gesichter nicht erkennbar" },
  { id: "laterne_fehlt", label: "Lichtquelle / Laterne fehlt" },
  { id: "kein_baum", label: "Hintergrund-Element fehlt" },
  { id: "mund_offen", label: "Falsche Mimik / Ausdruck" },
  { id: "hintergrund_unruhig", label: "Hintergrund zu unruhig / chaotisch" },
  { id: "loop_springt", label: "Loop springt / kein nahtloser Übergang" },
] as const;

// ── State-Typen ─────────────────────────────────────────────────────────────

interface ScenarioSetup {
  name: string;
  visualStyle: VisualStyle | null;
  mainSubject: string;
  prompt: string;
  negativePrompt: string;
  guidanceScale: number;
  stepsPerRound: number;
  inferenceSteps: number;  // Schritte bei der Generierung (rabbit_lake: 60, rainy_window: 75)
  loraConfig: string;      // "base_lora.yaml" (rank 16) oder "base_lora_v2.yaml" (rank 32)
}

interface YTVideo {
  id: string;
  title: string;
  url: string;
  channel: string;
  duration: string;
  thumbnail: string;
}

interface SourceClass {
  id: string;
  label: string;
  query: string;
  suggestions: string[];
  videos: YTVideo[];
  selectedIds: string[];
  manualFiles: File[];
  searching: boolean;
}

export interface VideoFeedback {
  dimensions: Record<FeedbackDimKey, number>;
  issues: string[];
  overall: number;
  freeText: string;
}

type Phase = "gallery" | "setup" | "sources" | "training" | "result" | "feedback";

interface SavedProject {
  id: string;
  name: string;
  created: string;
  updated: string;
  phase: Phase;
  setup: ScenarioSetup;
  sources: Omit<SourceClass, "searching">[];
  scenarioId: string | null;
}

function blankSetup(): ScenarioSetup {
  return {
    name: "",
    visualStyle: null,
    mainSubject: "",
    prompt: "",
    negativePrompt: STANDARD_NEGATIVE_PROMPT,
    guidanceScale: 9.0,
    stepsPerRound: 100,
    inferenceSteps: 60,
    loraConfig: "base_lora.yaml",
  };
}

// ── Trainings-Intensitäts-Presets ────────────────────────────────────────────
// Basierend auf rabbit_lake (R1–R8) und rainy_window (R1–R14) realen Werten.

const INTENSITY_PRESETS = [
  {
    id: "schnelltest",
    label: "Schnelltest",
    emoji: "⚡",
    desc: "Grober erster Eindruck in ~25 Min — gut um Prompt zu prüfen, bevor man Zeit investiert",
    stepsPerRound: 50,
    inferenceSteps: 30,
    loraConfig: "base_lora.yaml",
    hint: "50 Training-Steps · 30 Inference-Steps · LoRA rank 16",
  },
  {
    id: "standard",
    label: "Standard",
    emoji: "✦",
    desc: "Wie rabbit_lake R1–R8 und rainy_window R1–R14 — solides Ergebnis, ~45–60 Min pro Runde",
    stepsPerRound: 100,
    inferenceSteps: 60,
    loraConfig: "base_lora.yaml",
    hint: "100 Training-Steps · 60 Inference-Steps · LoRA rank 16",
    recommended: true,
  },
  {
    id: "intensiv",
    label: "Intensiv",
    emoji: "◈",
    desc: "Für Verfeinerungsrunden (Runde 2+) — mehr Details, mehr Kapazität, ~90 Min pro Runde",
    stepsPerRound: 200,
    inferenceSteps: 75,
    loraConfig: "base_lora_v2.yaml",
    hint: "200 Training-Steps · 75 Inference-Steps · LoRA rank 32 · LR 5e-5",
  },
] as const;

type IntensityPresetId = typeof INTENSITY_PRESETS[number]["id"];

export function blankFeedback(): VideoFeedback {
  return {
    dimensions: { stil_treue: 3, bewegung: 3, schaerfe: 3, stimmung: 3, motiv_genauigkeit: 3 },
    issues: [],
    overall: 0,
    freeText: "",
  };
}

// ── YouTube-Suchanfragen aus Haupt-Motiv ───────────────────────────────────
// Logik: Wie bei rabbit_lake — 2 Klassen mit unterschiedlichem Lernziel:
//   Klasse A = Charakter/Tier-Loops (kurze Animations-Loops, keine Compilations)
//   Klasse B = Hintergrund/Atmosphäre (Lo-Fi Animated Wallpaper, Live Wallpaper)
//
// Backend filtert Videos > 30min automatisch raus.
// Queries sind immer Englisch — YouTube-Algorithmus reagiert schlecht auf Deutsch.

const DE_TO_EN: Record<string, string> = {
  // Tiere (→ Klasse A: Charakter-Referenz)
  hase: "rabbit", hasen: "rabbit", kaninchen: "rabbit",
  katze: "cat", kater: "cat", kätzchen: "cat",
  hund: "dog", welpe: "puppy",
  fuchs: "fox", wolf: "wolf",
  bär: "bear", panda: "panda",
  ente: "duck", vogel: "bird",
  // Menschen
  mädchen: "girl", junge: "boy", frau: "woman", mann: "man",
  // Settings (→ Klasse B: Hintergrund-Referenz)
  see: "lake", teich: "pond", fluss: "river",
  meer: "ocean", strand: "beach",
  wald: "forest", baume: "trees", baum: "tree",
  regen: "rain", regentropfen: "rain",
  nacht: "night", dunkel: "dark",
  abend: "evening", dammerung: "dusk",
  fenster: "window", balkon: "balcony",
  cafe: "cafe", kaffee: "coffee",
  zimmer: "room", schreibtisch: "desk",
  garten: "garden", blumen: "flowers",
  berg: "mountain", schnee: "snow", winter: "winter",
  herbst: "autumn", blatter: "leaves",
  mond: "moon", sterne: "stars", stern: "star",
  laterne: "lantern", kerze: "candle",
  sakura: "sakura", kirsche: "cherry",
  brucke: "bridge", zug: "train",
  dach: "rooftop", buch: "book", bibliothek: "library",
  // Englische Wörter die direkt durchkommen
  lake: "lake", night: "night", rain: "rain", forest: "forest",
  rabbit: "rabbit", cat: "cat", window: "window",
};

const SKIP_WORDS = new Set([
  "vor", "einem", "bei", "im", "in", "an", "auf", "der", "die", "das",
  "ein", "eine", "und", "mit", "zu", "von", "am", "dem", "den", "des",
  "beim", "zum", "zur", "neben", "hinter", "uber", "unter", "zwischen",
  "durch", "gegen", "ohne", "seit", "fur", "ist", "sind", "und",
]);

const CHAR_KEYWORDS = new Set(["rabbit", "cat", "dog", "fox", "bear", "panda", "duck", "bird", "girl", "boy", "woman", "man", "puppy", "wolf"]);

function subjectToKeywords(subject: string): { chars: string[]; setting: string[] } {
  // Umlaute normalisieren für den Lookup
  const normalized = subject.toLowerCase()
    .replace(/ä/g, "a").replace(/ö/g, "o").replace(/ü/g, "u").replace(/ß/g, "ss")
    .replace(/[^a-z\s]/g, " ")
    .split(/\s+/)
    .filter(w => w.length > 2 && !SKIP_WORDS.has(w));

  const englishKws: string[] = [];
  for (const w of normalized) {
    const mapped = DE_TO_EN[w];
    if (mapped && !englishKws.includes(mapped)) englishKws.push(mapped);
    else if (!mapped && /^[a-z]{3,}$/.test(w) && !SKIP_WORDS.has(w) && !englishKws.includes(w)) {
      englishKws.push(w); // already English
    }
  }

  return {
    chars:   englishKws.filter(k => CHAR_KEYWORDS.has(k)).slice(0, 2),
    setting: englishKws.filter(k => !CHAR_KEYWORDS.has(k)).slice(0, 3),
  };
}

function generateSearchQueries(subject: string, _style: VisualStyle | null): [string, string] {
  const { chars, setting } = subjectToKeywords(subject);
  const charKw    = chars.length   ? chars.join(" ")          : "lofi girl";
  const settingKw = setting.length ? setting.slice(0, 2).join(" ") : "night nature";
  return [
    `lofi ${charKw} animated music background`,
    `lofi ${settingKw} animated music background`,
  ];
}

// Mehrere Suchvorschläge pro Klasse — als Chips anzeigen, direkt klickbar
function generateQuerySuggestions(subject: string, _style: VisualStyle | null): [string[], string[]] {
  const { chars, setting } = subjectToKeywords(subject);
  const charKw    = chars.length   ? chars.join(" ")          : "lofi girl";
  const settingKw = setting.length ? setting.slice(0, 2).join(" ") : "night nature";
  const char1     = chars[0] ?? "lofi girl";
  const set1      = setting[0] ?? "night";
  const set2      = setting[1] ?? "nature";

  return [
    // Klasse A — Motiv (wie "Lofi for Rabbits" 55min, 83k Views)
    [
      `lofi ${charKw} animated music background`,
      `lofi ${char1} music live wallpaper`,
      `${char1} lofi beats animated`,
      `lofi ${charKw} ${set1} music`,
    ],
    // Klasse B — Atmosphäre (wie "No AI Lofi by the Lake" 61min)
    [
      `lofi ${settingKw} animated music background`,
      `lofi ${set1} music live wallpaper`,
      `chill lofi ${set1} ${set2} beats`,
      `lo-fi hip hop ${settingKw} music video`,
    ],
  ];
}

// ── Haupt-Komponente ────────────────────────────────────────────────────────

function VideoPage() {
  const [phase, setPhase] = useState<Phase>("gallery");
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const [setup, setSetup] = useState<ScenarioSetup>(blankSetup());
  const [sources, setSources] = useState<SourceClass[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [resultVideoUrl, setResultVideoUrl] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<VideoFeedback>(blankFeedback());
  const [activeJobType, setActiveJobType] = useState<"train" | "generate">("train");
  const [currentProjectId, setCurrentProjectId] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved">("idle");

  const startNew = () => {
    setSetup(blankSetup());
    setSources([]);
    setJobId(null);
    setResultVideoUrl(null);
    setFeedback(blankFeedback());
    setScenarioId(null);
    setCurrentProjectId(null);
    setSaveStatus("idle");
    setPhase("setup");
  };

  const saveProject = async () => {
    setSaveStatus("saving");
    const id = currentProjectId ?? (setup.name || `projekt_${Date.now()}`);
    const sourcesForSave = sources.map(({ searching: _s, manualFiles: _m, ...rest }) => rest);
    try {
      await videoRequest<{ ok: boolean; id: string }>("/api/video/save-project", {
        method: "POST",
        body: JSON.stringify({
          id,
          name: setup.name || id,
          phase,
          setup,
          sources: sourcesForSave,
          scenarioId,
          created: currentProjectId ? undefined : new Date().toISOString(),
        }),
      });
      setCurrentProjectId(id);
      setSaveStatus("saved");
      setTimeout(() => setSaveStatus("idle"), 2500);
    } catch {
      setSaveStatus("idle");
    }
  };

  const loadProject = (p: SavedProject) => {
    setSetup(p.setup);
    setSources(
      (p.sources as SourceClass[]).map((sc) => ({
        ...sc,
        searching: false,
        manualFiles: [],
        suggestions: sc.suggestions ?? [],
      }))
    );
    setScenarioId(p.scenarioId);
    setCurrentProjectId(p.id);
    setJobId(null);
    setResultVideoUrl(null);
    setFeedback(blankFeedback());
    setSaveStatus("idle");
    setPhase(p.phase === "training" || p.phase === "result" || p.phase === "feedback"
      ? "sources" : p.phase);
  };

  return (
    <div className="space-y-7">
      <header className="border-b hairline pb-5">
        <div className="mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
          LTX-Video 13B + LoRA
        </div>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-foreground">Video</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Eigene Lo-Fi Szene trainieren und als animierten Loop / GIF exportieren.
        </p>
      </header>

      {/* Phasen-Stepper */}
      {phase !== "gallery" && (
        <PhaseBar current={phase} />
      )}

      {/* Speichern-Button (sichtbar ab Setup-Phase) */}
      {phase !== "gallery" && (
        <div className="flex justify-end">
          <button
            onClick={saveProject}
            disabled={saveStatus === "saving" || !setup.name}
            className="inline-flex h-8 items-center gap-1.5 rounded-full border hairline px-3 text-xs text-muted-foreground transition hover:border-warm/40 hover:text-foreground disabled:opacity-40"
          >
            {saveStatus === "saving" ? (
              <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Speichert…</>
            ) : saveStatus === "saved" ? (
              <><CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" /> Gespeichert</>
            ) : (
              <><Star className="h-3.5 w-3.5" /> Projekt speichern</>
            )}
          </button>
        </div>
      )}

      {phase === "gallery" && (
        <GalleryPhase onStartNew={startNew} onLoadProject={loadProject} />
      )}
      {phase === "setup" && (
        <SetupPhase
          setup={setup}
          onChange={setSetup}
          onBack={() => setPhase("gallery")}
          onNext={(sid) => {
            setScenarioId(sid);
            const [q1, q2] = generateSearchQueries(setup.mainSubject, setup.visualStyle);
            const [s1, s2] = generateQuerySuggestions(setup.mainSubject, setup.visualStyle);
            setSources([
              {
                id: "class_a",
                label: "Motiv-Referenz",
                query: q1,
                suggestions: s1,
                videos: [], selectedIds: [], manualFiles: [], searching: false,
              },
              {
                id: "class_b",
                label: "Hintergrund / Atmosphäre",
                query: q2,
                suggestions: s2,
                videos: [], selectedIds: [], manualFiles: [], searching: false,
              },
            ]);
            setPhase("sources");
          }}
        />
      )}
      {phase === "sources" && (
        <SourcesPhase
          sources={sources}
          setSources={setSources}
          scenarioId={scenarioId!}
          stepsPerRound={setup.stepsPerRound}
          onBack={() => setPhase("setup")}
          onStartTraining={(jid) => {
            setJobId(jid);
            setActiveJobType("train");
            setPhase("training");
          }}
        />
      )}
      {phase === "training" && (
        <TrainingPhase
          jobId={jobId}
          jobType={activeJobType}
          scenarioId={scenarioId!}
          onDone={(videoUrl) => {
            setResultVideoUrl(videoUrl);
            setPhase("result");
          }}
          onTrainDone={(jid) => {
            setJobId(jid);
            setActiveJobType("generate");
          }}
          onBack={() => setPhase(activeJobType === "train" ? "sources" : "result")}
        />
      )}
      {phase === "result" && (
        <ResultPhase
          videoUrl={resultVideoUrl}
          scenarioId={scenarioId!}
          onFeedback={() => {
            setFeedback(blankFeedback());
            setPhase("feedback");
          }}
          onBack={() => setPhase("sources")}
        />
      )}
      {phase === "feedback" && (
        <FeedbackPhase
          feedback={feedback}
          setFeedback={setFeedback}
          scenarioId={scenarioId!}
          onNewRound={(jid) => {
            setJobId(jid);
            setActiveJobType("generate");
            setPhase("training");
          }}
          onBack={() => setPhase("result")}
        />
      )}
    </div>
  );
}

// ── Phasen-Stepper ─────────────────────────────────────────────────────────

const PHASE_ORDER: Phase[] = ["setup", "sources", "training", "result", "feedback"];
const PHASE_LABELS: Record<Phase, string> = {
  gallery: "",
  setup: "1 · Szene",
  sources: "2 · Quellen",
  training: "3 · Training",
  result: "4 · Ergebnis",
  feedback: "5 · Feedback",
};

function PhaseBar({ current }: { current: Phase }) {
  const idx = PHASE_ORDER.indexOf(current);
  return (
    <div className="flex items-center gap-1 overflow-x-auto pb-1">
      {PHASE_ORDER.map((p, i) => (
        <div key={p} className="flex items-center gap-1">
          {i > 0 && <div className="h-px w-5 shrink-0 bg-hairline" />}
          <span
            className={[
              "mono shrink-0 rounded-full px-3 py-1 text-[11px] uppercase tracking-widest",
              i === idx ? "bg-warm/15 text-warm" :
              i < idx ? "text-emerald-400" : "text-muted-foreground",
            ].join(" ")}
          >
            {i < idx ? "✓ " : ""}{PHASE_LABELS[p]}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── Phase 0: Galerie ────────────────────────────────────────────────────────

const READY_GIFS = [
  {
    videoPath: "rabbit_lake/rounds/round_08/samples/r8_s80_g9_seed777.mp4",
    title: "Rabbits by the Lake",
    description: "Zwei Hasen am dunklen See bei Nacht · rabbit_lake R8",
  },
  {
    videoPath: "rainy_window/rounds/round_14/samples/step_00100_0.mp4",
    title: "Rainy Night by the Window",
    description: "Fensterszene im Regen, warmes Licht · rainy_window R14",
  },
  {
    videoPath: "rainy_window/rounds/round_09/samples/step_00075_0_hd2x_60fps_rife.mp4",
    title: "Rainy Window — HD",
    description: "2× HD, 60fps RIFE · rainy_window R9 Remaster",
  },
];

const PHASE_LABELS_SAVE: Record<string, string> = {
  setup: "Szene",
  sources: "Quellen",
  training: "Training",
  result: "Ergebnis",
  feedback: "Feedback",
};

function GalleryPhase({
  onStartNew,
  onLoadProject,
}: {
  onStartNew: () => void;
  onLoadProject: (p: SavedProject) => void;
}) {
  const projectsQuery = useQuery<{ projects: SavedProject[] }>({
    queryKey: ["video-projects"],
    queryFn: () => videoRequest<{ projects: SavedProject[] }>("/api/video/projects"),
    refetchOnWindowFocus: true,
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) =>
      videoRequest<{ ok: boolean }>("/api/video/delete-project", {
        method: "POST",
        body: JSON.stringify({ id }),
      }),
    onSuccess: () => projectsQuery.refetch(),
  });

  const projects = projectsQuery.data?.projects ?? [];

  return (
    <div className="space-y-5">
      {/* Fertige GIF-Previews */}
      <section className="rounded-2xl border hairline bg-surface/40 p-6">
        <div className="flex items-center gap-2">
          <Film className="h-4 w-4 text-warm" />
          <h2 className="text-base font-semibold text-foreground">Bereit zur Nutzung</h2>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          Beste generierte Loops aus dem Training — direkt abspielen oder als GIF exportieren.
        </p>

        <div className="mt-5 grid gap-4 sm:grid-cols-3">
          {READY_GIFS.map((gif) => (
            <div
              key={gif.videoPath}
              className="flex flex-col gap-2 rounded-xl border hairline bg-background/30 p-3"
            >
              <video
                src={videoMediaUrl(gif.videoPath)}
                autoPlay
                loop
                muted
                playsInline
                className="aspect-video w-full rounded-lg object-cover bg-black"
              />
              <span className="text-sm font-medium text-foreground px-1">{gif.title}</span>
              <span className="text-xs text-muted-foreground px-1">{gif.description}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Gespeicherte Projekte */}
      {projects.length > 0 && (
        <section className="rounded-2xl border hairline bg-surface/40 p-6">
          <div className="flex items-center gap-2">
            <Star className="h-4 w-4 text-warm" />
            <h2 className="text-base font-semibold text-foreground">Meine Projekte</h2>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            Gespeicherte Trainings und angefangene Szenen — direkt fortsetzen.
          </p>

          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {projects.map((p) => (
              <div
                key={p.id}
                className="flex flex-col gap-2 rounded-xl border hairline bg-background/30 p-4"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="truncate font-medium text-sm text-foreground">{p.name || p.id}</div>
                    <div className="mt-0.5 text-[11px] text-muted-foreground">
                      {new Date(p.updated).toLocaleDateString("de-DE", {
                        day: "2-digit", month: "2-digit", year: "numeric",
                        hour: "2-digit", minute: "2-digit",
                      })}
                    </div>
                  </div>
                  <button
                    onClick={() => deleteMutation.mutate(p.id)}
                    disabled={deleteMutation.isPending}
                    className="shrink-0 rounded p-1 text-muted-foreground/50 hover:text-red-400 transition"
                    title="Projekt löschen"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>

                <div className="flex flex-wrap gap-1.5">
                  <span className="mono rounded-full bg-warm/10 px-2 py-0.5 text-[10px] text-warm uppercase tracking-widest">
                    {PHASE_LABELS_SAVE[p.phase] ?? p.phase}
                  </span>
                  {p.setup.visualStyle && (
                    <span className="mono rounded-full border hairline px-2 py-0.5 text-[10px] text-muted-foreground uppercase tracking-widest">
                      {p.setup.visualStyle === "lofi_anime" ? "Anime" : "Illustration"}
                    </span>
                  )}
                  {(p.sources ?? []).length > 0 && (
                    <span className="mono rounded-full border hairline px-2 py-0.5 text-[10px] text-muted-foreground">
                      {(p.sources ?? []).reduce((n, sc) => n + (sc.selectedIds?.length ?? 0), 0)} Videos
                    </span>
                  )}
                </div>

                {p.setup.mainSubject && (
                  <p className="line-clamp-2 text-[11px] text-muted-foreground">
                    {p.setup.mainSubject}
                  </p>
                )}

                <button
                  onClick={() => onLoadProject(p)}
                  className="mt-auto inline-flex h-8 items-center justify-center gap-1.5 rounded-full bg-warm/10 px-4 text-xs font-medium text-warm transition hover:bg-warm/20"
                >
                  <ArrowRight className="h-3.5 w-3.5" />
                  Fortsetzen
                </button>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Eigenes GIF erstellen */}
      <section className="rounded-2xl border hairline bg-surface/40 p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-warm" />
              <h2 className="text-base font-semibold text-foreground">Kein passendes GIF dabei?</h2>
            </div>
            <p className="mt-2 max-w-xl text-xs text-muted-foreground">
              Eigene Szene mit Referenzbildern oder Clips anlegen und per Training + Bewertung
              Schritt für Schritt verbessern.
            </p>
          </div>
          <button
            onClick={onStartNew}
            className="inline-flex h-10 shrink-0 items-center gap-2 rounded-full bg-warm px-5 text-sm font-medium text-warm-foreground transition hover:brightness-110"
          >
            <Sparkles className="h-4 w-4" />
            Eigenes GIF erstellen
          </button>
        </div>
      </section>
    </div>
  );
}

// ── Phase 1: Szenen-Setup ───────────────────────────────────────────────────

function SetupPhase({
  setup,
  onChange,
  onBack,
  onNext,
}: {
  setup: ScenarioSetup;
  onChange: (s: ScenarioSetup) => void;
  onBack: () => void;
  onNext: (scenarioId: string) => void;
}) {
  const [showNeg, setShowNeg] = useState(false);
  const u = (patch: Partial<ScenarioSetup>) => onChange({ ...setup, ...patch });

  const isComplete =
    setup.name.trim().length >= 3 &&
    setup.visualStyle !== null &&
    setup.mainSubject.trim().length >= 5 &&
    setup.prompt.trim().length >= 20;

  const fillPrompt = () => {
    if (!setup.visualStyle) return;
    u({ prompt: buildPromptTemplate(setup.visualStyle, setup.mainSubject) });
  };

  const createMutation = useMutation({
    mutationFn: () =>
      videoRequest<{ scenario_id: string }>("/api/video/create-scenario", {
        method: "POST",
        body: JSON.stringify(setup),
      }),
    onSuccess: (data) => onNext(data.scenario_id),
    onError: () => {
      onNext(setup.name.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, ""));
    },
  });

  return (
    <section className="space-y-6 rounded-2xl border hairline bg-surface/40 p-6">
      <div className="flex items-center gap-2">
        <button
          onClick={onBack}
          className="mono inline-flex items-center gap-1 text-[11px] uppercase tracking-widest text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Zurück
        </button>
      </div>

      <div>
        <h2 className="text-lg font-semibold">Neue Szene</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Name + Stil + Motiv → Prompt vorschlagen lassen → bearbeiten → Quellen hinzufügen → Training starten.
        </p>
      </div>

      {/* Szenario-ID */}
      <Field label="Szenario-ID *" hint="Kurzname für den Ordner, z.B. rabbit_lake_v2 oder cat_window">
        <input
          value={setup.name}
          onChange={(e) => u({ name: e.target.value.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "") })}
          placeholder="meine_szene"
          className="h-10 w-full rounded-lg border hairline bg-background/40 px-3 font-mono text-sm outline-none focus:border-warm/60"
        />
      </Field>

      {/* Visueller Stil */}
      <ChipField label="Visueller Stil *" hint="Bestimmt Prompt-Template und guidance_scale">
        {VISUAL_STYLES.map((s) => (
          <Chip
            key={s.id}
            active={setup.visualStyle === s.id}
            onClick={() => u({ visualStyle: s.id, guidanceScale: s.guidance })}
            label={s.label}
            hint={s.hint}
          />
        ))}
      </ChipField>

      {/* Haupt-Motiv */}
      <Field
        label="Haupt-Motiv *"
        hint="Kurz beschreiben was zu sehen ist — wird als Basis für Prompt-Vorschlag und YouTube-Suche genutzt"
      >
        <input
          value={setup.mainSubject}
          onChange={(e) => u({ mainSubject: e.target.value })}
          placeholder="z.B. »Zwei Hasen an einem dunklen See bei Nacht, warme Laterne«"
          className="h-10 w-full rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
        />
      </Field>

      {/* Prompt */}
      <div className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div>
            <div className="text-xs font-medium text-foreground">Trainings-Prompt *</div>
            <div className="text-[11px] text-muted-foreground">
              Immer mit <code className="text-warm">lofi_girl,</code> — so haben rabbit_lake und rainy_window trainiert.
              Detaillierter = besser.
            </div>
          </div>
          <button
            onClick={fillPrompt}
            disabled={!setup.visualStyle || !setup.mainSubject.trim()}
            className="shrink-0 rounded-full border hairline px-3 py-1.5 text-[11px] text-muted-foreground hover:text-foreground disabled:opacity-40 transition"
          >
            <Wand2 className="inline h-3 w-3 mr-1" />
            Vorschlag
          </button>
        </div>
        <textarea
          value={setup.prompt}
          onChange={(e) => u({ prompt: e.target.value })}
          placeholder="lofi_girl, flat anime illustration, … — hier den vollen Prompt eingeben oder Vorschlag generieren"
          rows={6}
          className="w-full rounded-lg border hairline bg-background/40 px-3 py-2 font-mono text-xs leading-relaxed outline-none focus:border-warm/60"
        />
        <div className="text-[11px] text-muted-foreground">
          {setup.prompt.split(",").filter(Boolean).length} Keywords
        </div>
      </div>

      {/* Negativ-Prompt */}
      <div className="space-y-2">
        <button
          onClick={() => setShowNeg((v) => !v)}
          className="mono flex items-center gap-1 text-[11px] uppercase tracking-widest text-muted-foreground hover:text-foreground"
        >
          <span>{showNeg ? "▾" : "▸"}</span> Negativ-Prompt {showNeg ? "verbergen" : "bearbeiten"}
        </button>
        {showNeg && (
          <textarea
            value={setup.negativePrompt}
            onChange={(e) => u({ negativePrompt: e.target.value })}
            rows={4}
            className="w-full rounded-lg border hairline bg-background/40 px-3 py-2 font-mono text-xs leading-relaxed outline-none focus:border-warm/60"
          />
        )}
      </div>

      {/* Trainings-Intensität */}
      <div className="space-y-3">
        <div>
          <div className="text-xs font-medium text-foreground">Trainings-Intensität</div>
          <div className="text-[11px] text-muted-foreground">
            Bestimmt Training-Steps, Inference-Steps und LoRA-Rank. Werte bleiben manuell editierbar.
          </div>
        </div>
        <div className="grid gap-2 sm:grid-cols-3">
          {INTENSITY_PRESETS.map((p) => {
            const active = setup.stepsPerRound === p.stepsPerRound &&
              setup.inferenceSteps === p.inferenceSteps &&
              setup.loraConfig === p.loraConfig;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => u({
                  stepsPerRound: p.stepsPerRound,
                  inferenceSteps: p.inferenceSteps,
                  loraConfig: p.loraConfig,
                })}
                className={[
                  "relative flex flex-col gap-1 rounded-xl border p-3 text-left transition",
                  active
                    ? "border-warm/50 bg-warm/10"
                    : "hairline bg-background/20 hover:border-warm/25",
                ].join(" ")}
              >
                {"recommended" in p && p.recommended && (
                  <span className="absolute right-2 top-2 mono rounded-full bg-warm/20 px-1.5 py-0.5 text-[9px] uppercase tracking-widest text-warm">
                    Empfohlen
                  </span>
                )}
                <span className="text-base">{p.emoji}</span>
                <span className="text-sm font-medium text-foreground">{p.label}</span>
                <span className="text-[11px] text-muted-foreground leading-snug">{p.desc}</span>
                <span className="mono mt-1 text-[10px] text-muted-foreground/60">{p.hint}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Detail-Parameter (feinjustierbar) */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Field label="Training-Steps" hint="50 Test · 100 Standard · 200 Intensiv">
          <input
            type="number" step="25" min="25" max="500"
            value={setup.stepsPerRound}
            onChange={(e) => u({ stepsPerRound: parseInt(e.target.value) })}
            className="h-9 w-full rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
          />
        </Field>
        <Field label="Inference-Steps" hint="30 Test · 60 Standard · 75 Intensiv">
          <input
            type="number" step="5" min="10" max="150"
            value={setup.inferenceSteps}
            onChange={(e) => u({ inferenceSteps: parseInt(e.target.value) })}
            className="h-9 w-full rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
          />
        </Field>
        <Field label="guidance_scale" hint="Anime: 9.0 · Illustration: 5.0">
          <input
            type="number" step="0.5" min="1" max="20"
            value={setup.guidanceScale}
            onChange={(e) => u({ guidanceScale: parseFloat(e.target.value) })}
            className="h-9 w-full rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
          />
        </Field>
        <Field label="Auflösung / fps" hint="Fix: 832×480, 97 Frames @ 8fps">
          <input disabled value="832×480 · 97@8fps" className="h-9 w-full rounded-lg border hairline bg-background/20 px-3 text-xs text-muted-foreground" />
        </Field>
      </div>

      {!isComplete && (
        <p className="text-xs text-amber-400/80">
          * Szenario-ID, Stil, Motiv und Prompt sind Pflicht.
        </p>
      )}

      <button
        onClick={() => createMutation.mutate()}
        disabled={!isComplete || createMutation.isPending}
        className="inline-flex h-11 items-center gap-2 rounded-full bg-warm px-6 text-sm font-medium text-warm-foreground transition hover:brightness-110 disabled:opacity-50"
      >
        {createMutation.isPending ? (
          <><Loader2 className="h-4 w-4 animate-spin" /> Szenario anlegen…</>
        ) : (
          <><ArrowRight className="h-4 w-4" /> Weiter zu Quellen</>
        )}
      </button>
    </section>
  );
}

// ── Phase 2: Quellen (YouTube + Upload) ────────────────────────────────────

function SourcesPhase({
  sources,
  setSources,
  scenarioId,
  stepsPerRound,
  onBack,
  onStartTraining,
}: {
  sources: SourceClass[];
  setSources: (s: SourceClass[]) => void;
  scenarioId: string;
  stepsPerRound: number;
  onBack: () => void;
  onStartTraining: (jobId: string) => void;
}) {
  const mandatoryClasses = sources.slice(0, 2);
  const totalSelected = sources.reduce((sum, sc) => sum + sc.selectedIds.length + sc.manualFiles.length, 0);
  const mandatoryFilled = mandatoryClasses.every(
    (sc) => sc.selectedIds.length + sc.manualFiles.length >= 3,
  );

  const updateClass = (id: string, patch: Partial<SourceClass>) => {
    setSources(sources.map((sc) => (sc.id === id ? { ...sc, ...patch } : sc)));
  };

  const searchYouTube = async (classId: string, query: string) => {
    updateClass(classId, { searching: true });
    try {
      const data = await videoRequest<{ videos: YTVideo[] }>("/api/video/youtube-search", {
        method: "POST",
        body: JSON.stringify({ query }),
      });
      updateClass(classId, { videos: data.videos, searching: false });
    } catch {
      updateClass(classId, { videos: [], searching: false });
    }
  };

  const resolveUrl = async (classId: string, url: string) => {
    try {
      const data = await videoRequest<{ video: YTVideo }>("/api/video/resolve-url", {
        method: "POST",
        body: JSON.stringify({ url }),
      });
      const sc = sources.find((s) => s.id === classId);
      if (!sc) return;
      const exists = sc.videos.some((v) => v.id === data.video.id);
      const newVideos = exists ? sc.videos : [data.video, ...sc.videos];
      const newSelected = sc.selectedIds.includes(data.video.id)
        ? sc.selectedIds
        : [...sc.selectedIds, data.video.id];
      updateClass(classId, { videos: newVideos, selectedIds: newSelected });
    } catch {
      // URL konnte nicht aufgelöst werden — Fehler im UI
    }
  };

  const trainMutation = useMutation({
    mutationFn: () => {
      const selected_videos = sources.flatMap((sc) =>
        sc.selectedIds.map((id) => {
          const v = sc.videos.find((vid) => vid.id === id);
          return {
            id,
            url: v?.url ?? `https://www.youtube.com/watch?v=${id}`,
            title: v?.title ?? "",
            class_label: sc.id,
          };
        }),
      );
      return videoRequest<{ ok: boolean; job_id: string; round: number }>("/api/video/train", {
        method: "POST",
        body: JSON.stringify({ scenario_id: scenarioId, steps: stepsPerRound, selected_videos }),
      });
    },
    onSuccess: (data) => onStartTraining(data.job_id),
  });

  return (
    <section className="space-y-5">
      <div className="flex items-center gap-3">
        <button
          onClick={onBack}
          className="mono inline-flex items-center gap-1 text-[11px] uppercase tracking-widest text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Zurück
        </button>
        <div>
          <h2 className="text-base font-semibold">Trainingsquellen</h2>
          <p className="text-xs text-muted-foreground">
            Mindestens 2 Quell-Klassen mit je ≥ 3 Videos (≈ 10+ Clips). Verschiedene Klassen = bessere Generalisierung.
          </p>
        </div>
      </div>

      <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 px-4 py-3 text-xs text-blue-300 space-y-1">
        <div><strong>Klasse A — Motiv-Referenz:</strong> Zeigt dem Modell wie dein Haupt-Motiv (Tier, Figur, Objekt) aussieht und sich bewegt. Wie rabbit_lake: Hasen-Café-Clips für Anatomie + Pose.</div>
        <div><strong>Klasse B — Hintergrund / Atmosphäre:</strong> Lehrt Stil, Beleuchtung, Bewegungs-Qualität. Wie rabbit_lake: Bambuswald-Nacht + Blockhütte-Regen für Sternen-Glitzern + Lo-Fi-Atmosphäre.</div>
        <div className="text-blue-400/70">Je Klasse mindestens 3 Videos → Backend extrahiert automatisch 13s-Clips @ 832×480 für das Training.</div>
      </div>

      <div className="space-y-4">
        {sources.map((sc, idx) => (
          <SourceClassCard
            key={sc.id}
            sc={sc}
            isOptional={idx >= 2}
            onQueryChange={(q) => updateClass(sc.id, { query: q })}
            onSearch={() => searchYouTube(sc.id, sc.query)}
            onSuggestionClick={(q) => {
              updateClass(sc.id, { query: q });
              searchYouTube(sc.id, q);
            }}
            onResolveUrl={(url) => resolveUrl(sc.id, url)}
            onToggleVideo={(vid) => {
              const sel = sc.selectedIds.includes(vid)
                ? sc.selectedIds.filter((x) => x !== vid)
                : [...sc.selectedIds, vid];
              updateClass(sc.id, { selectedIds: sel });
            }}
            onAddFiles={(files) => updateClass(sc.id, { manualFiles: [...sc.manualFiles, ...files] })}
            onRemoveFile={(i) => updateClass(sc.id, { manualFiles: sc.manualFiles.filter((_, fi) => fi !== i) })}
          />
        ))}
      </div>

      <div className="flex items-center justify-between rounded-xl border hairline bg-surface/40 px-5 py-4">
        <div>
          <div className="text-sm font-medium">
            Gesamt: {totalSelected} ausgewählte Videos
          </div>
          <div className="text-xs text-muted-foreground">
            {mandatoryFilled ? "✓ Mindestanforderung erfüllt" : "Noch nicht genug für Klassen A & B (je ≥ 3)"}
          </div>
        </div>
        <button
          onClick={() => trainMutation.mutate()}
          disabled={!mandatoryFilled || trainMutation.isPending}
          className="inline-flex h-10 items-center gap-2 rounded-full bg-warm px-5 text-sm font-medium text-warm-foreground transition hover:brightness-110 disabled:opacity-50"
        >
          {trainMutation.isPending ? (
            <><Loader2 className="h-4 w-4 animate-spin" /> Startet…</>
          ) : (
            <><Sparkles className="h-4 w-4" /> Training starten</>
          )}
        </button>
      </div>
    </section>
  );
}

function SourceClassCard({
  sc,
  isOptional,
  onQueryChange,
  onSearch,
  onSuggestionClick,
  onResolveUrl,
  onToggleVideo,
  onAddFiles,
  onRemoveFile,
}: {
  sc: SourceClass;
  isOptional: boolean;
  onQueryChange: (q: string) => void;
  onSearch: () => void;
  onSuggestionClick: (q: string) => void;
  onResolveUrl: (url: string) => Promise<void>;
  onToggleVideo: (id: string) => void;
  onAddFiles: (f: File[]) => void;
  onRemoveFile: (i: number) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const dropRef = useRef<HTMLDivElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [urlInput, setUrlInput] = useState("");
  const [urlLoading, setUrlLoading] = useState(false);
  const [urlError, setUrlError] = useState("");
  const didAutoSearch = useRef(false);

  // Auto-Suche beim ersten Laden wenn Query vorausgefüllt ist
  useEffect(() => {
    if (!didAutoSearch.current && sc.query.trim() && sc.videos.length === 0 && !sc.searching) {
      didAutoSearch.current = true;
      onSearch();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleUrlAdd = async () => {
    const trimmed = urlInput.trim();
    if (!trimmed) return;
    setUrlLoading(true);
    setUrlError("");
    try {
      await onResolveUrl(trimmed);
      setUrlInput("");
    } catch {
      setUrlError("Link konnte nicht geladen werden.");
    } finally {
      setUrlLoading(false);
    }
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const files = Array.from(e.dataTransfer.files).filter(
        (f) => f.type.startsWith("video/") || f.type.startsWith("image/"),
      );
      if (files.length) onAddFiles(files);
    },
    [onAddFiles],
  );

  const totalPicked = sc.selectedIds.length + sc.manualFiles.length;

  return (
    <div className={["rounded-2xl border bg-surface/40 p-5", isOptional ? "hairline opacity-80" : "border-warm/20"].join(" ")}>
      {/* Kopfzeile */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={["mono rounded-full px-2 py-0.5 text-[10px] uppercase tracking-widest", isOptional ? "bg-surface-2/60 text-muted-foreground" : "bg-warm/15 text-warm"].join(" ")}>
            {sc.label}
          </span>
          {!isOptional && <span className="text-[10px] text-muted-foreground">Pflicht · ≥ 3 Videos</span>}
        </div>
        {totalPicked > 0 && (
          <span className="mono text-xs text-emerald-400">{totalPicked} ausgewählt</span>
        )}
      </div>

      {/* YouTube-Suche */}
      <div className="mb-2 flex gap-2">
        <input
          value={sc.query}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onSearch()}
          placeholder="Suchanfrage — z.B. lofi rabbit animated music background"
          className="h-9 min-w-0 flex-1 rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
        />
        <button
          onClick={onSearch}
          disabled={sc.searching || !sc.query.trim()}
          className="inline-flex h-9 shrink-0 items-center gap-1.5 rounded-lg border hairline px-3 text-sm text-muted-foreground transition hover:border-warm/50 hover:text-foreground disabled:opacity-50"
        >
          {sc.searching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
          {sc.searching ? "Suche…" : "Suchen"}
        </button>
      </div>

      {/* Suchvorschläge als klickbare Chips */}
      {sc.suggestions.length > 0 && (
        <div className="mb-4 flex flex-wrap gap-1.5">
          {sc.suggestions.map((s) => (
            <button
              key={s}
              onClick={() => onSuggestionClick(s)}
              className={[
                "mono rounded-full border px-2.5 py-1 text-[10px] uppercase tracking-widest transition",
                sc.query === s
                  ? "border-warm/50 bg-warm/10 text-warm"
                  : "hairline text-muted-foreground hover:border-warm/30 hover:text-foreground",
              ].join(" ")}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* YouTube-Link direkt einfügen */}
      <div className="mb-4">
        <div className="mono mb-1.5 text-[10px] uppercase tracking-widest text-muted-foreground">
          Oder: YouTube-Link direkt einfügen
        </div>
        <div className="flex gap-2">
          <input
            value={urlInput}
            onChange={(e) => { setUrlInput(e.target.value); setUrlError(""); }}
            onKeyDown={(e) => e.key === "Enter" && handleUrlAdd()}
            placeholder="https://www.youtube.com/watch?v=…"
            className="h-9 min-w-0 flex-1 rounded-lg border hairline bg-background/40 px-3 font-mono text-xs outline-none focus:border-warm/60"
          />
          <button
            onClick={handleUrlAdd}
            disabled={urlLoading || !urlInput.trim()}
            className="inline-flex h-9 shrink-0 items-center gap-1.5 rounded-lg border hairline px-3 text-sm text-muted-foreground transition hover:border-warm/50 hover:text-foreground disabled:opacity-50"
          >
            {urlLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
            {urlLoading ? "Lädt…" : "Hinzufügen"}
          </button>
        </div>
        {urlError && <p className="mt-1 text-[11px] text-red-400">{urlError}</p>}
      </div>

      {/* Lade-Spinner */}
      {sc.searching && sc.videos.length === 0 && (
        <div className="mb-4 flex items-center gap-2 py-4 text-xs text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          YouTube wird durchsucht…
        </div>
      )}

      {/* Video-Ergebnisse als Thumbnail-Kacheln */}
      {sc.videos.length > 0 && (
        <div className="mb-4">
          <div className="mono mb-2 text-[10px] uppercase tracking-widest text-muted-foreground">
            {sc.videos.length} Ergebnisse — mindestens 3 auswählen
          </div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {sc.videos.map((v) => {
              const sel = sc.selectedIds.includes(v.id);
              return (
                <button
                  key={v.id}
                  onClick={() => onToggleVideo(v.id)}
                  className={[
                    "group flex flex-col overflow-hidden rounded-xl border text-left transition",
                    sel
                      ? "border-warm/60 bg-warm/10 ring-1 ring-warm/30"
                      : "hairline bg-background/30 hover:border-warm/30",
                  ].join(" ")}
                >
                  {/* Thumbnail */}
                  <div className="relative aspect-video w-full overflow-hidden bg-black/30">
                    {v.thumbnail ? (
                      <img
                        src={v.thumbnail}
                        alt={v.title}
                        className="h-full w-full object-cover transition group-hover:brightness-90"
                        loading="lazy"
                        onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                      />
                    ) : (
                      <div className="flex h-full w-full items-center justify-center">
                        <Film className="h-8 w-8 text-muted-foreground/30" />
                      </div>
                    )}
                    {/* Dauer-Badge */}
                    <span className="absolute bottom-1 right-1 rounded bg-black/70 px-1 py-0.5 text-[10px] text-white">
                      {v.duration}
                    </span>
                    {/* Auswahl-Checkmark */}
                    {sel && (
                      <span className="absolute left-1.5 top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-warm text-[10px] text-warm-foreground">
                        ✓
                      </span>
                    )}
                  </div>
                  {/* Meta */}
                  <div className="p-2">
                    <div className="line-clamp-2 text-[11px] font-medium leading-snug text-foreground">
                      {v.title}
                    </div>
                    <div className="mt-0.5 truncate text-[10px] text-muted-foreground">
                      {v.channel}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Eigene Dateien */}
      <div>
        <div className="mono mb-1 text-[10px] uppercase tracking-widest text-muted-foreground">
          Oder: Eigene Clips &amp; Bilder hochladen
        </div>
        <div
          ref={dropRef}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          className={[
            "flex cursor-pointer flex-col items-center gap-1.5 rounded-lg border border-dashed px-4 py-4 text-center transition",
            dragOver ? "border-warm/60 bg-warm/5" : "hairline hover:bg-surface-2/40",
          ].join(" ")}
        >
          <UploadCloud className="h-5 w-5 text-warm" />
          <span className="text-xs text-muted-foreground">MP4-Clips oder Bilder (PNG/JPG) ablegen</span>
          <input
            ref={fileRef}
            type="file"
            accept="video/mp4,image/png,image/jpeg"
            multiple
            className="hidden"
            onChange={(e) => e.target.files && onAddFiles(Array.from(e.target.files))}
          />
        </div>
        {sc.manualFiles.length > 0 && (
          <ul className="mt-2 space-y-1">
            {sc.manualFiles.map((f, i) => (
              <li key={i} className="flex items-center justify-between gap-2 rounded-md border hairline bg-background/30 px-3 py-1.5 text-xs text-muted-foreground">
                <span className="truncate">{f.name}</span>
                <button onClick={() => onRemoveFile(i)} className="shrink-0 hover:text-red-400">
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

// ── Phase 3: Training / Generierung ────────────────────────────────────────

function TrainingPhase({
  jobId,
  jobType,
  scenarioId,
  onDone,
  onTrainDone,
  onBack,
}: {
  jobId: string | null;
  jobType: "train" | "generate";
  scenarioId: string;
  onDone: (videoUrl: string) => void;
  onTrainDone: (jobId: string) => void;
  onBack: () => void;
}) {
  type JobEntry = {
    done: boolean;
    videoUrl?: string;
    videoOutput?: string;
    output?: string;
    returncode?: number;
    progress?: number;
    phase?: string;
    trainStep?: number;
    trainTotal?: number;
  };
  const poll = useQuery<Record<string, JobEntry>>({
    queryKey: ["video-jobs"],
    queryFn: () => videoRequest<Record<string, JobEntry>>("/api/jobs"),
    refetchInterval: 2000,
    enabled: !!jobId,
  });

  const job: JobEntry | undefined = jobId ? (poll.data as Record<string, JobEntry> | undefined)?.[jobId] : undefined;
  const isDone = job?.done === true;
  const isFailed = isDone && job?.returncode !== 0;
  const progress = Math.min(100, Math.max(0, job?.progress ?? 0));

  // Generate-Jobs: automatisch zu ResultPhase navigieren wenn videoUrl vorhanden
  if (isDone && !isFailed && jobType === "generate" && job?.videoUrl) {
    const url = `${videoApiBase()}${job.videoUrl}`;
    setTimeout(() => onDone(url), 1000);
  }

  const genAfterTrainMutation = useMutation({
    mutationFn: () =>
      videoRequest<{ job_id: string; video_url: string }>("/api/video/generate", {
        method: "POST",
        body: JSON.stringify({ scenario_id: scenarioId, checkpoint: "auto" }),
      }),
    onSuccess: (data) => onTrainDone(data.job_id),
  });

  // Phasen-Ringe für die 4 Schritte (fest)
  const PHASES = ["1/4 · Download", "2/4 · Dataset", "3/4 · Preprocess", "4/4 · Training"];
  const currentPhaseIdx = job?.phase?.startsWith("4") ? 3
    : job?.phase?.startsWith("3") ? 2
    : job?.phase?.startsWith("2") ? 1
    : 0;

  return (
    <section className="rounded-2xl border hairline bg-surface/40 p-6 space-y-6">
      <button
        onClick={onBack}
        className="mono inline-flex items-center gap-1 text-[11px] uppercase tracking-widest text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-3.5 w-3.5" /> Zurück
      </button>

      {/* Titel */}
      <div className="flex items-center gap-3">
        {isDone && !isFailed ? (
          <CheckCircle2 className="h-5 w-5 text-emerald-400" />
        ) : isFailed ? (
          <AlertCircle className="h-5 w-5 text-destructive" />
        ) : (
          <Loader2 className="h-5 w-5 animate-spin text-warm" />
        )}
        <h2 className="text-base font-semibold">
          {jobType === "train" ? "LoRA-Training" : "Video-Generierung"} —{" "}
          {isDone && !isFailed ? "Fertig!" : isFailed ? "Fehler" : "Läuft…"}
        </h2>
        {!isDone && (
          <span className="ml-auto mono text-2xl font-bold tabular-nums text-warm">
            {progress}%
          </span>
        )}
      </div>

      {/* Großer Fortschrittsbalken */}
      <div className="space-y-1.5">
        <div className="h-3 w-full overflow-hidden rounded-full bg-surface-2/60">
          <div
            className={[
              "h-full rounded-full transition-all duration-700",
              isFailed ? "bg-destructive/70"
              : isDone ? "bg-emerald-500"
              : "bg-warm",
            ].join(" ")}
            style={{ width: `${progress}%` }}
          />
        </div>
        {/* Phasen-Fortschrittspunkte */}
        {jobType === "train" && (
          <div className="flex justify-between px-0.5">
            {PHASES.map((label, i) => (
              <div key={label} className="flex flex-col items-center gap-0.5">
                <div className={[
                  "h-1.5 w-1.5 rounded-full transition-colors",
                  i < currentPhaseIdx ? "bg-emerald-400"
                  : i === currentPhaseIdx && !isDone ? "bg-warm animate-pulse"
                  : isDone ? "bg-emerald-400"
                  : "bg-surface-2/60",
                ].join(" ")} />
                <span className="mono text-[9px] text-muted-foreground/60 hidden sm:block">
                  {label.split("·")[0].trim()}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Aktuelle Phase + Step-Counter */}
      <div className="flex items-center justify-between rounded-xl border hairline bg-background/30 px-4 py-3">
        <div>
          <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
            {isFailed ? "Fehler" : isDone ? "Abgeschlossen" : "Aktuelle Phase"}
          </div>
          <div className="mt-0.5 text-sm font-medium text-foreground">
            {job?.phase || (isDone ? "Fertig" : "Startet…")}
          </div>
        </div>
        {jobType === "train" && (job?.trainStep ?? 0) > 0 && !isDone && (
          <div className="text-right">
            <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">Training-Step</div>
            <div className="mono mt-0.5 text-sm font-semibold tabular-nums text-warm">
              {job?.trainStep} / {job?.trainTotal ?? "?"}
            </div>
          </div>
        )}
      </div>

      {/* Live-Ausgabe (letzte Zeile) */}
      {job?.output && !isDone && (
        <div className="rounded-lg bg-black/30 px-3 py-2 font-mono text-[11px] leading-relaxed text-muted-foreground">
          <span className="text-warm/50">▶ </span>{job.output}
        </div>
      )}

      {isFailed && job?.output && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 font-mono text-xs text-destructive/80">
          {job.output}
        </div>
      )}

      {/* Nach Training: Video generieren */}
      {isDone && !isFailed && jobType === "train" && (
        <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-5 space-y-3">
          <div className="text-sm font-medium text-emerald-400">
            LoRA-Training abgeschlossen — Modell kann jetzt ein Video generieren.
          </div>
          <p className="text-xs text-muted-foreground">
            LTX-Video 13B nutzt deinen Checkpoint um ein animiertes Loop-Video zu erzeugen (~5–10 Min bei 60 Steps).
          </p>
          {genAfterTrainMutation.isError && (
            <p className="text-xs text-destructive/80">Generierung konnte nicht gestartet werden — kein Checkpoint gefunden?</p>
          )}
          <button
            onClick={() => genAfterTrainMutation.mutate()}
            disabled={genAfterTrainMutation.isPending}
            className="inline-flex h-10 items-center gap-2 rounded-full bg-warm px-5 text-sm font-medium text-warm-foreground transition hover:brightness-110 disabled:opacity-50"
          >
            {genAfterTrainMutation.isPending ? (
              <><Loader2 className="h-4 w-4 animate-spin" /> Startet…</>
            ) : (
              <><Play className="h-4 w-4" /> Video generieren</>
            )}
          </button>
        </div>
      )}

      {!isDone && !isFailed && (
        <p className="text-xs text-muted-foreground">
          {jobType === "train"
            ? "Training läuft im Hintergrund — Seite kann offen bleiben oder geschlossen werden. Fortschritt unter »Läufe« weiter verfolgbar."
            : "Generierung läuft. Video erscheint automatisch wenn fertig (~5–10 Min bei 60 Steps)."}
        </p>
      )}
    </section>
  );
}

// ── Phase 4: Ergebnis ───────────────────────────────────────────────────────

function ResultPhase({
  videoUrl,
  scenarioId,
  onFeedback,
  onBack,
}: {
  videoUrl: string | null;
  scenarioId: string;
  onFeedback: () => void;
  onBack: () => void;
}) {
  const gifMutation = useMutation({
    mutationFn: (filename: string) => {
      // videoUrl hat Format: .../scenarios/{id}/web_outputs/gen_xxx.mp4
      const parts = videoUrl?.split("/") ?? [];
      const folder = parts.length >= 2 ? parts[parts.length - 2] : "web_outputs";
      return videoRequest("/api/video/make-gif", {
        method: "POST",
        body: JSON.stringify({ scenario_id: scenarioId, round: folder, filename }),
      });
    },
  });

  const filename = videoUrl?.split("/").pop() ?? "";

  return (
    <section className="space-y-5 rounded-2xl border hairline bg-surface/40 p-6">
      <div className="flex items-center gap-3">
        <button
          onClick={onBack}
          className="mono inline-flex items-center gap-1 text-[11px] uppercase tracking-widest text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Zurück
        </button>
        <h2 className="text-base font-semibold">Ergebnis</h2>
      </div>

      {videoUrl ? (
        <div className="overflow-hidden rounded-xl border hairline">
          <video
            src={videoUrl}
            autoPlay
            loop
            muted
            playsInline
            className="w-full"
          />
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed hairline bg-background/30 py-16">
          <Film className="h-8 w-8 text-muted-foreground" />
          <p className="text-xs text-muted-foreground">Video noch nicht verfügbar</p>
        </div>
      )}

      <div className="flex flex-wrap gap-3">
        {videoUrl && (
          <a
            href={videoUrl}
            download
            className="inline-flex h-10 items-center gap-2 rounded-full border hairline px-4 text-sm text-muted-foreground hover:text-foreground"
          >
            MP4 herunterladen
          </a>
        )}
        {filename && (
          <button
            onClick={() => gifMutation.mutate(filename)}
            disabled={gifMutation.isPending}
            className="inline-flex h-10 items-center gap-2 rounded-full border hairline px-4 text-sm text-muted-foreground transition hover:border-warm/50 hover:text-foreground disabled:opacity-50"
          >
            {gifMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Film className="h-4 w-4" />}
            Als GIF exportieren
          </button>
        )}
        {gifMutation.isSuccess && (
          <span className="flex items-center gap-1.5 text-xs text-emerald-400">
            <CheckCircle2 className="h-4 w-4" /> GIF gespeichert
          </span>
        )}
      </div>

      <AutoMetricsCard videoUrl={videoUrl} scenarioId={scenarioId} />

      <div className="rounded-lg border border-warm/20 bg-warm/5 px-4 py-4">
        <div className="text-sm font-medium">Wie gefällt dir das Ergebnis?</div>
        <p className="mt-1 text-xs text-muted-foreground">
          Strukturiertes Feedback steuert die nächste Generierung gezielt — jede Dimension mappt auf einen Parameter.
        </p>
        <button
          onClick={onFeedback}
          className="mt-3 inline-flex h-10 items-center gap-2 rounded-full bg-warm px-5 text-sm font-medium text-warm-foreground transition hover:brightness-110"
        >
          <Wand2 className="h-4 w-4" />
          Bewerten &amp; verbessern
        </button>
      </div>
    </section>
  );
}

// ── Auto-Metriken Karte ─────────────────────────────────────────────────────

type VideoMetrics = {
  temporal_ssim?: { mean: number };
  sharpness?: { normalized: number };
  flicker?: { mean: number };
  motion?: { mean: number; smoothness?: number };
  color_consistency?: { consistency: number };
  brightness_consistency?: { consistency: number; mean_brightness: number };
  overall_score?: number;
};

function MetricBar({ label, value, good, low, unit = "" }: {
  label: string; value: number; good: number; low: number; unit?: string;
}) {
  const pct = Math.min(100, Math.max(0, ((value - low) / (good - low)) * 100));
  const isGood = value >= good;
  const isMed = value >= low + (good - low) * 0.5;
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px]">
        <span className="text-muted-foreground">{label}</span>
        <span className={isGood ? "text-emerald-400" : isMed ? "text-amber-400" : "text-red-400"}>
          {value.toFixed(3)}{unit}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-2/60">
        <div
          className={["h-full rounded-full transition-all", isGood ? "bg-emerald-500" : isMed ? "bg-amber-500" : "bg-red-500/70"].join(" ")}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function AutoMetricsCard({ videoUrl, scenarioId }: { videoUrl: string | null; scenarioId: string }) {
  const videoRel = videoUrl
    ? videoUrl.split(`/api/video/media/${scenarioId}/`).pop() ?? ""
    : "";

  const metricsQuery = useQuery<{ ok: boolean; metrics?: VideoMetrics; clip_score?: number; error?: string }>({
    queryKey: ["video-metrics", scenarioId, videoRel],
    queryFn: () =>
      videoRequest(`/api/video/evaluate`, {
        method: "POST",
        body: JSON.stringify({ scenario_id: scenarioId, video_path: videoRel }),
      }),
    enabled: !!videoRel,
    staleTime: Infinity,
    retry: false,
  });

  if (!videoRel) return null;

  if (metricsQuery.isLoading) {
    return (
      <div className="flex items-center gap-2 rounded-lg border hairline bg-background/30 px-4 py-3 text-xs text-muted-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin" /> Metriken + CLIP-Score werden berechnet…
      </div>
    );
  }

  const m = metricsQuery.data?.metrics;
  const clipScore = metricsQuery.data?.clip_score;
  if (!m) return null;

  const overall = m.overall_score ?? 0;
  const overallGood = overall >= 0.75;
  const meanBrightness = m.brightness_consistency?.mean_brightness;
  const isOverexposed = meanBrightness !== undefined && meanBrightness > 200;

  return (
    <div className="rounded-lg border hairline bg-background/30 p-4 space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
          Automatische Metriken
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {clipScore !== undefined && (
            <span className={["mono rounded-full px-2 py-0.5 text-xs font-medium",
              clipScore >= 0.28 ? "bg-emerald-500/20 text-emerald-400" :
              clipScore >= 0.20 ? "bg-amber-500/20 text-amber-400" :
              "bg-red-500/20 text-red-400"].join(" ")}>
              CLIP {clipScore.toFixed(3)}
            </span>
          )}
          <span className={["mono rounded-full px-2 py-0.5 text-xs font-medium",
            overallGood ? "bg-emerald-500/20 text-emerald-400" : "bg-amber-500/20 text-amber-400"].join(" ")}>
            Score {overall.toFixed(3)}
          </span>
        </div>
      </div>

      {isOverexposed && (
        <div className="rounded border border-amber-500/30 bg-amber-500/8 px-3 py-2 text-[11px] text-amber-400">
          Überbelichtung erkannt (∅ Helligkeit: {meanBrightness?.toFixed(0)}/255) — Issue-Chip "Überbelichtet" setzen.
        </div>
      )}

      <div className="grid gap-2.5 sm:grid-cols-2">
        {m.temporal_ssim?.mean !== undefined && (
          <MetricBar label="Stabilität (SSIM)" value={m.temporal_ssim.mean} good={0.85} low={0.6} />
        )}
        {m.sharpness?.normalized !== undefined && (
          <MetricBar label="Schärfe" value={m.sharpness.normalized} good={0.22} low={0.05} />
        )}
        {m.motion?.mean !== undefined && (
          <MetricBar label="Bewegung" value={m.motion.mean} good={0.5} low={0} unit=" px" />
        )}
        {m.motion?.smoothness !== undefined && (
          <MetricBar label="Bewegungs-Gleichmäßigkeit" value={m.motion.smoothness} good={0.70} low={0.2} />
        )}
        {m.flicker?.mean !== undefined && (
          <MetricBar label="Flackern (niedr. = besser)" value={1 - Math.min(1, m.flicker.mean / 0.1)} good={0.95} low={0} />
        )}
        {m.color_consistency?.consistency !== undefined && (
          <MetricBar label="Farbstabilität" value={m.color_consistency.consistency} good={0.9} low={0.6} />
        )}
        {m.brightness_consistency?.consistency !== undefined && (
          <MetricBar label="Helligkeitsstabilität" value={m.brightness_consistency.consistency} good={0.85} low={0.5} />
        )}
      </div>

      {clipScore !== undefined && (
        <div className="text-[10px] text-muted-foreground/60 space-y-0.5">
          <div>CLIP ≥ 0.28 = Prompt-Treue gut · 0.20–0.28 = okay · &lt; 0.20 = Prompt überarbeiten</div>
          <div>Schärfe &gt; 0.22 · Bewegung &gt; 0.50 px/Frame · Gleichmäßigkeit &gt; 0.70 · Flackern &lt; 0.05</div>
        </div>
      )}
    </div>
  );
}

// ── Phase 5: Quantifiziertes Feedback ──────────────────────────────────────

export function FeedbackPhase({
  feedback,
  setFeedback,
  scenarioId,
  onNewRound,
  onBack,
}: {
  feedback: VideoFeedback;
  setFeedback: (f: VideoFeedback) => void;
  scenarioId: string;
  onNewRound: (jobId: string) => void;
  onBack: () => void;
}) {
  const u = (patch: Partial<VideoFeedback>) => setFeedback({ ...feedback, ...patch });

  const toggleIssue = (id: string) => {
    const issues = feedback.issues.includes(id)
      ? feedback.issues.filter((x) => x !== id)
      : [...feedback.issues, id];
    u({ issues });
  };

  const setDim = (key: FeedbackDimKey, val: number) => {
    u({ dimensions: { ...feedback.dimensions, [key]: val } });
  };

  const avgScore = Object.values(feedback.dimensions).reduce((a, b) => a + b, 0) / 5;

  const historyQuery = useQuery<{ ok: boolean; history: { lofi_qi?: number; timestamp?: string }[] }>({
    queryKey: ["feedback-history", scenarioId],
    queryFn: () => videoRequest(`/api/video/feedback-history?scenario_id=${encodeURIComponent(scenarioId)}`),
    staleTime: 30_000,
  });
  const qiHistory = (historyQuery.data?.history ?? [])
    .map((h) => h.lofi_qi)
    .filter((v): v is number => v !== undefined)
    .slice(-8);  // max 8 Punkte

  const submitMutation = useMutation({
    mutationFn: () =>
      videoRequest<{ ok: boolean; job_id?: string; needs_training: boolean; adjustments: string[] }>(
        "/api/video/quantified-feedback",
        {
          method: "POST",
          body: JSON.stringify({ scenario_id: scenarioId, feedback }),
        },
      ),
    onSuccess: (data) => {
      // Adjustments 2.5s anzeigen, dann zur nächsten Generierung navigieren
      if (data.job_id) {
        setTimeout(() => onNewRound(data.job_id!), 2500);
      }
    },
    onError: () => {
      videoRequest<{ job_id: string }>("/api/video/generate", {
        method: "POST",
        body: JSON.stringify({ scenario_id: scenarioId, checkpoint: "auto", steps: 60 }),
      }).then((r) => onNewRound(r.job_id));
    },
  });

  return (
    <section className="space-y-6 rounded-2xl border hairline bg-surface/40 p-6">
      <div className="flex items-center gap-3">
        <button
          onClick={onBack}
          className="mono inline-flex items-center gap-1 text-[11px] uppercase tracking-widest text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Zurück
        </button>
        <div>
          <h2 className="text-base font-semibold">Strukturiertes Feedback</h2>
          <p className="text-xs text-muted-foreground">
            Jede Dimension steuert einen konkreten Trainingsparameter — kein freies Raten.
          </p>
        </div>
      </div>

      {/* LOFI-QI Verlauf Sparkline */}
      {qiHistory.length >= 2 && (() => {
        const min = Math.min(...qiHistory, 0);
        const max = Math.max(...qiHistory, 1);
        const range = max - min || 1;
        const W = 200, H = 32, pad = 4;
        const pts = qiHistory.map((v, i) => {
          const x = pad + (i / (qiHistory.length - 1)) * (W - 2 * pad);
          const y = H - pad - ((v - min) / range) * (H - 2 * pad);
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        }).join(" ");
        const last = qiHistory[qiHistory.length - 1];
        const trend = qiHistory.length >= 2 ? last - qiHistory[qiHistory.length - 2] : 0;
        return (
          <div className="flex items-center gap-4 rounded-lg border hairline bg-background/20 px-4 py-2.5">
            <div className="flex-1 min-w-0">
              <div className="mono text-[9px] uppercase tracking-widest text-muted-foreground mb-1">LOFI-QI Verlauf</div>
              <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-8" style={{ maxWidth: 200 }}>
                <polyline points={pts} fill="none" stroke="currentColor" strokeOpacity="0.5" strokeWidth="1.5" className="text-warm" strokeLinecap="round" strokeLinejoin="round" />
                {qiHistory.map((v, i) => {
                  const x = pad + (i / (qiHistory.length - 1)) * (W - 2 * pad);
                  const y = H - pad - ((v - min) / range) * (H - 2 * pad);
                  return <circle key={i} cx={x} cy={y} r={i === qiHistory.length - 1 ? 3 : 2} fill="currentColor" className={i === qiHistory.length - 1 ? "text-warm" : "text-muted-foreground"} />;
                })}
              </svg>
            </div>
            <div className="text-right shrink-0">
              <div className="mono text-xl font-bold text-warm">{(last * 100).toFixed(0)}</div>
              <div className="mono text-[9px] text-muted-foreground">/ 100</div>
              {trend !== 0 && (
                <div className={`mono text-[10px] ${trend > 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {trend > 0 ? "▲" : "▼"} {(Math.abs(trend) * 100).toFixed(0)}
                </div>
              )}
            </div>
          </div>
        );
      })()}

      {/* Warum dimensionales Feedback */}
      <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 px-4 py-3 text-xs text-blue-300">
        <strong>Warum Dimensionen statt Sterne?</strong> RLHF-Forschung zeigt: dimensionale Scores
        (unabhängige Achsen) haben 40% weniger Inter-Rater-Varianz als eine einzige Gesamtnote.
        Jede Dimension hier mappt 1:1 auf einen Trainingsparameter — das Modell weiß genau was zu ändern ist.
      </div>

      {/* 5 Dimensionen */}
      <div className="space-y-5">
        {FEEDBACK_DIMS.map((dim) => {
          const val = feedback.dimensions[dim.key as FeedbackDimKey];
          return (
            <div key={dim.key}>
              <div className="mb-2 flex items-end justify-between">
                <div>
                  <div className="text-sm font-medium">{dim.label}</div>
                  <div className="mono text-[10px] text-muted-foreground">→ {dim.param}</div>
                </div>
                <span className={[
                  "mono rounded-full px-2 py-0.5 text-xs font-medium",
                  val <= 2 ? "bg-red-500/20 text-red-400" :
                  val === 3 ? "bg-amber-500/20 text-amber-400" :
                  "bg-emerald-500/20 text-emerald-400",
                ].join(" ")}>
                  {val}/5
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={5}
                step={1}
                value={val}
                onChange={(e) => setDim(dim.key as FeedbackDimKey, Number(e.target.value))}
                className="h-2 w-full accent-warm"
              />
              <div className="mt-1.5 flex justify-between">
                {[1, 2, 3, 4, 5].map((n) => (
                  <button
                    key={n}
                    onClick={() => setDim(dim.key as FeedbackDimKey, n)}
                    className={["h-6 w-6 rounded-full text-xs transition", val === n ? "bg-warm text-warm-foreground" : "text-muted-foreground hover:text-foreground"].join(" ")}
                  >
                    {n}
                  </button>
                ))}
              </div>
              {/* BARS anchor: verhaltensspezifische Beschreibung für den aktuellen Score */}
              <div className={[
                "mt-1.5 rounded px-2.5 py-1.5 text-[11px] leading-snug transition-colors",
                val <= 2 ? "bg-red-500/8 text-red-400/80" :
                val === 3 ? "bg-amber-500/8 text-amber-400/80" :
                "bg-emerald-500/8 text-emerald-400/80",
              ].join(" ")}>
                <span className="mono text-[10px] opacity-60">{val} → </span>
                {"anchors" in dim ? (dim as { anchors: Record<number, string> }).anchors[val] : ""}
              </div>
            </div>
          );
        })}
      </div>

      {/* Gesamteindruck (Sterne) */}
      <div>
        <div className="mono mb-2 text-[10px] uppercase tracking-widest text-muted-foreground">
          Gesamteindruck
        </div>
        <div className="flex gap-1.5">
          {[1, 2, 3, 4, 5].map((s) => (
            <button key={s} onClick={() => u({ overall: s })}>
              <Star className={`h-6 w-6 transition ${s <= feedback.overall ? "fill-warm text-warm" : "text-muted-foreground hover:text-warm/60"}`} />
            </button>
          ))}
        </div>
      </div>

      {/* Problem-Chips */}
      <div>
        <div className="mono mb-2 text-[10px] uppercase tracking-widest text-muted-foreground">
          Konkrete Probleme (optional)
        </div>
        <div className="flex flex-wrap gap-2">
          {ISSUE_PRESETS.map((issue) => (
            <button
              key={issue.id}
              onClick={() => toggleIssue(issue.id)}
              className={[
                "rounded-full border px-3 py-1.5 text-xs transition",
                feedback.issues.includes(issue.id)
                  ? "border-warm/50 bg-warm/15 text-warm"
                  : "hairline bg-background/40 text-muted-foreground hover:border-warm/40",
              ].join(" ")}
            >
              {issue.label}
            </button>
          ))}
        </div>
      </div>

      {/* Freitext */}
      <Field label="Weitere Wünsche (optional)">
        <textarea
          value={feedback.freeText}
          onChange={(e) => u({ freeText: e.target.value })}
          placeholder="z.B. Laterne soll heller leuchten, Hasen sollen näher zusammen sein"
          rows={2}
          className="w-full rounded-lg border hairline bg-background/40 px-3 py-2 text-sm outline-none focus:border-warm/60"
        />
      </Field>

      {/* Score-Zusammenfassung */}
      <div className="rounded-lg border hairline bg-background/30 px-4 py-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Ø Dimensions-Score</span>
          <span className={["mono text-sm font-medium", avgScore < 3 ? "text-amber-400" : "text-emerald-400"].join(" ")}>
            {avgScore.toFixed(1)} / 5
          </span>
        </div>
        {avgScore < 3 && (
          <p className="mt-1 text-xs text-amber-400/80">
            Niedrige Scores → nächste Runde mit angepassten Parametern und mehr Training-Steps.
          </p>
        )}
        {avgScore >= 4 && (
          <p className="mt-1 text-xs text-emerald-400/80">
            Hohe Scores → nächste Runde mit denselben Parametern, mehr Runden für Feintuning.
          </p>
        )}
      </div>

      {submitMutation.data && (
        <div className="space-y-2">
          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 space-y-1.5">
            <div className="flex items-center justify-between">
              <div className="text-xs font-medium text-emerald-400">
                Feedback angewendet — neue Generierung startet in 3 Sek…
              </div>
              {(submitMutation.data as { lofi_qi?: number }).lofi_qi !== undefined && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-emerald-400/60 uppercase tracking-widest mono">LOFI-QI</span>
                  <span className={[
                    "mono rounded-full px-2.5 py-0.5 text-sm font-bold",
                    (submitMutation.data as { lofi_qi?: number }).lofi_qi! >= 0.75
                      ? "bg-emerald-500/20 text-emerald-300"
                      : (submitMutation.data as { lofi_qi?: number }).lofi_qi! >= 0.50
                      ? "bg-amber-500/20 text-amber-300"
                      : "bg-red-500/20 text-red-300",
                  ].join(" ")}>
                    {((submitMutation.data as { lofi_qi?: number }).lofi_qi! * 100).toFixed(0)} / 100
                  </span>
                </div>
              )}
            </div>
            {(submitMutation.data.adjustments?.length ?? 0) > 0 ? (
              <div className="space-y-0.5">
                {submitMutation.data.adjustments.map((adj) => (
                  <div key={adj} className="mono text-[11px] text-emerald-400/70">+ {adj}</div>
                ))}
              </div>
            ) : (
              <div className="mono text-[11px] text-muted-foreground">Keine Parameter-Änderungen (Scores waren gut)</div>
            )}
          </div>
          {submitMutation.data.needs_training && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-4 py-3 text-xs text-amber-400">
              <strong>Tipp:</strong> Stil-Treue oder Bewegung waren niedrig — das bedeutet das LoRA-Modell selbst muss verbessert werden.
              Eine neue Trainingsrunde mit mehr oder besseren Referenzclips würde die Ergebnisse deutlich steigern.
            </div>
          )}
          {!submitMutation.data.job_id && (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-xs text-destructive/80">
              Kein Checkpoint gefunden — bitte zuerst ein Training durchführen.
            </div>
          )}
        </div>
      )}

      <button
        onClick={() => submitMutation.mutate()}
        disabled={submitMutation.isPending || feedback.overall === 0}
        className="inline-flex h-11 items-center gap-2 rounded-full bg-warm px-6 text-sm font-medium text-warm-foreground transition hover:brightness-110 disabled:opacity-50"
      >
        {submitMutation.isPending ? (
          <><Loader2 className="h-4 w-4 animate-spin" /> Verarbeite…</>
        ) : (
          <><Sparkles className="h-4 w-4" /> Neue Runde generieren</>
        )}
      </button>
      {feedback.overall === 0 && (
        <p className="text-xs text-muted-foreground">Gesamteindruck (Sterne) muss gesetzt sein.</p>
      )}
    </section>
  );
}

// ── Hilfs-Komponenten ───────────────────────────────────────────────────────

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div>
      <div className="mb-1.5">
        <span className="mono text-[10px] uppercase tracking-widest text-muted-foreground">{label}</span>
        {hint && <span className="ml-2 text-[10px] text-muted-foreground/60">{hint}</span>}
      </div>
      {children}
    </div>
  );
}

function ChipField({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div>
      <div className="mb-2">
        <span className="mono text-[10px] uppercase tracking-widest text-muted-foreground">{label}</span>
        {hint && <span className="ml-2 text-[10px] text-muted-foreground/60">{hint}</span>}
      </div>
      <div className="flex flex-wrap gap-2">{children}</div>
    </div>
  );
}

function Chip({ active, onClick, label, hint }: { active: boolean; onClick: () => void; label: string; hint?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "flex flex-col rounded-lg border px-3 py-2 text-left text-sm transition",
        active ? "border-warm/60 bg-warm/15 text-warm" : "hairline bg-background/30 text-muted-foreground hover:border-warm/40 hover:text-foreground",
      ].join(" ")}
    >
      <span className="font-medium">{label}</span>
      {hint && <span className="text-[10px] opacity-70">{hint}</span>}
    </button>
  );
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-b hairline pb-2">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mono mt-0.5 break-all text-xs">{value}</div>
    </div>
  );
}
