import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  Cpu,
  Database,
  Download,
  FileAudio,
  GitBranch,
  Image as ImageIcon,
  LineChart as LineChartIcon,
  Music2,
  Pause,
  Play,
  Radio,
  Repeat,
  Server,
  Settings2,
  Sparkles,
  Terminal,
  Users,
  Waves,
} from "lucide-react";
import {
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

export const Route = createFileRoute("/")({
  component: Index,
});

/* ---------- shared bits ---------- */

const nav = [
  { id: "projekt", label: "Projekt" },
  { id: "pipeline", label: "Pipeline" },
  { id: "musikmodell", label: "Musikmodell" },
  { id: "evaluation", label: "Evaluation" },
  { id: "websystem", label: "Websystem" },
  { id: "team", label: "Team" },
];

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="mono flex items-center gap-2 text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
      <span className="h-px w-6 bg-warm/60" />
      {children}
    </div>
  );
}

function useReveal<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setShown(true);
          io.disconnect();
        }
      },
      { threshold: 0.15 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return { ref, shown };
}

function Reveal({
  children,
  delay = 0,
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  const { ref, shown } = useReveal<HTMLDivElement>();
  return (
    <div
      ref={ref}
      className={className}
      style={{
        opacity: shown ? 1 : 0,
        transform: shown ? "translateY(0)" : "translateY(14px)",
        transition: `opacity 700ms cubic-bezier(0.2,0.7,0.2,1) ${delay}ms, transform 700ms cubic-bezier(0.2,0.7,0.2,1) ${delay}ms`,
      }}
    >
      {children}
    </div>
  );
}

/* ---------- hero waveform ---------- */

function HeroWave() {
  // Deterministic pseudo waveform
  const bars = Array.from({ length: 96 }, (_, i) => {
    const v =
      (Math.sin(i * 0.31) + Math.sin(i * 0.11 + 1.3) + Math.sin(i * 0.7 + 2.1)) /
        3 +
      1;
    return 0.2 + Math.abs(v) * 0.4;
  });
  return (
    <div className="relative h-full w-full overflow-hidden rounded-2xl border hairline bg-surface/40">
      <div className="absolute inset-0 grid-bg opacity-40" />
      {/* connection nodes SVG */}
      <svg
        viewBox="0 0 600 400"
        className="absolute inset-0 h-full w-full"
        aria-hidden
      >
        <defs>
          <linearGradient id="lg" x1="0" x2="1">
            <stop offset="0" stopColor="var(--warm)" stopOpacity="0.0" />
            <stop offset="0.5" stopColor="var(--warm)" stopOpacity="0.8" />
            <stop offset="1" stopColor="var(--cool)" stopOpacity="0.2" />
          </linearGradient>
        </defs>
        {[
          [60, 90, 220, 140],
          [220, 140, 380, 90],
          [380, 90, 540, 180],
          [220, 140, 300, 260],
          [300, 260, 460, 310],
          [60, 90, 300, 260],
        ].map(([x1, y1, x2, y2], i) => (
          <line
            key={i}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke="url(#lg)"
            strokeWidth="1"
            strokeDasharray="4 6"
            className="animate-dash"
            style={{ animationDelay: `${i * 0.4}s` }}
          />
        ))}
        {[
          [60, 90],
          [220, 140],
          [380, 90],
          [540, 180],
          [300, 260],
          [460, 310],
        ].map(([cx, cy], i) => (
          <g key={i}>
            <circle cx={cx} cy={cy} r="10" fill="var(--warm)" opacity="0.08" />
            <circle
              cx={cx}
              cy={cy}
              r="3"
              fill="var(--warm)"
              className="animate-pulse-soft"
              style={{ animationDelay: `${i * 0.35}s` }}
            />
          </g>
        ))}
      </svg>

      {/* waveform */}
      <div className="absolute inset-x-0 bottom-0 flex h-40 items-end gap-[3px] px-6 pb-6">
        {bars.map((h, i) => (
          <div
            key={i}
            className="flex-1 rounded-sm"
            style={{
              height: `${h * 100}%`,
              background:
                i % 8 === 0
                  ? "var(--warm)"
                  : "color-mix(in oklab, var(--foreground) 30%, transparent)",
              opacity: i % 8 === 0 ? 0.9 : 0.5,
              transition: "height 400ms ease",
            }}
          />
        ))}
      </div>

      {/* corner readouts */}
      <div className="mono absolute left-4 top-4 flex flex-col gap-1 text-[10px] uppercase tracking-widest text-muted-foreground">
        <span>node.pipeline</span>
        <span className="text-warm/80">status: idle · sr 44.1 kHz</span>
      </div>
      <div className="mono absolute right-4 top-4 text-[10px] uppercase tracking-widest text-muted-foreground">
        <span>lofi.gen / v0.1</span>
      </div>
    </div>
  );
}

/* ---------- pipeline ---------- */

const pipelineSteps = [
  {
    n: "01",
    title: "Datenerfassung",
    cardTitle: "Datenerfassung",
    desc: "Sammeln von Lo-Fi-Audiodaten und zugehörigen Metadaten aus definierten Quellen.",
    icon: Database,
  },
  {
    n: "02",
    title: "Audio- und Metadatenverarbeitung",
    cardTitle: "Audio & Metadaten",
    desc: "Normalisierung, Segmentierung und Bereinigung. Extraktion von Merkmalen wie Tempo und Klangfarbe.",
    icon: Waves,
  },
  {
    n: "03",
    title: "Trainingsdatenerstellung",
    cardTitle: "Trainingsdaten",
    desc: "Aufbereitung strukturierter Trainingsdaten mit konsistenter Sample Rate und Cliplänge.",
    icon: FileAudio,
  },
  {
    n: "04",
    title: "Modellanpassung und Training",
    cardTitle: "Modelltraining",
    desc: "Fine-Tuning eines generativen Audiomodells auf dem kuratierten Lo-Fi-Datensatz.",
    icon: Cpu,
  },
  {
    n: "05",
    title: "Lo-Fi-Musikgenerierung",
    cardTitle: "Musikgenerierung",
    desc: "Erzeugung kurzer, stimmungskonsistenter Audioclips mit optionaler Loop-Ausgabe.",
    icon: Music2,
  },
  {
    n: "06",
    title: "Automatisierte GIF-Produktion",
    cardTitle: "GIF-Produktion",
    desc: "Erzeugung passender visueller Loops zur generierten Musik durch ein separates Modell.",
    icon: ImageIcon,
  },
  {
    n: "07",
    title: "Evaluation und Ergebnisdarstellung",
    cardTitle: "Evaluation",
    desc: "Analyse der Ausgaben anhand definierter Scorer und Vergleich mit einer Baseline.",
    icon: Activity,
  },
];

function Pipeline() {
  const [active, setActive] = useState(0);
  return (
    <div>
      <div className="mb-10 grid gap-6 md:grid-cols-[1fr_auto] md:items-end">
        <div>
          <SectionLabel>04 · Pipeline</SectionLabel>
          <h2 className="mt-4 text-3xl font-semibold tracking-tight md:text-5xl">
            Sieben Schritte, ein reproduzierbarer Prozess.
          </h2>
        </div>
        <p className="max-w-md text-sm text-muted-foreground">
          Jeder Schritt kapselt eine klar abgegrenzte Verantwortung. Die Übergänge
          zwischen den Schritten sind versioniert und automatisiert.
        </p>
      </div>

      {/* horizontal on desktop, vertical on mobile */}
      <div className="relative rounded-2xl border hairline bg-surface/40 p-4 md:p-8">
        <div className="absolute inset-0 grid-bg opacity-30" />
        <div className="relative grid gap-3 sm:grid-cols-2 lg:grid-cols-4 2xl:grid-cols-7">
          {pipelineSteps.map((s, i) => {
            const Icon = s.icon;
            const isActive = i === active;
            return (
              <button
                key={s.n}
                onMouseEnter={() => setActive(i)}
                onFocus={() => setActive(i)}
                onClick={() => setActive(i)}
                className={`group relative flex min-h-[96px] min-w-0 flex-col justify-between overflow-hidden rounded-xl border p-3 text-left transition-all md:min-h-[110px] md:p-4 ${
                  isActive
                    ? "border-warm/60 bg-surface-2"
                    : "hairline bg-transparent hover:bg-surface-2/60"
                }`}
              >
                <div className="mono mb-2 flex items-center justify-between text-[10px] uppercase tracking-widest text-muted-foreground">
                  <span>{s.n}</span>
                  <Icon
                    className={`h-3.5 w-3.5 shrink-0 ${isActive ? "text-warm" : ""}`}
                  />
                </div>
                <div lang="de" className="max-w-full whitespace-normal text-sm font-medium leading-snug break-words hyphens-auto [overflow-wrap:anywhere]">
                  {s.cardTitle}
                </div>
                {i < pipelineSteps.length - 1 && (
                  <span className="pointer-events-none absolute right-[-6px] top-1/2 hidden h-px w-3 bg-hairline 2xl:block" />
                )}
              </button>
            );
          })}
        </div>

        <div className="relative mt-6 grid gap-6 rounded-xl border hairline bg-background/60 p-6 md:grid-cols-[auto_1fr_auto] md:items-center">
          <div className="mono text-4xl font-semibold text-warm">
            {pipelineSteps[active].n}
          </div>
          <div>
            <div className="text-lg font-medium">{pipelineSteps[active].title}</div>
            <p className="mt-1 text-sm text-muted-foreground">
              {pipelineSteps[active].desc}
            </p>
          </div>
          <div className="mono flex items-center gap-2 text-[10px] uppercase tracking-widest text-muted-foreground">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-warm animate-pulse-soft" />
            step {active + 1} / {pipelineSteps.length}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------- audio player ---------- */

const clips = [
  {
    name: "Warm Piano",
    params: "sr 44.1kHz · 32s · loop",
    seed: 1,
  },
  { name: "Rainy Night", params: "sr 44.1kHz · 28s · loop", seed: 2 },
  { name: "Dusty Drums", params: "sr 44.1kHz · 24s · one-shot", seed: 3 },
  {
    name: "Late Study Session",
    params: "sr 44.1kHz · 40s · loop",
    seed: 4,
  },
];

function Waveform({ seed, playing }: { seed: number; playing: boolean }) {
  const bars = Array.from({ length: 64 }, (_, i) => {
    const v =
      (Math.sin(i * (0.2 + seed * 0.03)) +
        Math.sin(i * 0.09 + seed) +
        Math.sin(i * 0.5 + seed * 0.7)) /
        3 +
      1;
    return 0.15 + Math.abs(v) * 0.5;
  });
  return (
    <div className="flex h-14 items-end gap-[2px]">
      {bars.map((h, i) => (
        <div
          key={i}
          className="flex-1 rounded-sm transition-all"
          style={{
            height: `${h * 100}%`,
            background:
              i < 22
                ? "var(--warm)"
                : "color-mix(in oklab, var(--foreground) 25%, transparent)",
            opacity: playing && i > 20 ? 0.8 : 0.55,
          }}
        />
      ))}
    </div>
  );
}

function ClipCard({ clip, idx }: { clip: (typeof clips)[number]; idx: number }) {
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(true);
  const dur = 32 + idx * 4;
  const cur = playing ? Math.floor(dur * 0.32) : 0;
  const fmt = (s: number) =>
    `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  return (
    <div className="group rounded-2xl border hairline bg-surface/60 p-5 transition-all hover:border-warm/40">
      <div className="mb-4 flex items-start justify-between">
        <div>
          <div className="text-base font-medium">{clip.name}</div>
          <div className="mono mt-1 text-[10px] uppercase tracking-widest text-muted-foreground">
            {clip.params}
          </div>
        </div>
        <button
          className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-warm/40 bg-warm/10 text-warm transition hover:bg-warm/20"
          onClick={() => setPlaying((p) => !p)}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? (
            <Pause className="h-4 w-4" />
          ) : (
            <Play className="h-4 w-4 translate-x-[1px]" />
          )}
        </button>
      </div>

      <Waveform seed={clip.seed} playing={playing} />

      <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-hairline/60">
        <div
          className="h-full bg-warm transition-all duration-500"
          style={{ width: playing ? "34%" : "0%" }}
        />
      </div>
      <div className="mono mt-2 flex items-center justify-between text-[10px] uppercase tracking-widest text-muted-foreground">
        <span>{fmt(cur)}</span>
        <span>{fmt(dur)}</span>
      </div>

      <div className="mt-4 flex items-center justify-between border-t hairline pt-4">
        <button
          onClick={() => setLoop((l) => !l)}
          className={`mono inline-flex items-center gap-2 rounded-md border px-2 py-1 text-[10px] uppercase tracking-widest transition ${
            loop
              ? "border-warm/40 bg-warm/10 text-warm"
              : "hairline text-muted-foreground hover:text-foreground"
          }`}
        >
          <Repeat className="h-3 w-3" />
          Loop
        </button>
        <button className="mono inline-flex items-center gap-2 text-[10px] uppercase tracking-widest text-muted-foreground transition hover:text-foreground">
          <Download className="h-3.5 w-3.5" />
          .mp3
        </button>
      </div>
    </div>
  );
}

/* ---------- evaluation chart ---------- */

const radarData = [
  { k: "Lautheit", baseline: 82, model: 78 },
  { k: "Bassanteil", baseline: 74, model: 71 },
  { k: "Dynamik", baseline: 65, model: 58 },
  { k: "Klangfarbe", baseline: 80, model: 76 },
  { k: "Rauschanteil", baseline: 68, model: 82 },
  { k: "Rhythmische Stabilität", baseline: 78, model: 73 },
  { k: "Lo-Fi-Stiltreue", baseline: 85, model: 79 },
  { k: "Loop-Qualität", baseline: 72, model: 68 },
];

function EvaluationChart() {
  return (
    <div className="h-[420px] w-full">
      <ResponsiveContainer>
        <RadarChart data={radarData} outerRadius="72%">
          <PolarGrid stroke="var(--hairline)" />
          <PolarAngleAxis
            dataKey="k"
            tick={{
              fill: "var(--muted-foreground)",
              fontSize: 11,
              fontFamily: "Inter",
            }}
          />
          <PolarRadiusAxis
            tick={{ fill: "transparent" }}
            axisLine={false}
            stroke="var(--hairline)"
          />
          <Radar
            name="Baseline"
            dataKey="baseline"
            stroke="var(--cool)"
            fill="var(--cool)"
            fillOpacity={0.15}
            strokeWidth={1.5}
          />
          <Radar
            name="Modell"
            dataKey="model"
            stroke="var(--warm)"
            fill="var(--warm)"
            fillOpacity={0.2}
            strokeWidth={1.8}
          />
          <Tooltip
            contentStyle={{
              background: "var(--surface-2)",
              border: "1px solid var(--hairline)",
              borderRadius: 8,
              fontSize: 12,
            }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ---------- gif mockups ---------- */

function GifMock({
  title,
  gradient,
  children,
}: {
  title: string;
  gradient: string;
  children: React.ReactNode;
}) {
  return (
    <div className="group relative overflow-hidden rounded-2xl border hairline bg-surface/40">
      <div
        className="relative aspect-[4/5] w-full"
        style={{ background: gradient }}
      >
        {children}
        <div className="mono absolute left-3 top-3 rounded-md border border-white/10 bg-black/30 px-2 py-1 text-[10px] uppercase tracking-widest text-white/80 backdrop-blur">
          loop · 6s
        </div>
      </div>
      <div className="flex items-center justify-between border-t hairline px-4 py-3">
        <div className="text-sm font-medium">{title}</div>
        <ImageIcon className="h-3.5 w-3.5 text-muted-foreground" />
      </div>
    </div>
  );
}

/* ---------- page ---------- */

function Index() {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 16);
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* NAV */}
      <header
        className={`fixed inset-x-0 top-0 z-50 transition-all ${
          scrolled
            ? "border-b hairline bg-background/80 backdrop-blur-md"
            : "border-b border-transparent"
        }`}
      >
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
          <a href="#top" className="mono flex items-center gap-2 text-sm font-semibold">
            <span className="inline-flex h-6 w-6 items-center justify-center rounded-md border border-warm/40 bg-warm/10">
              <span className="h-1.5 w-1.5 rounded-full bg-warm animate-pulse-soft" />
            </span>
            LOFI.GEN
          </a>
          <nav className="hidden items-center gap-7 md:flex">
            {nav.map((n) => (
              <a
                key={n.id}
                href={`#${n.id}`}
                className="text-sm text-muted-foreground transition hover:text-foreground"
              >
                {n.label}
              </a>
            ))}
          </nav>
          <div className="flex items-center gap-3">
            <a
              href="#pipeline"
              className="mono hidden items-center gap-2 rounded-full border border-warm/40 bg-warm/10 px-3.5 py-1.5 text-[11px] uppercase tracking-widest text-warm transition hover:bg-warm/20 sm:inline-flex"
            >
              Pipeline ansehen
              <ArrowRight className="h-3 w-3" />
            </a>
            <Link
              to="/studio"
              className="mono inline-flex items-center gap-2 rounded-full bg-warm px-3.5 py-1.5 text-[11px] uppercase tracking-widest text-warm-foreground transition hover:brightness-110"
            >
              Studio öffnen
              <ArrowUpRight className="h-3 w-3" />
            </Link>
          </div>
        </div>
      </header>

      {/* HERO */}
      <section id="top" className="relative overflow-hidden pt-32 md:pt-40">
        <div className="pointer-events-none absolute inset-0 grid-bg opacity-30" />
        <div
          className="pointer-events-none absolute -top-40 left-1/2 h-[520px] w-[880px] -translate-x-1/2 rounded-full opacity-40 blur-3xl"
          style={{
            background:
              "radial-gradient(closest-side, color-mix(in oklab, var(--warm) 40%, transparent), transparent)",
          }}
        />
        <div className="relative mx-auto max-w-7xl px-6 pb-24">
          <div className="grid gap-12 lg:grid-cols-[1.1fr_1fr] lg:items-center">
            <div>
              <Reveal>
                <SectionLabel>
                  Bachelorarbeit · Industrielle Automatisierungstechnik · 2026
                </SectionLabel>
              </Reveal>
              <Reveal delay={80}>
                <h1 className="mt-6 text-3xl font-semibold leading-[1.1] tracking-tight text-foreground md:text-5xl lg:text-6xl">
                  KI-gestützte Lo-Fi-Musikgenerierung und -bewertung
                </h1>
              </Reveal>
              <Reveal delay={160}>
                <p className="mt-6 max-w-xl text-base leading-relaxed text-muted-foreground md:text-lg">
                  Entwicklung einer reproduzierbaren Deep-Learning-Pipeline zur
                  Erfassung, Verarbeitung, Generierung und Bewertung von
                  Lo-Fi-Musik. Das System verbindet Datenaufbereitung,
                  Modelltraining, Audioerzeugung, GIF-Produktion und eine
                  webbasierte Benutzeroberfläche.
                </p>
              </Reveal>
              <Reveal delay={240}>
                <div className="mt-8 flex flex-wrap items-center gap-3">
                  <a
                    href="#pipeline"
                    className="inline-flex items-center gap-2 rounded-full bg-warm px-5 py-3 text-sm font-medium text-primary-foreground transition hover:brightness-110"
                  >
                    Pipeline entdecken
                    <ArrowRight className="h-4 w-4" />
                  </a>
                  <a
                    href="#musikmodell"
                    className="inline-flex items-center gap-2 rounded-full border hairline bg-surface/50 px-5 py-3 text-sm font-medium text-foreground transition hover:bg-surface-2"
                  >
                    Audioergebnisse ansehen
                    <ArrowUpRight className="h-4 w-4" />
                  </a>
                </div>
              </Reveal>
              <Reveal delay={320}>
                <div className="mono mt-10 grid max-w-lg grid-cols-3 gap-6 border-t hairline pt-6 text-[11px] uppercase tracking-widest text-muted-foreground">
                  <div>
                    <div className="text-2xl font-semibold text-foreground">7</div>
                    <div className="mt-1">Pipeline-Stufen</div>
                  </div>
                  <div>
                    <div className="text-2xl font-semibold text-foreground">2</div>
                    <div className="mt-1">Generative Modelle</div>
                  </div>
                  <div>
                    <div className="text-2xl font-semibold text-foreground">
                      8
                    </div>
                    <div className="mt-1">Bewertungs&shy;kategorien</div>
                  </div>
                </div>
              </Reveal>
            </div>

            <Reveal delay={200} className="h-[380px] md:h-[460px]">
              <HeroWave />
            </Reveal>
          </div>
        </div>
      </section>

      {/* PROJEKT */}
      <section id="projekt" className="relative py-24 md:py-32">
        <div className="mx-auto max-w-7xl px-6">
          <div className="grid gap-12 lg:grid-cols-[1fr_1.1fr]">
            <div>
              <SectionLabel>02 · Projekt</SectionLabel>
              <h2 className="mt-4 text-3xl font-semibold tracking-tight md:text-5xl">
                Von Rohdaten zur generierten Lo-Fi-Musik.
              </h2>
              <p className="mt-6 max-w-lg text-muted-foreground">
                Die Bachelorarbeit untersucht, wie eine reproduzierbare Pipeline
                aufgebaut werden kann, die Lo-Fi-Daten automatisiert erfasst,
                verarbeitet und für das Training eines generativen Musikmodells
                nutzt. Die generierten Ergebnisse werden anschließend analysiert,
                bewertet und mit einer Baseline verglichen.
              </p>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              {[
                {
                  t: "Datenerfassung",
                  d: "Automatisiertes Sammeln von Lo-Fi-Audio und strukturierten Metadaten.",
                  icon: Database,
                },
                {
                  t: "Modelltraining",
                  d: "Fine-Tuning eines Audiomodells auf dem kuratierten Datensatz.",
                  icon: Cpu,
                },
                {
                  t: "Musikgenerierung",
                  d: "Erzeugung stimmungskonsistenter Audioclips mit Loop-Option.",
                  icon: Music2,
                },
                {
                  t: "Evaluation",
                  d: "Bewertung anhand definierter Scorer und Vergleich mit Baseline.",
                  icon: Activity,
                },
              ].map((c, i) => {
                const Icon = c.icon;
                return (
                  <Reveal key={c.t} delay={i * 80}>
                    <div className="group h-full rounded-2xl border hairline bg-surface/50 p-6 transition hover:border-warm/40 hover:bg-surface-2">
                      <div className="flex items-start justify-between">
                        <div className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-warm/30 bg-warm/10 text-warm">
                          <Icon className="h-4 w-4" />
                        </div>
                        <span className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
                          0{i + 1}
                        </span>
                      </div>
                      <div className="mt-6 text-base font-medium">{c.t}</div>
                      <p className="mt-2 text-sm text-muted-foreground">{c.d}</p>
                      <div className="mt-6 h-8">
                        <svg viewBox="0 0 200 40" className="h-full w-full">
                          <path
                            d={`M0 30 ${Array.from({ length: 20 }, (_, k) => {
                              const y = 20 + Math.sin(k * 0.6 + i) * 10;
                              return `L${k * 10} ${y}`;
                            }).join(" ")}`}
                            fill="none"
                            stroke="var(--warm)"
                            strokeWidth="1"
                            strokeOpacity="0.6"
                          />
                        </svg>
                      </div>
                    </div>
                  </Reveal>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {/* PIPELINE */}
      <section id="pipeline" className="relative py-24 md:py-32">
        <div className="mx-auto max-w-7xl px-6">
          <Pipeline />
        </div>
      </section>

      {/* MUSIKMODELL */}
      <section id="musikmodell" className="relative py-24 md:py-32">
        <div className="mx-auto max-w-7xl px-6">
          <div className="grid gap-12 lg:grid-cols-[1fr_1fr] lg:items-start">
            <div>
              <SectionLabel>05 · Musikmodell</SectionLabel>
              <h2 className="mt-4 text-3xl font-semibold tracking-tight md:text-5xl">
                Generative Lo-Fi-Musik mit Deep Learning.
              </h2>
              <p className="mt-6 text-muted-foreground">
                Das Musikmodell wird mit aufbereiteten Lo-Fi-Audiodaten trainiert
                beziehungsweise angepasst. Ziel ist die Generierung kurzer,
                stimmungs&shy;konsistenter Audioclips mit ruhiger Variation,
                erkennbarer Stiltreue und möglichst sauberen Übergängen.
              </p>

              <div className="mt-8 grid grid-cols-2 gap-3">
                {[
                  { t: "Audioaufbereitung", i: Waves },
                  { t: "Trainingsdaten", i: Database },
                  { t: "Modellkonfiguration", i: Settings2 },
                  { t: "Musikgenerierung", i: Music2 },
                  { t: "Audioexport", i: FileAudio },
                  { t: "Loop-Erstellung", i: Repeat },
                ].map((x) => {
                  const I = x.i;
                  return (
                    <div
                      key={x.t}
                      className="flex items-center gap-2.5 rounded-lg border hairline bg-surface/40 px-3 py-2.5 text-sm"
                    >
                      <I className="h-3.5 w-3.5 text-warm" />
                      {x.t}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* model card */}
            <Reveal>
              <div className="relative rounded-2xl border hairline bg-surface/50 p-6 md:p-8">
                <div className="mono flex items-center justify-between text-[10px] uppercase tracking-widest text-muted-foreground">
                  <span>model.card</span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5 rounded-full bg-warm animate-pulse-soft" />
                    training · in bearbeitung
                  </span>
                </div>
                <div className="mt-4 text-xl font-medium">
                  Lo-Fi Music Generator
                </div>
                <div className="mono mt-1 text-xs text-muted-foreground">
                  fine-tuned · audio-diffusion (Projektwert)
                </div>

                <dl className="mt-6 grid grid-cols-2 gap-x-6 gap-y-4 border-t hairline pt-6">
                  {[
                    ["Modelltyp", "wird ergänzt"],
                    ["Sample Rate", "44.1 kHz"],
                    ["Clip-Länge", "wird ergänzt"],
                    ["Trainingsstatus", "in Bearbeitung"],
                    ["Datensatzgröße", "Projektwert"],
                    ["Trainingsschritte", "wird ergänzt"],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <dt className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
                        {k}
                      </dt>
                      <dd className="mono mt-1 text-sm text-foreground">{v}</dd>
                    </div>
                  ))}
                </dl>

                <div className="mt-6 rounded-lg border hairline bg-background/60 p-4">
                  <div className="mono mb-2 text-[10px] uppercase tracking-widest text-muted-foreground">
                    training loss (demo)
                  </div>
                  <div className="h-24">
                    <ResponsiveContainer>
                      <LineChart
                        data={Array.from({ length: 30 }, (_, i) => ({
                          x: i,
                          y: 2.4 * Math.exp(-i * 0.12) + 0.2 + Math.sin(i) * 0.05,
                        }))}
                        margin={{ top: 4, right: 4, left: 4, bottom: 0 }}
                      >
                        <Line
                          dataKey="y"
                          stroke="var(--warm)"
                          strokeWidth={1.5}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </Reveal>
          </div>

          {/* AUDIO PLAYER GRID */}
          <div className="mt-20">
            <div className="mb-8 flex items-end justify-between gap-6">
              <div>
                <SectionLabel>Audio · Ergebnisse</SectionLabel>
                <h3 className="mt-3 text-2xl font-semibold tracking-tight md:text-3xl">
                  Generierte Lo-Fi-Clips.
                </h3>
              </div>
              <span className="mono hidden text-[10px] uppercase tracking-widest text-muted-foreground md:block">
                demo · nicht endgültig
              </span>
            </div>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {clips.map((c, i) => (
                <Reveal key={c.name} delay={i * 60}>
                  <ClipCard clip={c} idx={i} />
                </Reveal>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* EVALUATION */}
      <section id="evaluation" className="relative py-24 md:py-32">
        <div className="mx-auto max-w-7xl px-6">
          <div className="mb-12 max-w-3xl">
            <SectionLabel>07 · Evaluation</SectionLabel>
            <h2 className="mt-4 text-3xl font-semibold tracking-tight md:text-5xl">
              Wie nah liegt die generierte Musik am Lo-Fi-Standard?
            </h2>
            <p className="mt-5 text-muted-foreground">
              Die generierten Audios werden anhand verschiedener Scorer bewertet
              und mit einer Baseline beziehungsweise einem Genre-Standard
              verglichen. Die Grafik zeigt, in welchen Kategorien das Modell
              näher oder weiter vom Standard entfernt liegt.
            </p>
          </div>

          <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
            <div className="rounded-2xl border hairline bg-surface/40 p-6 md:p-8">
              <div className="mb-4 flex items-center justify-between">
                <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  scorer.compare · radar
                </div>
                <div className="flex items-center gap-4 text-xs">
                  <span className="inline-flex items-center gap-2 text-muted-foreground">
                    <span className="h-2 w-2 rounded-full bg-cool" /> Baseline
                  </span>
                  <span className="inline-flex items-center gap-2 text-muted-foreground">
                    <span className="h-2 w-2 rounded-full bg-warm" /> Modell
                  </span>
                </div>
              </div>
              <EvaluationChart />
              <div className="mono mt-2 text-center text-[10px] uppercase tracking-widest text-muted-foreground">
                demo-daten · zur veranschaulichung
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 content-start">
              {[
                { t: "Gesamtbewertung", v: "73 / 100", sub: "über 8 Scorer" },
                { t: "Höchste Übereinstimmung", v: "Klangfarbe", sub: "Δ −4" },
                { t: "Größte Abweichung", v: "Rauschanteil", sub: "Δ +14" },
                { t: "Baseline-Differenz", v: "−6.2 pt", sub: "gewichtet" },
              ].map((s, i) => (
                <Reveal key={s.t} delay={i * 60}>
                  <div className="h-full rounded-2xl border hairline bg-surface/50 p-5">
                    <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
                      {s.t}
                    </div>
                    <div className="mt-3 text-2xl font-semibold text-foreground">
                      {s.v}
                    </div>
                    <div className="mono mt-1 text-[10px] uppercase tracking-widest text-warm/80">
                      {s.sub}
                    </div>
                  </div>
                </Reveal>
              ))}
              <div className="col-span-2 rounded-2xl border border-warm/20 bg-warm/5 p-4 text-xs text-muted-foreground">
                Alle abgebildeten Werte sind Platzhalter. Endgültige Ergebnisse
                folgen aus dem Projektabschluss und werden hier ergänzt.
              </div>
            </div>
          </div>

          {/* wissenschaftliche tabelle */}
          <div className="mt-16">
            <div className="mb-6 flex items-end justify-between gap-6">
              <div>
                <SectionLabel>Reproduzierbar. Vergleichbar. Messbar.</SectionLabel>
                <h3 className="mt-3 max-w-2xl text-2xl font-semibold tracking-tight md:text-3xl">
                  Die Pipeline erzeugt vergleichbare Ergebnisse für jede Iteration.
                </h3>
              </div>
              <span className="mono hidden text-[10px] uppercase tracking-widest text-muted-foreground md:block">
                run.log · demo
              </span>
            </div>
            <div className="overflow-x-auto rounded-2xl border hairline bg-surface/40">
              <table className="mono w-full min-w-[720px] text-sm">
                <thead>
                  <tr className="text-left text-[10px] uppercase tracking-widest text-muted-foreground">
                    {["Audio", "Modellversion", "Baseline", "Score", "Abweichung", "Status"].map(
                      (h) => (
                        <th key={h} className="border-b hairline px-5 py-4 font-normal">
                          {h}
                        </th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody className="text-foreground">
                  {[
                    ["warm_piano_01.mp3", "v0.3.1", "78", "74", "−4", "ok"],
                    ["rainy_night_04.mp3", "v0.3.1", "76", "82", "+6", "review"],
                    ["dusty_drums_02.mp3", "v0.3.0", "70", "63", "−7", "ok"],
                    ["late_study_07.mp3", "v0.3.1", "80", "77", "−3", "ok"],
                  ].map((r, i) => (
                    <tr key={i} className="border-b hairline last:border-b-0">
                      {r.map((c, j) => (
                        <td
                          key={j}
                          className={`px-5 py-4 ${
                            j === 5
                              ? c === "ok"
                                ? "text-warm"
                                : "text-cool"
                              : ""
                          }`}
                        >
                          {c}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mono mt-3 text-[10px] uppercase tracking-widest text-muted-foreground">
              * platzhalterwerte · demo-daten
            </div>
          </div>
        </div>
      </section>

      {/* WEBSYSTEM */}
      <section id="websystem" className="relative py-24 md:py-32">
        <div className="mx-auto max-w-7xl px-6">
          <div className="mb-12 grid gap-6 md:grid-cols-[1fr_auto] md:items-end">
            <div>
              <SectionLabel>08 · Websystem</SectionLabel>
              <h2 className="mt-4 text-3xl font-semibold tracking-tight md:text-5xl">
                Eine Oberfläche für die gesamte Pipeline.
              </h2>
            </div>
            <p className="max-w-md text-sm text-muted-foreground">
              Ein webbasiertes Frontend, das Prompt, Stilwahl, Generierung,
              Vorschau und Export in einer Ansicht verbindet.
            </p>
          </div>

          <div className="overflow-hidden rounded-2xl border hairline bg-surface/40">
            {/* window chrome */}
            <div className="flex items-center justify-between border-b hairline bg-surface-2/60 px-5 py-3">
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full bg-foreground/20" />
                <span className="h-2.5 w-2.5 rounded-full bg-foreground/20" />
                <span className="h-2.5 w-2.5 rounded-full bg-warm/60" />
              </div>
              <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
                lofi.gen / studio
              </div>
              <div className="mono flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-muted-foreground">
                <span className="h-1.5 w-1.5 rounded-full bg-warm animate-pulse-soft" />
                connected
              </div>
            </div>

            <div className="grid gap-6 p-6 md:grid-cols-[280px_1fr] md:p-8">
              {/* sidebar */}
              <div className="space-y-4">
                <div className="rounded-xl border hairline bg-background/60 p-4">
                  <div className="mono mb-2 text-[10px] uppercase tracking-widest text-muted-foreground">
                    prompt
                  </div>
                  <div className="rounded-md border hairline bg-surface/60 p-3 text-sm">
                    Ruhiges Piano, leiser Regen, langsames Beat.
                  </div>
                </div>

                <div className="rounded-xl border hairline bg-background/60 p-4">
                  <div className="mono mb-3 text-[10px] uppercase tracking-widest text-muted-foreground">
                    stil
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {["Warm Piano", "Rainy Night", "Dusty Drums", "Study"].map(
                      (s, i) => (
                        <span
                          key={s}
                          className={`rounded-full border px-2.5 py-1 text-xs ${
                            i === 0
                              ? "border-warm/50 bg-warm/10 text-warm"
                              : "hairline text-muted-foreground"
                          }`}
                        >
                          {s}
                        </span>
                      )
                    )}
                  </div>
                </div>

                <div className="rounded-xl border hairline bg-background/60 p-4">
                  <div className="mono mb-3 flex items-center justify-between text-[10px] uppercase tracking-widest text-muted-foreground">
                    <span>clip-länge</span>
                    <span className="text-foreground">32s</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-hairline/60">
                    <div className="h-full w-[45%] rounded-full bg-warm" />
                  </div>
                </div>

                <Link
                  to="/studio/generate"
                  className="mono flex w-full items-center justify-between rounded-xl border border-warm/40 bg-warm/10 px-4 py-3 text-xs uppercase tracking-widest text-warm transition hover:bg-warm/20"
                >
                  Generierung starten
                  <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>

              {/* main */}
              <div className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-xl border hairline bg-background/60 p-4">
                    <div className="mono mb-3 flex items-center justify-between text-[10px] uppercase tracking-widest text-muted-foreground">
                      <span>audio-vorschau</span>
                      <span className="text-warm">generating · 62%</span>
                    </div>
                    <Waveform seed={3} playing />
                    <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-hairline/60">
                      <div className="h-full w-[62%] bg-warm" />
                    </div>
                  </div>
                  <div className="rounded-xl border hairline bg-background/60 p-4">
                    <div className="mono mb-3 text-[10px] uppercase tracking-widest text-muted-foreground">
                      gif-vorschau
                    </div>
                    <div
                      className="relative aspect-video overflow-hidden rounded-md"
                      style={{
                        background:
                          "radial-gradient(120% 80% at 20% 30%, color-mix(in oklab, var(--warm) 25%, transparent), transparent), linear-gradient(160deg, #1c1a17, #2a221b)",
                      }}
                    >
                      <div className="absolute inset-0 grid-bg opacity-30" />
                      <div className="mono absolute bottom-2 left-2 text-[10px] uppercase tracking-widest text-white/70">
                        rendering
                      </div>
                    </div>
                  </div>
                </div>

                <div className="rounded-xl border hairline bg-background/60 p-4">
                  <div className="mono mb-3 flex items-center justify-between text-[10px] uppercase tracking-widest text-muted-foreground">
                    <span>evaluation · vs baseline</span>
                    <span>demo</span>
                  </div>
                  <div className="grid grid-cols-4 gap-2">
                    {radarData.slice(0, 4).map((r) => (
                      <div key={r.k} className="rounded-md bg-surface/60 p-3">
                        <div className="mono text-[9px] uppercase tracking-widest text-muted-foreground">
                          {r.k}
                        </div>
                        <div className="mono mt-1 flex items-baseline gap-2">
                          <span className="text-sm text-foreground">{r.model}</span>
                          <span className="text-[10px] text-cool">/ {r.baseline}</span>
                        </div>
                        <div className="mt-2 h-1 rounded-full bg-hairline/60">
                          <div
                            className="h-full rounded-full bg-warm"
                            style={{ width: `${r.model}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border hairline bg-background/60 p-4">
                  <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
                    export
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    {[".mp3", ".wav", ".gif", "report.json"].map((e) => (
                      <span
                        key={e}
                        className="mono inline-flex items-center gap-1.5 rounded-md border hairline px-2.5 py-1.5 text-[11px] text-muted-foreground"
                      >
                        <Download className="h-3 w-3" />
                        {e}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6 flex flex-col items-center justify-between gap-4 rounded-2xl border border-warm/30 bg-warm/5 p-6 text-center sm:flex-row sm:text-left">
            <div>
              <div className="mono text-[10px] uppercase tracking-widest text-warm">Live-System</div>
              <p className="mt-1 text-sm text-muted-foreground">
                Das echte Studio läuft lokal gegen die MusicGen + LoRA-Pipeline — Generierung,
                Vorgänge, Ausgaben, Bewertungen und Projektsteuerung in einer Oberfläche.
              </p>
            </div>
            <Link
              to="/studio"
              className="mono inline-flex shrink-0 items-center gap-2 rounded-full bg-warm px-5 py-2.5 text-xs uppercase tracking-widest text-warm-foreground transition hover:brightness-110"
            >
              Studio öffnen
              <ArrowUpRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>
      </section>

      {/* GIFs */}
      <section className="relative py-24 md:py-32">
        <div className="mx-auto max-w-7xl px-6">
          <div className="mb-12 max-w-2xl">
            <SectionLabel>09 · Visuelle Loops</SectionLabel>
            <h2 className="mt-4 text-3xl font-semibold tracking-tight md:text-5xl">
              Visuelle Loops passend zur generierten Musik.
            </h2>
            <p className="mt-5 text-muted-foreground">
              Ergänzend zur Musik erzeugt die Pipeline kurze GIFs, deren
              Stimmung mit dem gewählten Lo-Fi-Stil abgestimmt ist. Schwerpunkt:
              Inan Deniz Arduc.
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <GifMock
              title="Ruhiger Schreibtisch bei Nacht"
              gradient="linear-gradient(160deg,#141216 0%,#2a221b 60%,#3a2a1c 100%)"
            >
              <div
                className="absolute inset-x-0 bottom-0 h-1/2"
                style={{
                  background:
                    "radial-gradient(60% 100% at 30% 100%, color-mix(in oklab, var(--warm) 45%, transparent), transparent)",
                }}
              />
              <div
                className="absolute right-8 top-10 h-16 w-24 rounded-md border border-warm/30 bg-warm/10 animate-pulse-soft"
              />
              <div className="absolute right-6 bottom-24 h-1 w-32 rounded-full bg-warm/40" />
            </GifMock>
            <GifMock
              title="Zugfenster im Regen"
              gradient="linear-gradient(160deg,#0f141a 0%,#1a2530 60%,#20303f 100%)"
            >
              <svg className="absolute inset-0 h-full w-full opacity-40" aria-hidden>
                {Array.from({ length: 40 }).map((_, i) => (
                  <line
                    key={i}
                    x1={i * 24}
                    y1={-20}
                    x2={i * 24 - 30}
                    y2={400}
                    stroke="white"
                    strokeOpacity="0.15"
                    strokeWidth="1"
                  />
                ))}
              </svg>
              <div className="absolute inset-x-6 top-1/3 h-24 rounded-md border border-white/10 bg-white/5" />
            </GifMock>
            <GifMock
              title="Stadtansicht mit warmem Licht"
              gradient="linear-gradient(180deg,#1a1512 0%,#2b1e15 55%,#3a281a 100%)"
            >
              <div className="absolute inset-x-0 bottom-0 flex h-2/3 items-end gap-1 px-4">
                {[0.5, 0.7, 0.4, 0.9, 0.6, 0.85, 0.55, 0.75, 0.45].map((h, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-t-sm"
                    style={{
                      height: `${h * 100}%`,
                      background:
                        "linear-gradient(to top, #0e0a08, color-mix(in oklab, var(--warm) 30%, #1b120a))",
                    }}
                  />
                ))}
              </div>
              <div className="absolute right-8 top-8 h-8 w-8 rounded-full bg-warm/60 blur-xl" />
            </GifMock>
          </div>
        </div>
      </section>

      {/* TEAM */}
      <section id="team" className="relative py-24 md:py-32">
        <div className="mx-auto max-w-7xl px-6">
          <div className="mb-12 max-w-2xl">
            <SectionLabel>11 · Team</SectionLabel>
            <h2 className="mt-4 text-3xl font-semibold tracking-tight md:text-5xl">
              Zwei Schwerpunkte, eine gemeinsame Pipeline.
            </h2>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            {[
              {
                name: "Adil Berke Savcili",
                deg: "B. Sc. Maschinenbau",
                focus: [
                  "MP3-Musikerstellung",
                  "Audioverarbeitung",
                  "Training des Musikmodells",
                  "Musikgenerierung",
                  "Evaluation der Audioergebnisse",
                ],
                icon: Music2,
                accent: true,
              },
              {
                name: "Inan Deniz Arduc",
                deg: "M. Sc. Maschinenbau",
                focus: [
                  "Videogenerierung",
                  "Training des Videomodells",
                  "Visuelle Ausgabe der Pipeline",
                ],
                icon: ImageIcon,
              },
            ].map((p, i) => {
              const I = p.icon;
              return (
                <Reveal key={p.name} delay={i * 100}>
                  <div
                    className={`h-full rounded-2xl border p-6 md:p-8 ${
                      p.accent
                        ? "border-warm/40 bg-surface-2/60"
                        : "hairline bg-surface/50"
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="text-xl font-medium">{p.name}</div>
                        <div className="mono mt-1 text-xs text-muted-foreground">
                          {p.deg}
                        </div>
                      </div>
                      <div className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-warm/30 bg-warm/10 text-warm">
                        <I className="h-4 w-4" />
                      </div>
                    </div>
                    <div className="mono mt-6 text-[10px] uppercase tracking-widest text-muted-foreground">
                      schwerpunkte
                    </div>
                    <ul className="mt-3 space-y-2">
                      {p.focus.map((f) => (
                        <li key={f} className="flex items-start gap-2 text-sm">
                          <span className="mt-2 h-1 w-1 rounded-full bg-warm" />
                          {f}
                        </li>
                      ))}
                    </ul>
                  </div>
                </Reveal>
              );
            })}
          </div>

          <div className="mt-6 rounded-2xl border hairline bg-surface/40 p-6 md:p-8">
            <div className="flex items-start gap-3">
              <GitBranch className="mt-1 h-4 w-4 text-warm" />
              <div>
                <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  gemeinsame aufgaben
                </div>
                <div className="mt-3 grid gap-3 text-sm md:grid-cols-2">
                  {[
                    "Konzeption der KI-Pipeline",
                    "Integration der Teilkomponenten",
                    "Verbindung von Musik- und Videogenerierung",
                    "Entwicklung des webbasierten Systems",
                  ].map((t) => (
                    <div key={t} className="flex items-start gap-2">
                      <span className="mt-2 h-1 w-1 rounded-full bg-warm" />
                      {t}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* HOCHSCHULE */}
      <section className="relative py-24">
        <div className="mx-auto max-w-7xl px-6">
          <div className="rounded-2xl border hairline bg-surface/40 p-6 md:p-10">
            <div className="grid gap-8 md:grid-cols-[1.2fr_1fr] md:items-center">
              <div>
                <SectionLabel>12 · Hochschulkontext</SectionLabel>
                <div className="mt-4 text-xl font-medium leading-relaxed md:text-2xl">
                  Technische Universität Berlin
                  <br />
                  <span className="text-muted-foreground">
                    Institut für Werkzeugmaschinen und Fabrikbetrieb
                  </span>
                  <br />
                  <span className="text-muted-foreground">
                    Fachgebiet Industrielle Automatisierungstechnik
                  </span>
                </div>
              </div>
              <div className="grid gap-6 md:grid-cols-2">
                <div>
                  <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
                    Betreuung
                  </div>
                  <div className="mt-2 text-sm">
                    Prof. Dr.-Ing. Jörg Krüger
                  </div>
                </div>
                <div>
                  <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
                    Ansprechpartner
                  </div>
                  <div className="mt-2 text-sm">
                    Adam Michael Altenbuchner, M. Sc.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="border-t hairline py-12">
        <div className="mx-auto grid max-w-7xl gap-8 px-6 md:grid-cols-[1.4fr_1fr_1fr]">
          <div>
            <div className="mono flex items-center gap-2 text-sm font-semibold">
              <span className="inline-flex h-6 w-6 items-center justify-center rounded-md border border-warm/40 bg-warm/10">
                <span className="h-1.5 w-1.5 rounded-full bg-warm" />
              </span>
              LOFI.GEN
            </div>
            <p className="mt-4 max-w-sm text-sm text-muted-foreground">
              KI-gestützte Generierung und Bewertung von Lo-Fi-Musik ·
              Bachelorarbeit 2026 · Technische Universität Berlin.
            </p>
          </div>
          <div>
            <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
              Bearbeiter
            </div>
            <div className="mt-3 space-y-2 text-sm">
              <div>Adil Berke Savcili</div>
              <div>Inan Deniz Arduc</div>
            </div>
          </div>
          <div>
            <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
              Navigation
            </div>
            <div className="mt-3 grid grid-cols-2 gap-y-2 text-sm">
              {nav.map((n) => (
                <a
                  key={n.id}
                  href={`#${n.id}`}
                  className="text-muted-foreground transition hover:text-foreground"
                >
                  {n.label}
                </a>
              ))}
            </div>
          </div>
        </div>
        <div className="mono mx-auto mt-10 flex max-w-7xl flex-wrap items-center justify-between gap-3 px-6 text-[10px] uppercase tracking-widest text-muted-foreground">
          <span>© 2026 · lofi.gen</span>
          <span className="inline-flex items-center gap-2">
            <Terminal className="h-3 w-3" />
            build · v0.1 · demo-daten gekennzeichnet
          </span>
        </div>
      </footer>
    </div>
  );
}
