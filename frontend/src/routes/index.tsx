import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  Cpu,
  Database,
  FileAudio,
  Film,
  GitBranch,
  Image as ImageIcon,
  Music2,
  Radio,
  Server,
  Sparkles,
  Terminal,
  Users,
  Waves,
} from "lucide-react";

export const Route = createFileRoute("/")({
  component: Index,
});

/* ---------- shared bits ---------- */

const nav = [
  { id: "projekt", label: "Projekt" },
  { id: "pipeline", label: "Pipeline" },
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
        <div className="relative mx-auto max-w-4xl px-6 pb-24">
          <div className="flex flex-col items-center text-center">
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
              <p className="mx-auto mt-6 max-w-xl text-base leading-relaxed text-muted-foreground md:text-lg">
                Entwicklung einer reproduzierbaren Deep-Learning-Pipeline zur
                Erfassung, Verarbeitung, Generierung und Bewertung von
                Lo-Fi-Musik. Das System verbindet Datenaufbereitung,
                Modelltraining, Audioerzeugung, GIF-Produktion und eine
                webbasierte Benutzeroberfläche.
              </p>
            </Reveal>
            <Reveal delay={240}>
              <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
                <a
                  href="#pipeline"
                  className="inline-flex items-center gap-2 rounded-full bg-warm px-5 py-3 text-sm font-medium text-primary-foreground transition hover:brightness-110"
                >
                  Pipeline entdecken
                  <ArrowRight className="h-4 w-4" />
                </a>
              </div>
            </Reveal>
            <Reveal delay={320}>
              <div className="mono mx-auto mt-10 grid max-w-lg grid-cols-3 gap-6 border-t hairline pt-6 text-[11px] uppercase tracking-widest text-muted-foreground">
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
                    5
                  </div>
                  <div className="mt-1">Bewertungs&shy;kategorien</div>
                </div>
              </div>
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
                verarbeitet und für das Training generativer Audio- und
                Videomodelle nutzt. Die generierten Ergebnisse werden
                anschließend analysiert, bewertet und mit einer Baseline
                verglichen.
              </p>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              {[
                {
                  t: "Datenerfassung",
                  d: "Automatisiertes Sammeln von Lo-Fi-Audio, Referenzbildern und strukturierten Metadaten.",
                  icon: Database,
                },
                {
                  t: "Audiomodell-Training",
                  d: "Fine-Tuning eines Audiomodells auf dem kuratierten Lo-Fi-Datensatz.",
                  icon: Cpu,
                },
                {
                  t: "Musikgenerierung",
                  d: "Erzeugung stimmungskonsistenter Audioclips mit Loop-Option.",
                  icon: Music2,
                },
                {
                  t: "Videomodell-Training",
                  d: "Fine-Tuning eines Videomodells (LTX) auf ausgewählten visuellen Szenarien.",
                  icon: Film,
                },
                {
                  t: "Videogenerierung",
                  d: "Erzeugung kurzer visueller Loops passend zur Lo-Fi-Stimmung.",
                  icon: ImageIcon,
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
                deg: "M. Sc. Maschinenbau",
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
