import { Link, createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  BarChart3,
  Cpu,
  Database,
  FileAudio,
  GitBranch,
  ListChecks,
  Plus,
  Search,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import { api, profileLabel, type ProjectDatasetGenre, type Top10Candidate } from "../lib/api";
import { InfoNote, PageHeader, StatCard, StatusBadge } from "../components/ui-bits";

export const Route = createFileRoute("/studio/model")({
  head: () => ({
    meta: [
      { title: "LOFI.GEN · Projektsteuerung" },
      {
        name: "description",
        content: "Lokale Steuerung für Suche, Dataset, LoRA, Merkmale und Testaudios.",
      },
    ],
  }),
  component: ModelPage,
});

type ActionPayload = {
  action: string;
  body?: Record<string, unknown>;
};

function ModelPage() {
  const qc = useQueryClient();
  const overview = useQuery({
    queryKey: ["project-overview"],
    queryFn: api.projectOverview,
    refetchInterval: 5000,
  });
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 3000 });
  const top10 = useQuery({
    queryKey: ["top10-results"],
    queryFn: api.top10Results,
    refetchInterval: 5000,
  });

  const genres = overview.data?.genres ?? [];
  const firstGenre = genres[0]?.id ?? "chillhop_lofi";
  const [selectedGenre, setSelectedGenre] = useState(firstGenre);
  const [newGenre, setNewGenre] = useState("");
  const [newCaption, setNewCaption] = useState("");
  const [mood, setMood] = useState("calm relaxed lofi mood");
  const [bekannteErlauben, setBekannteErlauben] = useState(false);
  const [zielClips, setZielClips] = useState(5000);
  const [loopDauer, setLoopDauer] = useState("5m");
  const [loopCrossfade, setLoopCrossfade] = useState(5);
  const [loopFadeIn, setLoopFadeIn] = useState(5);
  const [loopFadeOut, setLoopFadeOut] = useState(5);
  const [loopGithubPush, setLoopGithubPush] = useState(true);
  const [checkpoint, setCheckpoint] = useState("training/musicgen/lora_training/adapter.pt");

  const genreKey = selectedGenre || firstGenre;
  const genreName = profileLabel(genreKey);
  const dataset = overview.data?.dataset;
  const lora = overview.data?.lora;
  const latestJob = jobs.data?.[0];

  const runAllTop10Mutation = useMutation({
    mutationFn: async () => {
      await Promise.all(
        genres.map((g) => api.runProjectAction("top10", { genre: g.id, stimmung: mood, bekannteErlauben })),
      );
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const runMutation = useMutation({
    mutationFn: ({ action, body = {} }: ActionPayload) => api.runProjectAction(action, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["project-overview"] });
      qc.invalidateQueries({ queryKey: ["top10-results"] });
    },
  });

  const genreMutation = useMutation({
    mutationFn: () =>
      api.addProjectGenre({
        label: newGenre,
        caption: newCaption || undefined,
      }),
    onSuccess: () => {
      setNewGenre("");
      setNewCaption("");
      qc.invalidateQueries({ queryKey: ["project-overview"] });
      qc.invalidateQueries({ queryKey: ["profiles"] });
    },
  });

  const genreRows = useMemo(() => {
    const entries = Object.entries(dataset?.genres ?? {});
    return entries.map(([key, value]) => ({
      key,
      label: profileLabel(key),
      value: value as ProjectDatasetGenre,
    }));
  }, [dataset?.genres]);

  function run(action: string, body: Record<string, unknown> = {}) {
    runMutation.mutate({ action, body });
  }

  return (
    <div className="space-y-7">
      <PageHeader
        title="Projektsteuerung"
        description="Alle lokalen Audio-Funktionen an einem Ort: Quellen suchen, Clips vorbereiten, Merkmale berechnen, LoRA trainieren und Testaudios erzeugen."
        actions={
          <>
            <div className="mono flex items-center gap-3 px-1 text-[11px] text-muted-foreground">
              <span>{overview.data?.status.device ?? "lokal"}</span>
              {latestJob && (
                <span className="flex items-center gap-1.5">
                  <span
                    className={[
                      "h-1.5 w-1.5 rounded-full",
                      latestJob.status === "running" || latestJob.status === "queued"
                        ? "bg-warm animate-pulse-soft"
                        : latestJob.status === "failed"
                          ? "bg-destructive"
                          : "bg-emerald-400",
                    ].join(" ")}
                  />
                  {latestJob.status}
                </span>
              )}
            </div>
            <Link
              to="/studio/jobs"
              className="inline-flex h-10 items-center gap-2 rounded-full border hairline px-3 text-sm hover:border-warm/50"
            >
              <ListChecks className="h-4 w-4" />
              Läufe
            </Link>
          </>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2">
        <StatCard
          label="Dataset"
          value={`${dataset?.selectedTotal ?? 0}/${dataset?.targetTotal ?? 5000}`}
          hint={dataset?.ready ? "bereit für LoRA" : `fehlen ${dataset?.missingTotal ?? 0}`}
          tone={dataset?.ready ? "primary" : "muted"}
        />
        <StatCard
          label="LoRA"
          value={<StatusBadge status={lora?.adapterExists ? "online" : "offline"} />}
          hint={lora?.adapter ?? "training/musicgen/lora_training/adapter.pt"}
          tone={lora?.adapterExists ? "accent" : "muted"}
        />
      </div>

      <section className="rounded-2xl border hairline bg-surface/40 p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-foreground">Genre & Quellen</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              Top-5-Suche und neue Genres nutzen die zentrale Projektkonfiguration.
            </p>
          </div>
          <StatusBadge status={overview.data?.status.online ? "online" : "offline"} />
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Field label="Aktives Genre">
            <select
              value={genreKey}
              onChange={(event) => setSelectedGenre(event.target.value)}
              className="h-10 w-full rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
            >
              {genres.map((genre) => (
                <option key={genre.id} value={genre.id}>
                  {genre.label}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Suchstimmung">
            <input
              value={mood}
              onChange={(event) => setMood(event.target.value)}
              className="h-10 w-full rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
            />
          </Field>
          <Field label="Optionen">
            <label className="flex h-10 items-center justify-between rounded-lg border hairline bg-background/40 px-3 text-sm">
              Bekannte Videos erlauben
              <input
                type="checkbox"
                checked={bekannteErlauben}
                onChange={(event) => setBekannteErlauben(event.target.checked)}
                className="h-4 w-4 accent-warm"
              />
            </label>
          </Field>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <ActionButton
            icon={<Search className="h-4 w-4" />}
            label={`Top 5 suchen: ${genreName}`}
            onClick={() =>
              run("top10", { genre: genreKey, stimmung: mood, bekannteErlauben })
            }
            busy={runMutation.isPending}
            variant="primary"
          />
          <ActionButton
            icon={<Search className="h-4 w-4" />}
            label="Alle Top5-Suchen aktualisieren (5 Genres)"
            onClick={() => runAllTop10Mutation.mutate()}
            busy={runAllTop10Mutation.isPending}
          />
          <ActionButton
            icon={<Database className="h-4 w-4" />}
            label="Fehlende Quellen laden"
            onClick={() => run("clips", { genres: genreKey, zielClips, mitDownloads: true })}
            busy={runMutation.isPending}
          />
        </div>
        {runAllTop10Mutation.isPending && (
          <p className="mt-2 text-xs text-muted-foreground">
            Läuft für alle 5 Genres parallel im Hintergrund - dauert je nach Suchpool
            ca. 15-20 Min. Fortschritt unter "Läufe" sichtbar.
          </p>
        )}

        <div className="mt-6 border-t hairline pt-4">
          <h3 className="mono mb-3 text-[10px] uppercase tracking-widest text-muted-foreground">
            Neues Genre anlegen
          </h3>
          <div className="grid gap-3 sm:grid-cols-[1fr_1fr_auto]">
            <input
              value={newGenre}
              onChange={(event) => setNewGenre(event.target.value)}
              placeholder="Neues Genre, z.B. Rainy Chill Lofi"
              className="h-10 rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
            />
            <input
              value={newCaption}
              onChange={(event) => setNewCaption(event.target.value)}
              placeholder="Optionaler Prompt / Caption"
              className="h-10 rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
            />
            <button
              onClick={() => genreMutation.mutate()}
              disabled={!newGenre.trim() || genreMutation.isPending}
              className="inline-flex h-10 items-center justify-center gap-2 rounded-full border hairline px-4 text-sm hover:border-warm/50 disabled:opacity-50"
            >
              <Plus className="h-4 w-4" />
              Genre speichern
            </button>
          </div>
        </div>

        <Top10Table rows={top10.data ?? []} />
      </section>

      <section className="rounded-2xl border hairline bg-surface/40 p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-foreground">Dataset & Analyse</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              Vorbereitung und Messung passieren lokal. LoRA-Training wird hier nicht automatisch gestartet.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            Zielclips
            <input
              type="number"
              min={100}
              step={100}
              value={zielClips}
              onChange={(event) => setZielClips(Number(event.target.value))}
              className="mono h-9 w-24 rounded-lg border hairline bg-background/40 px-2 outline-none focus:border-warm/60"
            />
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <ActionButton
            icon={<Database className="h-4 w-4" />}
            label="Clips aus lokalen MP3s"
            onClick={() => run("clips", { zielClips })}
            busy={runMutation.isPending}
          />
          <ActionButton
            icon={<SlidersHorizontal className="h-4 w-4" />}
            label="Trainingsdaten prüfen"
            onClick={() => run("trainingsdaten")}
            busy={runMutation.isPending}
          />
          <ActionButton
            icon={<BarChart3 className="h-4 w-4" />}
            label="Merkmale extrahieren"
            onClick={() => run("merkmale")}
            busy={runMutation.isPending}
          />
          <ActionButton
            icon={<BarChart3 className="h-4 w-4" />}
            label="Referenzvergleich"
            onClick={() => run("referenzvergleich")}
            busy={runMutation.isPending}
          />
          <div className="rounded-lg border hairline bg-background/40 p-2">
            <input
              value={loopDauer}
              onChange={(event) => setLoopDauer(event.target.value)}
              className="mb-2 h-8 w-full rounded-md border hairline bg-background/40 px-2 text-xs outline-none focus:border-warm/60"
              placeholder="5m"
            />
            <div className="mb-2 grid grid-cols-3 gap-2">
              <SmallNumber label="Fade" value={loopCrossfade} onChange={setLoopCrossfade} />
              <SmallNumber label="Start" value={loopFadeIn} onChange={setLoopFadeIn} />
              <SmallNumber label="Ende" value={loopFadeOut} onChange={setLoopFadeOut} />
            </div>
            <label className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
              GitHub
              <input
                type="checkbox"
                checked={loopGithubPush}
                onChange={(event) => setLoopGithubPush(event.target.checked)}
                className="h-3.5 w-3.5 accent-warm"
              />
            </label>
            <ActionButton
              icon={<FileAudio className="h-4 w-4" />}
              label="Clip-Pool loopen"
              onClick={() =>
                run("audio_loopen", {
                  genre: genreKey,
                  dauer: loopDauer,
                  crossfade: loopCrossfade,
                  fadeIn: loopFadeIn,
                  fadeOut: loopFadeOut,
                  githubPush: loopGithubPush,
                })
              }
              busy={runMutation.isPending}
            />
          </div>
        </div>

        {genreRows.length > 0 && (
          <div className="mt-5 overflow-hidden rounded-lg border hairline">
            <table className="w-full text-sm">
              <thead className="mono bg-surface-2/40 text-left text-[10px] uppercase tracking-widest text-muted-foreground">
                <tr>
                  <th className="px-3 py-2">Genre</th>
                  <th className="px-3 py-2">Clips</th>
                  <th className="px-3 py-2">Quellen</th>
                  <th className="px-3 py-2">Fehlen</th>
                </tr>
              </thead>
              <tbody>
                {genreRows.map((row) => (
                  <tr key={row.key} className="border-t hairline">
                    <td className="px-3 py-2">{row.label}</td>
                    <td className="mono px-3 py-2">
                      {Number(row.value.selected ?? 0)}/{Number(row.value.target ?? 0)}
                    </td>
                    <td className="mono px-3 py-2">{Number(row.value.sources ?? 0)}</td>
                    <td className="mono px-3 py-2">{Number(row.value.missing ?? 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="rounded-2xl border hairline bg-surface/40 p-5">
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-foreground">LoRA & Bewertung</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Diese Aktionen starten echte Prozesse. Testaudios und Trainingsclips werden wie im Terminal nach GitHub vorbereitet.
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            Jedes Training startet komplett neu (kein Fortsetzen/Erweitern mehr). Ein vorhandener Adapter wird davor
            automatisch in einen Zeitstempel-Ordner archiviert statt überschrieben.
          </p>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <ActionButton
            icon={<Cpu className="h-4 w-4" />}
            label="Neues Training starten"
            onClick={() => run("lora_training")}
            busy={runMutation.isPending}
            variant="primary"
          />
          <div className="rounded-lg border hairline bg-background/40 p-2">
            <input
              value={checkpoint}
              onChange={(event) => setCheckpoint(event.target.value)}
              className="mono mb-2 h-8 w-full rounded-md border hairline bg-background/40 px-2 text-xs outline-none focus:border-warm/60"
              placeholder="Checkpoint-Pfad"
            />
            <ActionButton
              icon={<GitBranch className="h-4 w-4" />}
              label="Checkpoint freigeben"
              onClick={() => run("lora_freigeben", { checkpoint })}
              busy={runMutation.isPending}
            />
          </div>
          <ActionButton
            icon={<Sparkles className="h-4 w-4" />}
            label="Testaudios"
            onClick={() => run("testaudios")}
            busy={runMutation.isPending}
          />
          <ActionButton
            icon={<FileAudio className="h-4 w-4" />}
            label="Trainingsclips"
            onClick={() => run("trainingsclip_test")}
            busy={runMutation.isPending}
          />
        </div>
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <InfoNote>
          Die Website startet dieselben lokalen Befehle wie das Terminal. Laufende Prozesse findest du unter{" "}
          <Link to="/studio/jobs" className="text-warm underline-offset-4 hover:underline">
            Läufe
          </Link>
          .
        </InfoNote>
        <InfoNote>
          Longform-Audios werden weiterhin auf der Seite{" "}
          <Link to="/studio/generate" className="text-warm underline-offset-4 hover:underline">
            Audio
          </Link>{" "}
          erzeugt. Dort wird MusicGen mit dem aktuellen LoRA-Adapter verwendet.
        </InfoNote>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mono text-[10px] uppercase tracking-widest text-muted-foreground">{label}</span>
      <div className="mt-2">{children}</div>
    </label>
  );
}

function ActionButton({
  icon,
  label,
  onClick,
  busy,
  variant = "secondary",
}: {
  icon: ReactNode;
  label: string;
  onClick: () => void;
  busy?: boolean;
  variant?: "primary" | "secondary";
}) {
  return (
    <button
      onClick={onClick}
      disabled={busy}
      className={[
        "inline-flex min-h-10 items-center justify-center gap-2 rounded-lg border px-3 py-2 text-sm transition disabled:opacity-50",
        variant === "primary"
          ? "border-warm/50 bg-warm/15 text-warm hover:bg-warm/25"
          : "hairline bg-background/40 hover:border-warm/50 hover:bg-warm/5",
      ].join(" ")}
    >
      {icon}
      <span>{busy ? "Startet ..." : label}</span>
    </button>
  );
}

function Top10Table({ rows }: { rows: Top10Candidate[] }) {
  if (!rows.length) return null;
  return (
    <div className="mt-5 overflow-hidden rounded-lg border hairline">
      <div className="mono border-b hairline bg-surface-2/30 px-3 py-2 text-xs text-muted-foreground">
        Neueste Top-5-Ergebnisse
      </div>
      <div className="max-h-80 overflow-auto">
        <table className="w-full text-sm">
          <thead className="mono sticky top-0 bg-surface text-left text-[10px] uppercase tracking-widest text-muted-foreground">
            <tr>
              <th className="px-3 py-2">Score</th>
              <th className="px-3 py-2">Aufrufe</th>
              <th className="px-3 py-2">Likes</th>
              <th className="px-3 py-2">Genre</th>
              <th className="px-3 py-2">Titel</th>
              <th className="px-3 py-2">Kanal</th>
              <th className="px-3 py-2">Link</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={`${row.video_id || row.url || index}`} className="border-t hairline">
                <td className="mono px-3 py-2">{formatNumber(row.gesamt_score)}</td>
                <td className="mono px-3 py-2">{formatNumber(row.view_count)}</td>
                <td className="mono px-3 py-2">{formatNumber(row.like_count)}</td>
                <td className="px-3 py-2">{row.genre || "-"}</td>
                <td className="max-w-[320px] truncate px-3 py-2">{row.titel || "-"}</td>
                <td className="px-3 py-2">{row.kanal || "-"}</td>
                <td className="px-3 py-2">
                  {row.url ? (
                    <a
                      href={row.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-warm underline-offset-4 hover:underline"
                    >
                      öffnen
                    </a>
                  ) : (
                    "-"
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SmallNumber({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-[10px] text-muted-foreground">{label}</span>
      <input
        type="number"
        min={0}
        max={20}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-8 w-full rounded-md border hairline bg-background/40 px-2 text-xs outline-none focus:border-warm/60"
      />
    </label>
  );
}

function formatNumber(value: unknown): string {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return new Intl.NumberFormat("de-DE", { maximumFractionDigits: 1 }).format(number);
}
