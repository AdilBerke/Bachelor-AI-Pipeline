import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import {
  CheckCircle2,
  Download,
  DownloadCloud,
  FileAudio,
  Link as LinkIcon,
  ListChecks,
  ListMusic,
  Plus,
  RefreshCw,
  Scissors,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { api, type YoutubeMp3ImportRequest } from "../lib/api";
import { InfoNote, PageHeader, StatusBadge } from "../components/ui-bits";

export const Route = createFileRoute("/studio/import")({
  head: () => ({
    meta: [
      { title: "LOFI.GEN · Quellenimport" },
      { name: "description", content: "Audioquellen lokal als MP3 importieren und in Trainingsclips schneiden." },
    ],
  }),
  component: ImportPage,
});

function ImportPage() {
  const queryClient = useQueryClient();
  const [url, setUrl] = useState("");
  const [genre, setGenre] = useState("chillhop_lofi");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [pickedNote, setPickedNote] = useState<{ status: string; grund: string | null } | null>(null);
  const sourceSectionRef = useRef<HTMLDivElement>(null);

  const handlePickMissing = (missingUrl: string, genreKey: string, status: string, grund: string | null) => {
    setUrl(missingUrl);
    setGenre(genreKey);
    setPickedNote({ status, grund });
    sourceSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const job = useQuery({
    queryKey: ["youtube-mp3-job", activeJobId],
    queryFn: () => api.youtubeMp3Job(activeJobId as string),
    enabled: !!activeJobId,
    refetchInterval: (q) => {
      const status = q.state.data?.job.status;
      return status === "queued" || status === "running" ? 1500 : false;
    },
  });

  const mutation = useMutation({
    mutationFn: (body: YoutubeMp3ImportRequest) => api.importYoutubeMp3(body),
    onSuccess: (res) => setActiveJobId(res.job_id),
  });

  const startImport = () => {
    mutation.mutate({
      url,
      genre,
      rightsConfirmed: true,
    });
  };

  const currentJob = job.data?.job ?? mutation.data?.job;
  const canStart = url.trim().length > 0 && !mutation.isPending;

  useEffect(() => {
    if (currentJob?.status === "completed") {
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["top10-status"] });
    }
  }, [currentJob?.status, queryClient]);

  return (
    <div>
      <PageHeader
        title="Quellenimport"
        description="Link einfügen oder MP3 hochladen, um sie lokal als Trainingsquelle zu speichern."
      />

      <div ref={sourceSectionRef} className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <section className="rounded-2xl border hairline bg-surface/40 p-6">
          <div className="flex items-center gap-2">
            <LinkIcon className="h-4 w-4 text-warm" />
            <h2 className="text-base font-semibold text-foreground">Audioquelle</h2>
          </div>

          <div className="mt-5 space-y-4">
            <Field label="Link">
              <input
                value={url}
                onChange={(e) => {
                  setUrl(e.target.value);
                  setPickedNote(null);
                }}
                placeholder="https://www.youtube.com/watch?v=..."
                className="w-full rounded-lg border hairline bg-background/40 px-3 py-2 text-sm outline-none focus:border-warm/60"
              />
            </Field>

            {pickedNote?.status === "manuell_noetig" && (
              <div className="rounded-lg border border-amber-500/25 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
                Dieses Video ist beim automatischen Download bereits gescheitert
                {pickedNote.grund ? ` (${pickedNote.grund})` : ""}. "Quelle importieren" wird wahrscheinlich
                erneut fehlschlagen, da derselbe Download-Weg genutzt wird. Stattdessen die MP3 selbst
                herunterladen (z.B. im Browser) und unten per Drag & Drop hochladen.
              </div>
            )}

            <Field label="Genre">
              <GenreSelect value={genre} onChange={setGenre} />
            </Field>
          </div>

          <button
            onClick={startImport}
            disabled={!canStart}
            className="mt-6 inline-flex items-center gap-2 rounded-full bg-warm px-5 py-2.5 text-sm font-medium text-warm-foreground transition hover:brightness-110 disabled:opacity-50"
          >
            <Download className="h-4 w-4" />
            {mutation.isPending ? "Import startet ..." : "Quelle importieren"}
          </button>

          {mutation.error && (
            <div className="mt-4 rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive-foreground">
              {(mutation.error as Error).message}
            </div>
          )}

          <UploadDropzone genre={genre} />
        </section>

        <aside className="space-y-4">
          <div className="rounded-2xl border hairline bg-surface/40 p-5">
            <div className="flex items-center gap-2">
              <RefreshCw className="h-4 w-4 text-warm" />
              <h3 className="text-base font-semibold text-foreground">Fortschritt</h3>
            </div>

            {!currentJob ? (
              <p className="mt-3 text-xs text-muted-foreground">
                Noch kein Import gestartet.
              </p>
            ) : (
              <div className="mt-4 space-y-4">
                <div className="flex items-center justify-between gap-3">
                  <span className="mono text-xs">{currentJob.job_id}</span>
                  <StatusBadge status={currentJob.status} />
                </div>

                <div>
                  <div className="flex justify-between text-[11px] text-muted-foreground">
                    <span>{stageLabel(currentJob.stage)}</span>
                    <span>{currentJob.progress_percent} %</span>
                  </div>
                  <div className="mt-1 h-2 overflow-hidden rounded-full bg-hairline/60">
                    <div
                      className="h-full bg-warm transition-all"
                      style={{ width: `${currentJob.progress_percent}%` }}
                    />
                  </div>
                </div>

                {currentJob.stage === "clipping" && currentJob.clip_total > 0 && (
                  <div className="rounded-lg border hairline bg-background/30 px-3 py-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Clips geprüft</span>
                      <span className="mono">
                        {currentJob.clip_current}/{currentJob.clip_total}
                      </span>
                    </div>
                    <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                      <div className="rounded-md bg-emerald-500/10 px-2 py-1 text-emerald-200">
                        Akzeptiert: {currentJob.clip_accepted}
                      </div>
                      <div className="rounded-md bg-destructive/10 px-2 py-1 text-destructive-foreground">
                        Verworfen: {currentJob.clip_rejected}
                      </div>
                    </div>
                    {currentJob.clip_last_reason && (
                      <div className="mt-2 text-[11px] text-muted-foreground">
                        Letzter Grund: {currentJob.clip_last_reason}
                      </div>
                    )}
                  </div>
                )}

                {currentJob.audio_path && (
                  <PathLine icon={<FileAudio className="h-4 w-4" />} label="MP3" value={currentJob.audio_path} />
                )}

                {currentJob.dataset_path && (
                  <PathLine
                    icon={<Scissors className="h-4 w-4" />}
                    label="Clips"
                    value={currentJob.dataset_path}
                  />
                )}

                {currentJob.status === "completed" && (
                  <div className="rounded-lg border border-emerald-500/25 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-200">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4" />
                      {currentJob.audio_exists
                        ? "Datei liegt lokal vor — geprüft auf der Festplatte."
                        : "Import abgeschlossen, aber Datei nicht gefunden — bitte prüfen."}
                    </div>
                    {currentJob.audio_exists && (
                      <div className="mono mt-2 space-y-1 text-[10px] uppercase tracking-widest text-emerald-200/80">
                        {currentJob.audio_path && <div className="break-all normal-case">{currentJob.audio_path}</div>}
                        {!!currentJob.file_size_bytes && <div>{formatBytes(currentJob.file_size_bytes)}</div>}
                      </div>
                    )}
                  </div>
                )}

                {currentJob.error && (
                  <div className="rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive-foreground">
                    {currentJob.error}
                  </div>
                )}
              </div>
            )}
          </div>

          <InfoNote>MP3-Import und Clip-Erstellung laufen lokal auf diesem Rechner.</InfoNote>
        </aside>
      </div>

      <Top10ChecklistPanel onPickMissing={handlePickMissing} />
      <SourcesPanel />
    </div>
  );
}

function GenreSelect({ value, onChange }: { value: string; onChange: (id: string) => void }) {
  const queryClient = useQueryClient();
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.promptProfiles });
  const [creating, setCreating] = useState(false);
  const [label, setLabel] = useState("");

  const createMutation = useMutation({
    mutationFn: (label: string) => api.addProjectGenre({ label }),
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["profiles"] });
      setCreating(false);
      setLabel("");
      const newId = String((res.genre as { id?: string }).id ?? "");
      if (newId) onChange(newId);
    },
  });

  return (
    <div>
      <select
        value={creating ? "__new__" : value}
        onChange={(e) => {
          if (e.target.value === "__new__") {
            setCreating(true);
          } else {
            onChange(e.target.value);
          }
        }}
        className="w-full rounded-lg border hairline bg-background/40 px-3 py-2 text-sm outline-none focus:border-warm/60"
      >
        {(profiles.data ?? []).map((item) => (
          <option key={item.id} value={item.id}>
            {item.label}
          </option>
        ))}
        <option value="__new__">+ Neues Genre...</option>
      </select>

      {creating && (
        <div className="mt-2 flex items-center gap-2">
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="z.B. Piano Lofi"
            className="w-full rounded-lg border hairline bg-background/40 px-3 py-2 text-sm outline-none focus:border-warm/60"
          />
          <button
            onClick={() => label.trim() && createMutation.mutate(label.trim())}
            disabled={!label.trim() || createMutation.isPending}
            className="mono inline-flex shrink-0 items-center gap-1 rounded-md bg-warm px-2.5 py-2 text-[10px] uppercase tracking-widest text-warm-foreground transition hover:brightness-110 disabled:opacity-50"
          >
            <Plus className="h-3.5 w-3.5" />
            Anlegen
          </button>
          <button
            onClick={() => setCreating(false)}
            className="rounded-md border hairline px-2.5 py-2 text-xs text-muted-foreground transition hover:text-foreground"
          >
            Abbrechen
          </button>
        </div>
      )}
      {createMutation.error && (
        <div className="mt-2 rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive-foreground">
          {(createMutation.error as Error).message}
        </div>
      )}
    </div>
  );
}

function Top10ChecklistPanel({
  onPickMissing,
}: {
  onPickMissing: (url: string, genreKey: string, status: string, grund: string | null) => void;
}) {
  const queryClient = useQueryClient();
  const status = useQuery({ queryKey: ["top10-status"], queryFn: api.top10Status });
  const genres = status.data ?? [];
  const offenCount = genres.reduce(
    (sum, g) => sum + g.entries.filter((entry) => entry.status === "offen").length,
    0,
  );

  const importAllMutation = useMutation({
    mutationFn: () => api.runProjectAction("import_pending", {}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  return (
    <section className="mt-8 rounded-2xl border hairline bg-surface/40 p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <ListChecks className="h-4 w-4 text-warm" />
          <h2 className="text-base font-semibold text-foreground">Top5 je Genre</h2>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => importAllMutation.mutate()}
            disabled={importAllMutation.isPending || offenCount === 0}
            className="mono inline-flex items-center gap-1.5 rounded-md border border-warm/40 bg-warm/10 px-2.5 py-1.5 text-[10px] uppercase tracking-widest text-warm transition hover:bg-warm/20 disabled:opacity-50"
          >
            <DownloadCloud className="h-3.5 w-3.5" />
            {importAllMutation.isPending
              ? "Startet ..."
              : `Alle offenen Videos importieren (${offenCount})`}
          </button>
          <button
            onClick={() => status.refetch()}
            className="mono inline-flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-muted-foreground transition hover:text-foreground"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Aktualisieren
          </button>
        </div>
      </div>

      <p className="mt-2 text-xs text-muted-foreground">
        Nur diese 5 Videos je Genre gehören in den Datensatz. Grün = bereits geladen. Gelb = vom Crawler
        gefunden, aber automatisiert nicht ladbar (z.B. Lizenz/Zugriff) — bitte manuell importieren. Grau = noch
        offen.
      </p>

      {importAllMutation.isPending && (
        <p className="mt-2 text-xs text-muted-foreground">
          Lädt alle offenen Videos nacheinander im Hintergrund herunter — einzelne Fehler (z.B. Lizenz) werden
          übersprungen, der Rest läuft weiter. Fortschritt unter{" "}
          <Link to="/studio/jobs" className="text-warm hover:underline">
            Läufe
          </Link>{" "}
          sichtbar.
        </p>
      )}
      {importAllMutation.error && (
        <div className="mt-2 rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive-foreground">
          {(importAllMutation.error as Error).message}
        </div>
      )}

      {status.isLoading ? (
        <p className="mt-4 text-xs text-muted-foreground">Lade Top5-Status...</p>
      ) : genres.length === 0 ? (
        <p className="mt-4 text-xs text-muted-foreground">
          Noch keine Top5-Suche gelaufen. Ueber "Steuerung" eine Top5-Suche starten.
        </p>
      ) : (
        <div className="mt-5 space-y-6">
          {genres.map((g) => (
            <div key={g.genreKey}>
              <div className="mono mb-2 text-[10px] uppercase tracking-widest text-muted-foreground">
                {g.label}
              </div>
              {g.entries.length === 0 ? (
                <div className="rounded-lg border border-dashed hairline bg-background/20 px-3 py-3 text-xs text-muted-foreground">
                  Keine verlässliche Top5-Suche vorhanden (fehlende oder unvollständige Abrufzahlen-Daten).{" "}
                  <Link to="/studio/model" className="text-warm hover:underline">
                    Top5-Suche in der Steuerung starten
                  </Link>
                  .
                </div>
              ) : (
                <div className="space-y-1.5">
                  {g.entries.map((entry) => (
                    <div
                      key={entry.videoId}
                      className="flex flex-wrap items-center justify-between gap-2 rounded-lg border hairline bg-background/30 px-3 py-2"
                    >
                      <div className="flex min-w-0 items-center gap-2">
                        <span
                          className={`h-2 w-2 shrink-0 rounded-full ${
                            entry.status === "im_datensatz"
                              ? "bg-emerald-400"
                              : entry.status === "manuell_noetig"
                                ? "bg-amber-400"
                                : "bg-muted-foreground/40"
                          }`}
                        />
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="truncate text-xs text-foreground">
                              #{entry.rang} {entry.titel || entry.videoId}
                            </span>
                            {entry.viewCount != null && (
                              <span className="mono shrink-0 text-[10px] text-muted-foreground">
                                {formatViews(entry.viewCount)} Aufrufe
                              </span>
                            )}
                          </div>
                          <a
                            href={entry.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="mono mt-0.5 block truncate text-[10px] text-warm hover:underline"
                          >
                            {entry.url}
                          </a>
                          {entry.grund && (
                            <div className="mono mt-0.5 text-[10px] uppercase tracking-widest text-amber-300">
                              {entry.grund}
                            </div>
                          )}
                        </div>
                      </div>
                      {entry.status !== "im_datensatz" && (
                        <button
                          onClick={() => onPickMissing(entry.url, g.genreKey, entry.status, entry.grund)}
                          className="mono inline-flex shrink-0 items-center gap-1.5 rounded-md border border-warm/40 bg-warm/10 px-2.5 py-1.5 text-[10px] uppercase tracking-widest text-warm transition hover:bg-warm/20"
                        >
                          <LinkIcon className="h-3 w-3" />
                          Manuell importieren
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function UploadDropzone({ genre }: { genre: string }) {
  const queryClient = useQueryClient();
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const uploadMutation = useMutation({
    mutationFn: (file: File) => api.uploadSource(file, genre),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["sources"] }),
  });

  const handleFile = (file: File | undefined | null) => {
    if (!file) return;
    uploadMutation.mutate(file);
  };

  return (
    <div className="mt-7 border-t hairline pt-6">
      <div className="mono mb-2 text-[10px] uppercase tracking-widest text-muted-foreground">
        Manuell hochladen
      </div>
      <p className="mb-3 text-xs text-muted-foreground">
        Video lässt sich nicht automatisch laden? MP3 im Browser herunterladen und hier per Drag & Drop
        oder Klick als Quelle hinzufügen.
      </p>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFile(e.dataTransfer.files?.[0]);
        }}
        onClick={() => inputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-4 py-6 text-center transition ${
          dragOver ? "border-warm/60 bg-warm/5" : "hairline hover:bg-surface-2/40"
        }`}
      >
        <UploadCloud className="h-5 w-5 text-warm" />
        <span className="text-xs text-muted-foreground">
          MP3/WAV/M4A hier ablegen oder klicken
        </span>
        <input
          ref={inputRef}
          type="file"
          accept=".mp3,.wav,.m4a,audio/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      {uploadMutation.isPending && (
        <p className="mt-3 text-xs text-muted-foreground">Datei wird hochgeladen ...</p>
      )}
      {uploadMutation.isSuccess && (
        <div className="mt-3 rounded-lg border border-emerald-500/25 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-200">
          Hochgeladen: {uploadMutation.data.title}
        </div>
      )}
      {uploadMutation.error && (
        <div className="mt-3 rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive-foreground">
          {(uploadMutation.error as Error).message}
        </div>
      )}
    </div>
  );
}

function SourcesPanel() {
  const queryClient = useQueryClient();
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [pendingPrune, setPendingPrune] = useState(false);
  const sources = useQuery({ queryKey: ["sources"], queryFn: api.sources });

  const deleteMutation = useMutation({
    mutationFn: (videoId: string) => api.deleteSource(videoId),
    onSuccess: () => {
      setPendingDelete(null);
      queryClient.invalidateQueries({ queryKey: ["sources"] });
    },
  });

  const pruneMutation = useMutation({
    mutationFn: () => api.pruneNonTop10(),
    onSuccess: () => {
      setPendingPrune(false);
      queryClient.invalidateQueries({ queryKey: ["sources"] });
    },
  });

  const rows = sources.data ?? [];

  return (
    <section className="mt-8 rounded-2xl border hairline bg-surface/40 p-6">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <ListMusic className="h-4 w-4 text-warm" />
          <h2 className="text-base font-semibold text-foreground">Importierte Quellen</h2>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => sources.refetch()}
            className="mono inline-flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-muted-foreground transition hover:text-foreground"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Aktualisieren
          </button>
          {pendingPrune ? (
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Nicht-Top5 wirklich entfernen?</span>
              <button
                onClick={() => pruneMutation.mutate()}
                disabled={pruneMutation.isPending}
                className="rounded-md bg-destructive/15 px-2.5 py-1 text-xs font-medium text-destructive-foreground transition hover:bg-destructive/25 disabled:opacity-50"
              >
                Ja, bereinigen
              </button>
              <button
                onClick={() => setPendingPrune(false)}
                className="rounded-md border hairline px-2.5 py-1 text-xs text-muted-foreground transition hover:text-foreground"
              >
                Abbrechen
              </button>
            </div>
          ) : (
            <button
              onClick={() => setPendingPrune(true)}
              className="mono inline-flex items-center gap-1.5 rounded-md border border-destructive/30 bg-destructive/10 px-2.5 py-1.5 text-[10px] uppercase tracking-widest text-destructive-foreground transition hover:bg-destructive/20"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Nicht-Top5 bereinigen
            </button>
          )}
        </div>
      </div>

      <p className="mt-2 text-xs text-muted-foreground">
        Schlechte Quellen manuell entfernen. Löscht die Rohdatei, den Metadaten-Eintrag und bereits daraus
        gebaute Trainingsclips. Manuell importierte/hochgeladene Quellen sind von der Top5-Bereinigung
        ausgenommen.
      </p>

      {pruneMutation.isSuccess && !pendingPrune && pruneMutation.data.aborted && (
        <div className="mt-2 rounded-lg border border-amber-500/25 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          Abgebrochen, nichts gelöscht: {pruneMutation.data.grund}
        </div>
      )}
      {pruneMutation.isSuccess && !pendingPrune && !pruneMutation.data.aborted && (
        <div className="mt-2 rounded-lg border border-emerald-500/25 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-200">
          Bereinigt: {pruneMutation.data.removedCount} entfernt, {pruneMutation.data.keptTop10} Top5-Treffer,{" "}
          {pruneMutation.data.keptManual} manuelle Quellen behalten.
        </div>
      )}

      {sources.isLoading ? (
        <p className="mt-5 text-xs text-muted-foreground">Lade Quellen...</p>
      ) : rows.length === 0 ? (
        <p className="mt-5 text-xs text-muted-foreground">Noch keine Quellen importiert.</p>
      ) : (
        <div className="mt-5 space-y-2">
          {rows.map((source) => (
            <div
              key={source.videoId}
              className="flex flex-wrap items-center justify-between gap-3 rounded-lg border hairline bg-background/30 px-3 py-2.5"
            >
              <div className="min-w-0">
                <div className="truncate text-sm text-foreground">{source.title}</div>
                <div className="mono mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] uppercase tracking-widest text-muted-foreground">
                  <span>{source.videoId}</span>
                  <span>{formatBytes(source.fileSizeBytes)}</span>
                  <span>{source.clipDirs} Clip-Set(s)</span>
                  {source.manual && <span className="text-warm">Manuell</span>}
                  {!source.audioExists && <span className="text-destructive-foreground">Datei fehlt</span>}
                </div>
              </div>

              {pendingDelete === source.videoId ? (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">Wirklich löschen?</span>
                  <button
                    onClick={() => deleteMutation.mutate(source.videoId)}
                    disabled={deleteMutation.isPending}
                    className="rounded-md bg-destructive/15 px-2.5 py-1 text-xs font-medium text-destructive-foreground transition hover:bg-destructive/25 disabled:opacity-50"
                  >
                    Ja, löschen
                  </button>
                  <button
                    onClick={() => setPendingDelete(null)}
                    className="rounded-md border hairline px-2.5 py-1 text-xs text-muted-foreground transition hover:text-foreground"
                  >
                    Abbrechen
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setPendingDelete(source.videoId)}
                  className="inline-flex items-center gap-1.5 rounded-md border border-destructive/30 bg-destructive/10 px-2.5 py-1.5 text-xs text-destructive-foreground transition hover:bg-destructive/20"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  Löschen
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {deleteMutation.isSuccess && pendingDelete === null && (
        <div className="mt-4 rounded-lg border border-emerald-500/25 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-200">
          Quelle entfernt: {deleteMutation.data.removedAudioFiles} Datei(en),{" "}
          {deleteMutation.data.removedClipDirs} Clip-Ordner, {deleteMutation.data.removedManifestRows}{" "}
          Trainings-Einträge bereinigt.
        </div>
      )}
      {deleteMutation.error && (
        <div className="mt-4 rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive-foreground">
          {(deleteMutation.error as Error).message}
        </div>
      )}
    </section>
  );
}

function formatBytes(bytes: number): string {
  if (!bytes) return "0 MB";
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(1)} MB`;
}

function formatViews(views: number): string {
  if (views >= 1_000_000) return `${(views / 1_000_000).toFixed(1)}M`;
  if (views >= 1_000) return `${(views / 1_000).toFixed(1)}K`;
  return `${views}`;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mono text-[10px] uppercase tracking-widest text-muted-foreground">{label}</span>
      <div className="mt-2">{children}</div>
    </label>
  );
}

function PathLine({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border hairline bg-background/30 px-3 py-2">
      <div className="mb-1 flex items-center gap-2 text-xs text-muted-foreground">
        {icon}
        {label}
      </div>
      <div className="mono break-all text-xs text-foreground">{value}</div>
    </div>
  );
}

function stageLabel(stage: string) {
  const labels: Record<string, string> = {
    queued: "Wartet",
    downloading: "MP3 wird geladen",
    downloaded: "MP3 geladen",
    preparing_clips: "Clip-Erstellung wird vorbereitet",
    clipping: "30s-Clips werden erstellt",
    done: "Fertig",
    failed: "Fehlgeschlagen",
  };
  return labels[stage] ?? stage;
}
