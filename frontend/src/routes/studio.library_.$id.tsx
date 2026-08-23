import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api, profileLabel, SCORE_CATEGORIES } from "../lib/api";
import { AudioPlayer } from "../components/AudioPlayer";
import { RadarComparison } from "../components/RadarComparison";
import { ArrowLeft, Check, FileText, Pencil, Sparkles, X } from "lucide-react";

export const Route = createFileRoute("/studio/library_/$id")({
  head: () => ({
    meta: [{ title: "LOFI.GEN · Bibliothek" }],
  }),
  component: LibraryDetailPage,
});

function fmtDur(sec: number) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

function LibraryDetailPage() {
  const { id } = Route.useParams();
  const qc = useQueryClient();
  const audioQuery = useQuery({ queryKey: ["audio"], queryFn: api.audio });
  const audio = (audioQuery.data ?? []).find((a) => a.id === id);

  const [editing, setEditing] = useState(false);
  const [draftName, setDraftName] = useState("");

  const renameMutation = useMutation({
    mutationFn: (name: string) => api.renameAudio(id, name),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["audio"] });
      setEditing(false);
    },
  });

  const [awaitingScore, setAwaitingScore] = useState(false);
  const bewertungMutation = useMutation({
    mutationFn: () => api.runProjectAction("audio_bewertung", { audioId: id }),
    onSuccess: () => {
      setAwaitingScore(true);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  useEffect(() => {
    if (!awaitingScore) return;
    const interval = setInterval(() => qc.invalidateQueries({ queryKey: ["audio"] }), 4000);
    return () => clearInterval(interval);
  }, [awaitingScore, qc]);

  const rating = audio?.scoreRating ?? audio?.score_rating;
  const rated = rating?.status === "finished" ? rating : undefined;

  useEffect(() => {
    if (awaitingScore && rated) setAwaitingScore(false);
  }, [awaitingScore, rated]);

  if (audioQuery.isLoading) {
    return <div className="text-sm text-muted-foreground">Lädt ...</div>;
  }

  if (!audio) {
    return (
      <div className="rounded-2xl border hairline bg-surface/40 p-10 text-center text-sm text-muted-foreground">
        Audio nicht gefunden.{" "}
        <Link to="/studio/library" className="text-warm hover:underline">
          Zurück zur Bibliothek
        </Link>
      </div>
    );
  }

  const wav = audio.formats === "wav" || audio.formats === "wav_mp3";
  const mp3 = audio.formats === "mp3" || audio.formats === "wav_mp3";

  return (
    <div>
      <Link
        to="/studio/library"
        className="mb-5 inline-flex items-center gap-1.5 text-xs text-muted-foreground transition hover:text-foreground"
      >
        <ArrowLeft className="h-3.5 w-3.5" /> Zurück zur Bibliothek
      </Link>

      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          {editing ? (
            <div className="flex items-center gap-1.5">
              <input
                autoFocus
                value={draftName}
                onChange={(e) => setDraftName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && draftName.trim()) renameMutation.mutate(draftName.trim());
                  if (e.key === "Escape") setEditing(false);
                }}
                className="h-10 rounded-md border hairline bg-background/60 px-2 text-2xl font-semibold text-foreground outline-none focus:border-warm/60"
              />
              <button
                onClick={() => draftName.trim() && renameMutation.mutate(draftName.trim())}
                disabled={renameMutation.isPending}
                className="rounded-md p-1.5 text-emerald-400 hover:bg-emerald-500/10 disabled:opacity-50"
              >
                <Check className="h-4 w-4" />
              </button>
              <button
                onClick={() => setEditing(false)}
                className="rounded-md p-1.5 text-muted-foreground hover:bg-surface-2"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          ) : (
            <div className="group flex items-center gap-2">
              <h1 className="text-2xl font-semibold tracking-tight text-foreground">{audio.title}</h1>
              <button
                onClick={() => {
                  setEditing(true);
                  setDraftName(audio.title);
                }}
                className="rounded-md p-1 text-muted-foreground opacity-0 transition hover:bg-surface-2 hover:text-foreground group-hover:opacity-100"
                title="Namen bearbeiten"
              >
                <Pencil className="h-3.5 w-3.5" />
              </button>
            </div>
          )}
          <div className="mono mt-1.5 text-xs text-muted-foreground">
            {new Date(audio.createdAt).toLocaleString("de-DE")} · {fmtDur(audio.durationSec)} · Genre{" "}
            <b className="text-foreground">{profileLabel(audio.promptProfile)}</b> · Checkpoint{" "}
            <span className="mono">{audio.checkpoint}</span>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {wav && (
            <a
              href={api.audioDownloadUrl(audio.id, "wav")}
              className="rounded-full border hairline px-3 py-1.5 text-xs hover:bg-surface-2"
            >
              Download WAV
            </a>
          )}
          {mp3 && (
            <a
              href={api.audioDownloadUrl(audio.id, "mp3")}
              className="rounded-full border hairline px-3 py-1.5 text-xs hover:bg-surface-2"
            >
              Download MP3
            </a>
          )}
          {audio.reportUrl && (
            <a
              href={api.absoluteUrl(audio.reportUrl)}
              className="inline-flex items-center gap-1 rounded-full border hairline px-3 py-1.5 text-xs hover:bg-surface-2"
            >
              <FileText className="h-3 w-3" /> Report
            </a>
          )}
        </div>
      </div>

      <AudioPlayer
        src={wav ? api.audioDownloadUrl(audio.id, "wav") : api.audioDownloadUrl(audio.id, "mp3")}
        title={audio.prompt}
        subtitle={`LoRA · ${audio.checkpoint}`}
        durationFallbackSec={audio.durationSec}
        downloadUrl={wav ? api.audioDownloadUrl(audio.id, "wav") : api.audioDownloadUrl(audio.id, "mp3")}
        metadata={audio.metadata}
      />

      <div className="mt-8">
        <div className="mono mb-4 text-[11px] uppercase tracking-widest text-muted-foreground">
          Qualitätsvergleich zum Genrestandard
        </div>

        {rated ? (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_260px]">
            <RadarComparison
              genrestandardScores={rated.genrestandard_scores}
              generierteScores={rated.generierte_audio_scores}
            />
            <div className="flex flex-col gap-1 rounded-xl border hairline bg-background/40 p-4">
              <div className="mb-1 grid grid-cols-[1fr_auto_auto] gap-2 text-[11px] uppercase tracking-wide text-muted-foreground">
                <span>Kategorie</span>
                <span className="text-cool">Standard</span>
                <span className="text-warm">Audio</span>
              </div>
              {SCORE_CATEGORIES.map((k) => {
                const standardWert = rated.genrestandard_scores?.[k];
                const audioWert = rated.generierte_audio_scores?.[k];
                return (
                  <div
                    key={k}
                    className="grid grid-cols-[1fr_auto_auto] items-baseline gap-2 border-t hairline py-1.5 text-sm first:border-t-0"
                  >
                    <span className="text-muted-foreground">{k}</span>
                    <span className="text-cool tabular-nums">
                      {standardWert === undefined ? "–" : standardWert.toFixed(1)}
                    </span>
                    <span className="font-medium text-warm tabular-nums">
                      {audioWert === undefined ? "–" : audioWert.toFixed(1)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed hairline bg-background/30 px-6 py-12 text-center">
            <p className="text-sm text-muted-foreground">Für diese Audio liegt noch keine Bewertung vor.</p>
            <button
              onClick={() => bewertungMutation.mutate()}
              disabled={bewertungMutation.isPending || awaitingScore}
              className="inline-flex items-center gap-2 rounded-full bg-warm px-5 py-2.5 text-sm font-medium text-warm-foreground transition hover:brightness-110 disabled:opacity-50"
            >
              <Sparkles className="h-4 w-4" />
              {bewertungMutation.isPending || awaitingScore ? "Bewertung läuft ..." : "Jetzt bewerten"}
            </button>
            {(bewertungMutation.isPending || awaitingScore) && (
              <p className="text-xs text-muted-foreground">
                Läuft im Hintergrund, dauert je nach Audiolänge ca. 1-2 Minuten. Fortschritt unter "Läufe"
                sichtbar, diese Seite aktualisiert sich automatisch.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
