import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { api, profileLabel, SCORE_CATEGORIES } from "../lib/api";
import { InfoNote, PageHeader } from "../components/ui-bits";
import { RadarComparison } from "../components/RadarComparison";
import { Sparkles } from "lucide-react";
import { FeedbackPhase, blankFeedback, videoRequest, type VideoFeedback } from "./studio.video";

export const Route = createFileRoute("/studio/evaluation")({
  head: () => ({
    meta: [
      { title: "LOFI.GEN · Bewertung" },
      { name: "description", content: "Generierte Audios und Video-Szenarien bewerten." },
    ],
  }),
  component: EvaluationPage,
});

function averageScore(scores: Record<string, number>): number {
  const values = SCORE_CATEGORIES.map((k) => Number(scores?.[k] ?? 0));
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function EvaluationPage() {
  const [tab, setTab] = useState<"audio" | "video">("audio");

  return (
    <div>
      <PageHeader
        title="Bewertung"
        description="Generierte Audios und Video-Szenarien bewerten."
      />

      <div className="mb-6 flex gap-2">
        <button
          onClick={() => setTab("audio")}
          className={[
            "mono rounded-full border px-4 py-1.5 text-[11px] uppercase tracking-widest transition",
            tab === "audio"
              ? "border-warm/45 bg-warm/15 text-warm"
              : "border-hairline hairline text-muted-foreground hover:text-foreground",
          ].join(" ")}
        >
          Audio
        </button>
        <button
          onClick={() => setTab("video")}
          className={[
            "mono rounded-full border px-4 py-1.5 text-[11px] uppercase tracking-widest transition",
            tab === "video"
              ? "border-warm/45 bg-warm/15 text-warm"
              : "border-hairline hairline text-muted-foreground hover:text-foreground",
          ].join(" ")}
        >
          Video
        </button>
      </div>

      {tab === "audio" ? <AudioEvaluation /> : <VideoEvaluation />}
    </div>
  );
}

function AudioEvaluation() {
  const qc = useQueryClient();
  const audioQuery = useQuery({ queryKey: ["audio"], queryFn: api.audio });
  const list = audioQuery.data ?? [];
  const profilesQuery = useQuery({ queryKey: ["profiles"], queryFn: api.promptProfiles });

  const entries = useMemo(
    () => list.map((a) => ({ audio: a, rating: a.scoreRating ?? a.score_rating })),
    [list],
  );

  const genres = useMemo(
    () => (profilesQuery.data ?? []).map((p) => p.label),
    [profilesQuery.data],
  );

  const [genre, setGenre] = useState(genres[0] ?? "");
  useEffect(() => {
    if (!genre && genres.length) setGenre(genres[0]);
  }, [genre, genres]);

  const inGenre = useMemo(
    () => entries.filter((x) => profileLabel(x.audio.promptProfile) === genre),
    [entries, genre],
  );

  const bestId = useMemo(() => {
    const scored = inGenre.filter((x) => x.rating?.status === "finished");
    if (!scored.length) return inGenre[0]?.audio.id ?? "";
    return [...scored].sort(
      (a, b) =>
        averageScore(b.rating!.generierte_audio_scores) - averageScore(a.rating!.generierte_audio_scores),
    )[0].audio.id;
  }, [inGenre]);

  const [audioId, setAudioId] = useState(bestId);
  useEffect(() => {
    setAudioId(bestId);
  }, [bestId]);

  const current = inGenre.find((x) => x.audio.id === audioId) ?? inGenre[0];
  const rating = current?.rating?.status === "finished" ? current.rating : undefined;

  const [awaitingScore, setAwaitingScore] = useState(false);

  const bewertungMutation = useMutation({
    mutationFn: (id: string) => api.runProjectAction("audio_bewertung", { audioId: id }),
    onSuccess: () => {
      setAwaitingScore(true);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  useEffect(() => {
    if (!awaitingScore) return;
    const interval = setInterval(() => {
      qc.invalidateQueries({ queryKey: ["audio"] });
    }, 4000);
    return () => clearInterval(interval);
  }, [awaitingScore, qc]);

  useEffect(() => {
    if (awaitingScore && rating) {
      setAwaitingScore(false);
    }
  }, [rating, awaitingScore]);

  return (
    <div>
      {list.length === 0 ? (
        <div className="rounded-2xl border hairline bg-surface/40 p-10 text-center text-sm text-muted-foreground">
          Noch keine Audios vorhanden.
        </div>
      ) : (
        <section className="rounded-2xl border hairline bg-surface/40 p-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <h2 className="text-base font-semibold text-foreground">Qualitätsvergleich</h2>
              <p className="mt-1 text-xs text-muted-foreground">
                Generierte Audio vs. Genrestandard. Standard je Genre: der beste einzelne Referenzclip.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <select
                value={genre}
                onChange={(e) => setGenre(e.target.value)}
                className="h-9 rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
              >
                {genres.map((g) => (
                  <option key={g} value={g}>
                    {g}
                  </option>
                ))}
              </select>
              <select
                value={current?.audio.id ?? ""}
                onChange={(e) => setAudioId(e.target.value)}
                className="h-9 max-w-[280px] rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
              >
                {inGenre.map((x) => (
                  <option key={x.audio.id} value={x.audio.id}>
                    {x.audio.title}
                    {x.rating?.status === "finished"
                      ? ` (${averageScore(x.rating.generierte_audio_scores).toFixed(0)})${
                          x.audio.id === bestId ? " · beste" : ""
                        }`
                      : " · nicht bewertet"}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {current && rating ? (
            <>
              <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_260px]">
                <RadarComparison
                  genrestandardScores={rating.genrestandard_scores}
                  generierteScores={rating.generierte_audio_scores}
                />
                <div className="flex flex-col gap-1 rounded-xl border hairline bg-background/40 p-4">
                  <div className="mb-1 grid grid-cols-[1fr_auto_auto] gap-2 text-[11px] uppercase tracking-wide text-muted-foreground">
                    <span>Kategorie</span>
                    <span className="text-cool">Standard</span>
                    <span className="text-warm">Audio</span>
                  </div>
                  {SCORE_CATEGORIES.map((k) => {
                    const standardWert = rating.genrestandard_scores?.[k];
                    const audioWert = rating.generierte_audio_scores?.[k];
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
              {rating.auswertung && (
                <div className="mt-2 rounded-xl border hairline bg-background/60 p-4 text-sm text-muted-foreground">
                  {rating.auswertung}
                </div>
              )}
            </>
          ) : (
            <div className="mt-6 flex flex-col items-center gap-3 rounded-xl border border-dashed hairline bg-background/30 px-6 py-12 text-center">
              <p className="text-sm text-muted-foreground">
                Für "{current?.audio.title}" liegt noch keine Bewertung vor.
              </p>
              <button
                onClick={() => current && bewertungMutation.mutate(current.audio.id)}
                disabled={!current || bewertungMutation.isPending || awaitingScore}
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
        </section>
      )}

      <div className="mt-6">
        <InfoNote>
          Gemessen werden 5 echte Kategorien: Technische Audioqualität, Musikalische Kohärenz, Genre-Treue,
          Übergangsqualität, Referenzähnlichkeit — berechnet aus der Audio selbst, kein Demo-Wert.
        </InfoNote>
      </div>
    </div>
  );
}

interface VideoScenario {
  id: string;
  description: string;
  rounds: number;
  samples: number;
  hasCheckpoint: boolean;
}

function VideoEvaluation() {
  const scenariosQuery = useQuery({
    queryKey: ["video-scenarios"],
    queryFn: () => videoRequest<VideoScenario[]>("/api/video/scenarios"),
  });
  const scenarios = (scenariosQuery.data ?? []).filter((s) => s.rounds > 0);

  const galleryQuery = useQuery({ queryKey: ["video-gallery"], queryFn: api.videoGallery });

  const [scenarioId, setScenarioId] = useState("");
  useEffect(() => {
    if (!scenarioId && scenarios.length) setScenarioId(scenarios[0].id);
  }, [scenarioId, scenarios]);

  const allSamples = useMemo(
    () =>
      (galleryQuery.data ?? [])
        .filter((g) => g.scenario === scenarioId)
        .sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt)),
    [galleryQuery.data, scenarioId],
  );
  const gifCount = allSamples.filter((s) => s.type === "gif").length;
  const mp4Count = allSamples.filter((s) => s.type === "mp4").length;

  const [typeFilter, setTypeFilter] = useState<"alle" | "gif" | "mp4">("alle");
  useEffect(() => setTypeFilter("alle"), [scenarioId]);

  const samples = useMemo(
    () => (typeFilter === "alle" ? allSamples : allSamples.filter((s) => s.type === typeFilter)),
    [allSamples, typeFilter],
  );

  const [sampleId, setSampleId] = useState("");
  useEffect(() => {
    setSampleId(samples[0]?.id ?? "");
  }, [samples]);

  const sample = samples.find((s) => s.id === sampleId);

  const [feedback, setFeedback] = useState<VideoFeedback>(blankFeedback());

  if (scenariosQuery.isLoading) {
    return <div className="text-sm text-muted-foreground">Lädt ...</div>;
  }

  if (!scenarios.length) {
    return (
      <div className="rounded-2xl border hairline bg-surface/40 p-10 text-center text-sm text-muted-foreground">
        Noch kein Video-Szenario mit mindestens einer Trainingsrunde vorhanden.
      </div>
    );
  }

  return (
    <section className="rounded-2xl border hairline bg-surface/40 p-6">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-foreground">Szenario-Feedback</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Dimensionales Feedback zu einem Video-Szenario abgeben — steuert direkt die nächste
            Trainingsrunde.
          </p>
        </div>
        <select
          value={scenarioId}
          onChange={(e) => {
            setScenarioId(e.target.value);
            setFeedback(blankFeedback());
          }}
          className="h-9 max-w-[320px] rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
        >
          {scenarios.map((s) => (
            <option key={s.id} value={s.id}>
              {s.id} ({s.rounds} Runden{s.hasCheckpoint ? "" : ", kein Checkpoint"})
            </option>
          ))}
        </select>
      </div>

      <div className="mb-5">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <span className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
            Erzeugte Datei ({samples.length} von {allSamples.length})
          </span>
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex gap-1.5">
              {(["alle", "gif", "mp4"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTypeFilter(t)}
                  className={[
                    "mono rounded-full border px-2.5 py-1 text-[10px] uppercase tracking-widest transition",
                    typeFilter === t
                      ? "border-warm/45 bg-warm/15 text-warm"
                      : "border-hairline hairline text-muted-foreground hover:text-foreground",
                  ].join(" ")}
                >
                  {t === "alle" ? `Alle (${allSamples.length})` : t === "gif" ? `GIF (${gifCount})` : `MP4 (${mp4Count})`}
                </button>
              ))}
            </div>
            {samples.length > 0 && (
              <select
                value={sampleId}
                onChange={(e) => setSampleId(e.target.value)}
                className="h-8 max-w-[280px] rounded-lg border hairline bg-background/40 px-2 text-xs outline-none focus:border-warm/60"
              >
                {samples.map((s) => (
                  <option key={s.id} value={s.id}>
                    Runde {s.round ?? "?"} · Step {s.step ?? "?"} · {s.type.toUpperCase()} ·{" "}
                    {new Date(s.modifiedAt).toLocaleDateString("de-DE")}
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>
        {sample ? (
          <div className="mx-auto max-w-xs overflow-hidden rounded-xl border hairline bg-background/30">
            {sample.type === "gif" ? (
              <img src={api.absoluteUrl(sample.url)} className="w-full" alt={sample.filename} />
            ) : (
              <video
                key={sample.id}
                src={api.absoluteUrl(sample.url)}
                controls
                autoPlay
                loop
                muted
                playsInline
                className="w-full"
              />
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed hairline bg-background/30 py-14 text-center">
            <p className="text-xs text-muted-foreground">
              Noch keine generierte Datei für dieses Szenario vorhanden.
            </p>
          </div>
        )}
      </div>

      {scenarioId && (
        <FeedbackPhase
          feedback={feedback}
          setFeedback={setFeedback}
          scenarioId={scenarioId}
          onNewRound={() => setFeedback(blankFeedback())}
          onBack={() => setScenarioId("")}
        />
      )}
    </section>
  );
}
