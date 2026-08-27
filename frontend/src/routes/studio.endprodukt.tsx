import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Combine } from "lucide-react";
import { api } from "../lib/api";
import { InfoNote, PageHeader, StatusBadge } from "../components/ui-bits";

export const Route = createFileRoute("/studio/endprodukt")({
  head: () => ({
    meta: [
      { title: "LOFI.GEN · Endprodukt" },
      {
        name: "description",
        content: "Ein fertiges Audio mit einem fertigen Video-Loop zu einem Endprodukt kombinieren.",
      },
    ],
  }),
  component: EndproduktPage,
});

function formatBytes(n: number): string {
  if (n >= 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${Math.round(n / 1024)} KB`;
}

function formatMinutes(sec: number): string {
  return `${Math.round(sec / 60)} min`;
}

function EndproduktPage() {
  const qc = useQueryClient();
  const audioQuery = useQuery({ queryKey: ["audio"], queryFn: api.audio });
  const videoQuery = useQuery({ queryKey: ["video-gallery"], queryFn: api.videoGallery });
  const productsQuery = useQuery({
    queryKey: ["final-products"],
    queryFn: api.finalProducts,
    refetchInterval: 5000,
  });

  const audios = audioQuery.data ?? [];
  const videos = (videoQuery.data ?? []).filter((v) => v.type === "mp4");

  const [audioId, setAudioId] = useState("");
  const [videoId, setVideoId] = useState("");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const effectiveAudioId = audioId || audios[0]?.id || "";
  const effectiveVideoId = videoId || videos[0]?.id || "";
  const selectedAudio = audios.find((a) => a.id === effectiveAudioId);
  const selectedVideo = videos.find((v) => v.id === effectiveVideoId);

  const jobQuery = useQuery({
    queryKey: ["job", activeJobId],
    queryFn: () => api.job(activeJobId as string),
    enabled: !!activeJobId,
    refetchInterval: 2000,
  });

  useEffect(() => {
    if (jobQuery.data?.status === "completed" || jobQuery.data?.status === "failed") {
      qc.invalidateQueries({ queryKey: ["final-products"] });
    }
  }, [jobQuery.data?.status, qc]);

  const createMutation = useMutation({
    mutationFn: () =>
      api.runProjectAction("endprodukt_erstellen", {
        audioId: effectiveAudioId,
        videoId: effectiveVideoId,
      }),
    onSuccess: (res) => setActiveJobId(res.jobId),
  });

  const busy =
    createMutation.isPending ||
    jobQuery.data?.status === "queued" ||
    jobQuery.data?.status === "running";

  return (
    <div className="space-y-7">
      <PageHeader
        title="Endprodukt"
        description="Ein fertig generiertes Audio mit einem fertigen Video-Loop kombinieren — das Video wird auf die Audiolänge geloopt, das Ergebnis ist ein herunterladbares MP4."
      />

      <InfoNote>
        Nur bereits vorhandene, fertige Audios (Bibliothek) und fertige MP4-Videos (kein GIF, da ohne
        Tonspur-Format) können kombiniert werden. Eine echte Videogenerierung auf Knopfdruck ist noch
        nicht angebunden.
      </InfoNote>

      <div className="grid gap-5 rounded-2xl border hairline bg-surface/50 p-5 lg:grid-cols-2">
        <div className="space-y-3">
          <label className="mono block text-[10px] uppercase tracking-widest text-muted-foreground">
            Audio
          </label>
          <select
            value={effectiveAudioId}
            onChange={(e) => setAudioId(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground"
          >
            {audios.map((a) => (
              <option key={a.id} value={a.id}>
                {a.title} · {formatMinutes(a.durationSec)}
              </option>
            ))}
          </select>
          {selectedAudio && (
            <audio
              key={selectedAudio.id}
              src={api.audioDownloadUrl(selectedAudio.id, "mp3")}
              controls
              className="w-full"
            />
          )}
        </div>

        <div className="space-y-3">
          <label className="mono block text-[10px] uppercase tracking-widest text-muted-foreground">
            Video-Loop
          </label>
          <select
            value={effectiveVideoId}
            onChange={(e) => setVideoId(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground"
          >
            {videos.map((v) => (
              <option key={v.id} value={v.id}>
                {v.scenario} · R{v.round ?? "–"} S{v.step ?? "–"}
              </option>
            ))}
          </select>
          {selectedVideo && (
            <video
              key={selectedVideo.id}
              src={api.absoluteUrl(selectedVideo.url)}
              autoPlay
              loop
              muted
              playsInline
              className="aspect-video w-full rounded-lg bg-black object-cover"
            />
          )}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          disabled={!effectiveAudioId || !effectiveVideoId || busy}
          onClick={() => createMutation.mutate()}
          className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Combine className="h-4 w-4" />
          {busy ? "Wird erstellt …" : "Endprodukt erstellen"}
        </button>
        {jobQuery.data && <StatusBadge status={jobQuery.data.status} />}
        {jobQuery.data?.error && (
          <span className="text-xs text-destructive">{jobQuery.data.error}</span>
        )}
      </div>

      <div>
        <h2 className="mb-3 text-lg font-semibold text-foreground">Fertige Endprodukte</h2>
        {(productsQuery.data?.length ?? 0) === 0 ? (
          <p className="text-sm text-muted-foreground">Noch keine Endprodukte erstellt.</p>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {productsQuery.data!.map((p) => (
              <div key={p.id} className="flex flex-col gap-2 rounded-xl border hairline bg-surface/50 p-3">
                <video
                  src={api.absoluteUrl(p.url)}
                  controls
                  className="aspect-video w-full rounded-lg bg-black object-cover"
                />
                <span className="text-sm font-medium text-foreground">{p.audioTitel}</span>
                <span className="text-xs text-muted-foreground">{p.videoLabel}</span>
                <div className="mono flex items-center justify-between text-[10px] text-muted-foreground">
                  <span>{formatBytes(p.sizeBytes)}</span>
                  <a href={api.absoluteUrl(p.url)} download className="text-warm hover:underline">
                    Download
                  </a>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
