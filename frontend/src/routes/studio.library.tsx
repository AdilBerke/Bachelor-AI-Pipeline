import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { api, profileLabel } from "../lib/api";
import { PageHeader } from "../components/ui-bits";
import { AudioPlayer } from "../components/AudioPlayer";
import { ScoreComparison } from "../components/ScoreComparison";
import { FileText } from "lucide-react";

export const Route = createFileRoute("/studio/library")({
  head: () => ({
    meta: [
      { title: "LOFI.GEN · Ausgaben" },
      { name: "description", content: "Alle generierten Lofi-Audios anhören und herunterladen." },
    ],
  }),
  component: LibraryPage,
});

function LibraryPage() {
  const audio = useQuery({ queryKey: ["audio"], queryFn: api.audio });

  function fmtDur(sec: number) {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return h > 0 ? `${h}h ${m}m` : `${m}m`;
  }

  return (
    <div>
      <PageHeader
        title="Ausgaben"
        description="Generierte Longform-Audios mit Player, Download und technischem Report."
      />

      {(audio.data ?? []).length === 0 ? (
        <div className="rounded-2xl border hairline bg-surface/40 p-10 text-center text-sm text-muted-foreground">
          Noch keine Audios vorhanden.
        </div>
      ) : (
        <div className="grid gap-5">
          {audio.data!.map((a) => {
            const wav = a.formats === "wav" || a.formats === "wav_mp3";
            const mp3 = a.formats === "mp3" || a.formats === "wav_mp3";
            const scoreRating = a.scoreRating ?? a.score_rating;
            return (
              <div key={a.id} className="rounded-2xl border hairline bg-surface/40 p-5">
                <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="text-lg font-semibold text-foreground">{a.title}</div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {new Date(a.createdAt).toLocaleString("de-DE")} · {fmtDur(a.durationSec)} ·
                      Genre <b className="text-foreground">{profileLabel(a.promptProfile)}</b> · Checkpoint{" "}
                      <span className="mono">{a.checkpoint}</span>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {wav && (
                      <a
                        href={api.audioDownloadUrl(a.id, "wav")}
                        className="rounded-full border hairline px-3 py-1.5 text-xs hover:bg-surface-2"
                      >
                        Download WAV
                      </a>
                    )}
                    {mp3 && (
                      <a
                        href={api.audioDownloadUrl(a.id, "mp3")}
                        className="rounded-full border hairline px-3 py-1.5 text-xs hover:bg-surface-2"
                      >
                        Download MP3
                      </a>
                    )}
                    {a.reportUrl && (
                      <a
                        href={api.absoluteUrl(a.reportUrl)}
                        className="inline-flex items-center gap-1 rounded-full border hairline px-3 py-1.5 text-xs hover:bg-surface-2"
                      >
                        <FileText className="h-3 w-3" /> Report
                      </a>
                    )}
                  </div>
                </div>

                <AudioPlayer
                  src={wav ? api.audioDownloadUrl(a.id, "wav") : api.audioDownloadUrl(a.id, "mp3")}
                  title={a.prompt}
                  subtitle={`LoRA · ${a.checkpoint}`}
                  durationFallbackSec={a.durationSec}
                  downloadUrl={
                    wav ? api.audioDownloadUrl(a.id, "wav") : api.audioDownloadUrl(a.id, "mp3")
                  }
                  metadata={a.metadata}
                />

                <ScoreComparison rating={scoreRating} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
