import { Link, createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Fragment, useState } from "react";
import { api, profileLabel } from "../lib/api";
import { PageHeader, StatusBadge } from "../components/ui-bits";
import { Download, FileText, Film, Play, RefreshCw, X } from "lucide-react";

export const Route = createFileRoute("/studio/jobs")({
  head: () => ({
    meta: [
      { title: "LOFI.GEN · Vorgänge" },
      { name: "description", content: "Status und Fortschritt aller lokalen Vorgänge." },
    ],
  }),
  component: JobsPage,
});

type PublicJob = {
  jobId: string;
  isVideoJob?: boolean;
  status: string;
  kind?: string;
  scenarioId?: string;
  phase?: string;
  trainStep?: number;
  trainTotal?: number;
  videoUrl?: string;
  output?: string;
  promptProfile?: string;
  targetDurationMin?: number;
  outputFormat?: string;
  startedAt?: string;
  finishedAt?: string;
  progress: number;
  audioId?: string;
  error?: string;
  logPath?: string;
  lastLines?: string[];
};

const VIDEO_API = "http://localhost:8000";

function jobTypeLabel(kind?: string): string {
  if (!kind) return "Video";
  if (kind.startsWith("video_train_")) return "LoRA-Training";
  if (kind.startsWith("video_gen_")) return "Generierung";
  return "Video";
}

function scenarioLabel(kind?: string, scenarioId?: string): string {
  const id = scenarioId || kind?.replace(/^video_(train|gen)_/, "").replace(/_\d{10}$/, "") || "?";
  return id;
}

function VideoJobRow({ j, onToggle, open }: { j: PublicJob; onToggle: () => void; open: boolean }) {
  const isTraining = j.kind?.startsWith("video_train_");
  const isGenerate = j.kind?.startsWith("video_gen_");
  const isDone = j.status === "completed";
  const isFailed = j.status === "failed";
  const isRunning = j.status === "running";

  const gifMutation = useMutation({
    mutationFn: async () => {
      const filename = j.videoUrl?.split("/").pop() ?? "";
      const folder = j.videoUrl?.split("/").slice(-2, -1)[0] ?? "web_outputs";
      const res = await fetch(`${VIDEO_API}/api/video/make-gif`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario_id: j.scenarioId, round: folder, filename }),
      });
      return res.json();
    },
  });

  return (
    <Fragment>
      <tr className="border-b hairline hover:bg-surface-2/30">
        {/* Typ */}
        <td className="px-4 py-3">
          <div className="flex items-center gap-2">
            <Film className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
            <div>
              <div className="text-sm font-medium text-foreground">{jobTypeLabel(j.kind)}</div>
              <div className="mono text-[10px] text-muted-foreground">{scenarioLabel(j.kind, j.scenarioId)}</div>
            </div>
          </div>
        </td>
        {/* Status */}
        <td className="px-4 py-3">
          <StatusBadge status={j.status} />
        </td>
        {/* Phase */}
        <td className="px-4 py-3 text-xs text-muted-foreground max-w-[180px] truncate">
          {j.phase || "—"}
        </td>
        {/* Start */}
        <td className="px-4 py-3 text-xs text-muted-foreground">
          {j.startedAt ? new Date(j.startedAt).toLocaleString("de-DE") : "—"}
        </td>
        {/* Fortschritt */}
        <td className="px-4 py-3">
          <div className="flex w-32 items-center gap-2">
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-hairline/60">
              <div
                className={["h-full rounded-full transition-all", isFailed ? "bg-destructive/70" : isDone ? "bg-emerald-500" : "bg-warm"].join(" ")}
                style={{ width: `${Math.round(j.progress * 100)}%` }}
              />
            </div>
            <span className="mono text-[11px] text-muted-foreground tabular-nums">
              {Math.round(j.progress * 100)}%
            </span>
          </div>
          {isRunning && j.trainStep && j.trainTotal && (
            <div className="mono mt-0.5 text-[10px] text-muted-foreground">
              Step {j.trainStep}/{j.trainTotal}
            </div>
          )}
        </td>
        {/* Aktionen */}
        <td className="px-4 py-3 text-right">
          <div className="inline-flex flex-wrap justify-end gap-2">
            {/* Log-Toggle */}
            <button
              onClick={onToggle}
              className="inline-flex items-center gap-1 rounded-full border hairline px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
            >
              <FileText className="h-3 w-3" /> Log
            </button>
            {/* Nach Training: weitertrainieren oder Video generieren → zur Video-Seite */}
            {isDone && isTraining && (
              <Link
                to="/studio/video"
                className="inline-flex items-center gap-1 rounded-full border border-warm/40 bg-warm/10 px-2 py-1 text-xs text-warm hover:bg-warm/20"
              >
                <Play className="h-3 w-3" /> Zur Video-Seite
              </Link>
            )}
            {/* Nach Generierung: GIF exportieren */}
            {isDone && isGenerate && j.videoUrl && (
              <>
                <a
                  href={`${VIDEO_API}${j.videoUrl}`}
                  download
                  className="inline-flex items-center gap-1 rounded-full border hairline px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
                >
                  <Download className="h-3 w-3" /> MP4
                </a>
                <button
                  onClick={() => gifMutation.mutate()}
                  disabled={gifMutation.isPending || gifMutation.isSuccess}
                  className="inline-flex items-center gap-1 rounded-full border hairline px-2 py-1 text-xs text-muted-foreground hover:text-foreground disabled:opacity-50"
                >
                  <Film className="h-3 w-3" />
                  {gifMutation.isPending ? "GIF…" : gifMutation.isSuccess ? "GIF ✓" : "Als GIF"}
                </button>
                <button
                  onClick={onToggle}
                  className="inline-flex items-center gap-1 rounded-full border hairline px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
                >
                  <Play className="h-3 w-3" /> Video
                </button>
              </>
            )}
          </div>
        </td>
      </tr>
      {/* Ausgeklappter Bereich: Video-Player oder Log */}
      {open && (
        <tr className="border-b hairline bg-surface-2/20">
          <td colSpan={6} className="px-4 py-4 space-y-3">
            {isDone && isGenerate && j.videoUrl && (
              <video
                src={`${VIDEO_API}${j.videoUrl}`}
                autoPlay loop muted playsInline
                className="max-h-72 rounded-xl border hairline object-contain"
              />
            )}
            {j.output && (
              <pre className="max-h-40 overflow-auto rounded-lg border hairline bg-background/50 p-3 text-xs text-muted-foreground whitespace-pre-wrap">
                {j.output}
              </pre>
            )}
          </td>
        </tr>
      )}
    </Fragment>
  );
}

function JobsPage() {
  const qc = useQueryClient();
  const [openJob, setOpenJob] = useState<string | null>(null);
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: api.jobs,
    refetchInterval: 3000,
  });

  const cancel = useMutation({
    mutationFn: (id: string) => api.cancelJob(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });

  const allJobs = (jobs.data ?? []) as PublicJob[];
  const videoJobs = allJobs.filter((j) => j.isVideoJob);
  const audioJobs = allJobs.filter((j) => !j.isVideoJob);

  const toggle = (id: string) => setOpenJob((prev) => (prev === id ? null : id));

  return (
    <div className="space-y-7">
      <PageHeader
        title="Vorgänge"
        description="Laufende und abgeschlossene Vorgänge — Audio-Generierungen, Video-Trainings und GIF-Exporte."
      />

      {/* ── Video-Jobs ─────────────────────────────────────────────── */}
      {videoJobs.length > 0 && (
        <div className="space-y-2">
          <div className="mono flex items-center gap-2 text-[10px] uppercase tracking-widest text-muted-foreground">
            <Film className="h-3.5 w-3.5" /> Video-Läufe
          </div>
          <div className="overflow-hidden rounded-2xl border hairline bg-surface/40">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="mono border-b hairline bg-surface-2/40 text-left text-[10px] uppercase tracking-widest text-muted-foreground">
                  <tr>
                    <th className="px-4 py-3">Vorgang</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3">Phase</th>
                    <th className="px-4 py-3">Start</th>
                    <th className="px-4 py-3">Fortschritt</th>
                    <th className="px-4 py-3 text-right">Aktionen</th>
                  </tr>
                </thead>
                <tbody>
                  {videoJobs.map((j) => (
                    <VideoJobRow
                      key={j.jobId}
                      j={j}
                      open={openJob === j.jobId}
                      onToggle={() => toggle(j.jobId)}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Audio-Jobs ─────────────────────────────────────────────── */}
      <div className="space-y-2">
        {videoJobs.length > 0 && (
          <div className="mono flex items-center gap-2 text-[10px] uppercase tracking-widest text-muted-foreground">
            Audio-Läufe
          </div>
        )}
        <div className="overflow-hidden rounded-2xl border hairline bg-surface/40">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="mono border-b hairline bg-surface-2/40 text-left text-[10px] uppercase tracking-widest text-muted-foreground">
                <tr>
                  <th className="px-4 py-3">Vorgang</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Genre</th>
                  <th className="px-4 py-3">Länge</th>
                  <th className="px-4 py-3">Format</th>
                  <th className="px-4 py-3">Start</th>
                  <th className="px-4 py-3">Fortschritt</th>
                  <th className="px-4 py-3 text-right">Aktionen</th>
                </tr>
              </thead>
              <tbody>
                {audioJobs.map((j) => (
                  <Fragment key={j.jobId}>
                    <tr className="border-b hairline hover:bg-surface-2/30">
                      <td className="mono px-4 py-3 text-xs">{j.jobId}</td>
                      <td className="px-4 py-3">
                        <StatusBadge status={j.status} />
                      </td>
                      <td className="px-4 py-3">{profileLabel(j.promptProfile ?? "")}</td>
                      <td className="px-4 py-3">{j.targetDurationMin} min</td>
                      <td className="px-4 py-3 uppercase">{j.outputFormat}</td>
                      <td className="px-4 py-3 text-xs text-muted-foreground">
                        {j.startedAt ? new Date(j.startedAt).toLocaleString("de-DE") : "—"}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex w-32 items-center gap-2">
                          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-hairline/60">
                            <div
                              className="h-full bg-warm"
                              style={{ width: `${j.progress * 100}%` }}
                            />
                          </div>
                          <span className="mono text-[11px] text-muted-foreground">
                            {Math.round(j.progress * 100)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <div className="inline-flex gap-2">
                          <button
                            onClick={() => toggle(j.jobId)}
                            className="inline-flex items-center gap-1 rounded-full border hairline px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
                          >
                            <FileText className="h-3 w-3" /> Log
                          </button>
                          {(j.status === "running" || j.status === "queued") && (
                            <button
                              onClick={() => cancel.mutate(j.jobId)}
                              className="inline-flex items-center gap-1 rounded-full border hairline px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
                            >
                              <X className="h-3 w-3" /> Abbrechen
                            </button>
                          )}
                          {j.status === "completed" && j.audioId && (
                            <a
                              href={api.audioDownloadUrl(j.audioId, "wav")}
                              className="inline-flex items-center gap-1 rounded-full bg-warm/15 px-2 py-1 text-xs text-warm hover:bg-warm/20"
                            >
                              <Download className="h-3 w-3" /> Download
                            </a>
                          )}
                        </div>
                      </td>
                    </tr>
                    {openJob === j.jobId && (
                      <tr className="border-b hairline bg-surface-2/20">
                        <td colSpan={8} className="px-4 py-3">
                          <div className="mono mb-2 text-[11px] text-muted-foreground">
                            {j.logPath || "kein Logpfad"}
                          </div>
                          <pre className="max-h-64 overflow-auto rounded-lg border hairline bg-background/50 p-3 text-xs text-muted-foreground">
                            {(j.lastLines ?? []).length
                              ? (j.lastLines ?? []).join("\n")
                              : "Noch keine Logzeilen vorhanden."}
                          </pre>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
                {audioJobs.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-4 py-10 text-center text-sm text-muted-foreground">
                      Keine Audio-Vorgänge vorhanden.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {allJobs.length === 0 && (
        <div className="flex flex-col items-center gap-3 py-16 text-muted-foreground">
          <RefreshCw className="h-6 w-6 opacity-40" />
          <p className="text-sm">Noch keine Vorgänge gestartet.</p>
        </div>
      )}
    </div>
  );
}
