import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Fragment, useState } from "react";
import { api, profileLabel } from "../lib/api";
import { PageHeader, StatusBadge } from "../components/ui-bits";
import { Download, FileText, X } from "lucide-react";

export const Route = createFileRoute("/studio/jobs")({
  head: () => ({
    meta: [
      { title: "LOFI.GEN · Vorgänge" },
      { name: "description", content: "Status und Fortschritt aller lokalen Audio-Vorgänge." },
    ],
  }),
  component: JobsPage,
});

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

  return (
    <div>
      <PageHeader
        title="Vorgänge"
        description="Laufende und abgeschlossene Audio-Generierungen. Die Tabelle aktualisiert sich automatisch."
      />

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
              {(jobs.data ?? []).map((j) => (
                <Fragment key={j.jobId}>
                  <tr className="border-b hairline hover:bg-surface-2/30">
                    <td className="mono px-4 py-3 text-xs">{j.jobId}</td>
                    <td className="px-4 py-3">
                      <StatusBadge status={j.status} />
                    </td>
                    <td className="px-4 py-3">{profileLabel(j.promptProfile)}</td>
                    <td className="px-4 py-3">{j.targetDurationMin} min</td>
                    <td className="px-4 py-3 uppercase">{j.outputFormat}</td>
                    <td className="px-4 py-3 text-xs text-muted-foreground">
                      {new Date(j.startedAt).toLocaleString("de-DE")}
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
                          onClick={() => setOpenJob(openJob === j.jobId ? null : j.jobId)}
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
              {(jobs.data ?? []).length === 0 && (
                <tr>
                  <td colSpan={8} className="px-4 py-10 text-center text-sm text-muted-foreground">
                    Keine Vorgänge vorhanden.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
