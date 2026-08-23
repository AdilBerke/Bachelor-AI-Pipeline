import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { Play, X } from "lucide-react";
import {
  api,
  profileLabel,
  type GenerateRequest,
  type OutputFormat,
  type PromptProfile,
} from "../lib/api";
import { getSettings } from "../lib/settings";
import { StatusBadge } from "../components/ui-bits";

export const Route = createFileRoute("/studio/generate")({
  head: () => ({
    meta: [
      { title: "Lofi Studio · Generieren" },
      { name: "description", content: "Longform-Audio mit MusicGen + LoRA generieren." },
    ],
  }),
  component: GeneratePage,
});

const DURATIONS = [
  { label: "5 min", value: "5m" },
  { label: "15 min", value: "15m" },
  { label: "20 min", value: "20m" },
  { label: "30 min", value: "30m" },
  { label: "1 h", value: "1h" },
  { label: "3 h", value: "3h" },
];

const SEGMENT_PRESETS = [30, 60, 90, 120];

const stepLabels: Record<string, string> = {
  model_loading: "Modell laden",
  segment_generation: "Clips generieren",
  segment_check: "Qualität prüfen",
  crossfade: "Übergänge bauen",
  export: "Exportieren",
  done: "Fertig",
};

function parseDurationInput(value: string, fallback = 20): number {
  const text = value.trim().toLowerCase().replace(",", ".").replace(/\s+/g, "");
  if (!text) return fallback;
  const unit = ["s", "m", "h"].includes(text.at(-1) ?? "") ? text.at(-1) : "m";
  const numberText = unit === "s" || unit === "m" || unit === "h" ? text.slice(0, -1) : text;
  const number = Number(numberText);
  if (!Number.isFinite(number) || number <= 0) return fallback;
  if (unit === "h") return Math.max(1, Math.round(number * 60));
  if (unit === "s") return Math.max(1, Math.round(number / 60));
  return Math.max(1, Math.round(number));
}

function GeneratePage() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const settings = getSettings();
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.promptProfiles });

  const [durationInput, setDurationInput] = useState("");
  const [profile, setProfile] = useState<PromptProfile | "">("");
  const [customPrompt, setCustomPrompt] = useState("");
  const [targetBpm, setTargetBpm] = useState<number | "">("");
  const [instruments, setInstruments] = useState("");
  const [format, setFormat] = useState<OutputFormat>(settings.defaultFormat);
  const [segment, setSegment] = useState<number | "">("");
  const [crossfade, setCrossfade] = useState<number>(settings.defaultCrossfadeSec);
  const [githubPush, setGithubPush] = useState(true);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const job = useQuery({
    queryKey: ["job", activeJobId],
    queryFn: () => api.job(activeJobId as string),
    enabled: !!activeJobId,
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      return status === "queued" || status === "running" ? 1500 : false;
    },
  });

  const startMutation = useMutation({
    mutationFn: (body: GenerateRequest) => api.generate(body),
    onSuccess: (res) => {
      setActiveJobId(res.jobId);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const cancelMutation = useMutation({
    mutationFn: (id: string) => api.cancelJob(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["job", activeJobId] }),
  });

  useEffect(() => {
    if (job.data?.status === "completed") {
      qc.invalidateQueries({ queryKey: ["jobs"] });
    }
  }, [job.data?.status, qc]);

  const finalDuration = parseDurationInput(durationInput, settings.defaultDurationMin);
  const progress = Math.round((job.data?.progress ?? 0) * 100);
  const isRunning =
    startMutation.isPending ||
    (!!activeJobId && (job.data?.status === "running" || job.data?.status === "queued"));

  const submit = () => {
    if (!profile || !durationInput) return;
    const body: GenerateRequest = {
      targetDurationMin: finalDuration,
      durationInput,
      promptProfile: profile,
      customPrompt: customPrompt || undefined,
      targetBpm: targetBpm === "" ? undefined : targetBpm,
      instruments: instruments || undefined,
      segmentDurationSec: segment === "" ? undefined : segment,
      crossfadeSec: crossfade,
      seed: 0,
      outputFormat: format,
      adapterPath: "training/musicgen/lora_training/adapter.pt",
      githubPush,
    };
    startMutation.mutate(body);
  };

  return (
    <div className="space-y-5">
      <header className="border-b hairline pb-5">
        <div className="mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
          MusicGen + LoRA
        </div>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-foreground">
          Longform generieren
        </h1>
      </header>

      <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
        <section className="rounded-2xl border hairline bg-surface/40 p-5">
          <div className="grid gap-5">
            <Field label="Genre">
              <div className="grid gap-2 sm:grid-cols-2">
                {(profiles.data ?? []).map((item) => (
                  <button
                    key={item.id}
                    onClick={() => setProfile(item.id)}
                    className={[
                      "rounded-lg border px-3 py-3 text-left transition",
                      profile === item.id
                        ? "border-warm/50 bg-warm/10"
                        : "hairline bg-background/40 hover:border-warm/40",
                    ].join(" ")}
                  >
                    <div className="text-sm font-medium text-foreground">{item.label}</div>
                    <div className="mt-1 text-xs text-muted-foreground">{item.description}</div>
                  </button>
                ))}
              </div>
            </Field>

            <Field label="Dauer">
              <div className="grid gap-2 sm:grid-cols-6">
                {DURATIONS.map((item) => (
                  <button
                    key={item.value}
                    onClick={() => {
                      setDurationInput(item.value);
                    }}
                    className={[
                      "h-10 rounded-lg border text-sm transition",
                      durationInput === item.value
                        ? "border-warm/50 bg-warm text-warm-foreground"
                        : "hairline bg-background/40 text-muted-foreground hover:text-foreground",
                    ].join(" ")}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
              <input
                value={durationInput}
                onChange={(event) => setDurationInput(event.target.value)}
                placeholder="Eigene Dauer, z.B. 20m, 45m, 1h"
                className="mt-2 h-10 w-full rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
              />
            </Field>

            <Field label="Prompt">
              <textarea
                value={customPrompt}
                onChange={(event) => setCustomPrompt(event.target.value)}
                placeholder="optional, z.B. ruhiger Chill Lofi mit mellow piano, warmem Bass und soft drums"
                rows={3}
                className="w-full rounded-lg border hairline bg-background/40 px-3 py-2 text-sm outline-none focus:border-warm/60"
              />
            </Field>

            <div className="grid gap-4 sm:grid-cols-[180px_1fr]">
              <Field label="Ziel-BPM">
                <input
                  type="number"
                  min={50}
                  max={120}
                  value={targetBpm}
                  onChange={(event) =>
                    setTargetBpm(event.target.value === "" ? "" : Number(event.target.value))
                  }
                  placeholder="z.B. 78"
                  className="h-10 w-full rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
                />
              </Field>
              <Field label="Instrumente">
                <input
                  value={instruments}
                  onChange={(event) => setInstruments(event.target.value)}
                  placeholder="z.B. mellow piano, warm bass, soft drums"
                  className="h-10 w-full rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
                />
              </Field>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Format">
                <Select value={format} onChange={(value) => setFormat(value as OutputFormat)}>
                  <option value="wav">WAV</option>
                  <option value="mp3">MP3</option>
                  <option value="wav_mp3">WAV + MP3</option>
                </Select>
              </Field>
              <Field label={`Crossfade ${crossfade}s`}>
                <input
                  type="range"
                  min={0}
                  max={10}
                  value={crossfade}
                  onChange={(event) => setCrossfade(Number(event.target.value))}
                  className="h-10 w-full accent-warm"
                />
              </Field>
            </div>

            <Field label="Segmentlänge">
              <div className="grid grid-cols-4 gap-2">
                {SEGMENT_PRESETS.map((value) => (
                  <button
                    key={value}
                    onClick={() => setSegment(value)}
                    className={[
                      "h-10 rounded-lg border text-sm transition",
                      segment === value
                        ? "border-warm/50 bg-warm text-warm-foreground"
                        : "hairline bg-background/40 text-muted-foreground hover:text-foreground",
                    ].join(" ")}
                  >
                    {value}s
                  </button>
                ))}
              </div>
              <input
                type="number"
                min={5}
                step={5}
                value={segment}
                onChange={(event) =>
                  setSegment(event.target.value === "" ? "" : Math.max(5, Number(event.target.value)))
                }
                placeholder="Eigene Sekunden, z.B. 45"
                className="mt-2 h-10 w-full rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
              />
            </Field>

            <label className="flex items-center justify-between rounded-lg border hairline bg-background/40 px-3 py-3 text-sm">
              Audio nach GitHub vorbereiten
              <input
                type="checkbox"
                checked={githubPush}
                onChange={(event) => setGithubPush(event.target.checked)}
                className="h-4 w-4 accent-warm"
              />
            </label>

            <button
              onClick={submit}
              disabled={isRunning || !profile || !durationInput}
              className="inline-flex h-11 items-center justify-center gap-2 rounded-full bg-warm px-4 text-sm font-medium text-warm-foreground transition hover:brightness-110 disabled:opacity-50"
            >
              <Play className="h-4 w-4" />
              {startMutation.isPending
                ? "Startet"
                : !profile
                  ? "Genre wählen"
                  : !durationInput
                    ? "Dauer wählen"
                    : "Generierung starten"}
            </button>
          </div>
        </section>

        <aside className="rounded-2xl border hairline bg-surface/40 p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-foreground">Status</h2>
            <StatusBadge status={job.data?.status ?? (activeJobId ? "queued" : "offline")} />
          </div>

          <div className="space-y-3">
            <StatusRow label="Vorgang" value={activeJobId ?? "kein aktiver Vorgang"} />
            <StatusRow label="Genre" value={profile ? profileLabel(profile) : "noch nicht gewählt"} />
            <StatusRow label="Dauer" value={durationInput ? `${finalDuration} min` : "noch nicht gewählt"} />
            <StatusRow label="BPM" value={targetBpm === "" ? "Standard (78)" : `${targetBpm}`} />
            <StatusRow label="GitHub" value={githubPush ? "aktiv" : "aus"} />
            <StatusRow label="Schritt" value={stepLabels[job.data?.step ?? "model_loading"]} />

            <div>
              <div className="mb-2 flex justify-between text-xs text-muted-foreground">
                <span>Fortschritt</span>
                <span>{progress}%</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-hairline/60">
                <div className="h-full bg-warm" style={{ width: `${progress}%` }} />
              </div>
            </div>

            {(job.data?.status === "running" || job.data?.status === "queued") && activeJobId && (
              <button
                onClick={() => cancelMutation.mutate(activeJobId)}
                className="inline-flex h-9 items-center gap-2 rounded-full border hairline px-3 text-sm text-muted-foreground hover:text-foreground"
              >
                <X className="h-4 w-4" />
                Abbrechen
              </button>
            )}

            {job.data?.status === "completed" && (
              <button
                onClick={() => navigate({ to: "/studio/library" })}
                className="inline-flex h-9 items-center rounded-full border hairline px-3 text-sm hover:border-warm/50"
              >
                Ausgabe öffnen
              </button>
            )}
          </div>
        </aside>
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

function Select({
  value,
  onChange,
  children,
}: {
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
}) {
  return (
    <select
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="h-10 w-full rounded-lg border hairline bg-background/40 px-3 text-sm outline-none focus:border-warm/60"
    >
      {children}
    </select>
  );
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-b hairline pb-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mono mt-1 break-words text-xs">{value}</div>
    </div>
  );
}
