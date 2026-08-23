import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { api, profileLabel, SCORE_CATEGORIES } from "../lib/api";
import { PageHeader } from "../components/ui-bits";
import { Check, Music2, Pencil, Play, X } from "lucide-react";

export const Route = createFileRoute("/studio/library")({
  head: () => ({
    meta: [
      { title: "LOFI.GEN · Bibliothek" },
      { name: "description", content: "Alle generierten Lofi-Audios und GIFs durchsuchen." },
    ],
  }),
  component: LibraryPage,
});

const GENRE_COLOR: Record<string, string> = {
  jazz_lofi: "var(--cool)",
  chillhop_lofi: "var(--warm)",
  dreamy_lofi: "var(--chart-3)",
  study_lofi: "var(--chart-5)",
  guitar_lofi: "var(--chart-4)",
};

function averageScore(scores?: Record<string, number>): number | undefined {
  if (!scores) return undefined;
  const values = SCORE_CATEGORIES.map((k) => scores[k]).filter((v) => v !== undefined);
  if (!values.length) return undefined;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function fmtDur(sec: number) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

function LibraryPage() {
  const [tab, setTab] = useState<"audio" | "gifs">("audio");

  return (
    <div>
      <PageHeader
        title="Bibliothek"
        description="Alle generierten Longform-Audios und GIFs."
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
          onClick={() => setTab("gifs")}
          className={[
            "mono rounded-full border px-4 py-1.5 text-[11px] uppercase tracking-widest transition",
            tab === "gifs"
              ? "border-warm/45 bg-warm/15 text-warm"
              : "border-hairline hairline text-muted-foreground hover:text-foreground",
          ].join(" ")}
        >
          Gifs
        </button>
      </div>

      {tab === "audio" ? <AudioLibrary /> : <GifLibrary />}
    </div>
  );
}

function AudioLibrary() {
  const qc = useQueryClient();
  const audioQuery = useQuery({ queryKey: ["audio"], queryFn: api.audio });
  const profilesQuery = useQuery({ queryKey: ["profiles"], queryFn: api.promptProfiles });
  const list = audioQuery.data ?? [];
  const [genre, setGenre] = useState("Alle");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");

  const renameMutation = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => api.renameAudio(id, name),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["audio"] });
      setEditingId(null);
    },
  });

  function startEditing(id: string, currentTitle: string) {
    setEditingId(id);
    setDraftName(currentTitle);
  }

  function saveRename(id: string) {
    const trimmed = draftName.trim();
    if (!trimmed) return;
    renameMutation.mutate({ id, name: trimmed });
  }

  const chips = ["Alle", ...(profilesQuery.data ?? []).map((p) => p.label)];
  const filtered = list.filter((a) => genre === "Alle" || profileLabel(a.promptProfile) === genre);

  if (list.length === 0) {
    return (
      <div className="rounded-2xl border hairline bg-surface/40 p-10 text-center text-sm text-muted-foreground">
        Noch keine Audios vorhanden.
      </div>
    );
  }

  return (
    <>
      <div className="mb-6 flex flex-wrap gap-2">
        {chips.map((c) => (
          <button
            key={c}
            onClick={() => setGenre(c)}
            className={[
              "mono rounded-full border px-4 py-1.5 text-[11px] uppercase tracking-widest transition",
              c === genre
                ? "border-warm/45 bg-warm/15 text-warm"
                : "border-hairline hairline text-muted-foreground hover:text-foreground",
            ].join(" ")}
          >
            {c}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-5">
        {filtered.map((a) => {
          const rating = a.scoreRating ?? a.score_rating;
          const rated = rating?.status === "finished";
          const avg = rated ? averageScore(rating!.generierte_audio_scores) : undefined;
          const color = GENRE_COLOR[a.promptProfile] ?? "var(--warm)";
          const isEditing = editingId === a.id;
          return (
            <div key={a.id}>
              <Link to="/studio/library/$id" params={{ id: a.id }} className="group block">
                <div
                  className="relative aspect-square overflow-hidden rounded-xl border hairline"
                  style={{ background: `color-mix(in oklab, ${color} 22%, var(--surface))` }}
                >
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Music2 className="h-8 w-8 opacity-80" style={{ color }} />
                  </div>
                  {rated ? (
                    <span
                      className="mono absolute right-2 top-2 rounded-full bg-background/60 px-2 py-0.5 text-[10px] font-semibold backdrop-blur"
                      style={{ color }}
                    >
                      ⌀ {avg!.toFixed(0)}
                    </span>
                  ) : (
                    <span className="mono absolute right-2 top-2 rounded-full border hairline bg-background/60 px-2 py-0.5 text-[9px] uppercase tracking-wide text-muted-foreground backdrop-blur">
                      offen
                    </span>
                  )}
                  <span className="absolute bottom-2 left-2 grid h-7 w-7 place-items-center rounded-full bg-warm text-warm-foreground opacity-0 transition group-hover:opacity-100">
                    <Play className="h-3 w-3 translate-x-[1px]" />
                  </span>
                </div>
              </Link>

              {isEditing ? (
                <div className="mt-2 flex items-center gap-1">
                  <input
                    autoFocus
                    value={draftName}
                    onChange={(e) => setDraftName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") saveRename(a.id);
                      if (e.key === "Escape") setEditingId(null);
                    }}
                    className="h-7 w-0 min-w-0 flex-1 rounded-md border hairline bg-background/60 px-1.5 text-xs text-foreground outline-none focus:border-warm/60"
                  />
                  <button
                    onClick={() => saveRename(a.id)}
                    disabled={renameMutation.isPending}
                    className="shrink-0 rounded-md p-1 text-emerald-400 hover:bg-emerald-500/10 disabled:opacity-50"
                  >
                    <Check className="h-3.5 w-3.5" />
                  </button>
                  <button
                    onClick={() => setEditingId(null)}
                    className="shrink-0 rounded-md p-1 text-muted-foreground hover:bg-surface-2"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ) : (
                <div className="group/title mt-2 flex items-center gap-1">
                  <Link
                    to="/studio/library/$id"
                    params={{ id: a.id }}
                    className="truncate text-sm font-medium text-foreground hover:text-warm"
                  >
                    {a.title}
                  </Link>
                  <button
                    onClick={() => startEditing(a.id, a.title)}
                    className="shrink-0 rounded-md p-0.5 text-muted-foreground opacity-0 transition hover:bg-surface-2 hover:text-foreground group-hover/title:opacity-100"
                    title="Namen bearbeiten"
                  >
                    <Pencil className="h-3 w-3" />
                  </button>
                </div>
              )}
              <div className="mono mt-0.5 truncate text-[10.5px] text-muted-foreground">
                {profileLabel(a.promptProfile)} · {fmtDur(a.durationSec)}
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}

function GifLibrary() {
  const galleryQuery = useQuery({ queryKey: ["video-gallery"], queryFn: api.videoGallery });
  const gifs = useMemo(
    () => (galleryQuery.data ?? []).filter((g) => g.type === "gif").sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt)),
    [galleryQuery.data],
  );

  const scenarios = useMemo(() => ["Alle", ...Array.from(new Set(gifs.map((g) => g.scenario))).sort()], [gifs]);
  const [scenario, setScenario] = useState("Alle");
  const filtered = gifs.filter((g) => scenario === "Alle" || g.scenario === scenario);

  if (galleryQuery.isLoading) {
    return <div className="text-sm text-muted-foreground">Lädt ...</div>;
  }

  if (gifs.length === 0) {
    return (
      <div className="rounded-2xl border hairline bg-surface/40 p-10 text-center text-sm text-muted-foreground">
        Noch keine GIFs vorhanden.
      </div>
    );
  }

  return (
    <>
      <div className="mb-6 flex flex-wrap gap-2">
        {scenarios.map((s) => (
          <button
            key={s}
            onClick={() => setScenario(s)}
            className={[
              "mono rounded-full border px-4 py-1.5 text-[11px] uppercase tracking-widest transition",
              s === scenario
                ? "border-warm/45 bg-warm/15 text-warm"
                : "border-hairline hairline text-muted-foreground hover:text-foreground",
            ].join(" ")}
          >
            {s} {s !== "Alle" && `(${gifs.filter((g) => g.scenario === s).length})`}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-5">
        {filtered.map((g) => (
          <a key={g.id} href={api.absoluteUrl(g.url)} target="_blank" rel="noreferrer" className="group block">
            <div className="relative aspect-square overflow-hidden rounded-xl border hairline bg-surface">
              <img
                src={api.absoluteUrl(g.url)}
                alt={g.filename}
                className="h-full w-full object-cover"
                loading="lazy"
              />
              <span className="mono absolute right-2 top-2 rounded-full bg-background/60 px-2 py-0.5 text-[9px] uppercase tracking-wide text-warm backdrop-blur">
                GIF
              </span>
            </div>
            <div className="mt-2 truncate text-sm font-medium text-foreground group-hover:text-warm">
              {g.scenario}
            </div>
            <div className="mono mt-0.5 truncate text-[10.5px] text-muted-foreground">
              Runde {g.round ?? "?"} · Step {g.step ?? "?"} · {new Date(g.modifiedAt).toLocaleDateString("de-DE")}
            </div>
          </a>
        ))}
      </div>
    </>
  );
}
