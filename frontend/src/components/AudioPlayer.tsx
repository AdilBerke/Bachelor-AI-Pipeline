import { useEffect, useRef, useState } from "react";
import { Download, Pause, Play, Volume2 } from "lucide-react";

interface Props {
  src?: string;
  title: string;
  subtitle?: string;
  durationFallbackSec?: number;
  downloadUrl?: string;
  metadata?: Record<string, string | number>;
}

function fmt(sec: number): string {
  if (!isFinite(sec) || sec < 0) return "00:00";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const mm = String(m).padStart(2, "0");
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function AudioPlayer({
  src,
  title,
  subtitle,
  durationFallbackSec,
  downloadUrl,
  metadata,
}: Props) {
  const ref = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [t, setT] = useState(0);
  const [dur, setDur] = useState(durationFallbackSec ?? 0);
  const [vol, setVol] = useState(0.8);

  useEffect(() => {
    const a = ref.current;
    if (!a) return;
    const onTime = () => setT(a.currentTime);
    const onDur = () => setDur(a.duration || durationFallbackSec || 0);
    const onEnd = () => setPlaying(false);
    a.addEventListener("timeupdate", onTime);
    a.addEventListener("loadedmetadata", onDur);
    a.addEventListener("ended", onEnd);
    a.volume = vol;
    return () => {
      a.removeEventListener("timeupdate", onTime);
      a.removeEventListener("loadedmetadata", onDur);
      a.removeEventListener("ended", onEnd);
    };
  }, [src, vol, durationFallbackSec]);

  const toggle = async () => {
    const a = ref.current;
    if (!a || !src) return;
    if (playing) {
      a.pause();
      setPlaying(false);
    } else {
      try {
        await a.play();
        setPlaying(true);
      } catch (e) {
        console.warn("playback failed", e);
      }
    }
  };

  const seek = (pct: number) => {
    const a = ref.current;
    if (!a || !dur) return;
    a.currentTime = (pct / 100) * dur;
    setT(a.currentTime);
  };

  const noSrc = !src;

  return (
    <div className="rounded-2xl border hairline bg-surface/50 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium text-foreground">{title}</div>
          {subtitle && (
            <div className="mono mt-0.5 truncate text-[11px] uppercase tracking-widest text-muted-foreground">
              {subtitle}
            </div>
          )}
        </div>
        {downloadUrl && (
          <a
            href={downloadUrl}
            className="mono inline-flex shrink-0 items-center gap-1.5 rounded-full border hairline px-2.5 py-1.5 text-[10px] uppercase tracking-widest text-muted-foreground transition hover:bg-surface-2 hover:text-foreground"
          >
            <Download className="h-3.5 w-3.5" /> Download
          </a>
        )}
      </div>

      <div className="mt-4 flex items-center gap-3">
        <button
          onClick={toggle}
          disabled={noSrc}
          className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-warm text-warm-foreground transition hover:brightness-110 disabled:opacity-40"
          aria-label={playing ? "Pause" : "Abspielen"}
        >
          {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 translate-x-[1px]" />}
        </button>

        <div className="flex-1">
          <input
            type="range"
            min={0}
            max={100}
            value={dur ? (t / dur) * 100 : 0}
            onChange={(e) => seek(Number(e.target.value))}
            disabled={noSrc}
            className="w-full accent-warm"
          />
          <div className="mono mt-1 flex justify-between text-[11px] text-muted-foreground">
            <span>{fmt(t)}</span>
            <span>{fmt(dur)}</span>
          </div>
        </div>

        <div className="hidden w-32 items-center gap-2 sm:flex">
          <Volume2 className="h-4 w-4 text-muted-foreground" />
          <input
            type="range"
            min={0}
            max={100}
            value={vol * 100}
            onChange={(e) => setVol(Number(e.target.value) / 100)}
            className="w-full accent-warm"
          />
        </div>
      </div>

      {noSrc && (
        <div className="mt-3 rounded-lg border border-dashed hairline bg-background/40 p-2 text-[11px] text-muted-foreground">
          Keine Audio-URL vorhanden. Player wird funktional, sobald das lokale Backend die Datei
          über <span className="mono">/api/audio/&lt;id&gt;/download</span> liefert.
        </div>
      )}

      {metadata && Object.keys(metadata).length > 0 && (
        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
          {Object.entries(metadata).map(([k, v]) => (
            <div key={k} className="rounded-lg bg-surface-2/60 px-2 py-1.5">
              <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
                {k}
              </div>
              <div className="mono text-xs text-foreground">{String(v)}</div>
            </div>
          ))}
        </div>
      )}

      <audio ref={ref} src={src} preload="metadata" />
    </div>
  );
}
