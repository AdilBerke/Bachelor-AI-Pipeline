import { SCORE_CATEGORIES, type AudioScoreRating, type ScoreMap } from "../lib/api";

function scoreValue(scores: ScoreMap, category: string): number {
  const value = Number(scores?.[category] ?? 0);
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

function scorePoints(scores: ScoreMap) {
  const width = 520;
  const height = 220;
  const left = 44;
  const right = 16;
  const top = 18;
  const bottom = 58;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;

  return SCORE_CATEGORIES.map((category, index) => {
    const x = left + (plotWidth / (SCORE_CATEGORIES.length - 1)) * index;
    const y = top + plotHeight - (scoreValue(scores, category) / 100) * plotHeight;
    return { category, x, y, value: scoreValue(scores, category) };
  });
}

function ScoreGraph({ title, scores }: { title: string; scores: ScoreMap }) {
  const points = scorePoints(scores);
  const path = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`)
    .join(" ");

  return (
    <section className="rounded-xl border hairline bg-background/60 p-4">
      <h3 className="mono mb-3 text-[10px] uppercase tracking-widest text-muted-foreground">
        {title}
      </h3>
      <svg viewBox="0 0 520 220" role="img" aria-label={title} className="h-auto w-full">
        <line x1="44" y1="18" x2="44" y2="162" className="stroke-hairline" />
        <line x1="44" y1="162" x2="504" y2="162" className="stroke-hairline" />
        {[0, 25, 50, 75, 100].map((tick) => {
          const y = 162 - (tick / 100) * 144;
          return (
            <g key={tick}>
              <line x1="44" y1={y} x2="504" y2={y} className="stroke-hairline opacity-60" />
              <text x="12" y={y + 4} className="fill-muted-foreground text-[10px]">
                {tick}
              </text>
            </g>
          );
        })}
        <path d={path} className="fill-none stroke-warm" strokeWidth="3" strokeLinecap="round" />
        {points.map((point, index) => (
          <g key={point.category}>
            <circle cx={point.x} cy={point.y} r="5" className="fill-warm stroke-background" strokeWidth="2" />
            <text x={point.x} y={point.y - 10} textAnchor="middle" className="fill-foreground text-[10px]">
              {Math.round(point.value)}
            </text>
            <text
              x={point.x}
              y="186"
              textAnchor="middle"
              className="fill-muted-foreground text-[9px]"
            >
              {index + 1}
            </text>
          </g>
        ))}
      </svg>
      <div className="mt-2 grid gap-1 text-xs text-muted-foreground">
        {SCORE_CATEGORIES.map((category, index) => (
          <div key={category}>
            {index + 1}. {category}
          </div>
        ))}
      </div>
    </section>
  );
}

function ScoreList({ title, scores }: { title: string; scores: ScoreMap }) {
  return (
    <section className="rounded-xl border hairline bg-background/60 p-4">
      <h3 className="mono mb-3 text-[10px] uppercase tracking-widest text-muted-foreground">
        {title}
      </h3>
      <div className="grid gap-2">
        {SCORE_CATEGORIES.map((category) => (
          <div key={category} className="flex items-center justify-between gap-3 text-sm">
            <span className="text-muted-foreground">{category}</span>
            <span className="mono text-foreground">{scoreValue(scores, category).toFixed(1)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

export function ScoreComparison({ rating }: { rating?: AudioScoreRating }) {
  if (!rating || rating.status !== "finished") return null;

  return (
    <div className="mt-5 grid gap-4">
      <div className="text-sm">
        Genre: <span className="font-medium text-warm">{rating.genre}</span>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <ScoreGraph title="Genrestandard" scores={rating.genrestandard_scores} />
        <ScoreGraph title="Generierte Audio" scores={rating.generierte_audio_scores} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <ScoreList title="Scores des Genrestandards" scores={rating.genrestandard_scores} />
        <ScoreList title="Scores der generierten Audio" scores={rating.generierte_audio_scores} />
      </div>

      {rating.auswertung && (
        <section className="rounded-xl border hairline bg-background/60 p-4 text-sm text-muted-foreground">
          {rating.auswertung}
        </section>
      )}
    </div>
  );
}
