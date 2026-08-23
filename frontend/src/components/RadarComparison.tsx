import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { SCORE_CATEGORIES, type ScoreMap } from "../lib/api";

export function RadarComparison({
  genrestandardScores,
  generierteScores,
}: {
  genrestandardScores: ScoreMap;
  generierteScores: ScoreMap;
}) {
  // Kategorien, die fuer diese Audio (noch) gar nicht berechnet wurden, werden
  // ausgelassen statt als 0 geplottet - 0 waere eine echte, sehr schlechte
  // Messung und liesse sich sonst nicht von "nicht vorhanden" unterscheiden.
  const data = SCORE_CATEGORIES.filter(
    (k) => genrestandardScores?.[k] !== undefined || generierteScores?.[k] !== undefined,
  ).map((k) => ({
    k,
    baseline: Number(genrestandardScores?.[k] ?? 0),
    model: Number(generierteScores?.[k] ?? 0),
  }));

  return (
    <div className="h-[380px] w-full">
      <ResponsiveContainer>
        <RadarChart data={data} outerRadius="72%">
          <PolarGrid stroke="var(--hairline)" />
          <PolarAngleAxis
            dataKey="k"
            tick={{ fill: "var(--muted-foreground)", fontSize: 11, fontFamily: "Inter" }}
          />
          <PolarRadiusAxis
            domain={[0, 100]}
            tick={{ fill: "transparent" }}
            axisLine={false}
            stroke="var(--hairline)"
          />
          <Radar
            name="Genrestandard"
            dataKey="baseline"
            stroke="var(--cool)"
            fill="var(--cool)"
            fillOpacity={0.15}
            strokeWidth={1.5}
          />
          <Radar
            name="Generierte Audio"
            dataKey="model"
            stroke="var(--warm)"
            fill="var(--warm)"
            fillOpacity={0.2}
            strokeWidth={1.8}
          />
          <Tooltip
            contentStyle={{
              background: "var(--surface-2)",
              border: "1px solid var(--hairline)",
              borderRadius: 8,
              fontSize: 12,
            }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
