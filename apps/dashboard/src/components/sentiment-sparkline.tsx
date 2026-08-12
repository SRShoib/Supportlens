// A "diverging line vs baseline" micro-chart (dataviz skill: polarity job ->
// diverging color, two hues + a neutral gray midpoint -- never a hue at the
// midpoint, and warm/cool poles so they read as opposite). Each message's
// signed sentiment score plots above (positive) or below (negative) a zero
// baseline; the y-position is the primary signal, color a reinforcing
// secondary one -- never the only channel, since every point also carries a
// hover title and the endpoint gets a direct text label.

const WIDTH = 160;
const HEIGHT = 36;
const PADDING_X = 6;
const AMPLITUDE = HEIGHT / 2 - 6;

const DOT_FILL_CLASS: Record<string, string> = {
  positive: "fill-emerald-500 dark:fill-emerald-400",
  neutral: "fill-zinc-400 dark:fill-zinc-500",
  negative: "fill-rose-500 dark:fill-rose-400",
};

const LABEL_TEXT_CLASS: Record<string, string> = {
  positive: "text-emerald-600 dark:text-emerald-400",
  neutral: "text-zinc-500 dark:text-zinc-400",
  negative: "text-rose-600 dark:text-rose-400",
};

interface SentimentSparklineProps {
  sequence: string[];
  scores: number[];
  resolutionQuality: number;
}

export function SentimentSparkline({
  sequence,
  scores,
  resolutionQuality,
}: SentimentSparklineProps) {
  if (sequence.length === 0) {
    return null;
  }

  const baselineY = HEIGHT / 2;
  const stepX = sequence.length > 1 ? (WIDTH - PADDING_X * 2) / (sequence.length - 1) : 0;
  const points = sequence.map((label, i) => ({
    x: sequence.length > 1 ? PADDING_X + stepX * i : WIDTH / 2,
    y: baselineY - (scores[i] ?? 0) * AMPLITUDE,
    label,
    score: scores[i] ?? 0,
  }));
  const path = points.map((p) => `${p.x},${p.y}`).join(" ");
  const finalPoint = points[points.length - 1];

  return (
    <div className="animate-fade-in flex items-center gap-2">
      <svg
        width={WIDTH}
        height={HEIGHT}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`Sentiment trajectory across ${sequence.length} message${
          sequence.length === 1 ? "" : "s"
        }: ${sequence.join(" then ")}`}
        className="shrink-0"
      >
        <line
          x1={PADDING_X}
          y1={baselineY}
          x2={WIDTH - PADDING_X}
          y2={baselineY}
          strokeWidth={1}
          className="stroke-zinc-200 dark:stroke-zinc-700"
        />
        {points.length > 1 && (
          <polyline
            points={path}
            fill="none"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            className="stroke-zinc-300 dark:stroke-zinc-600"
          />
        )}
        {points.map((p, i) => (
          <circle
            key={i}
            cx={p.x}
            cy={p.y}
            r={4}
            strokeWidth={2}
            className={`${DOT_FILL_CLASS[p.label] ?? DOT_FILL_CLASS.neutral} stroke-white dark:stroke-zinc-950`}
          >
            <title>{`Message ${i + 1}: ${p.label} (${p.score.toFixed(2)})`}</title>
          </circle>
        ))}
      </svg>
      <div className="text-xs">
        <span className={`font-medium ${LABEL_TEXT_CLASS[finalPoint.label] ?? LABEL_TEXT_CLASS.neutral}`}>
          ending {finalPoint.label}
        </span>
        <span className="ml-1 text-zinc-400 dark:text-zinc-500">
          · resolution {resolutionQuality >= 0 ? "+" : ""}
          {resolutionQuality.toFixed(2)}
        </span>
      </div>
    </div>
  );
}
