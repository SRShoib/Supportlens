// A multi-line "ticket volume per topic per week" trend chart (SPEC M7:
// "topics-over-time chart"), hand-rolled inline SVG following
// sentiment-sparkline.tsx's convention exactly: Tailwind SVG utility
// classes (never fill=/stroke= attributes), role="img" + aria-label, a
// <title> per point for hover detail.
//
// Categorical color job (dataviz skill): blue/amber/violet/cyan/fuchsia/
// lime, fixed order, one hue per topic identity -- deliberately avoids
// emerald/rose (already used elsewhere in this app for sentiment's
// diverging positive/negative, sentiment-sparkline.tsx) and red (reserved
// below as the "emerging issue" status ring, never a topic identity color
// -- the skill's status palette is never reused for series identity).
// indigo was tried and dropped for the 6th slot: next to blue (slot 1) on
// the hue wheel, the two read as the same color at chart line weight --
// lime sits far enough around the wheel from every other slot to stay
// distinct. The legend below the chart pairs every color swatch with its
// label and total count, so identity is never color-alone.
//
// At most MAX_SERIES topics are plotted (raw ticket volume, not the
// backend's normalized "share" the z-score is actually computed on --
// share is the right basis for *detection*, but count is the more legible
// read for a volume-over-time chart). SPEC M7 wants >=30 topics total and
// a 30-line chart is unreadable; the rest fold into a text note below the
// legend (dataviz skill, series-count ladder: "5-6 soft cap; legend or
// small multiples").

import type { TopicVolumeSeries } from "@/lib/api";

const MAX_SERIES = 6;
const WIDTH = 640;
const HEIGHT = 200;
const PADDING_LEFT = 12;
const PADDING_RIGHT = 12;
const PADDING_TOP = 12;
const PADDING_BOTTOM = 24;

const SERIES_STROKE_CLASS = [
  "stroke-blue-500 dark:stroke-blue-400",
  "stroke-amber-500 dark:stroke-amber-400",
  "stroke-violet-500 dark:stroke-violet-400",
  "stroke-cyan-500 dark:stroke-cyan-400",
  "stroke-fuchsia-500 dark:stroke-fuchsia-400",
  "stroke-lime-500 dark:stroke-lime-400",
];
const SERIES_FILL_CLASS = [
  "fill-blue-500 dark:fill-blue-400",
  "fill-amber-500 dark:fill-amber-400",
  "fill-violet-500 dark:fill-violet-400",
  "fill-cyan-500 dark:fill-cyan-400",
  "fill-fuchsia-500 dark:fill-fuchsia-400",
  "fill-lime-500 dark:fill-lime-400",
];
const SERIES_SWATCH_CLASS = [
  "bg-blue-500 dark:bg-blue-400",
  "bg-amber-500 dark:bg-amber-400",
  "bg-violet-500 dark:bg-violet-400",
  "bg-cyan-500 dark:bg-cyan-400",
  "bg-fuchsia-500 dark:bg-fuchsia-400",
  "bg-lime-500 dark:bg-lime-400",
];
const EMERGING_RING_CLASS = "stroke-red-600 dark:stroke-red-500";

interface TopicsOverTimeChartProps {
  weeks: string[];
  series: TopicVolumeSeries[];
}

function formatWeek(week: string): string {
  return new Date(`${week}T00:00:00Z`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

function seriesTotal(series: TopicVolumeSeries): number {
  return series.points.reduce((sum, point) => sum + point.count, 0);
}

export function TopicsOverTimeChart({ weeks, series }: TopicsOverTimeChartProps) {
  if (weeks.length === 0 || series.length === 0) {
    return (
      <p className="text-sm text-zinc-500 dark:text-zinc-400">
        No topic volume yet -- run <code className="font-mono">make assign-topics</code> against the
        API first.
      </p>
    );
  }

  const ranked = [...series].sort((a, b) => seriesTotal(b) - seriesTotal(a));
  const shown = ranked.slice(0, MAX_SERIES);
  const hiddenCount = ranked.length - shown.length;

  const maxCount = Math.max(1, ...shown.flatMap((s) => s.points.map((p) => p.count)));
  const plotWidth = WIDTH - PADDING_LEFT - PADDING_RIGHT;
  const plotHeight = HEIGHT - PADDING_TOP - PADDING_BOTTOM;
  const stepX = weeks.length > 1 ? plotWidth / (weeks.length - 1) : 0;
  const xFor = (weekIndex: number) =>
    PADDING_LEFT + (weeks.length > 1 ? stepX * weekIndex : plotWidth / 2);
  const yFor = (count: number) => PADDING_TOP + plotHeight * (1 - count / maxCount);
  const baselineY = HEIGHT - PADDING_BOTTOM;

  return (
    <div>
      <svg
        width={WIDTH}
        height={HEIGHT}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`Weekly ticket volume for ${shown.length} topics across ${weeks.length} weeks, ${formatWeek(weeks[0])} through ${formatWeek(weeks[weeks.length - 1])}`}
        className="w-full"
      >
        <line
          x1={PADDING_LEFT}
          y1={baselineY}
          x2={WIDTH - PADDING_RIGHT}
          y2={baselineY}
          strokeWidth={1}
          className="stroke-zinc-200 dark:stroke-zinc-700"
        />
        {weeks.map((week, i) => (
          <text
            key={week}
            x={xFor(i)}
            y={HEIGHT - 6}
            textAnchor="middle"
            className="fill-zinc-400 text-[9px] dark:fill-zinc-500"
          >
            {formatWeek(week)}
          </text>
        ))}

        {shown.map((s, seriesIndex) => {
          const pointByWeek = new Map(s.points.map((p) => [p.week, p]));
          const coords = weeks.map((week, i) => ({
            week,
            x: xFor(i),
            y: yFor(pointByWeek.get(week)?.count ?? 0),
            point: pointByWeek.get(week),
          }));
          const path = coords.map((c) => `${c.x},${c.y}`).join(" ");
          const strokeClass = SERIES_STROKE_CLASS[seriesIndex % SERIES_STROKE_CLASS.length];
          const fillClass = SERIES_FILL_CLASS[seriesIndex % SERIES_FILL_CLASS.length];

          return (
            <g key={s.topic_id}>
              <polyline
                points={path}
                fill="none"
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
                className={strokeClass}
              />
              {coords.map((c) => (
                <g key={c.week}>
                  {c.point?.is_emerging && (
                    <circle cx={c.x} cy={c.y} r={6} fill="none" strokeWidth={2} className={EMERGING_RING_CLASS} />
                  )}
                  <circle
                    cx={c.x}
                    cy={c.y}
                    r={3}
                    strokeWidth={1.5}
                    className={`${fillClass} stroke-white dark:stroke-zinc-950`}
                  >
                    <title>
                      {`${s.label} · ${formatWeek(c.week)}: ${c.point?.count ?? 0} tickets`}
                      {c.point?.is_emerging ? " · emerging issue (z > 2)" : ""}
                    </title>
                  </circle>
                </g>
              ))}
            </g>
          );
        })}
      </svg>

      <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5 text-xs">
        {shown.map((s, seriesIndex) => (
          <li key={s.topic_id} className="flex items-center gap-1.5">
            <span
              className={`h-2 w-2 shrink-0 rounded-full ${SERIES_SWATCH_CLASS[seriesIndex % SERIES_SWATCH_CLASS.length]}`}
            />
            <span className="text-zinc-700 dark:text-zinc-300">{s.label}</span>
            <span className="text-zinc-400 dark:text-zinc-500">({seriesTotal(s)})</span>
            {s.points.some((p) => p.is_emerging) && (
              <span className="rounded-full bg-red-100 px-1.5 py-0.5 font-medium text-red-700 dark:bg-red-950 dark:text-red-400">
                emerging
              </span>
            )}
          </li>
        ))}
      </ul>
      {hiddenCount > 0 && (
        <p className="mt-2 text-xs text-zinc-400 dark:text-zinc-500">
          +{hiddenCount} more topic{hiddenCount === 1 ? "" : "s"} not shown, ranked by volume.
        </p>
      )}
    </div>
  );
}
