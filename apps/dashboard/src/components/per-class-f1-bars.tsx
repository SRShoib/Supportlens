// Per-class F1 bar chart (SPEC M9: "per-class F1"). Dataviz skill: magnitude
// job -> sequential color, single hue, same blue as confusion-matrix.tsx
// (both live in the same classification-task card, so they read as one
// system). Sorted ascending -- weakest class first, the same convention
// scripts/generate_m3_report.py's model cards already use for this exact
// table. Value is direct-labeled at the bar's end (selective direct
// labeling, not a number on every point elsewhere) since there's no axis
// here to read the value off of otherwise.

const TRACK_WIDTH_PX = 100;

interface PerClassF1BarsProps {
  perClassF1: Record<string, number>;
}

export function PerClassF1Bars({ perClassF1 }: PerClassF1BarsProps) {
  const entries = Object.entries(perClassF1).sort(([, a], [, b]) => a - b);
  if (entries.length === 0) {
    return <p className="text-sm text-zinc-500 dark:text-zinc-400">No per-class F1 yet.</p>;
  }

  return (
    <ul className="space-y-1.5">
      {entries.map(([label, f1], index) => (
        <li key={label} className="flex items-center gap-2 text-xs">
          <span className="w-28 shrink-0 truncate text-right text-zinc-600 dark:text-zinc-300" title={label}>
            {label}
          </span>
          <span
            className="h-3.5 shrink-0 overflow-hidden rounded-full bg-zinc-100 dark:bg-white/5"
            style={{ width: `${TRACK_WIDTH_PX}px` }}
          >
            <span
              className="animate-grow-x block h-full rounded-full bg-gradient-to-r from-blue-500 to-indigo-500 dark:from-blue-400 dark:to-indigo-400"
              style={{
                width: `${Math.max(0, Math.min(1, f1)) * 100}%`,
                animationDelay: `${index * 25}ms`,
              }}
            />
          </span>
          <span className="w-10 shrink-0 tabular-nums text-zinc-500 dark:text-zinc-400">
            {f1.toFixed(2)}
          </span>
        </li>
      ))}
    </ul>
  );
}
