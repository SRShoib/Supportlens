// M4's per-entity-type span metrics (SPEC M9's "per-class F1", entity-typed
// variant -- span_metrics.py's per_type has no confusion matrix equivalent,
// spans aren't a fixed-size classification grid, so this is a table instead
// of confusion-matrix.tsx's heatmap). F1 column gets the same sequential
// blue intensity as the other two magnitude charts on this page, applied to
// a table cell rather than a bar -- still text-first, color reinforcing.

import type { SpanTypeMetrics } from "@/lib/api";

const SEQUENTIAL_STEPS = [
  "bg-blue-50 dark:bg-blue-950",
  "bg-blue-100 dark:bg-blue-900",
  "bg-blue-200 dark:bg-blue-800/80",
  "bg-blue-300 dark:bg-blue-700/80",
  "bg-blue-400 dark:bg-blue-600",
  "bg-blue-500 dark:bg-blue-500",
];
const LIGHT_TEXT_STEP_INDEX = 4;

function bucketIndex(value: number): number {
  return Math.min(SEQUENTIAL_STEPS.length - 1, Math.floor(value * SEQUENTIAL_STEPS.length));
}

interface SpanMetricsTableProps {
  perType: Record<string, SpanTypeMetrics>;
}

export function SpanMetricsTable({ perType }: SpanMetricsTableProps) {
  const entries = Object.entries(perType);
  if (entries.length === 0) {
    return <p className="text-sm text-zinc-500 dark:text-zinc-400">No span metrics yet.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead>
          <tr className="text-zinc-500 dark:text-zinc-400">
            <th className="py-1 pr-3 font-medium">Entity</th>
            <th className="py-1 pr-3 text-right font-medium">F1</th>
            <th className="py-1 pr-3 text-right font-medium">Precision</th>
            <th className="py-1 pr-3 text-right font-medium">Recall</th>
            <th className="py-1 text-right font-medium">Support</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([label, m]) => {
            const step = bucketIndex(m.f1);
            return (
              <tr
                key={label}
                className="border-t border-zinc-100 transition-colors duration-150 hover:bg-indigo-50/50 dark:border-white/5 dark:hover:bg-indigo-500/5"
              >
                <td className="py-1 pr-3 text-zinc-700 dark:text-zinc-300">{label}</td>
                <td
                  className={`py-1 pr-3 text-right tabular-nums ${SEQUENTIAL_STEPS[step]} ${
                    step >= LIGHT_TEXT_STEP_INDEX ? "text-white" : "text-zinc-700 dark:text-zinc-200"
                  }`}
                  title={`95% CI [${m.f1_ci_low.toFixed(2)}, ${m.f1_ci_high.toFixed(2)}]`}
                >
                  {m.f1.toFixed(3)}
                </td>
                <td className="py-1 pr-3 text-right tabular-nums text-zinc-600 dark:text-zinc-300">
                  {m.precision.toFixed(3)}
                </td>
                <td className="py-1 pr-3 text-right tabular-nums text-zinc-600 dark:text-zinc-300">
                  {m.recall.toFixed(3)}
                </td>
                <td className="py-1 text-right tabular-nums text-zinc-500 dark:text-zinc-400">
                  {m.support}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
