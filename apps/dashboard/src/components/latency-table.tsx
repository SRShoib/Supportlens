// SPEC M9's "latency percentiles" area, every row a
// scripts/generate_m9_latency_report.py-persisted EvalRun
// (task=<task>, split="latency"). Status is a status-palette job (state,
// not identity/magnitude) -- emerald "OK" / red "OVER", the same
// alarm-color convention emerging-issues-panel.tsx and drift-panel.tsx use,
// shipped as an icon-free colored pill + text label so it never reads by
// color alone.

import type { EvalRun, LatencyMetrics } from "@/lib/api";

// SPEC §3's per-request CPU latency budgets, ms.
const BUDGET_MS_BY_TASK: Record<string, number> = {
  intent: 150,
  urgency: 150,
  entities: 250,
  sentiment: 150,
  emotion: 150,
  thread_summary: 3000,
  embedding: 100,
};

interface LatencyTableProps {
  runs: EvalRun[];
}

export function LatencyTable({ runs }: LatencyTableProps) {
  const rows = runs.filter((r) => r.split === "latency");
  if (rows.length === 0) {
    return (
      <p className="text-sm text-zinc-500 dark:text-zinc-400">
        No latency benchmarks yet -- run <code className="font-mono">make eval-latency</code>.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead>
          <tr className="text-zinc-500 dark:text-zinc-400">
            <th className="py-1 pr-3 font-medium">Task</th>
            <th className="py-1 pr-3 font-medium">Model</th>
            <th className="py-1 pr-3 text-right font-medium">p50</th>
            <th className="py-1 pr-3 text-right font-medium">p95</th>
            <th className="py-1 pr-3 text-right font-medium">Budget</th>
            <th className="py-1 font-medium">Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((run) => {
            const metrics = run.metrics as unknown as LatencyMetrics;
            const budget = BUDGET_MS_BY_TASK[run.task];
            const overBudget = budget !== undefined && metrics.p50_ms >= budget;
            return (
              <tr
                key={run.id}
                className="border-t border-zinc-100 transition-colors duration-150 hover:bg-indigo-50/50 dark:border-white/5 dark:hover:bg-indigo-500/5"
              >
                <td className="py-1.5 pr-3 text-zinc-700 dark:text-zinc-300">{run.task}</td>
                <td className="py-1.5 pr-3 font-mono text-zinc-600 dark:text-zinc-400">
                  {run.model_version}
                </td>
                <td className="py-1.5 pr-3 text-right tabular-nums text-zinc-700 dark:text-zinc-200">
                  {metrics.p50_ms.toFixed(1)} ms
                </td>
                <td className="py-1.5 pr-3 text-right tabular-nums text-zinc-500 dark:text-zinc-400">
                  {metrics.p95_ms.toFixed(1)} ms
                </td>
                <td className="py-1.5 pr-3 text-right tabular-nums text-zinc-400 dark:text-zinc-500">
                  {budget !== undefined ? `${budget} ms` : "—"}
                </td>
                <td className="py-1.5">
                  {budget === undefined ? (
                    <span className="text-zinc-400 dark:text-zinc-500">—</span>
                  ) : overBudget ? (
                    <span className="rounded-full bg-red-100 px-2 py-0.5 font-medium text-red-700 dark:bg-red-500/15 dark:text-red-400">
                      over
                    </span>
                  ) : (
                    <span className="rounded-full bg-emerald-100 px-2 py-0.5 font-medium text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-400">
                      ok
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
