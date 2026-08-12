// SPEC M9's "retrieval metrics" area: M8's dense-vs-rerank hit-rate@5
// comparison, read back from the same eval_runs rows
// scripts/generate_m8_report.py persists (model_version="dense_v1" vs.
// "dense_rerank_v1"). Same magnitude job/single-hue treatment as
// per-class-f1-bars.tsx, direct-labeled since there's no shared axis here.

import type { EvalRun, RetrievalRunMetrics } from "@/lib/api";

const TRACK_WIDTH_PX = 160;

interface RetrievalPanelProps {
  runs: EvalRun[];
}

export function RetrievalPanel({ runs }: RetrievalPanelProps) {
  const latestByVariant = new Map<string, EvalRun>();
  for (const run of runs) {
    const existing = latestByVariant.get(run.model_version);
    if (!existing || run.started_at > existing.started_at) {
      latestByVariant.set(run.model_version, run);
    }
  }
  const variants = [...latestByVariant.values()].sort(
    (a, b) => (a.metrics as unknown as RetrievalRunMetrics).hit_rate_at_k -
      (b.metrics as unknown as RetrievalRunMetrics).hit_rate_at_k,
  );

  if (variants.length === 0) {
    return (
      <p className="text-sm text-zinc-500 dark:text-zinc-400">
        No retrieval eval runs yet -- run <code className="font-mono">make eval-search</code>.
      </p>
    );
  }

  return (
    <ul className="space-y-1.5">
      {variants.map((run) => {
        const metrics = run.metrics as unknown as RetrievalRunMetrics;
        return (
          <li key={run.model_version} className="flex items-center gap-2 text-xs">
            <span className="w-32 shrink-0 truncate text-right font-mono text-zinc-600 dark:text-zinc-300">
              {run.model_version}
            </span>
            <span
              className="h-3.5 shrink-0 rounded-sm bg-zinc-100 dark:bg-zinc-800"
              style={{ width: `${TRACK_WIDTH_PX}px` }}
            >
              <span
                className="block h-full rounded-sm bg-blue-500 dark:bg-blue-400"
                style={{ width: `${Math.max(0, Math.min(1, metrics.hit_rate_at_k)) * 100}%` }}
              />
            </span>
            <span className="w-28 shrink-0 tabular-nums text-zinc-500 dark:text-zinc-400">
              {metrics.hit_rate_at_k.toFixed(3)} (n={metrics.n_queries})
            </span>
          </li>
        );
      })}
    </ul>
  );
}
