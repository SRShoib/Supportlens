// Generic "eval runs over time" table (SPEC M9: "per-model eval runs over
// time") -- model_version/dataset/split/started_at are always shown; each
// caller supplies its own task-specific headline column(s) (macro-F1, NPMI,
// ROUGE, judge scores, ...) rather than this component knowing every
// task's metrics shape.

import type { ReactNode } from "react";
import type { EvalRun } from "@/lib/api";
import { formatDateTime } from "@/lib/format";

export interface EvalRunsTableColumn {
  header: string;
  render: (run: EvalRun) => ReactNode;
}

interface EvalRunsTableProps {
  runs: EvalRun[];
  columns: EvalRunsTableColumn[];
  emptyMessage: string;
}

export function EvalRunsTable({ runs, columns, emptyMessage }: EvalRunsTableProps) {
  if (runs.length === 0) {
    return <p className="text-sm text-zinc-500 dark:text-zinc-400">{emptyMessage}</p>;
  }

  return (
    <div className="overflow-x-auto rounded-lg">
      <table className="w-full text-left text-xs">
        <thead>
          <tr className="text-zinc-500 dark:text-zinc-400">
            <th className="py-1.5 pr-3 font-medium">Model</th>
            <th className="py-1.5 pr-3 font-medium">Dataset</th>
            <th className="py-1.5 pr-3 font-medium">Split</th>
            {columns.map((col) => (
              <th key={col.header} className="py-1.5 pr-3 text-right font-medium">
                {col.header}
              </th>
            ))}
            <th className="py-1.5 font-medium">Run at</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr
              key={run.id}
              className="border-t border-zinc-100 transition-colors duration-150 hover:bg-indigo-50/50 dark:border-white/5 dark:hover:bg-indigo-500/5"
            >
              <td className="py-1.5 pr-3 font-mono text-zinc-700 dark:text-zinc-300">
                {run.model_version}
              </td>
              <td className="py-1.5 pr-3 text-zinc-500 dark:text-zinc-400">{run.dataset}</td>
              <td className="py-1.5 pr-3 text-zinc-500 dark:text-zinc-400">{run.split}</td>
              {columns.map((col) => (
                <td
                  key={col.header}
                  className="py-1.5 pr-3 text-right tabular-nums text-zinc-700 dark:text-zinc-200"
                >
                  {col.render(run)}
                </td>
              ))}
              <td className="py-1.5 text-zinc-400 dark:text-zinc-500">
                {formatDateTime(run.started_at)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
