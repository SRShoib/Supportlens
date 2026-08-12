// Card list of currently-flagged emerging topics (SPEC M7: "an 'emerging
// issues' panel that fires on an injected synthetic spike"). Status color
// (dataviz skill: the status palette is reserved, never reused for series
// identity) -- red/critical accent + icon + label, matching
// topics-over-time-chart.tsx's emerging ring color, never color alone.

import type { EmergingIssue } from "@/lib/api";

interface EmergingIssuesPanelProps {
  issues: EmergingIssue[];
}

export function EmergingIssuesPanel({ issues }: EmergingIssuesPanelProps) {
  if (issues.length === 0) {
    return (
      <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
        <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">Emerging issues</h2>
        <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
          No topic&apos;s weekly volume has spiked (z-score &gt; 2) in the current window.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-950/30">
      <h2 className="flex items-center gap-1.5 text-sm font-semibold text-red-800 dark:text-red-300">
        <span aria-hidden className="text-red-600 dark:text-red-400">
          ▲
        </span>
        Emerging issues
      </h2>
      <ul className="mt-3 space-y-2">
        {issues.map((issue) => (
          <li
            key={`${issue.topic_id}-${issue.week}`}
            className="flex items-center justify-between gap-4 rounded-md bg-white px-3 py-2 text-sm dark:bg-zinc-950"
          >
            <div className="min-w-0">
              <p className="truncate font-medium text-zinc-900 dark:text-zinc-50">{issue.label}</p>
              <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">
                week of {issue.week} · {issue.count} tickets
              </p>
            </div>
            <span className="shrink-0 rounded-full bg-red-100 px-2 py-1 text-xs font-medium text-red-700 dark:bg-red-900 dark:text-red-300">
              z {issue.z_score.toFixed(1)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
