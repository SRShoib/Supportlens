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
      <div className="surface-card p-4">
        <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">Emerging issues</h2>
        <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
          No topic&apos;s weekly volume has spiked (z-score &gt; 2) in the current window.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-red-200/70 bg-red-50/80 p-4 shadow-sm backdrop-blur-sm dark:border-red-500/20 dark:bg-red-500/[0.06]">
      <h2 className="flex items-center gap-1.5 text-sm font-semibold text-red-800 dark:text-red-300">
        <span aria-hidden className="animate-pulse-ring inline-block text-red-600 dark:text-red-400">
          ▲
        </span>
        Emerging issues
      </h2>
      <ul className="mt-3 space-y-2">
        {issues.map((issue, index) => (
          <li
            key={`${issue.topic_id}-${issue.week}`}
            className="animate-fade-in-up flex items-center justify-between gap-4 rounded-xl bg-white/90 px-3 py-2 text-sm shadow-sm transition-transform duration-150 hover:-translate-y-0.5 dark:bg-zinc-950/60"
            style={{ animationDelay: `${index * 40}ms` }}
          >
            <div className="min-w-0">
              <p className="truncate font-medium text-zinc-900 dark:text-zinc-50">{issue.label}</p>
              <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">
                week of {issue.week} · {issue.count} tickets
              </p>
            </div>
            <span className="shrink-0 rounded-full bg-red-100 px-2 py-1 text-xs font-medium text-red-700 dark:bg-red-500/15 dark:text-red-300">
              z {issue.z_score.toFixed(1)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
