interface ThreadSummaryProps {
  summary: string;
}

// SPEC M6: "2-line summary shown at top of every ticket view" -- line-clamp-2
// is the literal implementation of that, not a stylistic choice.
export function ThreadSummary({ summary }: ThreadSummaryProps) {
  return (
    <div className="mt-3 rounded-xl border border-indigo-100 bg-gradient-to-br from-indigo-50/80 to-violet-50/50 px-3.5 py-2.5 shadow-sm dark:border-indigo-400/15 dark:from-indigo-500/[0.07] dark:to-violet-500/[0.05]">
      <span className="text-xs font-semibold tracking-wide text-indigo-600 uppercase dark:text-indigo-300">
        Summary
      </span>
      <p className="mt-0.5 line-clamp-2 text-sm text-zinc-700 dark:text-zinc-300">{summary}</p>
    </div>
  );
}
