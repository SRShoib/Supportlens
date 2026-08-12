interface ThreadSummaryProps {
  summary: string;
}

// SPEC M6: "2-line summary shown at top of every ticket view" -- line-clamp-2
// is the literal implementation of that, not a stylistic choice.
export function ThreadSummary({ summary }: ThreadSummaryProps) {
  return (
    <div className="mt-3 rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 dark:border-zinc-800 dark:bg-zinc-900">
      <span className="text-xs font-medium tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
        Summary
      </span>
      <p className="mt-0.5 line-clamp-2 text-sm text-zinc-700 dark:text-zinc-300">{summary}</p>
    </div>
  );
}
