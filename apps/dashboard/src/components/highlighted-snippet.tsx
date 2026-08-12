import type { ReactNode } from "react";

import type { SearchHighlight } from "@/lib/api";

interface HighlightedSnippetProps {
  text: string;
  highlights: SearchHighlight[];
}

// `span.start`/`span.end` are offsets into exactly this `text` string
// (ml/inference/highlight.py's contract) -- same "slice the unmodified
// string" rule entity-highlighted-text.tsx follows for M4's entity spans,
// simplified here since search highlights carry no label/score, just a
// span to mark.
export function HighlightedSnippet({ text, highlights }: HighlightedSnippetProps) {
  const nodes: ReactNode[] = [];
  let cursor = 0;

  for (const span of [...highlights].sort((a, b) => a.start - b.start)) {
    if (span.start < cursor || span.end > text.length || span.start >= span.end) {
      continue;
    }
    if (span.start > cursor) {
      nodes.push(text.slice(cursor, span.start));
    }
    nodes.push(
      <mark
        key={`${span.start}-${span.end}`}
        className="rounded bg-amber-100 px-0.5 text-inherit dark:bg-amber-900/40"
      >
        {text.slice(span.start, span.end)}
      </mark>,
    );
    cursor = span.end;
  }
  if (cursor < text.length) {
    nodes.push(text.slice(cursor));
  }

  return <>{nodes}</>;
}
