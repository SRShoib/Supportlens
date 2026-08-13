import type { ReactNode } from "react";

import type { EntitySpan } from "@/lib/api";

const LABEL_STYLES: Record<string, string> = {
  ORDER_ID: "bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-200",
  PRODUCT: "bg-violet-100 text-violet-900 dark:bg-violet-900/40 dark:text-violet-200",
  DATE: "bg-sky-100 text-sky-900 dark:bg-sky-900/40 dark:text-sky-200",
  AMOUNT: "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-200",
  ACCOUNT_REF: "bg-rose-100 text-rose-900 dark:bg-rose-900/40 dark:text-rose-200",
};
const DEFAULT_LABEL_STYLE = "bg-zinc-200 text-zinc-900 dark:bg-zinc-700 dark:text-zinc-100";

interface EntityHighlightedTextProps {
  text: string;
  entities: EntitySpan[];
}

// `span.start`/`span.end` are offsets into exactly this `text` string (the
// offset contract every M4 component shares -- see
// ml.inference.base.EntitySpan's docstring), so slicing directly is safe as
// long as text is passed unmodified from what was sent to /predict/entities.
export function EntityHighlightedText({ text, entities }: EntityHighlightedTextProps) {
  const nodes: ReactNode[] = [];
  let cursor = 0;

  for (const span of [...entities].sort((a, b) => a.start - b.start)) {
    // The hybrid predictor (ml/inference/hybrid_ner.py) routes each entity
    // type independently to rules or the model -- spans from different
    // types aren't guaranteed non-overlapping the way one system's own
    // output is. Drop anything that overlaps what's already rendered rather
    // than slicing text twice.
    if (span.start < cursor || span.end > text.length || span.start >= span.end) {
      continue;
    }
    if (span.start > cursor) {
      nodes.push(text.slice(cursor, span.start));
    }
    nodes.push(
      <mark
        key={`${span.start}-${span.end}-${span.label}`}
        title={`${span.label} · ${Math.round(span.score * 100)}% confidence`}
        className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-inherit shadow-sm transition-transform duration-150 hover:-translate-y-px hover:shadow ${
          LABEL_STYLES[span.label] ?? DEFAULT_LABEL_STYLE
        }`}
      >
        {text.slice(span.start, span.end)}
        <span className="rounded-sm bg-white/60 px-1 text-[10px] font-semibold tracking-wide uppercase dark:bg-black/20">
          {span.label}
        </span>
      </mark>,
    );
    cursor = span.end;
  }
  if (cursor < text.length) {
    nodes.push(text.slice(cursor));
  }

  return <>{nodes}</>;
}
