"use client";

import { useActionState, useRef } from "react";

import {
  type ClassificationComparison,
  tryItAction,
  type TryItActionState,
} from "@/app/try/actions";
import { EntityHighlightedText } from "@/components/entity-highlighted-text";
import type { TaskResult } from "@/lib/api";

const TASK_LABEL: Record<ClassificationComparison["task"], string> = {
  intent: "Intent",
  urgency: "Urgency",
  sentiment: "Sentiment",
  emotion: "Emotion",
};

// Server Action modules ("use server" at the top of src/app/try/actions.ts)
// may only export async functions -- same boundary rule
// suggested-reply-panel.tsx and search-form.tsx already document, so the
// initial-state constant lives here instead, next to its only consumer.
const initialState: TryItActionState = {
  text: "",
  classifications: [],
  entities: null,
  error: null,
};

const EXAMPLES = [
  "I've been waiting THREE WEEKS for order #48213 and nobody has replied to my last two emails. This is unacceptable, I want a refund NOW.",
  "hey quick question, does the pro plan include api access or do i need to upgrade again",
];

function ResultBlock({ label, result }: { label: string; result: TaskResult | null }) {
  return (
    <div>
      <p className="text-[11px] font-semibold tracking-wide text-zinc-400 uppercase dark:text-zinc-500">
        {label}
      </p>
      {result ? (
        <>
          <p className="mt-0.5 text-sm font-semibold text-zinc-900 dark:text-zinc-50">
            {result.label}
          </p>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            {result.probabilities
              ? `${(result.score * 100).toFixed(1)}% confidence`
              : `decision margin ${result.score.toFixed(2)}`}
          </p>
        </>
      ) : (
        <p className="mt-0.5 text-sm text-zinc-400 dark:text-zinc-500">not available</p>
      )}
    </div>
  );
}

function ClassificationCard({ comparison }: { comparison: ClassificationComparison }) {
  const { task, baseline, transformer } = comparison;
  const agree =
    baseline && transformer ? baseline.label === transformer.label : null;

  return (
    <div className="surface-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">
          {TASK_LABEL[task]}
        </h3>
        {agree !== null && (
          <span
            className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
              agree
                ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-400"
                : "bg-amber-100 text-amber-700 dark:bg-amber-500/15 dark:text-amber-400"
            }`}
          >
            {agree ? "agree" : "disagree"}
          </span>
        )}
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3">
        <ResultBlock label="Baseline" result={baseline} />
        <ResultBlock label="Transformer" result={transformer} />
      </div>
    </div>
  );
}

export function TryItForm() {
  const [state, formAction, pending] = useActionState(tryItAction, initialState);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  function fillExample(example: string) {
    if (!textareaRef.current) return;
    textareaRef.current.value = example;
    formRef.current?.requestSubmit();
  }

  return (
    <div>
      <form ref={formRef} action={formAction}>
        <textarea
          ref={textareaRef}
          name="text"
          defaultValue={state.text}
          rows={4}
          required
          placeholder="Paste a customer message to analyze..."
          className="w-full rounded-xl border border-zinc-300/80 bg-white/90 px-3.5 py-3 text-sm text-zinc-900 shadow-sm placeholder:text-zinc-400 transition-all duration-200 focus:border-pink-400 focus:ring-4 focus:ring-pink-500/15 focus:outline-none dark:border-white/10 dark:bg-white/5 dark:text-zinc-50 dark:placeholder:text-zinc-500"
        />
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => fillExample(example)}
                className="max-w-[16rem] truncate rounded-full bg-zinc-100 px-3 py-1 text-left text-xs text-zinc-600 transition-colors duration-150 hover:bg-zinc-200 dark:bg-white/5 dark:text-zinc-300 dark:hover:bg-white/10"
                title={example}
              >
                {example}
              </button>
            ))}
          </div>
          <button
            type="submit"
            disabled={pending}
            className="rounded-xl bg-gradient-to-r from-pink-500 to-rose-600 px-4 py-2.5 text-sm font-medium text-white shadow-sm shadow-pink-500/30 transition-all duration-200 hover:shadow-md hover:shadow-pink-500/40 active:scale-[0.98] disabled:opacity-50"
          >
            {pending ? "Analyzing…" : "Analyze"}
          </button>
        </div>
      </form>

      {state.error && <p className="mt-3 text-sm text-red-600 dark:text-red-400">{state.error}</p>}

      {state.classifications.length > 0 && (
        <div className="animate-fade-in-up mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
          {state.classifications.map((comparison) => (
            <ClassificationCard key={comparison.task} comparison={comparison} />
          ))}
        </div>
      )}

      {state.entities !== null && (
        <div
          className="surface-card animate-fade-in-up mt-4 p-4"
          style={{ animationDelay: "80ms" }}
        >
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">
            Extracted entities
          </h3>
          <p className="mt-2 text-sm whitespace-pre-wrap text-zinc-800 dark:text-zinc-100">
            {state.entities.length > 0 ? (
              <EntityHighlightedText text={state.text} entities={state.entities} />
            ) : (
              state.text
            )}
          </p>
          {state.entities.length === 0 && (
            <p className="mt-2 text-xs text-zinc-400 dark:text-zinc-500">
              No entities found in this text.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
