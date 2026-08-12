"use client";

import { useActionState } from "react";

import {
  generateSuggestedReplyAction,
  type SuggestedReplyActionState,
} from "@/app/tickets/[id]/actions";

interface SuggestedReplyPanelProps {
  ticketId: string;
}

// Server Action modules ("use server" at the top of
// src/app/tickets/[id]/actions.ts) may only export async functions -- a
// plain object export like an initial-state constant doesn't survive that
// boundary, so it's defined here instead, next to its only consumer.
const initialSuggestedReplyState: SuggestedReplyActionState = {
  reply: null,
  error: null,
  requested: false,
};

// SPEC M8: "agent sees draft + sources side by side". Draft generation is
// gated behind an explicit button, not fetched on page load -- see
// src/app/tickets/[id]/actions.ts's module docstring for why.
export function SuggestedReplyPanel({ ticketId }: SuggestedReplyPanelProps) {
  const boundAction = generateSuggestedReplyAction.bind(null, ticketId);
  const [state, formAction, pending] = useActionState(boundAction, initialSuggestedReplyState);

  return (
    <section className="surface-card animate-fade-in-up mt-8 p-4">
      <div className="flex items-center justify-between gap-4">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-zinc-900 dark:text-zinc-50">
          <span aria-hidden>✨</span> Suggested reply
        </h2>
        <form action={formAction}>
          <button
            type="submit"
            disabled={pending}
            className="rounded-lg bg-gradient-to-r from-indigo-500 to-violet-600 px-3.5 py-1.5 text-xs font-medium text-white shadow-sm shadow-indigo-500/30 transition-all duration-200 hover:shadow-md hover:shadow-indigo-500/40 active:scale-[0.98] disabled:opacity-50"
          >
            {pending ? "Generating…" : state.requested ? "Regenerate" : "Generate suggested reply"}
          </button>
        </form>
      </div>

      {state.error && <p className="mt-3 text-sm text-red-600 dark:text-red-400">{state.error}</p>}

      {state.reply?.refused && (
        <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
          {state.reply.refusal_reason ?? "No confident match was found for this issue."}
        </p>
      )}

      {state.reply && !state.reply.refused && (
        <div className="animate-fade-in-up mt-4 grid gap-4 sm:grid-cols-2">
          <div>
            <h3 className="text-xs font-semibold tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
              Draft
            </h3>
            <p className="mt-2 rounded-lg border border-zinc-200/70 bg-zinc-50/70 p-3 text-sm whitespace-pre-wrap text-zinc-800 dark:border-white/10 dark:bg-white/5 dark:text-zinc-100">
              {state.reply.draft}
            </p>
            <p className="mt-2 text-xs text-zinc-400 dark:text-zinc-500">
              {state.reply.cached ? "Served from cache" : `Cost: $${state.reply.cost_usd.toFixed(4)}`}
            </p>
          </div>
          <div>
            <h3 className="text-xs font-semibold tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
              Sources
            </h3>
            <ol className="mt-2 space-y-2">
              {state.reply.sources.map((source) => (
                <li
                  key={source.index}
                  className={`rounded-lg border p-2.5 text-xs transition-colors duration-150 ${
                    state.reply?.cited_indices.includes(source.index)
                      ? "border-amber-300/80 bg-amber-50 dark:border-amber-400/25 dark:bg-amber-400/10"
                      : "border-zinc-200/70 dark:border-white/10"
                  }`}
                >
                  <p className="font-medium text-zinc-700 dark:text-zinc-300">
                    [{source.index}] {source.title ?? "Similar resolved case"}
                  </p>
                  <p className="mt-1 line-clamp-3 whitespace-pre-wrap text-zinc-500 dark:text-zinc-400">
                    {source.text}
                  </p>
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}
    </section>
  );
}
