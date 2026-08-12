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
    <section className="mt-8 rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
      <div className="flex items-center justify-between gap-4">
        <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">Suggested reply</h2>
        <form action={formAction}>
          <button
            type="submit"
            disabled={pending}
            className="rounded-lg bg-zinc-900 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50 dark:bg-zinc-50 dark:text-zinc-900"
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
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <div>
            <h3 className="text-xs font-semibold tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
              Draft
            </h3>
            <p className="mt-2 text-sm whitespace-pre-wrap text-zinc-800 dark:text-zinc-100">
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
                  className={`rounded-md border p-2 text-xs ${
                    state.reply?.cited_indices.includes(source.index)
                      ? "border-amber-300 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/30"
                      : "border-zinc-200 dark:border-zinc-800"
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
