"use client";

import Link from "next/link";
import { useActionState } from "react";

import { type SearchActionState, searchAction } from "@/app/search/actions";
import { HighlightedSnippet } from "@/components/highlighted-snippet";
import type { SearchResultSource } from "@/lib/api";

const SOURCE_LABEL: Record<SearchResultSource, string> = {
  ticket: "Resolved ticket",
  kb_article: "KB article",
};

// Two distinct hues, one per result source -- a real category (result kind),
// not a magnitude, so categorical color is the right call here (dataviz
// skill). Kept close to Search's own amber section identity for "ticket"
// since resolved tickets are this page's primary result type; KB articles
// get teal so the two never read as the same thing in a mixed results list.
const SOURCE_BADGE_CLASS: Record<SearchResultSource, string> = {
  ticket: "bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300",
  kb_article: "bg-teal-50 text-teal-700 dark:bg-teal-500/10 dark:text-teal-300",
};

// Server Action modules ("use server" at the top of
// src/app/search/actions.ts) may only export async functions -- a plain
// object export like an initial-state constant doesn't survive that
// boundary, so it's defined here instead, next to its only consumer.
const initialSearchState: SearchActionState = {
  results: [],
  reranked: true,
  query: "",
  error: null,
};

export function SearchForm() {
  const [state, formAction, pending] = useActionState(searchAction, initialSearchState);

  return (
    <div>
      <form action={formAction} className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          name="query"
          defaultValue={state.query}
          placeholder="Search resolved tickets and KB articles..."
          required
          className="min-w-64 flex-1 rounded-xl border border-zinc-300/80 bg-white/90 px-3.5 py-2.5 text-sm text-zinc-900 shadow-sm placeholder:text-zinc-400 transition-all duration-200 focus:border-amber-400 focus:ring-4 focus:ring-amber-500/15 focus:outline-none dark:border-white/10 dark:bg-white/5 dark:text-zinc-50 dark:placeholder:text-zinc-500"
        />
        <label className="flex items-center gap-2 text-sm text-zinc-600 dark:text-zinc-400">
          <input
            type="checkbox"
            name="rerank"
            defaultChecked
            className="rounded border-zinc-300 text-amber-600 focus:ring-amber-500 dark:border-zinc-700"
          />
          Rerank
        </label>
        <button
          type="submit"
          disabled={pending}
          className="rounded-xl bg-gradient-to-r from-amber-500 to-orange-600 px-4 py-2.5 text-sm font-medium text-white shadow-sm shadow-amber-500/30 transition-all duration-200 hover:shadow-md hover:shadow-amber-500/40 active:scale-[0.98] disabled:opacity-50"
        >
          {pending ? "Searching…" : "Search"}
        </button>
      </form>

      {state.error && <p className="mt-3 text-sm text-red-600 dark:text-red-400">{state.error}</p>}

      {!state.error && state.query && (
        <p className="mt-4 text-xs text-zinc-500 dark:text-zinc-400">
          {state.results.length} result{state.results.length === 1 ? "" : "s"} for &ldquo;
          {state.query}&rdquo; · {state.reranked ? "reranked" : "dense only"}
        </p>
      )}

      <ul className="mt-3 space-y-3">
        {state.results.map((result, index) => (
          <li
            key={`${result.source}-${result.id}`}
            className="surface-card surface-card-interactive animate-fade-in-up p-4"
            style={{ animationDelay: `${Math.min(index, 12) * 40}ms` }}
          >
            <div className="flex items-center justify-between gap-2">
              <span
                className={`rounded-full px-2 py-0.5 text-xs font-medium ${SOURCE_BADGE_CLASS[result.source]}`}
              >
                {SOURCE_LABEL[result.source]}
              </span>
              <span className="text-xs text-zinc-400 dark:text-zinc-500">
                score {result.score.toFixed(3)}
              </span>
            </div>
            {result.title && (
              <p className="mt-2 text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                {result.title}
              </p>
            )}
            <p className="mt-1 line-clamp-3 text-sm whitespace-pre-wrap text-zinc-700 dark:text-zinc-300">
              <HighlightedSnippet text={result.snippet} highlights={result.highlights} />
            </p>
            {result.source === "ticket" && (
              <Link
                href={`/tickets/${result.id}`}
                className="mt-2 inline-block text-xs font-medium text-amber-600 hover:underline dark:text-amber-400"
              >
                View ticket →
              </Link>
            )}
          </li>
        ))}
      </ul>

      {!state.error && state.query && state.results.length === 0 && (
        <p className="mt-6 text-center text-sm text-zinc-500 dark:text-zinc-400">
          No results found.
        </p>
      )}
    </div>
  );
}
