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
          className="min-w-64 flex-1 rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-zinc-500 focus:outline-none dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-50"
        />
        <label className="flex items-center gap-2 text-sm text-zinc-600 dark:text-zinc-400">
          <input
            type="checkbox"
            name="rerank"
            defaultChecked
            className="rounded border-zinc-300 dark:border-zinc-700"
          />
          Rerank
        </label>
        <button
          type="submit"
          disabled={pending}
          className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-zinc-50 dark:text-zinc-900"
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
        {state.results.map((result) => (
          <li
            key={`${result.source}-${result.id}`}
            className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-xs font-medium text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
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
                className="mt-2 inline-block text-xs text-zinc-500 hover:underline dark:text-zinc-400"
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
