"use server";

import { search, type SearchResult } from "@/lib/api";

export interface SearchActionState {
  results: SearchResult[];
  reranked: boolean;
  query: string;
  error: string | null;
}

export async function searchAction(
  _prevState: SearchActionState,
  formData: FormData,
): Promise<SearchActionState> {
  const query = String(formData.get("query") ?? "").trim();
  const rerank = formData.get("rerank") === "on";

  if (!query) {
    return { results: [], reranked: rerank, query: "", error: "Enter a search query." };
  }

  try {
    const response = await search(query, { rerank });
    return { results: response.results, reranked: response.reranked, query, error: null };
  } catch (error) {
    return {
      results: [],
      reranked: rerank,
      query,
      error: error instanceof Error ? error.message : "Search failed.",
    };
  }
}
