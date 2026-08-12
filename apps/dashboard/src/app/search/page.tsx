import { SearchForm } from "@/components/search-form";

export const metadata = { title: "Search — supportlens" };

export default function SearchPage() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50">Search</h1>
      <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
        Semantic search over resolved tickets and knowledge-base articles.
      </p>
      <div className="mt-8">
        <SearchForm />
      </div>
    </div>
  );
}
