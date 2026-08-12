import { SearchIcon } from "@/components/icons";
import { SearchForm } from "@/components/search-form";

export const metadata = { title: "Search — supportlens" };

export default function SearchPage() {
  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <div className="animate-fade-in-up">
        <div className="flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-100 text-amber-600 dark:bg-amber-500/15 dark:text-amber-400">
            <SearchIcon className="h-5 w-5" />
          </span>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            Search
          </h1>
        </div>
        <p className="mt-3 text-sm text-zinc-500 dark:text-zinc-400">
          Semantic search over resolved tickets and knowledge-base articles.
        </p>
      </div>
      <div className="animate-fade-in-up mt-8" style={{ animationDelay: "80ms" }}>
        <SearchForm />
      </div>
    </div>
  );
}
