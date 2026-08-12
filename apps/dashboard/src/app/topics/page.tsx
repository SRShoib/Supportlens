import { EmergingIssuesPanel } from "@/components/emerging-issues-panel";
import { TagIcon } from "@/components/icons";
import { TopicsOverTimeChart } from "@/components/topics-over-time-chart";
import { getEmergingIssues, getTopicVolume, listTopics } from "@/lib/api";

export const metadata = { title: "Topics — supportlens" };

export default async function TopicsPage() {
  const [topics, volume, emerging] = await Promise.all([
    listTopics(),
    getTopicVolume(),
    getEmergingIssues(),
  ]);

  // topic_key=-1 is HDBSCAN's outlier cluster (never counted toward SPEC
  // M7's "≥ 30 coherent topics") -- still rendered in the catalog list
  // below for transparency, just excluded from this header count.
  const discoveredTopics = topics.filter((t) => t.topic_key !== -1);

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <div className="animate-fade-in-up">
        <div className="flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-100 text-violet-600 dark:bg-violet-500/15 dark:text-violet-400">
            <TagIcon className="h-5 w-5" />
          </span>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            Topics
          </h1>
        </div>
        <p className="mt-3 text-sm text-zinc-500 dark:text-zinc-400">
          {discoveredTopics.length} topic{discoveredTopics.length === 1 ? "" : "s"} discovered from
          the real-ticket corpus
        </p>
      </div>

      <section className="animate-fade-in-up mt-8" style={{ animationDelay: "60ms" }}>
        <h2 className="flex items-center gap-2 text-sm font-semibold text-zinc-900 dark:text-zinc-50">
          <span className="h-3.5 w-1 rounded-full bg-violet-500" />
          Volume over time
        </h2>
        <div className="surface-card mt-3 p-4">
          <TopicsOverTimeChart weeks={volume.weeks} series={volume.series} />
        </div>
      </section>

      <section className="animate-fade-in-up mt-8" style={{ animationDelay: "120ms" }}>
        <EmergingIssuesPanel issues={emerging} />
      </section>

      <section className="animate-fade-in-up mt-8" style={{ animationDelay: "180ms" }}>
        <h2 className="flex items-center gap-2 text-sm font-semibold text-zinc-900 dark:text-zinc-50">
          <span className="h-3.5 w-1 rounded-full bg-violet-500" />
          All topics
        </h2>
        <ul className="surface-card mt-3 divide-y divide-zinc-200/70 dark:divide-white/10">
          {topics.map((topic) => (
            <li
              key={topic.id}
              className="flex items-center justify-between gap-4 px-4 py-3.5 transition-colors duration-150 hover:bg-violet-50/60 dark:hover:bg-violet-500/5"
            >
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-zinc-900 dark:text-zinc-50">
                  {topic.topic_key === -1 ? "Outliers (uncategorized)" : topic.label}
                </p>
                <p className="mt-0.5 truncate text-xs text-zinc-500 dark:text-zinc-400">
                  {topic.keywords.join(", ") || "—"}
                </p>
              </div>
              <span className="shrink-0 rounded-full bg-violet-50 px-2.5 py-1 text-xs font-medium text-violet-700 dark:bg-violet-500/10 dark:text-violet-300">
                {topic.size} ticket{topic.size === 1 ? "" : "s"}
              </span>
            </li>
          ))}
          {topics.length === 0 && (
            <li className="py-10 text-center text-sm text-zinc-500 dark:text-zinc-400">
              No topics yet. Run <code className="font-mono">make assign-topics</code> against the
              API first.
            </li>
          )}
        </ul>
      </section>
    </div>
  );
}
