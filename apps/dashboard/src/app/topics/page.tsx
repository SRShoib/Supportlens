import { EmergingIssuesPanel } from "@/components/emerging-issues-panel";
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
    <div className="mx-auto max-w-4xl px-6 py-10">
      <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50">Topics</h1>
      <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
        {discoveredTopics.length} topic{discoveredTopics.length === 1 ? "" : "s"} discovered from the
        real-ticket corpus
      </p>

      <section className="mt-8">
        <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">Volume over time</h2>
        <div className="mt-3 rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
          <TopicsOverTimeChart weeks={volume.weeks} series={volume.series} />
        </div>
      </section>

      <section className="mt-8">
        <EmergingIssuesPanel issues={emerging} />
      </section>

      <section className="mt-8">
        <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">All topics</h2>
        <ul className="mt-3 divide-y divide-zinc-200 dark:divide-zinc-800">
          {topics.map((topic) => (
            <li key={topic.id} className="flex items-center justify-between gap-4 py-3">
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-zinc-900 dark:text-zinc-50">
                  {topic.topic_key === -1 ? "Outliers (uncategorized)" : topic.label}
                </p>
                <p className="mt-0.5 truncate text-xs text-zinc-500 dark:text-zinc-400">
                  {topic.keywords.join(", ") || "—"}
                </p>
              </div>
              <span className="shrink-0 rounded-full bg-zinc-100 px-2 py-1 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
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
