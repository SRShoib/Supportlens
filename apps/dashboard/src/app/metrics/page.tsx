// SPEC M9: "/metrics area in the dashboard: per-model eval runs over time,
// confusion matrices, per-class F1, retrieval metrics, latency
// percentiles" + drift monitoring. Every section reads straight from
// GET /eval-runs / GET /drift -- both pure Postgres reads (see
// apps/api/routers/eval_runs.py, drift.py) -- so nothing on this page is
// computed client- or server-side beyond picking which already-persisted
// run to feature (SPEC's accept criterion: "all metrics render from
// Postgres eval runs (no hardcoded numbers)").

import { ConfusionMatrix } from "@/components/confusion-matrix";
import { DriftPanel } from "@/components/drift-panel";
import { EvalRunsTable } from "@/components/eval-runs-table";
import { LatencyTable } from "@/components/latency-table";
import { PerClassF1Bars } from "@/components/per-class-f1-bars";
import { RetrievalPanel } from "@/components/retrieval-panel";
import { SpanMetricsTable } from "@/components/span-metrics-table";
import type {
  ClassificationMetrics,
  CoherenceMetrics,
  EvalRun,
  LLMJudgeMetrics,
  SpanMetrics,
  SummarizationMetrics,
} from "@/lib/api";
import { getDrift, listEvalRuns } from "@/lib/api";

export const metadata = { title: "Metrics — supportlens" };

const CLASSIFICATION_TASKS = ["intent", "urgency", "sentiment", "emotion"] as const;

function isAccuracyRun(run: EvalRun): boolean {
  return run.split !== "latency";
}

function newestFirst(runs: EvalRun[]): EvalRun[] {
  return [...runs].sort((a, b) => b.started_at.localeCompare(a.started_at));
}

function ClassificationSection({ task, runs }: { task: string; runs: EvalRun[] }) {
  if (runs.length === 0) {
    return (
      <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
        <h3 className="text-sm font-semibold text-zinc-900 capitalize dark:text-zinc-50">{task}</h3>
        <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">No eval runs yet.</p>
      </div>
    );
  }

  const ranked = [...runs].sort(
    (a, b) =>
      (b.metrics as unknown as ClassificationMetrics).macro_f1 -
      (a.metrics as unknown as ClassificationMetrics).macro_f1,
  );
  const featured = ranked[0];
  const featuredMetrics = featured.metrics as unknown as ClassificationMetrics;

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 className="text-sm font-semibold text-zinc-900 capitalize dark:text-zinc-50">{task}</h3>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          best: <span className="font-mono">{featured.model_version}</span> · macro-F1{" "}
          {featuredMetrics.macro_f1.toFixed(4)}
        </p>
      </div>
      <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div>
          <h4 className="text-xs font-semibold text-zinc-500 dark:text-zinc-400">
            Confusion matrix
          </h4>
          <div className="mt-2">
            <ConfusionMatrix
              labels={featuredMetrics.labels}
              matrix={featuredMetrics.confusion_matrix}
            />
          </div>
        </div>
        <div>
          <h4 className="text-xs font-semibold text-zinc-500 dark:text-zinc-400">Per-class F1</h4>
          <div className="mt-2">
            <PerClassF1Bars perClassF1={featuredMetrics.per_class_f1} />
          </div>
        </div>
      </div>
      {ranked.length > 1 && (
        <div className="mt-4">
          <h4 className="text-xs font-semibold text-zinc-500 dark:text-zinc-400">
            All eval runs
          </h4>
          <div className="mt-2">
            <EvalRunsTable
              runs={ranked}
              columns={[
                {
                  header: "macro-F1",
                  render: (r) => (r.metrics as unknown as ClassificationMetrics).macro_f1.toFixed(4),
                },
              ]}
              emptyMessage="No eval runs yet."
            />
          </div>
        </div>
      )}
    </div>
  );
}

function EntitiesSection({ runs }: { runs: EvalRun[] }) {
  const goldRuns = runs.filter((r) => r.split === "gold");
  const featured = [...goldRuns].sort(
    (a, b) =>
      (b.metrics as unknown as SpanMetrics).micro_f1 - (a.metrics as unknown as SpanMetrics).micro_f1,
  )[0];

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
      {featured ? (
        <>
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              Gold set (span-level F1)
            </h3>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              best: <span className="font-mono">{featured.model_version}</span> · micro-F1{" "}
              {(featured.metrics as unknown as SpanMetrics).micro_f1.toFixed(4)}
            </p>
          </div>
          <div className="mt-3">
            <SpanMetricsTable perType={(featured.metrics as unknown as SpanMetrics).per_type} />
          </div>
        </>
      ) : (
        <p className="text-sm text-zinc-500 dark:text-zinc-400">No gold-set eval runs yet.</p>
      )}
      {runs.length > 0 && (
        <div className="mt-4">
          <h4 className="text-xs font-semibold text-zinc-500 dark:text-zinc-400">
            All eval runs (every split)
          </h4>
          <div className="mt-2">
            <EvalRunsTable
              runs={newestFirst(runs)}
              columns={[
                {
                  header: "micro-F1",
                  render: (r) => (r.metrics as unknown as SpanMetrics).micro_f1.toFixed(4),
                },
              ]}
              emptyMessage="No eval runs yet."
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default async function MetricsPage() {
  const [allRuns, drift] = await Promise.all([listEvalRuns({ limit: 500 }), getDrift()]);

  const runsByTask = new Map<string, EvalRun[]>();
  for (const run of allRuns) {
    const list = runsByTask.get(run.task) ?? [];
    list.push(run);
    runsByTask.set(run.task, list);
  }

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50">Metrics</h1>
      <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
        Every number on this page renders from a persisted{" "}
        <code className="font-mono">eval_runs</code> row -- nothing here is hardcoded (SPEC M9).
      </p>

      <section className="mt-8">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">Classification</h2>
        <div className="mt-3 space-y-4">
          {CLASSIFICATION_TASKS.map((task) => (
            <ClassificationSection
              key={task}
              task={task}
              runs={(runsByTask.get(task) ?? []).filter(isAccuracyRun)}
            />
          ))}
        </div>
      </section>

      <section className="mt-10">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">Entities</h2>
        <div className="mt-3">
          <EntitiesSection runs={(runsByTask.get("entities") ?? []).filter(isAccuracyRun)} />
        </div>
      </section>

      <section className="mt-10">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">Retrieval</h2>
        <div className="mt-3 rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
          <RetrievalPanel runs={runsByTask.get("retrieval") ?? []} />
        </div>
      </section>

      <section className="mt-10">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">Latency</h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          CPU, single request, per SPEC §3&apos;s per-task budgets.
        </p>
        <div className="mt-3 rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
          <LatencyTable runs={allRuns} />
        </div>
      </section>

      <section className="mt-10">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">Drift monitoring</h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Embedding-distribution distance + prediction-distribution shift (PSI), reference week vs.
          live window, real traffic vs. a simulated topically-different injection.
        </p>
        <div className="mt-3">
          <DriftPanel drift={drift} />
        </div>
      </section>

      <section className="mt-10">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">Summarization</h2>
        <div className="mt-3 space-y-4">
          <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
            <h3 className="text-xs font-semibold text-zinc-500 dark:text-zinc-400">
              ROUGE (per dataset test split)
            </h3>
            <div className="mt-2">
              <EvalRunsTable
                runs={newestFirst((runsByTask.get("thread_summary") ?? []).filter(isAccuracyRun))}
                columns={[
                  {
                    header: "ROUGE-1",
                    render: (r) => (r.metrics as unknown as SummarizationMetrics).rouge1.toFixed(4),
                  },
                  {
                    header: "ROUGE-2",
                    render: (r) => (r.metrics as unknown as SummarizationMetrics).rouge2.toFixed(4),
                  },
                  {
                    header: "ROUGE-L",
                    render: (r) => (r.metrics as unknown as SummarizationMetrics).rougeL.toFixed(4),
                  },
                ]}
                emptyMessage="No summarization eval runs yet."
              />
            </div>
          </div>
          <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
            <h3 className="text-xs font-semibold text-zinc-500 dark:text-zinc-400">
              LLM-judge (real supportlens tickets, 1-5 rubric)
            </h3>
            <div className="mt-2">
              <EvalRunsTable
                runs={newestFirst(runsByTask.get("thread_summary_judge") ?? [])}
                columns={[
                  {
                    header: "Faithfulness",
                    render: (r) =>
                      (r.metrics as unknown as LLMJudgeMetrics).mean_faithfulness.toFixed(2),
                  },
                  {
                    header: "Coverage",
                    render: (r) => (r.metrics as unknown as LLMJudgeMetrics).mean_coverage.toFixed(2),
                  },
                ]}
                emptyMessage="No LLM-judge eval runs yet."
              />
            </div>
          </div>
        </div>
      </section>

      <section className="mt-10 mb-10">
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">Topics</h2>
        <div className="mt-3 rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
          <EvalRunsTable
            runs={newestFirst(runsByTask.get("topics") ?? [])}
            columns={[
              {
                header: "Mean NPMI",
                render: (r) => (r.metrics as unknown as CoherenceMetrics).mean_npmi.toFixed(4),
              },
              {
                header: "# topics",
                render: (r) => String((r.metrics as unknown as CoherenceMetrics).n_topics),
              },
            ]}
            emptyMessage="No topic eval runs yet."
          />
        </div>
      </section>
    </div>
  );
}
