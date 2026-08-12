// SPEC M9: "/metrics area in the dashboard: per-model eval runs over time,
// confusion matrices, per-class F1, retrieval metrics, latency
// percentiles" + drift monitoring. Every section reads straight from
// GET /eval-runs / GET /drift -- both pure Postgres reads (see
// apps/api/routers/eval_runs.py, drift.py) -- so nothing on this page is
// computed client- or server-side beyond picking which already-persisted
// run to feature (SPEC's accept criterion: "all metrics render from
// Postgres eval runs (no hardcoded numbers)").

import type { ReactNode } from "react";

import { ConfusionMatrix } from "@/components/confusion-matrix";
import { DriftPanel } from "@/components/drift-panel";
import { EvalRunsTable } from "@/components/eval-runs-table";
import { ActivityIcon } from "@/components/icons";
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

function SectionHeading({ children }: { children: ReactNode }) {
  return (
    <h2 className="flex items-center gap-2 text-lg font-semibold text-zinc-900 dark:text-zinc-50">
      <span className="h-4 w-1 rounded-full bg-gradient-to-b from-cyan-500 to-blue-600" />
      {children}
    </h2>
  );
}

function ClassificationSection({ task, runs }: { task: string; runs: EvalRun[] }) {
  if (runs.length === 0) {
    return (
      <div className="surface-card p-4">
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
    <div className="surface-card surface-card-interactive p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 className="text-sm font-semibold text-zinc-900 capitalize dark:text-zinc-50">{task}</h3>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          best: <span className="font-mono">{featured.model_version}</span> · macro-F1{" "}
          <span className="font-semibold text-indigo-600 dark:text-indigo-400">
            {featuredMetrics.macro_f1.toFixed(4)}
          </span>
        </p>
      </div>
      <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div>
          <h4 className="text-xs font-semibold tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
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
          <h4 className="text-xs font-semibold tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
            Per-class F1
          </h4>
          <div className="mt-2">
            <PerClassF1Bars perClassF1={featuredMetrics.per_class_f1} />
          </div>
        </div>
      </div>
      {ranked.length > 1 && (
        <div className="mt-4 border-t border-zinc-200/70 pt-4 dark:border-white/10">
          <h4 className="text-xs font-semibold tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
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
    <div className="surface-card p-4">
      {featured ? (
        <>
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              Gold set (span-level F1)
            </h3>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              best: <span className="font-mono">{featured.model_version}</span> · micro-F1{" "}
              <span className="font-semibold text-indigo-600 dark:text-indigo-400">
                {(featured.metrics as unknown as SpanMetrics).micro_f1.toFixed(4)}
              </span>
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
        <div className="mt-4 border-t border-zinc-200/70 pt-4 dark:border-white/10">
          <h4 className="text-xs font-semibold tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
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
    <div className="mx-auto max-w-7xl px-6 py-10">
      <div className="animate-fade-in-up">
        <div className="flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-100 text-cyan-600 dark:bg-cyan-500/15 dark:text-cyan-400">
            <ActivityIcon className="h-5 w-5" />
          </span>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            Metrics
          </h1>
        </div>
        <p className="mt-3 text-sm text-zinc-500 dark:text-zinc-400">
          Every number on this page renders from a persisted{" "}
          <code className="font-mono">eval_runs</code> row -- nothing here is hardcoded (SPEC M9).
        </p>
      </div>

      <section className="animate-fade-in-up mt-8" style={{ animationDelay: "40ms" }}>
        <SectionHeading>Classification</SectionHeading>
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

      <section className="animate-fade-in-up mt-10" style={{ animationDelay: "80ms" }}>
        <SectionHeading>Entities</SectionHeading>
        <div className="mt-3">
          <EntitiesSection runs={(runsByTask.get("entities") ?? []).filter(isAccuracyRun)} />
        </div>
      </section>

      <section className="animate-fade-in-up mt-10" style={{ animationDelay: "120ms" }}>
        <SectionHeading>Retrieval</SectionHeading>
        <div className="surface-card mt-3 p-4">
          <RetrievalPanel runs={runsByTask.get("retrieval") ?? []} />
        </div>
      </section>

      <section className="animate-fade-in-up mt-10" style={{ animationDelay: "160ms" }}>
        <SectionHeading>Latency</SectionHeading>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          CPU, single request, per SPEC §3&apos;s per-task budgets.
        </p>
        <div className="surface-card mt-3 p-4">
          <LatencyTable runs={allRuns} />
        </div>
      </section>

      <section className="animate-fade-in-up mt-10" style={{ animationDelay: "200ms" }}>
        <SectionHeading>Drift monitoring</SectionHeading>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Embedding-distribution distance + prediction-distribution shift (PSI), reference week vs.
          live window, real traffic vs. a simulated topically-different injection.
        </p>
        <div className="mt-3">
          <DriftPanel drift={drift} />
        </div>
      </section>

      <section className="animate-fade-in-up mt-10" style={{ animationDelay: "240ms" }}>
        <SectionHeading>Summarization</SectionHeading>
        <div className="mt-3 space-y-4">
          <div className="surface-card p-4">
            <h3 className="text-xs font-semibold tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
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
          <div className="surface-card p-4">
            <h3 className="text-xs font-semibold tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
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

      <section className="animate-fade-in-up mt-10 mb-10" style={{ animationDelay: "280ms" }}>
        <SectionHeading>Topics</SectionHeading>
        <div className="surface-card mt-3 p-4">
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
