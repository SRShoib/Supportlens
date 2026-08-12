// Landing dashboard. Every figure here is read straight from the same
// sources the rest of the app already trusts -- listEvalRuns/getDrift back
// /metrics (SPEC M9: "all metrics render from Postgres eval runs"),
// listTopics/getEmergingIssues back /topics, listTickets backs /tickets --
// this page just re-surfaces the best/latest of each as a colorful
// headline, it never computes or hardcodes a number of its own.

import type { ComponentType } from "react";

import Link from "next/link";

import {
  AlertTriangleIcon,
  ArrowRightIcon,
  GaugeIcon,
  LayersIcon,
  TargetIcon,
  TrendingUpIcon,
} from "@/components/icons";
import { COLOR_STYLES, NAV_ITEMS } from "@/lib/nav";
import {
  type ClassificationMetrics,
  type CoherenceMetrics,
  type DriftStatus,
  type EvalRun,
  type PredictionDriftMetrics,
  type RetrievalRunMetrics,
  getDrift,
  getEmergingIssues,
  listEvalRuns,
  listTickets,
  listTopics,
} from "@/lib/api";
import { ticketSourceBadgeClass } from "@/lib/format";

export const metadata = { title: "supportlens — Overview" };

function bestMacroF1(runs: EvalRun[], task: string): number | null {
  const rows = runs.filter((r) => r.task === task && r.split !== "latency");
  if (rows.length === 0) return null;
  return Math.max(...rows.map((r) => (r.metrics as unknown as ClassificationMetrics).macro_f1));
}

function bestHitRate(runs: EvalRun[]): number | null {
  const rows = runs.filter((r) => r.task === "retrieval");
  if (rows.length === 0) return null;
  return Math.max(...rows.map((r) => (r.metrics as unknown as RetrievalRunMetrics).hit_rate_at_k));
}

function latestTopicCoherence(runs: EvalRun[]): CoherenceMetrics | null {
  const rows = [...runs]
    .filter((r) => r.task === "topics")
    .sort((a, b) => b.started_at.localeCompare(a.started_at));
  return rows[0] ? (rows[0].metrics as unknown as CoherenceMetrics) : null;
}

function driftStatusOf(run: EvalRun | null): DriftStatus | null {
  return (run?.metrics as unknown as PredictionDriftMetrics | undefined)?.status ?? null;
}

function formatPercent(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

type TileColor = "blue" | "rose" | "violet" | "amber" | "red" | "cyan";

const TILE_STYLES: Record<TileColor, { iconBg: string; iconText: string }> = {
  blue: { iconBg: "bg-blue-100 dark:bg-blue-500/15", iconText: "text-blue-600 dark:text-blue-400" },
  rose: { iconBg: "bg-rose-100 dark:bg-rose-500/15", iconText: "text-rose-600 dark:text-rose-400" },
  violet: {
    iconBg: "bg-violet-100 dark:bg-violet-500/15",
    iconText: "text-violet-600 dark:text-violet-400",
  },
  amber: {
    iconBg: "bg-amber-100 dark:bg-amber-500/15",
    iconText: "text-amber-600 dark:text-amber-400",
  },
  red: { iconBg: "bg-red-100 dark:bg-red-500/15", iconText: "text-red-600 dark:text-red-400" },
  cyan: { iconBg: "bg-cyan-100 dark:bg-cyan-500/15", iconText: "text-cyan-600 dark:text-cyan-400" },
};

const DRIFT_BADGE_CLASS: Record<DriftStatus, string> = {
  stable: "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-400",
  watch: "bg-amber-100 text-amber-700 dark:bg-amber-500/15 dark:text-amber-400",
  alarm: "bg-red-100 text-red-700 dark:bg-red-500/15 dark:text-red-400",
};

function StatTile({
  href,
  icon: Icon,
  color,
  label,
  value,
  hint,
  delayMs,
}: {
  href: string;
  icon: ComponentType<{ className?: string }>;
  color: TileColor;
  label: string;
  value: string;
  hint?: string;
  delayMs: number;
}) {
  const styles = TILE_STYLES[color];
  return (
    <Link
      href={href}
      className="surface-card surface-card-interactive animate-fade-in-up group flex flex-col gap-3 p-5"
      style={{ animationDelay: `${delayMs}ms` }}
    >
      <span
        className={`flex h-10 w-10 items-center justify-center rounded-xl transition-transform duration-150 group-hover:scale-105 ${styles.iconBg} ${styles.iconText}`}
      >
        <Icon className="h-5 w-5" />
      </span>
      <div>
        <p className="text-2xl font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
          {value}
        </p>
        <p className="mt-0.5 text-xs font-medium text-zinc-500 dark:text-zinc-400">{label}</p>
      </div>
      {hint && <p className="text-xs text-zinc-400 dark:text-zinc-500">{hint}</p>}
    </Link>
  );
}

function DriftMiniBadge({ label, status }: { label: string; status: DriftStatus | null }) {
  return (
    <div className="flex items-center gap-2 rounded-xl border border-zinc-200/70 bg-white/70 px-3 py-2 dark:border-white/10 dark:bg-white/5">
      <span className="text-xs text-zinc-500 dark:text-zinc-400">{label}</span>
      <span
        className={`rounded-full px-2 py-0.5 text-xs font-medium ${
          status ? DRIFT_BADGE_CLASS[status] : "bg-zinc-100 text-zinc-500 dark:bg-white/10 dark:text-zinc-400"
        }`}
      >
        {status ?? "not computed"}
      </span>
    </div>
  );
}

export default async function OverviewPage() {
  const [allRuns, topics, emerging, drift, recentTickets] = await Promise.all([
    listEvalRuns({ limit: 500 }),
    listTopics(),
    getEmergingIssues(),
    getDrift(),
    listTickets({ limit: 6 }),
  ]);

  const discoveredTopics = topics.filter((t) => t.topic_key !== -1);
  const coherence = latestTopicCoherence(allRuns);

  return (
    <div className="mx-auto max-w-7xl px-6 py-10">
      <div className="animate-fade-in-up relative overflow-hidden rounded-3xl border border-zinc-200/70 bg-gradient-to-br from-indigo-500/10 via-violet-500/10 to-fuchsia-500/10 p-8 dark:border-white/10">
        <div className="pointer-events-none absolute -top-20 -right-20 h-56 w-56 rounded-full bg-gradient-to-br from-indigo-400/30 to-fuchsia-500/20 blur-3xl" />
        <p className="text-xs font-semibold tracking-wide text-indigo-600 uppercase dark:text-indigo-400">
          Customer support intelligence
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          Overview
        </h1>
        <p className="mt-2 max-w-xl text-sm text-zinc-600 dark:text-zinc-400">
          Classical baselines, fine-tuned transformers, and RAG watching the same ticket stream --
          every figure below reads from a persisted eval run or a live API call (SPEC M9), never
          hardcoded.
        </p>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
        <StatTile
          href="/metrics"
          icon={TargetIcon}
          color="blue"
          label="Intent macro-F1"
          value={formatPercent(bestMacroF1(allRuns, "intent"))}
          delayMs={0}
        />
        <StatTile
          href="/metrics"
          icon={AlertTriangleIcon}
          color="rose"
          label="Urgency macro-F1"
          value={formatPercent(bestMacroF1(allRuns, "urgency"))}
          delayMs={30}
        />
        <StatTile
          href="/metrics"
          icon={TrendingUpIcon}
          color="violet"
          label="Sentiment macro-F1"
          value={formatPercent(bestMacroF1(allRuns, "sentiment"))}
          delayMs={60}
        />
        <StatTile
          href="/topics"
          icon={LayersIcon}
          color="amber"
          label="Topics discovered"
          value={String(discoveredTopics.length)}
          hint={coherence ? `mean NPMI ${coherence.mean_npmi.toFixed(3)}` : undefined}
          delayMs={90}
        />
        <StatTile
          href="/topics"
          icon={AlertTriangleIcon}
          color="red"
          label="Emerging issues"
          value={String(emerging.length)}
          hint="z-score > 2 this window"
          delayMs={120}
        />
        <StatTile
          href="/search"
          icon={GaugeIcon}
          color="cyan"
          label="Retrieval hit-rate@5"
          value={formatPercent(bestHitRate(allRuns))}
          delayMs={150}
        />
      </div>

      <section className="animate-fade-in-up mt-6" style={{ animationDelay: "180ms" }}>
        <div className="surface-card flex flex-wrap items-center justify-between gap-4 p-5">
          <div>
            <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              Drift monitoring
            </h2>
            <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">
              Prediction-distribution shift, reference week vs. live window.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <DriftMiniBadge label="Real traffic" status={driftStatusOf(drift.real.prediction)} />
            <DriftMiniBadge
              label="Simulated spike"
              status={driftStatusOf(drift.simulated.prediction)}
            />
          </div>
        </div>
      </section>

      <section className="animate-fade-in-up mt-8" style={{ animationDelay: "220ms" }}>
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">Recent tickets</h2>
          <Link
            href="/tickets"
            className="inline-flex items-center gap-1 text-xs font-medium text-blue-600 transition-colors hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
          >
            All tickets <ArrowRightIcon className="h-3.5 w-3.5" />
          </Link>
        </div>
        <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {recentTickets.map((ticket, index) => (
            <Link
              key={ticket.id}
              href={`/tickets/${ticket.id}`}
              className="surface-card surface-card-interactive animate-fade-in-up flex items-center justify-between gap-3 p-4"
              style={{ animationDelay: `${260 + index * 30}ms` }}
            >
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-zinc-900 dark:text-zinc-50">
                  {ticket.messages[0]?.text_clean ?? "(no messages)"}
                </p>
                <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                  {ticket.source} · {ticket.messages.length} message
                  {ticket.messages.length === 1 ? "" : "s"}
                </p>
              </div>
              <span
                className={`max-w-[7rem] shrink-0 truncate rounded-full px-2.5 py-1 text-xs font-medium ${ticketSourceBadgeClass(ticket.source)}`}
                title={ticket.brand ?? ticket.source}
              >
                {ticket.brand ?? ticket.source}
              </span>
            </Link>
          ))}
          {recentTickets.length === 0 && (
            <p className="surface-card p-6 text-center text-sm text-zinc-500 dark:text-zinc-400 sm:col-span-2">
              No tickets ingested yet. Run <code className="font-mono">make seed</code>.
            </p>
          )}
        </div>
      </section>

      <section className="animate-fade-in-up mt-8 mb-10" style={{ animationDelay: "300ms" }}>
        <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">Explore</h2>
        <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {NAV_ITEMS.filter((item) => item.href !== "/").map((item) => {
            const styles = COLOR_STYLES[item.color];
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className="surface-card surface-card-interactive group flex flex-col gap-3 p-5"
              >
                <span
                  className={`flex h-10 w-10 items-center justify-center rounded-xl transition-transform duration-150 group-hover:scale-105 ${styles.iconBg} ${styles.iconText}`}
                >
                  <Icon className="h-5 w-5" />
                </span>
                <span className="flex items-center gap-1 text-sm font-medium text-zinc-900 dark:text-zinc-50">
                  {item.label}
                  <ArrowRightIcon className="h-3.5 w-3.5 text-zinc-400 transition-transform duration-150 group-hover:translate-x-0.5" />
                </span>
              </Link>
            );
          })}
        </div>
      </section>
    </div>
  );
}
