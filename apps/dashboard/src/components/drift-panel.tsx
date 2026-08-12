// SPEC M9's drift monitoring: "embedding-distribution distance... +
// prediction-distribution shift (PSI) between a reference week and the live
// window, with a simulated drift scenario... watch the alarms fire".
// Status-palette job (state: stable/watch/alarm), reserved and never reused
// for series identity elsewhere on this page -- red matches
// emerging-issues-panel.tsx's existing "something needs attention" alarm
// color exactly (same status class, same visual language), amber is the
// intermediate "watch" band PSI has and embedding distance doesn't, emerald
// matches sentiment-sparkline.tsx's "positive" convention for "stable".
// Every badge ships an icon + text label, never color alone.

import type { Drift, DriftStatus, EmbeddingDriftMetrics, PredictionDriftMetrics } from "@/lib/api";

const STATUS_BADGE_CLASS: Record<DriftStatus, string> = {
  stable: "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-400",
  watch: "bg-amber-100 text-amber-700 dark:bg-amber-500/15 dark:text-amber-400",
  alarm: "bg-red-100 text-red-700 dark:bg-red-500/15 dark:text-red-400",
};
const STATUS_ICON: Record<DriftStatus, string> = { stable: "●", watch: "▲", alarm: "▲" };

function StatusBadge({ status }: { status: DriftStatus }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_BADGE_CLASS[status]} ${
        status === "alarm" ? "animate-pulse-ring" : ""
      }`}
    >
      <span aria-hidden>{STATUS_ICON[status]}</span>
      {status}
    </span>
  );
}

function EmbeddingCell({ metrics }: { metrics: EmbeddingDriftMetrics | null }) {
  if (!metrics) {
    return <p className="text-xs text-zinc-400 dark:text-zinc-500">not computed yet</p>;
  }
  const status: DriftStatus = metrics.is_alarm ? "alarm" : "stable";
  return (
    <div>
      <p className="text-lg font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
        {metrics.cosine_shift.toFixed(4)}
      </p>
      <p className="text-xs text-zinc-400 dark:text-zinc-500">
        cosine shift (alarm &gt; {metrics.threshold}) · n={metrics.reference_n} vs {metrics.live_n}
      </p>
      <div className="mt-1.5">
        <StatusBadge status={status} />
      </div>
    </div>
  );
}

function PredictionCell({ metrics }: { metrics: PredictionDriftMetrics | null }) {
  if (!metrics) {
    return <p className="text-xs text-zinc-400 dark:text-zinc-500">not computed yet</p>;
  }
  return (
    <div>
      <p className="text-lg font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
        PSI {metrics.psi.toFixed(4)}
      </p>
      <p className="text-xs text-zinc-400 dark:text-zinc-500">
        urgency label · n={metrics.reference_n} vs {metrics.live_n}
      </p>
      <div className="mt-1.5">
        <StatusBadge status={metrics.status} />
      </div>
    </div>
  );
}

interface DriftPanelProps {
  drift: Drift;
}

export function DriftPanel({ drift }: DriftPanelProps) {
  const realEmbedding = drift.real.embedding?.metrics as unknown as EmbeddingDriftMetrics | undefined;
  const realPrediction = drift.real.prediction?.metrics as unknown as
    | PredictionDriftMetrics
    | undefined;
  const simEmbedding = drift.simulated.embedding?.metrics as unknown as
    | EmbeddingDriftMetrics
    | undefined;
  const simPrediction = drift.simulated.prediction?.metrics as unknown as
    | PredictionDriftMetrics
    | undefined;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      <div className="surface-card surface-card-interactive p-4">
        <h3 className="text-xs font-semibold tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
          Real (reference week vs. live window)
        </h3>
        <p className="mt-0.5 text-xs text-zinc-400 dark:text-zinc-500">
          Normal week-to-week traffic -- expected: no alarm.
        </p>
        <div className="mt-3 space-y-3">
          <EmbeddingCell metrics={realEmbedding ?? null} />
          <PredictionCell metrics={realPrediction ?? null} />
        </div>
      </div>
      <div className="surface-card surface-card-interactive p-4">
        <h3 className="text-xs font-semibold tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
          Simulated (reference week vs. Bitext-injected slice)
        </h3>
        <p className="mt-0.5 text-xs text-zinc-400 dark:text-zinc-500">
          A topically-different injected slice (SPEC M9) -- expected: alarm fires.
        </p>
        <div className="mt-3 space-y-3">
          <EmbeddingCell metrics={simEmbedding ?? null} />
          <PredictionCell metrics={simPrediction ?? null} />
        </div>
      </div>
    </div>
  );
}
