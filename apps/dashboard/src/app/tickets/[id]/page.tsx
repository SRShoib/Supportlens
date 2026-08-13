import Link from "next/link";
import { notFound } from "next/navigation";

import { EntityHighlightedText } from "@/components/entity-highlighted-text";
import { SentimentSparkline } from "@/components/sentiment-sparkline";
import { SuggestedReplyPanel } from "@/components/suggested-reply-panel";
import { ThreadSummary } from "@/components/thread-summary";
import {
  type EntityResult,
  getSentimentTrajectory,
  getThreadSummary,
  getTicket,
  type SentimentTrajectoryPayload,
  predictEntities,
} from "@/lib/api";
import { formatDateTime } from "@/lib/format";

// Best-effort: the routing file or model export may not have been generated
// yet (`make eval-ner`), and a ticket page is a read of already-ingested
// data -- it shouldn't 500 just because entity extraction is unavailable.
// Falling back to plain message text degrades gracefully instead.
async function getMessageEntities(texts: string[]): Promise<EntityResult[] | null> {
  try {
    return await predictEntities(texts);
  } catch {
    return null;
  }
}

// Same best-effort contract: scripts/compute_sentiment_trajectories.py may
// simply not have run yet against this ticket -- getSentimentTrajectory
// already returns null for that case, but the API itself could still be
// unreachable, so this stays defensive the same way getMessageEntities is.
async function getTrajectory(ticketId: string): Promise<SentimentTrajectoryPayload | null> {
  try {
    return await getSentimentTrajectory(ticketId);
  } catch {
    return null;
  }
}

// Same best-effort contract: scripts/compute_thread_summaries.py skips
// single-message tickets entirely and may simply not have run yet against
// this one -- getThreadSummary already returns null for that case, but the
// API itself could still be unreachable, so this stays defensive too.
async function getSummary(ticketId: string): Promise<string | null> {
  try {
    return await getThreadSummary(ticketId);
  } catch {
    return null;
  }
}

export default async function TicketDetailPage({ params }: PageProps<"/tickets/[id]">) {
  const { id } = await params;
  const ticket = await getTicket(id);
  if (!ticket) {
    notFound();
  }

  const [entityResults, trajectory, summary] = await Promise.all([
    getMessageEntities(ticket.messages.map((message) => message.text_clean)),
    getTrajectory(ticket.id),
    getSummary(ticket.id),
  ]);

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <Link
        href="/tickets"
        className="group inline-flex items-center gap-1 text-sm text-zinc-500 transition-colors duration-200 hover:text-blue-600 dark:text-zinc-400 dark:hover:text-blue-400"
      >
        <span className="transition-transform duration-200 group-hover:-translate-x-0.5">←</span> All
        tickets
      </Link>

      <header className="animate-fade-in-up mt-4">
        <h1 className="text-xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          Ticket {ticket.external_id}
        </h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          {ticket.source} · {ticket.channel}
          {ticket.brand ? ` · ${ticket.brand}` : ""}
          {ticket.created_at ? ` · ${formatDateTime(ticket.created_at)}` : ""}
        </p>
        {summary && <ThreadSummary summary={summary} />}
        {trajectory && (
          <div className="mt-3">
            <SentimentSparkline
              sequence={trajectory.sequence}
              scores={trajectory.scores}
              resolutionQuality={trajectory.resolution_quality}
            />
          </div>
        )}
      </header>

      <ol className="mt-8 space-y-4">
        {ticket.messages.map((message, index) => (
          <li
            key={message.id}
            className={`animate-fade-in-up rounded-xl border p-4 shadow-sm transition-shadow duration-200 hover:shadow-md ${
              message.author_role === "customer"
                ? "border-zinc-200/80 bg-white/90 dark:border-white/10 dark:bg-zinc-900/60"
                : "border-blue-200/70 bg-blue-50/80 dark:border-blue-400/20 dark:bg-blue-500/10"
            }`}
            style={{ animationDelay: `${Math.min(index, 12) * 40}ms` }}
          >
            <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400">
              <span
                className={`font-semibold tracking-wide uppercase ${
                  message.author_role === "customer"
                    ? "text-zinc-500 dark:text-zinc-400"
                    : "text-blue-600 dark:text-blue-300"
                }`}
              >
                {message.author_role}
              </span>
              {message.sent_at && <span>{formatDateTime(message.sent_at)}</span>}
            </div>
            <p className="mt-2 text-sm whitespace-pre-wrap text-zinc-800 dark:text-zinc-100">
              {entityResults?.[index] ? (
                <EntityHighlightedText
                  text={message.text_clean}
                  entities={entityResults[index].entities}
                />
              ) : (
                message.text_clean
              )}
            </p>
          </li>
        ))}
        {ticket.messages.length === 0 && (
          <li className="surface-card py-10 text-center text-sm text-zinc-500 dark:text-zinc-400">
            This ticket has no messages.
          </li>
        )}
      </ol>

      <SuggestedReplyPanel ticketId={ticket.id} />
    </div>
  );
}
