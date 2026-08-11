import Link from "next/link";
import { notFound } from "next/navigation";

import { getTicket } from "@/lib/api";
import { formatDateTime } from "@/lib/format";

export default async function TicketDetailPage({ params }: PageProps<"/tickets/[id]">) {
  const { id } = await params;
  const ticket = await getTicket(id);
  if (!ticket) {
    notFound();
  }

  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <Link href="/tickets" className="text-sm text-zinc-500 hover:underline dark:text-zinc-400">
        ← All tickets
      </Link>

      <header className="mt-4">
        <h1 className="text-xl font-semibold text-zinc-900 dark:text-zinc-50">
          Ticket {ticket.external_id}
        </h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          {ticket.source} · {ticket.channel}
          {ticket.brand ? ` · ${ticket.brand}` : ""}
          {ticket.created_at ? ` · ${formatDateTime(ticket.created_at)}` : ""}
        </p>
      </header>

      <ol className="mt-8 space-y-4">
        {ticket.messages.map((message) => (
          <li
            key={message.id}
            className={`rounded-lg border p-4 ${
              message.author_role === "customer"
                ? "border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950"
                : "border-blue-100 bg-blue-50 dark:border-blue-900 dark:bg-blue-950/30"
            }`}
          >
            <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400">
              <span className="font-medium tracking-wide uppercase">{message.author_role}</span>
              {message.sent_at && <span>{formatDateTime(message.sent_at)}</span>}
            </div>
            <p className="mt-2 text-sm whitespace-pre-wrap text-zinc-800 dark:text-zinc-100">
              {message.text_clean}
            </p>
          </li>
        ))}
        {ticket.messages.length === 0 && (
          <li className="py-10 text-center text-sm text-zinc-500 dark:text-zinc-400">
            This ticket has no messages.
          </li>
        )}
      </ol>
    </div>
  );
}
