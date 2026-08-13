import Link from "next/link";

import { InboxIcon } from "@/components/icons";
import { listTickets, type TicketSource } from "@/lib/api";
import { ticketSourceBadgeClass } from "@/lib/format";

export const metadata = { title: "Tickets — supportlens" };

function isTicketSource(value: string | undefined): value is TicketSource {
  return value === "bitext" || value === "twitter";
}

export default async function TicketsPage({ searchParams }: PageProps<"/tickets">) {
  const resolvedParams = await searchParams;
  const rawSource = resolvedParams.source;
  const sourceParam = Array.isArray(rawSource) ? rawSource[0] : rawSource;
  const source = isTicketSource(sourceParam) ? sourceParam : undefined;

  const tickets = await listTickets({ source, limit: 50 });

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <div className="animate-fade-in-up">
        <div className="flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-100 text-blue-600 dark:bg-blue-500/15 dark:text-blue-400">
            <InboxIcon className="h-5 w-5" />
          </span>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            Tickets
          </h1>
        </div>
        <p className="mt-3 text-sm text-zinc-500 dark:text-zinc-400">
          {tickets.length} ticket{tickets.length === 1 ? "" : "s"}
          {source ? ` · source: ${source}` : ""}
        </p>

        <nav className="mt-5 flex gap-2 text-sm">
          <FilterLink label="All" href="/tickets" active={!source} />
          <FilterLink label="Twitter" href="/tickets?source=twitter" active={source === "twitter"} />
          <FilterLink label="Bitext" href="/tickets?source=bitext" active={source === "bitext"} />
        </nav>
      </div>

      <ul className="mt-6 space-y-2">
        {tickets.map((ticket, index) => (
          <li
            key={ticket.id}
            className="animate-fade-in-up"
            style={{ animationDelay: `${Math.min(index, 12) * 30}ms` }}
          >
            <Link
              href={`/tickets/${ticket.id}`}
              className="surface-card surface-card-interactive flex items-center justify-between gap-4 px-4 py-4"
            >
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-zinc-900 dark:text-zinc-50">
                  {ticket.messages[0]?.text_clean ?? "(no messages)"}
                </p>
                <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                  {ticket.source} · {ticket.channel} · {ticket.messages.length} message
                  {ticket.messages.length === 1 ? "" : "s"}
                </p>
              </div>
              <span
                className={`max-w-[7rem] shrink-0 truncate rounded-full px-2.5 py-1 text-xs font-medium sm:max-w-[10rem] ${ticketSourceBadgeClass(ticket.source)}`}
                title={ticket.brand ?? ticket.source}
              >
                {ticket.brand ?? ticket.source}
              </span>
            </Link>
          </li>
        ))}
        {tickets.length === 0 && (
          <li className="surface-card py-10 text-center text-sm text-zinc-500 dark:text-zinc-400">
            No tickets found. Run <code className="font-mono">make seed</code> or{" "}
            <code className="font-mono">make ingest-twitter</code> against the API first.
          </li>
        )}
      </ul>
    </div>
  );
}

function FilterLink({
  label,
  href,
  active,
}: {
  label: string;
  href: string;
  active: boolean;
}) {
  return (
    <Link
      href={href}
      className={`rounded-full px-3.5 py-1.5 font-medium transition-all duration-200 ${
        active
          ? "bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-sm shadow-blue-500/30"
          : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200 dark:bg-white/5 dark:text-zinc-300 dark:hover:bg-white/10"
      }`}
    >
      {label}
    </Link>
  );
}
