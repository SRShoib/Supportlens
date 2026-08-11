import Link from "next/link";

import { listTickets, type TicketSource } from "@/lib/api";

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
    <div className="mx-auto max-w-4xl px-6 py-10">
      <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50">Tickets</h1>
      <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
        {tickets.length} ticket{tickets.length === 1 ? "" : "s"}
        {source ? ` · source: ${source}` : ""}
      </p>

      <nav className="mt-4 flex gap-2 text-sm">
        <FilterLink label="All" href="/tickets" active={!source} />
        <FilterLink label="Twitter" href="/tickets?source=twitter" active={source === "twitter"} />
        <FilterLink label="Bitext" href="/tickets?source=bitext" active={source === "bitext"} />
      </nav>

      <ul className="mt-6 divide-y divide-zinc-200 dark:divide-zinc-800">
        {tickets.map((ticket) => (
          <li key={ticket.id}>
            <Link
              href={`/tickets/${ticket.id}`}
              className="flex items-center justify-between gap-4 py-4 hover:bg-zinc-50 dark:hover:bg-zinc-900"
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
              <span className="shrink-0 rounded-full bg-zinc-100 px-2 py-1 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
                {ticket.brand ?? ticket.source}
              </span>
            </Link>
          </li>
        ))}
        {tickets.length === 0 && (
          <li className="py-10 text-center text-sm text-zinc-500 dark:text-zinc-400">
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
      className={`rounded-full px-3 py-1 ${
        active
          ? "bg-zinc-900 text-white dark:bg-zinc-50 dark:text-zinc-900"
          : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
      }`}
    >
      {label}
    </Link>
  );
}
