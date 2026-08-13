import Link from "next/link";

export default function TicketNotFound() {
  return (
    <div className="animate-fade-in-up mx-auto max-w-3xl px-6 py-24 text-center">
      <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-600 text-lg text-white shadow-sm shadow-blue-500/30">
        ?
      </div>
      <p className="mt-4 text-sm text-zinc-500 dark:text-zinc-400">No ticket with that id exists.</p>
      <Link
        href="/tickets"
        className="mt-4 inline-block text-sm font-medium text-blue-600 hover:underline dark:text-blue-400"
      >
        ← All tickets
      </Link>
    </div>
  );
}
