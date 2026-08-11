import Link from "next/link";

export default function TicketNotFound() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-16 text-center">
      <p className="text-sm text-zinc-500 dark:text-zinc-400">
        No ticket with that id exists.
      </p>
      <Link
        href="/tickets"
        className="mt-4 inline-block text-sm font-medium text-zinc-900 hover:underline dark:text-zinc-50"
      >
        ← All tickets
      </Link>
    </div>
  );
}
