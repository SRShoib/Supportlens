import type { TicketSource } from "@/lib/api";

export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

// Chrome-level identity color per ticket source (twitter/bitext), not a
// data-encoding color -- same "reserve a hue for wayfinding, not chart
// meaning" convention src/components/sidebar-nav.tsx's COLOR_STYLES follows
// for the sidebar. Kept here (not JSX) since both the Overview page and the
// tickets list render the same badge.
const TICKET_SOURCE_BADGE_CLASS: Record<TicketSource, string> = {
  twitter: "bg-sky-50 text-sky-700 dark:bg-sky-500/10 dark:text-sky-300",
  bitext: "bg-violet-50 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300",
};

export function ticketSourceBadgeClass(source: TicketSource): string {
  return TICKET_SOURCE_BADGE_CLASS[source];
}
