// Server-side client for apps/api (apps/api/routers/tickets.py). Every call
// here runs in a Server Component or Route Handler, never in the browser,
// so there's no CORS surface to configure on the FastAPI side and no API
// base URL to leak to the client bundle.

const DEFAULT_API_BASE_URL = "http://localhost:8000";

function apiBaseUrl(): string {
  return process.env.API_BASE_URL ?? DEFAULT_API_BASE_URL;
}

export type TicketSource = "bitext" | "twitter";
export type AuthorRole = "customer" | "agent";

export interface Message {
  id: string;
  ticket_id: string;
  seq: number;
  author_role: AuthorRole;
  text_raw: string;
  text_clean: string;
  sent_at: string | null;
  lang: string | null;
  lang_confidence: number | null;
  content_hash: string;
  external_id: string;
  meta: Record<string, unknown>;
}

export interface Ticket {
  id: string;
  source: TicketSource;
  external_id: string;
  created_at: string | null;
  channel: string;
  customer_id: string | null;
  brand: string | null;
  lang: string | null;
  meta: Record<string, unknown>;
  ingested_at: string;
  messages: Message[];
}

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function apiFetch<T>(path: string): Promise<T> {
  const response = await fetch(`${apiBaseUrl()}${path}`, {
    // The dashboard shows live ticket state -- never serve a stale cached
    // response for what is, functionally, a database read.
    cache: "no-store",
  });
  if (!response.ok) {
    throw new ApiError(
      response.status,
      `${path} failed: ${response.status} ${response.statusText}`,
    );
  }
  return (await response.json()) as T;
}

export interface ListTicketsParams {
  source?: TicketSource;
  limit?: number;
  offset?: number;
}

export async function listTickets(params: ListTicketsParams = {}): Promise<Ticket[]> {
  const search = new URLSearchParams();
  if (params.source) search.set("source", params.source);
  if (params.limit !== undefined) search.set("limit", String(params.limit));
  if (params.offset !== undefined) search.set("offset", String(params.offset));
  const query = search.toString();
  return apiFetch<Ticket[]>(`/tickets${query ? `?${query}` : ""}`);
}

export async function getTicket(id: string): Promise<Ticket | null> {
  try {
    return await apiFetch<Ticket>(`/tickets/${id}`);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return null;
    }
    throw error;
  }
}
