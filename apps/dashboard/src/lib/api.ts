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

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl()}${path}`, {
    // The dashboard shows live ticket state -- never serve a stale cached
    // response for what is, functionally, a database read.
    cache: "no-store",
    ...init,
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

// Mirrors apps/api/schemas/predict.py's EntitySpanOut/EntityResultOut.
// `start`/`end` are character offsets into exactly the string that was
// sent -- never re-clean text_clean before slicing it with these.
export interface EntitySpan {
  start: number;
  end: number;
  label: string;
  text: string;
  score: number;
}

export interface EntityResult {
  entities: EntitySpan[];
  truncated: boolean;
}

// Mirrors ml/inference/sentiment_trajectory.py::Trajectory.to_payload() --
// the exact JSONB shape scripts/compute_sentiment_trajectories.py writes
// into Prediction.payload for task="sentiment_trajectory".
export interface SentimentTrajectoryPayload {
  sequence: string[];
  scores: number[];
  final_customer_label: string;
  resolution_quality: number;
  urgency_label: string;
  urgency_score: number;
  urgency_model_version: string;
}

export interface Prediction {
  id: string;
  ticket_id: string | null;
  message_id: string | null;
  task: string;
  label: string | null;
  score: number | null;
  payload: Record<string, unknown>;
  model_version: string;
  eval_run_id: string | null;
  created_at: string;
}

// GET /tickets/{id}/predictions reads durably-stored Predictions (SPEC M5) --
// unlike predictEntities above, this never recomputes live. Returns [] (not
// an error) for a ticket that exists but has no predictions yet, e.g.
// scripts/compute_sentiment_trajectories.py hasn't run against this ticket.
export async function getTicketPredictions(
  ticketId: string,
  task?: string,
): Promise<Prediction[]> {
  const search = new URLSearchParams();
  if (task) search.set("task", task);
  const query = search.toString();
  try {
    return await apiFetch<Prediction[]>(
      `/tickets/${ticketId}/predictions${query ? `?${query}` : ""}`,
    );
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return [];
    }
    throw error;
  }
}

export async function getSentimentTrajectory(
  ticketId: string,
): Promise<SentimentTrajectoryPayload | null> {
  const predictions = await getTicketPredictions(ticketId, "sentiment_trajectory");
  const payload = predictions[0]?.payload;
  return (payload as unknown as SentimentTrajectoryPayload | undefined) ?? null;
}

// scripts/compute_thread_summaries.py stores the summary text directly on
// Prediction.label (not payload) -- there's no secondary structure to a
// summary the way sentiment_trajectory has sequence/scores, so a plain
// string field is the whole payload. null for tickets the backfill script
// skipped (< 2 messages) or hasn't reached yet -- same best-effort contract
// as getSentimentTrajectory.
export async function getThreadSummary(ticketId: string): Promise<string | null> {
  const predictions = await getTicketPredictions(ticketId, "thread_summary");
  return predictions[0]?.label ?? null;
}

// POST /predict/entities caps a single request at 100 texts
// (apps/api/schemas/predict.py's PredictRequest), so a ticket with an
// unusually long message history is chunked rather than truncated.
const ENTITY_PREDICT_BATCH_SIZE = 100;

// `model: "transformer"` returns the hybrid rules+model predictor's output
// (ml/inference/hybrid_ner.py): each entity type routed to whichever system
// docs/m4-rules-vs-model-report.md found actually wins it on the gold set,
// not pure model output.
export async function predictEntities(texts: string[]): Promise<EntityResult[]> {
  if (texts.length === 0) {
    return [];
  }
  const results: EntityResult[] = [];
  for (let offset = 0; offset < texts.length; offset += ENTITY_PREDICT_BATCH_SIZE) {
    const chunk = texts.slice(offset, offset + ENTITY_PREDICT_BATCH_SIZE);
    const response = await apiFetch<{ results: EntityResult[] }>("/predict/entities", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: chunk, model: "transformer" }),
    });
    results.push(...response.results);
  }
  return results;
}

// Mirrors apps/api/schemas/topic.py -- SPEC M7's topic catalog + weekly
// volume trend + emerging-issues surfaces. Unlike predictEntities above,
// none of these ever compute live: they read straight from what
// scripts/assign_topics.py already wrote to Postgres (GET /topics/* never
// loads an embedding or topic model, see docs/decisions.md).
export interface Topic {
  id: string;
  topic_key: number;
  label: string;
  keywords: string[];
  size: number;
  model_version: string;
  created_at: string;
}

export interface TopicVolumePoint {
  week: string;
  count: number;
  share: number;
  z_score: number | null;
  is_emerging: boolean;
}

export interface TopicVolumeSeries {
  topic_id: number;
  label: string;
  points: TopicVolumePoint[];
}

export interface TopicVolumeResponse {
  weeks: string[];
  series: TopicVolumeSeries[];
}

export interface EmergingIssue {
  topic_id: number;
  label: string;
  week: string;
  count: number;
  share: number;
  z_score: number;
}

export async function listTopics(): Promise<Topic[]> {
  return apiFetch<Topic[]>("/topics");
}

export async function getTopicVolume(): Promise<TopicVolumeResponse> {
  return apiFetch<TopicVolumeResponse>("/topics/volume");
}

export async function getEmergingIssues(): Promise<EmergingIssue[]> {
  return apiFetch<EmergingIssue[]>("/topics/emerging");
}
