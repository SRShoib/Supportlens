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

async function predictEntitiesWithModel(
  texts: string[],
  model: PredictModel,
): Promise<EntityResult[]> {
  const response = await apiFetch<{ results: EntityResult[] }>("/predict/entities", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts, model }),
  });
  return response.results;
}

// `model: "transformer"` returns the hybrid rules+model predictor's output
// (ml/inference/hybrid_ner.py): each entity type routed to whichever system
// docs/m4-rules-vs-model-report.md found actually wins it on the gold set,
// not pure model output. Falls back to model="baseline" (pure rules) on a
// 503 -- the transformer entity export is a multi-hundred-MB opt-in
// artifact some deployments deliberately don't have (e.g. a free-tier host
// with no RAM headroom for it, see docs/decisions.md), and the rules
// extractor can never 503 (apps/api/routers/predict.py's
// _get_entity_predictor) and already wins 4 of this task's 5 entity types
// on the real gold set -- so this fallback is barely a downgrade even when
// it triggers, and ticket pages keep their entity highlighting either way.
async function predictEntitiesChunk(texts: string[]): Promise<EntityResult[]> {
  try {
    return await predictEntitiesWithModel(texts, "transformer");
  } catch (error) {
    if (error instanceof ApiError && error.status === 503) {
      return predictEntitiesWithModel(texts, "baseline");
    }
    throw error;
  }
}

export async function predictEntities(texts: string[]): Promise<EntityResult[]> {
  if (texts.length === 0) {
    return [];
  }
  const results: EntityResult[] = [];
  for (let offset = 0; offset < texts.length; offset += ENTITY_PREDICT_BATCH_SIZE) {
    const chunk = texts.slice(offset, offset + ENTITY_PREDICT_BATCH_SIZE);
    results.push(...(await predictEntitiesChunk(chunk)));
  }
  return results;
}

// Mirrors apps/api/schemas/predict.py's PredictRequest/PredictResponse --
// the four classification endpoints (/predict/intent, /predict/urgency,
// /predict/sentiment, /predict/emotion) all share this exact request/
// response shape, unlike /predict/entities and /predict/summary above.
export type PredictModel = "baseline" | "transformer";
export type ClassificationTask = "intent" | "urgency" | "sentiment" | "emotion";

export interface TaskResult {
  label: string;
  score: number;
  probabilities: Record<string, number> | null;
}

// Single text in, single result out -- the live "try it" playground's use
// case, unlike the ticket-detail page's predictEntities which batches a
// whole thread. Unauthenticated model="transformer" 503s (model export
// missing, e.g. `make eval` hasn't been run) propagate as ApiError so the
// caller can degrade per-task rather than fail the whole page.
export async function predictClassification(
  task: ClassificationTask,
  text: string,
  model: PredictModel,
): Promise<TaskResult> {
  const response = await apiFetch<PredictResponse>(`/predict/${task}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts: [text], model }),
  });
  return response.results[0];
}

interface PredictResponse {
  results: TaskResult[];
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

// Mirrors apps/api/schemas/search.py -- SPEC M8's dense-retrieval-plus-
// optional-rerank search endpoint. `highlights` are char offsets into
// exactly `snippet` (ml/inference/highlight.py's contract, same "offsets
// into the unmodified string" rule EntitySpan above already follows).
export interface SearchHighlight {
  start: number;
  end: number;
}

export type SearchResultSource = "ticket" | "kb_article";

export interface SearchResult {
  source: SearchResultSource;
  id: string;
  title: string | null;
  snippet: string;
  score: number;
  highlights: SearchHighlight[];
}

export interface SearchResponse {
  results: SearchResult[];
  reranked: boolean;
}

export interface SearchParams {
  topK?: number;
  rerank?: boolean;
}

export async function search(query: string, params: SearchParams = {}): Promise<SearchResponse> {
  return apiFetch<SearchResponse>("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: params.topK ?? 5,
      rerank: params.rerank ?? true,
    }),
  });
}

// Mirrors apps/api/schemas/rag.py -- SPEC M8's RAG suggested-reply
// endpoint. Unlike every getX above, this triggers a real (cached,
// budget-guarded) OpenAI call server-side on every un-cached ticket, so
// callers must only invoke it from an explicit user action (a button
// click), never from a page's initial data-fetch -- see
// src/app/tickets/[id]/actions.ts.
export interface RagSource {
  index: number;
  kind: SearchResultSource;
  id: string;
  title: string | null;
  text: string;
  score: number;
}

export interface SuggestedReply {
  refused: boolean;
  refusal_reason: string | null;
  draft: string | null;
  cited_indices: number[];
  cached: boolean;
  cost_usd: number;
  sources: RagSource[];
}

export async function getSuggestedReply(ticketId: string): Promise<SuggestedReply> {
  return apiFetch<SuggestedReply>(`/tickets/${ticketId}/suggested-reply`, { method: "POST" });
}

// Mirrors apps/api/schemas/eval_run.py -- SPEC M9's /metrics dashboard data
// source. `metrics` is deliberately untyped here (its shape depends on
// `task`: ml/evaluation/metrics.py's ClassificationMetrics for
// intent/urgency/sentiment/emotion, span_metrics.py's SpanMetrics for
// entities, rouge_metrics.py's SummarizationMetrics for thread_summary,
// llm_judge_metrics.py's LLMJudgeMetrics for thread_summary_judge,
// topic_metrics.py's CoherenceMetrics for topics, retrieval_metrics.py's
// RetrievalMetrics for retrieval, latency.py's LatencyResult for
// split="latency" rows, and drift_metrics.py's EmbeddingDriftResult/
// PredictionDriftResult for drift_embedding/drift_prediction) -- callers
// narrow with the specific *Metrics interfaces below, matching how
// getSentimentTrajectory already casts Prediction.payload.
export interface EvalRun {
  id: string;
  task: string;
  model_version: string;
  dataset: string;
  split: string;
  metrics: Record<string, unknown>;
  params: Record<string, unknown>;
  git_sha: string | null;
  status: string;
  started_at: string;
  finished_at: string | null;
}

export interface ClassificationMetrics {
  macro_f1: number;
  per_class_f1: Record<string, number>;
  confusion_matrix: number[][];
  labels: string[];
}

export interface SpanTypeMetrics {
  precision: number;
  recall: number;
  f1: number;
  tp: number;
  fp: number;
  fn: number;
  support: number;
  f1_ci_low: number;
  f1_ci_high: number;
}

export interface SpanMetrics {
  per_type: Record<string, SpanTypeMetrics>;
  micro_precision: number;
  micro_recall: number;
  micro_f1: number;
  macro_f1: number;
  boundary_f1: number;
  partial_f1: number;
  labels: string[];
  n_documents: number;
  n_gold_spans: number;
}

export interface SummarizationMetrics {
  rouge1: number;
  rouge2: number;
  rougeL: number;
  n: number;
}

export interface LLMJudgeMetrics {
  n: number;
  mean_faithfulness: number;
  mean_coverage: number;
  parsed_ok_rate: number;
}

export interface CoherenceMetrics {
  mean_npmi: number;
  n_topics: number;
}

export interface RetrievalRunMetrics {
  hit_rate_at_k: number;
  k: number;
  n_queries: number;
}

export interface LatencyMetrics {
  n_runs: number;
  mean_ms: number;
  p50_ms: number;
  p95_ms: number;
  max_ms: number;
}

// Mirrors ml/evaluation/drift_metrics.py's to_metrics_dict() shapes.
export interface EmbeddingDriftMetrics {
  cosine_shift: number;
  is_alarm: boolean;
  reference_n: number;
  live_n: number;
  threshold: number;
}

export type DriftStatus = "stable" | "watch" | "alarm";

export interface PredictionDriftMetrics {
  psi: number;
  status: DriftStatus;
  reference_dist: Record<string, number>;
  live_dist: Record<string, number>;
  reference_n: number;
  live_n: number;
  watch_threshold: number;
  alarm_threshold: number;
}

export interface ListEvalRunsParams {
  task?: string;
  modelVersion?: string;
  limit?: number;
}

export async function listEvalRuns(params: ListEvalRunsParams = {}): Promise<EvalRun[]> {
  const search = new URLSearchParams();
  if (params.task) search.set("task", params.task);
  if (params.modelVersion) search.set("model_version", params.modelVersion);
  if (params.limit !== undefined) search.set("limit", String(params.limit));
  const query = search.toString();
  return apiFetch<EvalRun[]>(`/eval-runs${query ? `?${query}` : ""}`);
}

// Mirrors apps/api/schemas/drift.py -- SPEC M9's drift panel, the latest
// real-vs-simulated x embedding-vs-prediction EvalRuns
// scripts/compute_drift.py persisted. A leaf is null until that script has
// run at least once.
export interface DriftScenario {
  embedding: EvalRun | null;
  prediction: EvalRun | null;
}

export interface Drift {
  real: DriftScenario;
  simulated: DriftScenario;
}

export async function getDrift(): Promise<Drift> {
  return apiFetch<Drift>("/drift");
}
