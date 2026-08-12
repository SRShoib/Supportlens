"use server";

// Backs the "Try it live" playground (src/components/try-it-form.tsx): runs
// arbitrary pasted text through the real, already-trained models -- nothing
// here writes to the database, this is read-only inference (SPEC's
// baseline-vs-transformer story, made interactive). Every classification
// task runs baseline AND transformer in parallel so the two can be shown
// side by side, the same comparison CLAUDE.md rule #2 asks every module to
// report formally, just live instead of from a persisted eval run.

import { type ClassificationTask, type EntitySpan, predictClassification, predictEntities, type TaskResult } from "@/lib/api";

const CLASSIFICATION_TASKS: ClassificationTask[] = ["intent", "urgency", "sentiment", "emotion"];

export interface ClassificationComparison {
  task: ClassificationTask;
  baseline: TaskResult | null;
  transformer: TaskResult | null;
}

export interface TryItActionState {
  text: string;
  classifications: ClassificationComparison[];
  entities: EntitySpan[] | null;
  error: string | null;
}

// A model export can legitimately be missing (`make eval`/`make eval-ner`
// hasn't been run for that task yet) -- that degrades to "not available"
// for just that one cell, not a failed page, matching the best-effort
// contract src/app/tickets/[id]/page.tsx already uses for entities/
// trajectory/summary.
async function safePredict(
  task: ClassificationTask,
  text: string,
  model: "baseline" | "transformer",
): Promise<TaskResult | null> {
  try {
    return await predictClassification(task, text, model);
  } catch {
    return null;
  }
}

async function safeEntities(text: string): Promise<EntitySpan[] | null> {
  try {
    const results = await predictEntities([text]);
    return results[0]?.entities ?? null;
  } catch {
    return null;
  }
}

export async function tryItAction(
  prevState: TryItActionState,
  formData: FormData,
): Promise<TryItActionState> {
  const text = String(formData.get("text") ?? "").trim();

  if (!text) {
    return { ...prevState, error: "Paste a message first." };
  }

  const [classifications, entities] = await Promise.all([
    Promise.all(
      CLASSIFICATION_TASKS.map(async (task) => {
        const [baseline, transformer] = await Promise.all([
          safePredict(task, text, "baseline"),
          safePredict(task, text, "transformer"),
        ]);
        return { task, baseline, transformer };
      }),
    ),
    safeEntities(text),
  ]);

  return { text, classifications, entities, error: null };
}
