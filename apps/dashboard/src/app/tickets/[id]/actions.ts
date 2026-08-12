"use server";

import { getSuggestedReply, type SuggestedReply } from "@/lib/api";

export interface SuggestedReplyActionState {
  reply: SuggestedReply | null;
  error: string | null;
  requested: boolean;
}

// Bound with the ticket id (`generateSuggestedReplyAction.bind(null,
// ticketId)`, see src/components/suggested-reply-panel.tsx) -- this is a
// real, cached-but-sometimes-billed OpenAI call
// (ml/inference/llm_client.py), so it only ever runs from this explicit
// button-click action, never from a page's initial server-rendered fetch.
// useActionState calls the bound result as (state, formData); both are
// unused here (no form fields, ticketId is already bound in) so neither is
// declared -- a function with fewer params still satisfies that call shape.
export async function generateSuggestedReplyAction(
  ticketId: string,
): Promise<SuggestedReplyActionState> {
  try {
    const reply = await getSuggestedReply(ticketId);
    return { reply, error: null, requested: true };
  } catch (error) {
    return {
      reply: null,
      error: error instanceof Error ? error.message : "Failed to generate a suggested reply.",
      requested: true,
    };
  }
}
