# Thread summarization failure modes (M6)

SPEC M6: "Document hallucination examples found." The five examples below are the lowest-faithfulness
real supportlens tickets from the 50-example LLM-as-judge pass (`ml/data/llm_judge_summaries.py`,
`gpt-4o-mini`, 1-5 rubric) against `transformer_thread_summary_flan-t5-small_v1` — every one scored
faithfulness=1/5. Unlike `docs/m6-comparison-report.md` (fully generated from persisted eval runs), this
doc is a hand-written read of the actual hallucinated output, because "why is this wrong" requires reading
the conversation, not just a metric.

They fall into three distinct failure modes, not one generic "the model hallucinates" story.

## 1. Speaker/role misattribution

The model attributes the customer's statement or experience to the agent, or vice versa — a pure subject-swap,
not a fabricated fact per se.

**Ticket `549fb83f-6881-5cce-b979-aeaa9e5dc9ef`** (hotel noise complaint):

> [customer] `<USER>` what's the policy when you can't sleep because there is music blasting out downstairs?!
> [agent] `<USER>` Thanks for staying with us & sorry to see your dismay regarding the issues w/ noise. Kindly reach out to the Duty Manager...

**Generated summary:** "Agent has a complaint about the music blasting out downstairs." (faithfulness 1,
coverage 1)

It's the *customer* who can't sleep and is complaining — the agent is the one responding to a complaint,
not making one. The model correctly extracted the topic ("music blasting downstairs") but assigned it to
the wrong party.

**Ticket `7daa0765-7fc6-5b6a-9882-cde8bb2d4fea`** (support-quality complaint):

> [customer] `<USER>` sucked the life out of me today. 😞
> [agent] `<USER>` Oh no, that's not healthy. Please shoot me a DM #HereForYou ... *JawedK
> [customer] ... Jawed = 8th person I talked to today, took a lot of my time but couldn't help me...

**Generated summary:** "Jawed sucked the life out of him today." (faithfulness 1, coverage 1)

"Jawed" is the agent's handle (`*JawedK`), signed on their own reply — not the subject of the customer's
complaint. The model read the agent's name off the sign-off and pasted it into the customer's figurative
complaint as the grammatical subject, producing a sentence that reads as a false, mildly reputation-damaging
claim about a named (if pseudonymous) support agent. This is the most concerning variant of the pattern:
role-swap failures aren't just "wrong," they can fabricate a negative claim about a real person's name that
appears verbatim in the source text.

## 2. Fabricated resolution status

The model asserts a request was resolved, confirmed, or agreed to, when the conversation actually shows the
matter still open, unresolved, or explicitly declined by the customer.

**Ticket `27872993-b031-54d6-9ddd-80e59e299437`** (delay-repay claim status):

> [customer] `<USER>` how do I find out my delay repay claim forms have been received
> [agent] `<USER>` Hello, at the bottom of this page you can see which date we are currently processing: `<URL>`

**Generated summary:** "The delay repayment claim forms have been received." (faithfulness 1, coverage 2)

The customer is *asking how to check* whether their forms arrived — a question. The agent points to a
status page, without confirming receipt. The model converted an open question into a settled fact that
directly contradicts what actually happened (nothing was confirmed as received).

**Ticket `ca104a32-2857-5e4f-81ee-b9fc96dc11fa`** (expired product refund):

> [customer] I would have loved to... But it's a few days out of date
> [agent] ...you can return the packaging to any Aldi and we'll provide a refund.
> [customer] Checked and we don't [have the receipt]. I guess that's something that's going to be ignored then...

**Generated summary:** "Stephen got home from shopping with a DM. He will return the packaging to Aldi. "
(faithfulness 1, coverage 1)

The agent's refund offer required a receipt; the customer confirmed they didn't have one and explicitly
gave up pursuing it ("I guess that's something that's going to be ignored then"). The summary reports the
refund path as a decided future action ("he will return the packaging") — the opposite of how the customer
actually left it. "got home from shopping with a DM" is also a garbled non-sequitur, folding the earlier
DM-support offer into the unrelated shopping-trip sentence.

## 3. Narrative fabrication from keyword co-occurrence

The model stitches surface-level words from different, unrelated remarks into a coherent-sounding but
entirely invented storyline.

**Ticket `3b2b2d60-1fd7-5161-b0c8-effb2032c656`** (turbulent flight):

> [customer] #1 most terrifying turbulence experience of my life... Someone needs to buy these pilots a box
> of cigars for their flying `<USER>`
> [agent] Oh no, Jacob! Are you okay? *HDG
> [customer] `<USER>` `<USER>` Welcome to the Rockies. Worst turbulence for me was Las Vegas>Denver...

**Generated summary:** "Jacob is going to the Rockies to buy the pilots a box of cigars for their flying."
(faithfulness 1, coverage 1)

Three unrelated fragments — a joke about buying the pilots cigars, a name ("Jacob") the agent used to
address the customer, and a different reply-thread participant's remark ("Welcome to the Rockies," a
comparison to a different rough flight, not a travel plan) — get merged into one fabricated sentence with
a fake purpose ("going to the Rockies *to buy* the pilots a box of cigars") that never appears anywhere in
the conversation. Nobody is going anywhere to buy anything; the customer is recounting a scary flight and
joking that the pilots deserve a reward.

## Takeaways

- All five examples score faithfulness=1 *and* mostly coverage=1-2 together — low faithfulness here isn't
  "a good summary with one wrong detail," it's summaries that would actively mislead an agent skimming the
  ticket queue.
- Two of five involve inventing or misassigning a **name** (`Jacob`, `Jawed`) present in the raw text but
  attached to the wrong action or claim — a specific pattern worth watching for in any future guardrail
  work (e.g., checking that named entities in the summary appear near the *same* speaker role in the
  source).
- Real support-Twitter dialogue (multi-turn, informal, cross-referenced handles like `*HDG`/`*AMR`/`*JawedK`
  sign-offs) is a harder distribution than samsum/dialogsum's cleaner two-party chats — consistent with
  `docs/m6-comparison-report.md`'s own limitations section noting the domain gap was never separately
  measured beyond this 50-example judge sample.
- These are 5 of 50 judged tickets (10%) scoring the minimum faithfulness — see
  `docs/m6-comparison-report.md` for the full aggregate (mean faithfulness/coverage across all 50).
