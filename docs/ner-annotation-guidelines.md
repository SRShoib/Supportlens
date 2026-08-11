# NER annotation guidelines (v1)

Defines the five entity types for M4 (SPEC §4) sharply enough to annotate against. Written before any
annotation happens; referenced by version (`v1`) from `data/gold/ner_gold_v1.meta.json` and from the
synthetic generator's templates. If a v2 ever changes a definition, the gold set gets re-versioned too —
these guidelines and the gold set they produced must always agree.

## Global rules

- Spans are **flat and non-overlapping**. On a genuine conflict, the longest span wins; if equal length, the
  more specific type wins per the disambiguation notes below.
- A span never includes leading/trailing whitespace, and never trailing sentence punctuation.
- Mask tokens (`<URL>`, `<USER>`, `<EMAIL>`, `<PHONE>`, `<EMOJI:...>` — see `ml/data/masking.py`) are **never**
  annotated. They already replaced the raw text; there is nothing left to label.
- Every occurrence of an entity gets its own span, including repeats within one message.
- If still uncertain after applying the tie-breaks below, leave the span unannotated and add the example id
  to the **Contested cases** section at the bottom of this file. That list is itself useful — it documents
  where the schema itself is ambiguous, not just where an annotator hesitated.

## ORDER_ID

A merchant-assigned identifier for **a specific transaction or shipment**.

- **Covers:** order numbers, confirmation numbers, booking references, tracking numbers, RMA/return numbers,
  invoice numbers, flight numbers (`DL404` identifies the journey being asked about).
- **Span:** the identifier token only. Excludes the trigger word and a leading `#`/`:`
  (`order #99321` → span is `99321`).
- **Tie-break:** *an id is something you could paste into a lookup box.*
- **Not:** account/membership/policy/case numbers (→ ACCOUNT_REF); prices; dates; bare quantities
  (`I ordered 3 of them` — `3` is not an id).
- **Edges:**
  - `orders 111, 222 and 333` → three separate spans, digits only.
  - `order 12345` where `12345` could be read as a quantity → ORDER_ID only if the sentence is about
    looking up or tracking a specific order.

## PRODUCT

A nameable good or service the customer bought or uses.

- **Span:** the full product name as written, brand included when contiguous with the model:
  `iPhone 12 Pro Max`, `Samsung Galaxy S22`, `Apple Watch Series 8`, `Xbox Game Pass`, `Spotify Premium`,
  `Delta Comfort+`.
- **Brand vs model — the sharp edge:**
  - Brand adjacent to a model name → one PRODUCT span covering both.
  - Brand alone, naming the company being complained *to* → **not** PRODUCT, it's the vendor:
    `<USER> this is unacceptable`, `Verizon, fix this`.
  - Brand alone, unambiguously denoting the purchased good → PRODUCT: `my Kindle broke`.
- **Not:** generic category nouns (`my phone`, `the package`, `the app`, `my order`); bare plan words
  (`my plan`, `my subscription`) — but a *named* plan is PRODUCT (`Unlimited Plus`, `Prime`).
- Casing is a strong signal but not a requirement: `iphone 12 pro` is still PRODUCT.

## DATE

An expression that points at a time, absolute or relative.

- **Absolute:** `March 3`, `Mar 3rd`, `03/12/2023`, `2023-12-03`, `Dec 2022`, `Black Friday`, `Christmas`.
- **Relative:** `yesterday`, `this morning`, `last Tuesday`, `two weeks ago`, `over the weekend`, `3 days back`.
- **Boundary rule:** the span starts at the first token carrying temporal content and ends at the last.
  Leading prepositions (`on`, `since`, `by`, `from`, `after`) are **excluded** (`since Friday` → span is
  `Friday`). Determiners that are part of the expression are **included** (`last Tuesday`, `this morning`).
  `ago`/`back` are **included**.
- **Relative-date edge — the most contested one, stated with examples both ways:** a duration that *fixes a
  point in time* is DATE (`3 days ago`, `two weeks back`); a pure duration that does not fix a point is
  **not** DATE (`I waited 3 days`, `a 2-hour wait`, `within 24 hours`).
- **Not:** a bare time of day (`at 3pm`) — v1 has no TIME type; `3pm yesterday` → span is `yesterday` only.
  `20 mins` alone → nothing. A bare year (`2023`) → DATE.

## AMOUNT

A monetary quantity.

- **Span includes** the currency symbol or code: `$49.99`, `£12.50`, `€1,299.00`, `USD 40`, `40 dollars`,
  `twenty quid`. Include the symbol even when space-separated (`$ 49.99` → one span).
- **Currency vs bare number — the sharp edge:** a bare number is AMOUNT **only when the governing context is
  unambiguously monetary** — `refund of 1,299.00`, `charged me 49.99`, `they took 200 off my card`.
  **Tie-break:** *would substituting a different currency amount keep the sentence sensible?* A number with
  no monetary governor (`3 items`, `order 12345`, `2 weeks`) is not AMOUNT.
- **Not:** percentages (`20% off` — v1 has no PERCENT type; in `20% of $50` the span is `$50` only);
  loyalty points/miles/credits (`5000 miles`, `200 points` — a real and frequent twcs airline pattern,
  called out explicitly so it isn't mislabeled); `free`/`no charge`.
- **Ranges:** `$40-$60` → two separate spans.

## ACCOUNT_REF

A customer-side identifier for **the relationship**, not the transaction.

- **This is the definition SPEC leaves open, and the masking pipeline settles it:** `ml/data/masking.py`
  already replaces `@handles` with `<USER>` before any NER text is ever seen. So ACCOUNT_REF explicitly does
  **not** mean social handles or usernames — those are gone by the time this task sees the text.
- **Covers:** account numbers, member/loyalty ids, policy numbers, support case/ticket/reference numbers,
  subscriber ids, card last-4 (`ending in 4432` → span is `4432`).
- **Span:** the identifier only, not the trigger word (`account 4455-9911` → span is `4455-9911`).
- **Disambiguation vs ORDER_ID — genuinely arguable, stated explicitly:** if the id identifies *a purchase or
  shipment* → ORDER_ID; if it identifies *the customer or their ongoing relationship/interaction* →
  ACCOUNT_REF. Support case numbers are ACCOUNT_REF.
- **Not:** emails/phones (already masked); IP addresses; device serials/IMEIs (device identity — out of
  scope in v1, no span).

## Contested cases

Real example ids get logged here during gold-set annotation whenever the rules above don't cleanly resolve
a case. Empty until annotation starts.
