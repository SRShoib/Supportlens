"""Generates SPEC M8's "small synthetic KB (~40 articles)" -- purely
templated, no LLM call. SPEC §5's M8 budget line ("RAG reply drafting (demo
+ cache warm) ~= $1.50") is earmarked for reply drafting specifically, not
for writing the KB itself, so this stays free and offline (docs/decisions.md).

40 articles = one per Bitext intent (27, the exact real intent categories
ingested into Postgres -- see `ml/training/splits.py`'s
`ticket.meta.get("intent")`) + 13 hand-picked from the real M7 Twitter
topic catalog (`topics` table, model_version="topics_bertopic_v1"). The 13
are chosen, not derived mechanically from the raw c-TF-IDF keyword strings:
several real topics are noise clusters ("wtf, does, jpg, love";
"fuck, worst, suck, hate") or too vague ("service, customer, hold, chat")
to make a coherent KB article, and c-TF-IDF keyword lists read as word
soup, not prose -- rendering them directly would produce a KB a portfolio
demo shouldn't show. Each of the 13 still cites its source topic_key in
`source_key`/`tags` for traceability back to the real cluster it's
grounded in.

Ids are deterministic (`ml/data/ids.py::deterministic_id`, keyed off
generator_version + source_kind + source_key) so re-running this script
after an edit is a natural upsert-by-title, never a duplicate -- unlike
Topic/Prediction's random-uuid4 + delete-everything-first convention,
appropriate there because those are recomputed from a fresh model fit each
time; these are static, hand-authored content.

Run: uv run python -m ml.data.kb_generate
"""

from dataclasses import dataclass

from api.db.models import KbArticle
from api.db.session import SessionLocal
from sqlalchemy.orm import Session

from ml.data.ids import deterministic_id

GENERATOR_VERSION = "kb_template_v1"


@dataclass(frozen=True)
class ArticleSpec:
    source_kind: str  # "intent" | "topic"
    source_key: str
    title: str
    intro: str
    steps: list[str]
    tags: list[str]


def render_body(intro: str, steps: list[str]) -> str:
    lines = [intro, "", "**Steps:**"]
    lines += [f"{i}. {step}" for i, step in enumerate(steps, start=1)]
    lines.append("")
    lines.append(
        "If this doesn't resolve the issue, contact customer support and reference this article."
    )
    return "\n".join(lines)


# One per real Bitext intent (source_key is the exact intent string stored in
# Ticket.meta["intent"], see ml/data/loaders/bitext.py).
_INTENT_ARTICLES: list[ArticleSpec] = [
    ArticleSpec(
        "intent",
        "cancel_order",
        "How to Cancel an Order",
        "You can cancel an order before it ships from your account's order history.",
        [
            "Go to Order History in your account and select the order you want to cancel.",
            "Click Cancel Order. This option is only available while the order status is still Processing.",
            "If the order has already shipped, use the return process instead once it arrives.",
            "A cancellation confirmation email is sent once the cancellation is processed.",
        ],
        ["cancel_order", "orders"],
    ),
    ArticleSpec(
        "intent",
        "change_order",
        "How to Change an Order After Placing It",
        "Orders can be edited while they're still being processed.",
        [
            "Open the order in Order History and select Edit Order.",
            "You can change quantity, size, or shipping address only while the order is still Processing.",
            "Once the order moves to Preparing for Shipment, changes can no longer be made -- cancel and reorder instead.",
            "Save your changes; a confirmation email will summarize the updated order.",
        ],
        ["change_order", "orders"],
    ),
    ArticleSpec(
        "intent",
        "change_shipping_address",
        "How to Update Your Shipping Address",
        "Your default shipping address, and an unshipped order's address, can both be updated.",
        [
            "Go to Account Settings > Addresses.",
            "Edit an existing address or add a new one, then set it as the default shipping address.",
            "For an order already placed but not yet shipped, edit the order directly instead of your account default.",
            "Double-check the postal code and unit number; carriers reject packages with mismatched postal codes.",
        ],
        ["change_shipping_address", "shipping"],
    ),
    ArticleSpec(
        "intent",
        "check_cancellation_fee",
        "Understanding Cancellation Fees",
        "Whether a cancellation fee applies depends on how far the order has progressed.",
        [
            "Orders cancelled before shipment are never charged a cancellation fee.",
            "Orders cancelled after shipment are treated as a return and may be subject to a restocking fee shown on the product page.",
            "Subscription cancellations follow the terms in your plan; check Account > Subscriptions for the exact fee, if any.",
            "The exact fee for your order, if applicable, is always shown before you confirm cancellation.",
        ],
        ["check_cancellation_fee", "orders", "billing"],
    ),
    ArticleSpec(
        "intent",
        "check_invoice",
        "How to Find and Download Your Invoice",
        "Invoices are available as a PDF for every completed order.",
        [
            "Go to Order History and select the order.",
            "Click Download Invoice to get a PDF copy.",
            "Invoices are generated once payment is confirmed, usually within a few minutes of checkout.",
            "For a missing invoice on an older order, use the Get Invoice request form instead.",
        ],
        ["check_invoice", "billing"],
    ),
    ArticleSpec(
        "intent",
        "check_payment_methods",
        "Accepted Payment Methods",
        "A summary of which payment methods are supported at checkout.",
        [
            "We accept major credit and debit cards, PayPal, and store gift cards.",
            "Payment methods can be managed under Account Settings > Payment Methods.",
            "Some payment methods may not be available in all regions or for all order types.",
            "For payment issues at checkout, see the Troubleshooting a Failed Payment article.",
        ],
        ["check_payment_methods", "billing"],
    ),
    ArticleSpec(
        "intent",
        "check_refund_policy",
        "Our Refund Policy",
        "The rules that determine whether a returned item qualifies for a refund.",
        [
            "Items can be returned within 30 days of delivery for a full refund.",
            "The item must be in its original condition and packaging.",
            "Refunds are issued to the original payment method within 5-10 business days of the return being received.",
            "Digital and final-sale items are not eligible for refund; this is noted on the product page.",
        ],
        ["check_refund_policy", "refunds"],
    ),
    ArticleSpec(
        "intent",
        "complaint",
        "How to File a Complaint",
        "How to formally report an issue and get it tracked to resolution.",
        [
            "Go to Contact Us and select Complaint as the topic.",
            "Include your order number and a description of the issue for the fastest resolution.",
            "You'll receive a case number by email; use it to check the status of your complaint.",
            "Serious complaints are escalated to a specialist within 1 business day.",
        ],
        ["complaint"],
    ),
    ArticleSpec(
        "intent",
        "contact_customer_service",
        "How to Contact Customer Service",
        "The available channels for reaching support, and when to use each.",
        [
            "Use the Contact Us form for non-urgent requests; expect a reply within 24 hours.",
            "Live chat is available from the Help widget during business hours for faster help.",
            "For account security issues, use the dedicated Security contact option instead.",
            "Have your order number or account email ready to speed up verification.",
        ],
        ["contact_customer_service"],
    ),
    ArticleSpec(
        "intent",
        "contact_human_agent",
        "How to Reach a Live Agent",
        "How to skip past automated support when you need a person.",
        [
            "Open the Help widget and select Talk to an Agent.",
            'If a bot answers first, type "agent" or "human" to skip ahead in the queue.',
            "Live agents are available during business hours; outside those hours, your request is queued for the next available agent.",
            "Average wait time is shown in the chat window before you connect.",
        ],
        ["contact_human_agent"],
    ),
    ArticleSpec(
        "intent",
        "create_account",
        "How to Create an Account",
        "Setting up a new account takes under a minute.",
        [
            "Click Sign Up and enter your email address and a password.",
            "Verify your email using the confirmation link sent to your inbox.",
            "Complete your profile with a shipping address to speed up future checkouts.",
            "If you already ordered as a guest, use the same email to link past orders automatically.",
        ],
        ["create_account", "account"],
    ),
    ArticleSpec(
        "intent",
        "delete_account",
        "How to Delete Your Account",
        "Account deletion permanently removes your saved data.",
        [
            "Go to Account Settings > Privacy > Delete Account.",
            "Confirm the request; this permanently removes your saved addresses, payment methods, and order history.",
            "Any pending orders must be completed or cancelled before deletion can proceed.",
            "Account deletion cannot be undone; download any records you need first.",
        ],
        ["delete_account", "account"],
    ),
    ArticleSpec(
        "intent",
        "delivery_options",
        "Available Delivery Options",
        "What shipping speeds and pickup options are offered at checkout.",
        [
            "Standard, expedited, and next-day delivery are available at checkout, where offered.",
            "Available options depend on your address and the items in your cart.",
            "In-store pickup is offered for select locations as an alternative to shipping.",
            "Delivery option and cost are confirmed before you complete your order.",
        ],
        ["delivery_options", "shipping"],
    ),
    ArticleSpec(
        "intent",
        "delivery_period",
        "Estimated Delivery Times",
        "Typical delivery windows by shipping speed.",
        [
            "Standard delivery typically takes 3-7 business days after the order ships.",
            "Expedited delivery typically takes 1-3 business days.",
            "Estimated delivery dates are shown at checkout and in your shipping confirmation email.",
            "Delays can occur during peak seasons or due to carrier disruptions in your area.",
        ],
        ["delivery_period", "shipping"],
    ),
    ArticleSpec(
        "intent",
        "edit_account",
        "How to Edit Your Account Details",
        "Updating your name, contact details, or saved information.",
        [
            "Go to Account Settings to update your name, email, or phone number.",
            "Changing your email requires verifying the new address before it takes effect.",
            "Password changes are made separately under Account Settings > Security.",
            "Saved addresses and payment methods can be edited from their own sections.",
        ],
        ["edit_account", "account"],
    ),
    ArticleSpec(
        "intent",
        "get_invoice",
        "How to Request an Invoice Copy",
        "Getting an invoice for an order that isn't in your account, or an older order.",
        [
            "Go to Order History, select the order, and click Download Invoice.",
            "If the order isn't listed (e.g. a guest checkout), use the Request Invoice form with your order number and email.",
            "Requested invoices are emailed within one business day.",
            "Invoices older than 24 months may require a manual request through customer service.",
        ],
        ["get_invoice", "billing"],
    ),
    ArticleSpec(
        "intent",
        "get_refund",
        "How to Request a Refund",
        "Starting a return and getting your money back.",
        [
            "Go to Order History, select the order, and choose Return or Refund.",
            "Select a reason and confirm; a prepaid return label is provided when applicable.",
            "Once the returned item is received and inspected, the refund is issued to your original payment method.",
            "See the Refund Policy article for eligibility windows and item conditions.",
        ],
        ["get_refund", "refunds"],
    ),
    ArticleSpec(
        "intent",
        "newsletter_subscription",
        "Managing Your Newsletter Subscription",
        "Subscribing or unsubscribing from marketing email.",
        [
            "Go to Account Settings > Communication Preferences to subscribe or unsubscribe.",
            "You can also unsubscribe using the link at the bottom of any newsletter email.",
            "Preference changes can take up to 48 hours to fully apply.",
            "Transactional emails (order confirmations, shipping updates) are sent regardless of newsletter preference.",
        ],
        ["newsletter_subscription", "account"],
    ),
    ArticleSpec(
        "intent",
        "payment_issue",
        "Troubleshooting a Failed Payment",
        "Why a payment might be declined and how to fix it.",
        [
            "Confirm your card details, billing address, and expiration date are entered correctly.",
            "Check with your bank -- some declines are due to a fraud hold on the card, not an issue on our end.",
            "Try an alternate payment method if the issue persists.",
            "If you were charged but the order didn't complete, the charge is automatically reversed within 5-7 business days.",
        ],
        ["payment_issue", "billing"],
    ),
    ArticleSpec(
        "intent",
        "place_order",
        "How to Place an Order",
        "The standard checkout flow from cart to confirmation.",
        [
            "Add items to your cart and proceed to checkout.",
            "Enter or select a shipping address and payment method.",
            "Review your order summary, then click Place Order to confirm.",
            "A confirmation email with your order number is sent immediately after checkout.",
        ],
        ["place_order", "orders"],
    ),
    ArticleSpec(
        "intent",
        "recover_password",
        "How to Recover Your Password",
        "Resetting a forgotten password.",
        [
            "Click Forgot Password on the sign-in page and enter your account email.",
            "Follow the reset link sent to your inbox; it expires after 30 minutes for security.",
            "If no email arrives, check your spam folder or confirm you're using the email your account was created with.",
            "Still stuck? Contact customer service to verify your identity and reset access manually.",
        ],
        ["recover_password", "account"],
    ),
    ArticleSpec(
        "intent",
        "registration_problems",
        "Troubleshooting Registration Problems",
        "Common reasons sign-up fails, and how to fix each.",
        [
            "Make sure your password meets the minimum requirements (8+ characters, one number).",
            '"Email already in use" means an account already exists -- try Forgot Password instead of signing up again.',
            "Clear your browser cache or try a different browser if the form won't submit.",
            "If verification emails aren't arriving, check spam or request a new verification link.",
        ],
        ["registration_problems", "account"],
    ),
    ArticleSpec(
        "intent",
        "review",
        "How to Leave a Product Review",
        "Reviewing an item you've purchased.",
        [
            "Go to Order History and select Write a Review next to a delivered item.",
            "Reviews can only be submitted for items you've purchased and received.",
            "Ratings and written reviews are both optional, but photos help other customers most.",
            "Reviews are typically published within 24 hours after a moderation check.",
        ],
        ["review"],
    ),
    ArticleSpec(
        "intent",
        "set_up_shipping_address",
        "How to Set Up a Shipping Address",
        "Adding a new address to your account.",
        [
            "Go to Account Settings > Addresses > Add Address.",
            "Fill in the full address including apartment/unit number and postal code.",
            "Mark an address as default to have it pre-filled at checkout.",
            "You can save multiple addresses and choose between them during checkout.",
        ],
        ["set_up_shipping_address", "shipping"],
    ),
    ArticleSpec(
        "intent",
        "switch_account",
        "How to Switch Between Accounts",
        "Moving between two separate accounts on the same device.",
        [
            "Click your profile icon and select Switch Account from the menu.",
            "Sign in with the other account's email and password when prompted.",
            "Each account keeps its own order history, saved addresses, and payment methods separately.",
            "To merge two accounts' order histories, contact customer service -- this can't be done automatically.",
        ],
        ["switch_account", "account"],
    ),
    ArticleSpec(
        "intent",
        "track_order",
        "How to Track Your Order",
        "Following a shipment from warehouse to doorstep.",
        [
            "Go to Order History and select Track Package on the relevant order.",
            "Tracking becomes active once the carrier scans the package, usually within 24 hours of shipment.",
            "You can also track directly on the carrier's website using the tracking number in your shipping confirmation email.",
            "If tracking hasn't updated in 5+ business days, contact customer service to open an investigation with the carrier.",
        ],
        ["track_order", "shipping"],
    ),
    ArticleSpec(
        "intent",
        "track_refund",
        "How to Track a Refund",
        "Checking where a refund is in the process.",
        [
            "Go to Order History and select the order to see its refund status.",
            "Refund status moves from Requested to Approved to Issued as it's processed.",
            "Once issued, it typically takes 5-10 business days to appear on your original payment method's statement.",
            "If it's been longer than 10 business days since Issued, contact your bank first, then customer service.",
        ],
        ["track_refund", "refunds"],
    ),
]

# 13 hand-picked from the real M7 Twitter topic catalog (topic_key: label),
# grounded in a real cluster but rewritten as clean, brand-agnostic
# prose -- see module docstring for why these 13 and not a mechanical pass
# over every topic.
_TOPIC_ARTICLES: list[ArticleSpec] = [
    ArticleSpec(
        "topic",
        "topic:0",
        "Food & Store Order Issues",
        "Issues with a food or in-store pickup order -- wrong items, missing items, or an order mixed up with someone else's.",
        [
            "Check your order confirmation to confirm which items and store location were included.",
            "Report a wrong or missing item through the order's Help option within 24 hours for the fastest resolution.",
            "Refunds or credits for order errors are typically issued to your original payment method or account balance.",
            "For a repeated issue at the same location, mention it in your report -- store-level patterns get escalated separately.",
        ],
        ["topic_0", "food, store"],
    ),
    ArticleSpec(
        "topic",
        "topic:1",
        "Flight Booking & Travel Issues",
        "Delays, cancellations, gate changes, and rebooking for a flight.",
        [
            "Check your airline's app or website first -- gate and delay information updates there fastest.",
            "For a cancelled or significantly delayed flight, you're usually entitled to a free rebooking on the next available flight.",
            "Compensation eligibility (meal vouchers, hotel, refund) depends on the cause and length of the delay -- ask the gate agent or airline support directly.",
            "Keep your boarding pass and any receipts for expenses if you plan to file a compensation claim.",
        ],
        ["topic_1", "flight, flights"],
    ),
    ArticleSpec(
        "topic",
        "topic:2",
        "Gaming Account & Login Issues",
        "Trouble signing in to your gaming account, a locked account, or a forgotten password.",
        [
            "Use the platform's Forgot Password flow first; most login issues are resolved this way.",
            "If your account shows as locked or suspended, check your email for a notice explaining why.",
            "Two-factor authentication issues (lost device, lost backup codes) require identity verification through account recovery.",
            'Avoid sharing your account credentials, even with people offering to "fix" an issue -- this is a common scam vector.',
        ],
        ["topic_2", "xbox, game, account"],
    ),
    ArticleSpec(
        "topic",
        "topic:3",
        "Ride-Share Trip Issues",
        "A problem with a ride -- wrong route, driver cancellation, disputed fare, or a lost item.",
        [
            "Report the issue from the trip receipt in your app within a few days of the ride for the fastest review.",
            "Fare disputes are reviewed against the trip's GPS and time data, not just the receipt total.",
            "For a lost item, use the in-app Lost Item flow to message the driver directly -- this is faster than general support.",
            "Safety-related issues should be reported immediately through the app's dedicated safety report option.",
        ],
        ["topic_3", "uber, driver, ride"],
    ),
    ArticleSpec(
        "topic",
        "topic:4",
        "Train Ticket Issues",
        "Problems with a train ticket -- booking errors, delays, or refund eligibility.",
        [
            "Delayed or cancelled services are usually eligible for a Delay Repay claim through the operator's website.",
            "Booking errors (wrong date, wrong station) can often be corrected online before the travel date for a change fee.",
            "Keep your ticket or booking reference; it's required for any refund or delay claim.",
            "For a missed connection caused by a delay, ask staff about being placed on the next available service at no extra cost.",
        ],
        ["topic_4", "train, trains, ticket"],
    ),
    ArticleSpec(
        "topic",
        "topic:5",
        "Streaming Service Playback Issues",
        "Playback problems -- buffering, songs skipping, or a playlist that won't sync.",
        [
            "Check your internet connection; switch to a lower audio quality setting if streaming on mobile data.",
            "Log out and back in, or reinstall the app, if playback issues persist across multiple songs.",
            "Playlist sync issues usually resolve after forcing a manual sync from the app's settings menu.",
            "If a specific song or album won't play, it may be a regional licensing restriction rather than a technical fault.",
        ],
        ["topic_5", "spotify, music, songs"],
    ),
    ArticleSpec(
        "topic",
        "topic:7",
        "Mobile OS Update Issues",
        "Problems after a phone's operating system update -- apps crashing, battery drain, or a stuck update.",
        [
            "Restart the device after the update completes; many post-update issues clear on the first reboot.",
            "Update individual apps from the app store -- older app versions are the most common cause of post-update crashes.",
            "If the update itself is stuck, ensure at least 20% battery and a stable Wi-Fi connection, then retry.",
            "Battery drain in the first day or two after an update is common while the system re-indexes; it typically settles within 48 hours.",
        ],
        ["topic_7", "ios, iphone, update"],
    ),
    ArticleSpec(
        "topic",
        "topic:8",
        "Internet Outage & Connectivity Issues",
        "No internet connection, or an outage affecting your area.",
        [
            "Check the provider's outage map or status page to see if there's a known outage in your area.",
            "If no outage is reported, power-cycle your router and modem (unplug both for 30 seconds, then plug the modem in first).",
            "Check that all cables are firmly connected, especially after any recent storms or power cuts.",
            "If the issue persists with no reported outage, request a technician visit through your account portal.",
        ],
        ["topic_8", "internet, outage"],
    ),
    ArticleSpec(
        "topic",
        "topic:9",
        "Package Delivery Issues",
        "A package that's missing, delayed, or marked delivered but not received.",
        [
            "Check with neighbors and any safe-drop locations noted in the delivery photo, if one was provided.",
            'Wait 24 hours after a "delivered" status -- some carriers mark packages delivered slightly before the driver\'s final stop.',
            "If it's still missing after 24 hours, file a claim with the carrier using your tracking number.",
            "For a high-value item, the seller or retailer can often open an investigation directly with the carrier on your behalf.",
        ],
        ["topic_9", "package, delivery"],
    ),
    ArticleSpec(
        "topic",
        "topic:10",
        "Checked Baggage Issues",
        "Delayed, damaged, or lost checked baggage.",
        [
            "Report missing or damaged baggage at the airport's baggage service desk before leaving -- this is required to file a claim.",
            "Keep your baggage claim tag; it's needed to track a delayed bag's status.",
            "Delayed bags are usually delivered to your address within 24-48 hours at no cost to you.",
            "For damaged bags or missing contents, file a claim online within the airline's stated claim window (often 7 days).",
        ],
        ["topic_10", "bag, baggage, luggage"],
    ),
    ArticleSpec(
        "topic",
        "topic:11",
        "Card Payment Issues",
        "A declined card, an unrecognized charge, or a card that needs updating.",
        [
            "For a declined card, confirm the card isn't expired and that billing details match what's on file with your bank.",
            "For a charge you don't recognize, check the merchant name against recent orders -- many merchants bill under a different trading name.",
            "If a charge is confirmed unauthorized, contact your card issuer to dispute it and request a replacement card.",
            "Update saved card details under Payment Methods before they expire to avoid failed renewal or subscription charges.",
        ],
        ["topic_11", "card, credit, cards"],
    ),
    ArticleSpec(
        "topic",
        "topic:14",
        "Software Update Issues",
        "An app or software update that fails to install or causes problems afterward.",
        [
            "Ensure you have enough free storage space -- most failed installs are caused by insufficient space.",
            "Restart the device and retry the update over a stable Wi-Fi connection rather than mobile data.",
            "If an update causes a specific feature to break, check for a follow-up patch before rolling back.",
            "As a last resort, uninstall and reinstall the app; back up any local data first if the app supports export.",
        ],
        ["topic_14", "windows, adobe, update"],
    ),
    ArticleSpec(
        "topic",
        "topic:16",
        "TV Channel & Streaming Access Issues",
        "A channel or streaming service that won't load, is missing, or shows an authorization error.",
        [
            "Sign out and back in to refresh your service authorization, especially after a recent plan change.",
            "Check your plan or package to confirm the channel is included -- some channels require an add-on.",
            "Restart your streaming device or set-top box if a channel loads for others but not you.",
            "Live sports and events sometimes have separate regional restrictions from your regular channel package.",
        ],
        ["topic_16", "channel, tv, channels"],
    ),
]


def build_articles() -> list[ArticleSpec]:
    return _INTENT_ARTICLES + _TOPIC_ARTICLES


def persist_articles(session: Session, specs: list[ArticleSpec]) -> int:
    written = 0
    for spec in specs:
        article_id = deterministic_id(
            "kb_article", GENERATOR_VERSION, spec.source_kind, spec.source_key
        )
        body = render_body(spec.intro, spec.steps)
        existing = session.get(KbArticle, article_id)
        if existing is None:
            session.add(
                KbArticle(
                    id=article_id,
                    title=spec.title,
                    body=body,
                    tags=spec.tags,
                    source_kind=spec.source_kind,
                    source_key=spec.source_key,
                    generator_version=GENERATOR_VERSION,
                )
            )
        else:
            existing.title = spec.title
            existing.body = body
            existing.tags = spec.tags
        written += 1
    session.commit()
    return written


def main() -> None:
    specs = build_articles()
    session = SessionLocal()
    try:
        written = persist_articles(session, specs)
        print(f"wrote {written} kb articles ({GENERATOR_VERSION})")
    finally:
        session.close()


if __name__ == "__main__":
    main()
