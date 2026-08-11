"""Entity value pools for the synthetic NER generator. Every sampler takes
an explicit random.Random (never the module-level `random`) so a run is
reproducible from one seed.

Two hard constraints, unit-tested in tests/unit/test_ner_pools.py:
  1. Every emitted value must be unchanged by ml.data.masking.mask_entities()
     -- an ACCOUNT_REF shaped like a phone number would become <PHONE>.
  2. Every emitted value must appear verbatim, at the expected offset, in
     clean_text(f"prefix {value} suffix") -- e.g. no run of 4+ identical
     characters, which clean_text.collapse_repeats() would shrink to 3.
"""

import random

_DIGITS = "0123456789"
_UPPER_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_UPPER_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _no_run_of_four(chars: list[str], candidate: str) -> bool:
    return not (len(chars) >= 3 and chars[-1] == chars[-2] == chars[-3] == candidate)


def _random_from(rng: random.Random, alphabet: str, n: int) -> str:
    """n characters from alphabet, rejecting any 4th consecutive repeat --
    clean_text.collapse_repeats() would otherwise shrink e.g. "AAAA" to
    "AAA" and silently corrupt the value."""
    chars: list[str] = []
    while len(chars) < n:
        candidate = rng.choice(alphabet)
        if _no_run_of_four(chars, candidate):
            chars.append(candidate)
    return "".join(chars)


def _random_digits(rng: random.Random, n: int) -> str:
    return _random_from(rng, _DIGITS, n)


def _random_upper_alnum(rng: random.Random, n: int) -> str:
    return _random_from(rng, _UPPER_ALNUM, n)


def _random_upper_alpha(rng: random.Random, n: int) -> str:
    return _random_from(rng, _UPPER_ALPHA, n)


# --- ORDER_ID -----------------------------------------------------------
# Procedural, not a fixed list of strings -- otherwise the transformer just
# memorizes ~50 literal ids instead of learning the format families.


def sample_order_id(rng: random.Random) -> str:
    family = rng.choice(("numeric", "ord_prefix", "letter_digit_pair", "carrier_tracking", "amz"))
    if family == "numeric":
        return _random_digits(rng, rng.randint(5, 9))
    if family == "ord_prefix":
        return f"ORD-{_random_digits(rng, 5)}"
    if family == "letter_digit_pair":
        return f"{_random_upper_alpha(rng, 2)}{_random_digits(rng, 4)}-{_random_digits(rng, 4)}"
    if family == "carrier_tracking":
        return f"1Z{_random_upper_alnum(rng, 16)}"
    return f"AMZ-{_random_digits(rng, 3)}-{_random_digits(rng, 7)}"


# --- ACCOUNT_REF ----------------------------------------------------------
# "4455-9911" (a 4-4 digit split) is deliberately never a 3-3-4 split --
# ml.data.masking._PHONE_RE would mask a genuine 3-3-4 shape as <PHONE>.


def sample_account_ref(rng: random.Random) -> str:
    family = rng.choice(("acc_prefix", "digit_dash_digit", "case_prefix", "ref_prefix"))
    if family == "acc_prefix":
        return f"ACC-{_random_digits(rng, 6)}"
    if family == "digit_dash_digit":
        return f"{_random_digits(rng, 4)}-{_random_digits(rng, 4)}"
    if family == "case_prefix":
        return f"CASE{_random_digits(rng, rng.randint(2, 5))}"
    return f"REF{_random_digits(rng, rng.randint(4, 7))}"


def sample_card_last_four(rng: random.Random) -> str:
    return _random_digits(rng, 4)


# --- AMOUNT -----------------------------------------------------------------

_CURRENCY_SYMBOLS = ("$", "£", "€")
_CURRENCY_CODES = ("USD", "EUR", "GBP", "CAD", "AUD")
_SPELLED_CURRENCIES = ("dollars", "bucks", "quid", "pounds", "euros")


def _random_amount_value(rng: random.Random) -> str:
    whole = rng.randint(1, 999)
    cents = rng.randint(0, 99)
    return f"{whole}.{cents:02d}" if rng.random() < 0.7 else str(whole)


def sample_amount(rng: random.Random) -> str:
    family = rng.choice(("symbol", "code", "spelled"))
    value = _random_amount_value(rng)
    if family == "symbol":
        return f"{rng.choice(_CURRENCY_SYMBOLS)}{value}"
    if family == "code":
        return f"{rng.choice(_CURRENCY_CODES)} {value}"
    return f"{value} {rng.choice(_SPELLED_CURRENCIES)}"


# --- DATE -------------------------------------------------------------------
# Sampled ~21% absolute / ~79% relative to match the measured real-corpus
# ratio (docs/decisions.md), and deliberately includes relative phrasing
# ml.inference.rules_ner does not enumerate (a couple of days ago, the other
# day, ...) -- that gap is the headroom the transformer has to win on DATE.

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip
_ORDINAL_SUFFIXES = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd", 31: "st"}
_HOLIDAYS = ("Christmas", "Black Friday", "Cyber Monday", "Thanksgiving", "New Year's Day")
_WEEKDAYS = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
)  # fmt: skip
_CLOSED_LIST_RELATIVE = (
    "yesterday", "today", "tomorrow", "this morning", "this afternoon", "this evening",
    "last week", "last month", "over the weekend",
)  # fmt: skip
# Not matched by ml.inference.rules_ner's DATE rules -- the model's headroom.
_OPEN_CLASS_RELATIVE = (
    "a couple of days ago", "a few weeks back", "the other day", "earlier this week",
    "not long ago", "quite a while back", "the day before yesterday",
    "a little while ago", "just the other week",
)  # fmt: skip
_DURATION_UNITS = ("day", "week", "month")


def _sample_absolute_date(rng: random.Random) -> str:
    family = rng.choice(("month_day", "numeric", "holiday"))
    if family == "holiday":
        return rng.choice(_HOLIDAYS)
    if family == "numeric":
        month_num = rng.randint(1, 12)
        day = rng.randint(1, 28)
        year = rng.randint(2021, 2025)
        return f"{month_num:02d}/{day:02d}/{year}"
    month = rng.choice(_MONTHS)
    day = rng.randint(1, 28)
    suffix = _ORDINAL_SUFFIXES.get(day, "th") if rng.random() < 0.5 else ""
    return f"{month} {day}{suffix}"


def _sample_relative_date(rng: random.Random) -> str:
    family = rng.choice(("closed_list", "last_weekday", "duration", "open_class"))
    if family == "closed_list":
        return rng.choice(_CLOSED_LIST_RELATIVE)
    if family == "last_weekday":
        return f"last {rng.choice(_WEEKDAYS)}"
    if family == "duration":
        n = rng.randint(2, 6)
        unit = rng.choice(_DURATION_UNITS)
        plural = "s" if n != 1 else ""
        return f"{n} {unit}{plural} {rng.choice(('ago', 'back'))}"
    return rng.choice(_OPEN_CLASS_RELATIVE)


def sample_date(rng: random.Random) -> str:
    if rng.random() < 0.21:
        return _sample_absolute_date(rng)
    return _sample_relative_date(rng)


# --- PRODUCT ------------------------------------------------------------
# Mostly a gazetteer (real product names), plus a combinable brand+noun
# generator so training exposure isn't limited to a fixed, memorizable list
# -- PRODUCT is the entity type the rules baseline structurally cannot win.

_NAMED_PRODUCTS = (
    "iPhone 15 Pro Max", "iPhone 14 Pro", "iPhone 13", "iPhone 12 Pro Max", "iPhone SE",
    "iPad Pro", "iPad Air", "MacBook Pro", "MacBook Air", "Apple Watch Series 9", "AirPods Pro",
    "Samsung Galaxy S23", "Samsung Galaxy S22", "Samsung Galaxy Watch", "Galaxy Buds",
    "Kindle Paperwhite", "Kindle Fire", "PlayStation 5", "Xbox Series X", "Xbox Game Pass",
    "Nintendo Switch", "Spotify Premium", "Amazon Prime", "Disney Plus", "Google Pixel 8",
    "Fitbit Charge", "Echo Dot", "Surface Pro", "Delta Comfort+", "Unlimited Plus",
)  # fmt: skip
_BRANDS = (
    "Samsung", "Sony", "LG", "Dell", "HP", "Lenovo", "Bose", "JBL", "Garmin", "Verizon",
    "AT&T", "Comcast", "Delta", "United", "Marriott", "Hilton", "Chase", "Capital One",
)  # fmt: skip
_PRODUCT_NOUNS = (
    "speaker", "laptop", "router", "monitor", "headphones", "smartwatch", "tablet",
    "charger", "plan", "subscription", "account", "card", "membership",
)  # fmt: skip


def sample_product(rng: random.Random) -> str:
    if rng.random() < 0.6:
        return rng.choice(_NAMED_PRODUCTS)
    return f"{rng.choice(_BRANDS)} {rng.choice(_PRODUCT_NOUNS)}"
