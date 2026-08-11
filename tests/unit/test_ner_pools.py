import random

import pytest

from ml.data.cleaning import clean_text
from ml.data.masking import MaskToken, mask_entities
from ml.data.ner.pools import (
    sample_account_ref,
    sample_amount,
    sample_card_last_four,
    sample_date,
    sample_order_id,
    sample_product,
)

_SAMPLERS = {
    "order_id": sample_order_id,
    "account_ref": sample_account_ref,
    "amount": sample_amount,
    "date": sample_date,
    "product": sample_product,
    "card_last_four": sample_card_last_four,
}
N_SAMPLES = 300


@pytest.mark.parametrize("name", list(_SAMPLERS))
def test_sampled_values_survive_masking_unchanged(name: str) -> None:
    sampler = _SAMPLERS[name]
    rng = random.Random(42)
    for _ in range(N_SAMPLES):
        value = sampler(rng)
        assert mask_entities(value) == value, f"{name} value {value!r} was masked"


@pytest.mark.parametrize("name", list(_SAMPLERS))
def test_sampled_values_survive_clean_text_in_context_at_expected_offset(name: str) -> None:
    sampler = _SAMPLERS[name]
    rng = random.Random(42)
    for _ in range(N_SAMPLES):
        value = sampler(rng)
        text = f"prefix {value} suffix"
        cleaned = clean_text(text)
        assert cleaned == text, f"{name} value {value!r} changed under clean_text: {cleaned!r}"
        start = text.index(value)
        assert cleaned[start : start + len(value)] == value


@pytest.mark.parametrize("name", list(_SAMPLERS))
def test_sampled_values_never_have_a_run_of_four_identical_characters(name: str) -> None:
    sampler = _SAMPLERS[name]
    rng = random.Random(7)
    for _ in range(N_SAMPLES):
        value = sampler(rng)
        for i in range(len(value) - 3):
            assert len(set(value[i : i + 4])) > 1, f"{name} value {value!r} has a 4-char run"


def test_order_id_and_account_ref_never_look_like_a_phone_number() -> None:
    # ml.data.masking._PHONE_RE matches a 3-3-4 digit grouping; neither
    # sampler should ever accidentally produce that exact shape.
    rng = random.Random(123)
    for sampler in (sample_order_id, sample_account_ref):
        for _ in range(N_SAMPLES):
            value = sampler(rng)
            assert MaskToken.PHONE.value not in mask_entities(f"contact info: {value}")


def test_sample_order_id_is_deterministic_given_seed() -> None:
    first = [sample_order_id(random.Random(42)) for _ in range(20)]
    second = [sample_order_id(random.Random(42)) for _ in range(20)]
    assert first == second


def test_sample_date_matches_measured_absolute_relative_ratio_roughly() -> None:
    # SPEC/docs/decisions.md: ~21% absolute / ~79% relative, matching the
    # measured real-corpus ratio. Not exact -- just not inverted or 50/50.
    rng = random.Random(42)
    absolute_markers = ("/", "January", "February", "March", "April", "May", "June", "July",
                         "August", "September", "October", "November", "December",
                         "Christmas", "Black Friday", "Cyber Monday", "Thanksgiving",
                         "New Year's Day")  # fmt: skip
    n = 2000
    n_absolute = sum(1 for _ in range(n) if any(m in sample_date(rng) for m in absolute_markers))
    ratio = n_absolute / n
    assert 0.10 < ratio < 0.35
