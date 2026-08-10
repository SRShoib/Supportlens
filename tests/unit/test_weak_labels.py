from ml.data.weak_labels import compute_urgency_score, weak_label_urgency


class TestLowUrgency:
    def test_plain_informational_question(self) -> None:
        assert weak_label_urgency("can you help me reset my password") == "low"

    def test_short_acknowledgement(self) -> None:
        assert weak_label_urgency("thx!!") == "low"

    def test_neutral_factual_report(self) -> None:
        assert weak_label_urgency("<USER> my phone overheated") == "low"

    def test_positive_message(self) -> None:
        assert weak_label_urgency("Ready for takeoff! Great flight so far.") == "low"

    def test_empty_string(self) -> None:
        assert weak_label_urgency("") == "low"


class TestMediumUrgency:
    def test_still_waiting(self) -> None:
        assert weak_label_urgency("<USER> <USER> still waiting...") == "medium"

    def test_frustration_word(self) -> None:
        assert (
            weak_label_urgency("I don't know what I have to do to remove a damn account")
            == "medium"
        )

    def test_repeated_contact_count(self) -> None:
        assert weak_label_urgency("I sent three DMs about this") == "medium"

    def test_not_working(self) -> None:
        assert weak_label_urgency("the app is not working today") == "medium"

    def test_compounding_medium_signals_reach_high(self) -> None:
        """Multiple medium-strength signals together are more urgent than any
        one alone — this is intentional compounding, not a bug."""
        assert weak_label_urgency("I sent three DMs, please reply") == "high"


class TestHighUrgency:
    def test_legal_threat(self) -> None:
        assert weak_label_urgency("If this isn't fixed I will contact my lawyer") == "high"

    def test_fraud_language(self) -> None:
        assert weak_label_urgency("Someone hacked my account and stole my card details") == "high"

    def test_all_caps_with_punctuation(self) -> None:
        assert weak_label_urgency("WHY IS MY ORDER STILL NOT HERE???") == "high"

    def test_strong_negative_sentiment(self) -> None:
        text = "<USER> <USER> <USER> Yea, this is horrible! foliofirst is stealing our stock!"
        assert weak_label_urgency(text) == "high"

    def test_extreme_repeated_contact(self) -> None:
        assert (
            weak_label_urgency("<USER> <USER> 113th day requesting any kind of response") == "high"
        )

    def test_escalation_to_authority(self) -> None:
        text = "Unbelievable contempt shown. Contact your MP if you don't pay!"
        assert weak_label_urgency(text) == "high"


class TestScoreProperties:
    def test_score_is_monotonic_with_added_signals(self) -> None:
        base = "the service is down"
        assert compute_urgency_score(base) <= compute_urgency_score(base + " this is unacceptable")

    def test_mask_tokens_do_not_inflate_caps_ratio(self) -> None:
        # <USER>/<URL>/<EMAIL> are already uppercase; without stripping them
        # first, a plain low-urgency message with several mentions would
        # score artificially high on the caps-ratio signal alone.
        text = "<USER> <USER> <USER> can you help me with my order"
        assert weak_label_urgency(text) == "low"

    def test_score_never_negative(self) -> None:
        assert compute_urgency_score("") >= 0.0
        assert compute_urgency_score("just a normal message") >= 0.0
