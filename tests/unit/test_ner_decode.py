from ml.inference.token_classification import decode_spans

LABELS = ["O", "B-ORDER_ID", "I-ORDER_ID", "B-DATE", "I-DATE"]


def test_simple_b_i_sequence_merges_into_one_span() -> None:
    text = "order ORD-99321 shipped"
    offsets = [(0, 0), (6, 9), (9, 15), (0, 0)]
    label_ids = [-100, 1, 2, -100]  # special B-ORDER_ID I-ORDER_ID special
    scores = [0.0, 0.9, 0.8, 0.0]

    spans = decode_spans(text, offsets, label_ids, scores, LABELS)

    assert len(spans) == 1
    assert spans[0].label == "ORDER_ID"
    assert spans[0].text == "ORD-99321"
    assert spans[0].start == 6
    assert spans[0].end == 15
    assert spans[0].score == (0.9 + 0.8) / 2


def test_b_b_same_type_splits_into_two_spans() -> None:
    text = "abc def"
    offsets = [(0, 3), (4, 7)]
    label_ids = [1, 1]  # B-ORDER_ID B-ORDER_ID
    scores = [0.9, 0.9]

    spans = decode_spans(text, offsets, label_ids, scores, LABELS)

    assert len(spans) == 2
    assert [s.text for s in spans] == ["abc", "def"]


def test_bare_i_with_nothing_open_still_opens_a_span() -> None:
    text = "abc"
    offsets = [(0, 3)]
    label_ids = [2]  # I-ORDER_ID, no preceding B-
    scores = [0.7]

    spans = decode_spans(text, offsets, label_ids, scores, LABELS)

    assert len(spans) == 1
    assert spans[0].label == "ORDER_ID"
    assert spans[0].text == "abc"


def test_o_closes_an_open_span() -> None:
    text = "abc def ghi"
    offsets = [(0, 3), (4, 7), (8, 11)]
    label_ids = [1, 0, 1]  # B-ORDER_ID O B-ORDER_ID
    scores = [0.9, 0.9, 0.9]

    spans = decode_spans(text, offsets, label_ids, scores, LABELS)

    assert len(spans) == 2
    assert [s.text for s in spans] == ["abc", "ghi"]


def test_different_type_i_tag_closes_and_opens_new_span() -> None:
    text = "abc def"
    offsets = [(0, 3), (4, 7)]
    label_ids = [1, 4]  # B-ORDER_ID then I-DATE (different type, no B-DATE)
    scores = [0.9, 0.8]

    spans = decode_spans(text, offsets, label_ids, scores, LABELS)

    assert len(spans) == 2
    assert spans[0].label == "ORDER_ID"
    assert spans[1].label == "DATE"


def test_special_tokens_skipped_via_zero_width_offset() -> None:
    text = "abc"
    offsets = [(0, 0), (0, 3), (0, 0)]
    label_ids = [-100, 1, -100]
    scores = [0.0, 0.9, 0.0]

    spans = decode_spans(text, offsets, label_ids, scores, LABELS)

    assert len(spans) == 1
    assert spans[0].text == "abc"


def test_out_of_range_label_id_treated_as_closing_special() -> None:
    text = "abc def"
    offsets = [(0, 3), (4, 7)]
    label_ids = [1, -100]  # B-ORDER_ID then padding
    scores = [0.9, 0.0]

    spans = decode_spans(text, offsets, label_ids, scores, LABELS)

    assert len(spans) == 1
    assert spans[0].text == "abc"


def test_no_entities_returns_empty_list() -> None:
    text = "nothing here"
    offsets = [(0, 7), (8, 12)]
    label_ids = [0, 0]
    scores = [0.9, 0.9]

    assert decode_spans(text, offsets, label_ids, scores, LABELS) == []


def test_score_is_mean_of_per_token_scores_across_the_span() -> None:
    text = "abcdefghi"
    offsets = [(0, 3), (3, 6), (6, 9)]
    label_ids = [1, 2, 2]  # B-ORDER_ID I-ORDER_ID I-ORDER_ID
    scores = [1.0, 0.5, 0.0]

    spans = decode_spans(text, offsets, label_ids, scores, LABELS)

    assert spans[0].score == 0.5


def test_every_returned_span_offset_matches_text() -> None:
    text = "order ORD-99321 shipped yesterday, ref ACC-1"
    offsets = [(0, 0), (6, 9), (9, 15), (16, 23), (24, 33), (40, 45), (0, 0)]
    label_ids = [-100, 1, 2, 0, 3, 3, -100]
    scores = [0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0]

    for span in decode_spans(text, offsets, label_ids, scores, LABELS):
        assert text[span.start : span.end] == span.text


def test_trims_whitespace_defensively() -> None:
    # WordPiece offsets shouldn't include surrounding whitespace, but the
    # trim is defensive in case a future tokenizer swap does include it.
    text = "abc def"
    offsets = [(0, 4)]  # deliberately includes the trailing space
    label_ids = [1]
    scores = [0.9]

    spans = decode_spans(text, offsets, label_ids, scores, LABELS)

    assert spans[0].text == "abc"
    assert spans[0].end == 3
