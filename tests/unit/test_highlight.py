from ml.inference.highlight import HighlightSpan, highlight_matches


def test_highlights_case_insensitive_query_term_matches() -> None:
    document = "Where is my Order? I need to Track it."

    spans = highlight_matches("track my order", document)

    highlighted_words = {document[s.start : s.end] for s in spans}
    assert highlighted_words == {"Order", "Track"}


def test_excludes_stopwords_from_query_terms() -> None:
    spans = highlight_matches("how do I track my order", "my order shipped")

    # "my" is a stopword and shouldn't be highlighted even though it appears
    # in both query and document; "order" should be.
    document = "my order shipped"
    highlighted_words = {document[s.start : s.end] for s in spans}
    assert highlighted_words == {"order"}


def test_query_of_only_stopwords_returns_no_spans() -> None:
    assert highlight_matches("the a is", "the order is ready") == []


def test_no_overlap_returns_no_spans() -> None:
    assert highlight_matches("refund", "totally unrelated text here") == []


def test_span_offsets_index_into_the_exact_document_passed_in() -> None:
    document = "order 12345 shipped"
    spans = highlight_matches("order", document)

    assert spans == [HighlightSpan(0, 5)]
    assert document[spans[0].start : spans[0].end] == "order"
