from pathlib import Path
from unittest.mock import MagicMock

import pytest
from api.config import Settings

import ml.data.ner.paraphrase as paraphrase_module
from ml.data.ner.markup import render
from ml.data.ner.paraphrase import (
    PassStats,
    _already_paraphrased_ids,
    _candidate_examples,
    _paraphrase_id,
    _try_paraphrase,
    main,
    run_paraphrase_pass,
)
from ml.data.ner.schema import CharSpan, NerExample, read_jsonl, write_jsonl
from ml.inference.llm_client import LLMCallResult, LLMClient


def _example(**overrides: object) -> NerExample:
    defaults: dict[str, object] = {
        "id": "tpl:000001",
        "text": "order ORD-99321 shipped yesterday",
        "entities": [
            CharSpan(6, 15, "ORDER_ID", "ORD-99321"),
            CharSpan(24, 33, "DATE", "yesterday"),
        ],
        "source": "template",
        "split": "train",
        "template_id": "order_shipped_date",
    }
    defaults.update(overrides)
    return NerExample(**defaults)  # type: ignore[arg-type]


class TestParaphraseId:
    def test_prefixes_with_para(self) -> None:
        assert _paraphrase_id("tpl:000001") == "para:tpl:000001"


class TestCandidateExamples:
    def test_only_template_sourced_examples_are_candidates(self, tmp_path: Path) -> None:
        path = tmp_path / "ner_v1.jsonl"
        template_example = _example(id="tpl:1", source="template")
        shell_example = _example(id="shell:m1:wrap:1", source="shell_wrap")
        write_jsonl([template_example, shell_example], path)

        candidates = _candidate_examples(path)

        assert [c.id for c in candidates] == ["tpl:1"]


class TestAlreadyParaphrasedIds:
    def test_empty_set_when_output_does_not_exist(self, tmp_path: Path) -> None:
        assert _already_paraphrased_ids(tmp_path / "does_not_exist.jsonl") == set()

    def test_reads_existing_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "ner_v1_paraphrase.jsonl"
        write_jsonl([_example(id="para:tpl:1")], path)
        assert _already_paraphrased_ids(path) == {"para:tpl:1"}


class TestTryParaphrase:
    def test_valid_markup_accepted_with_correct_offsets(self) -> None:
        example = _example()
        response = "hey your order [ORD-99321|ORDER_ID] went out [yesterday|DATE]"

        result = _try_paraphrase(example, response)

        assert result is not None
        assert result.id == "para:tpl:000001"
        assert result.source == "paraphrase"
        assert result.split == example.split
        assert result.template_id == example.template_id
        for span in result.entities:
            assert result.text[span.start : span.end] == span.text
        assert {(s.label, s.text) for s in result.entities} == {
            ("ORDER_ID", "ORD-99321"),
            ("DATE", "yesterday"),
        }

    def test_dropped_block_rejected(self) -> None:
        example = _example()
        # Missing the DATE block entirely.
        response = "your order [ORD-99321|ORDER_ID] went out yesterday"
        assert _try_paraphrase(example, response) is None

    def test_duplicated_block_rejected(self) -> None:
        example = _example()
        response = (
            "order [ORD-99321|ORDER_ID] went out [yesterday|DATE], "
            "order [ORD-99321|ORDER_ID] confirmed"
        )
        assert _try_paraphrase(example, response) is None

    def test_altered_surface_rejected(self) -> None:
        example = _example()
        response = "order [ORD-00000|ORDER_ID] went out [yesterday|DATE]"
        assert _try_paraphrase(example, response) is None

    def test_malformed_markup_rejected(self) -> None:
        example = _example()
        response = "order [ORD-99321 ORDER_ID] went out [yesterday|DATE]"
        assert _try_paraphrase(example, response) is None

    def test_unknown_label_rejected(self) -> None:
        example = _example()
        response = "order [ORD-99321|NOT_A_LABEL] went out [yesterday|DATE]"
        assert _try_paraphrase(example, response) is None

    def test_round_trip_through_render_is_accepted(self) -> None:
        # The exact shape a well-behaved LLM response looks like: the
        # original markup, verbatim (a trivial "paraphrase").
        example = _example()
        response = render(example.text, example.entities)
        result = _try_paraphrase(example, response)
        assert result is not None
        assert result.text == example.text


class TestRunParaphrasePass:
    def _fake_client(self, responses: list[str], *, cached: bool = False) -> MagicMock:
        client = MagicMock(spec=LLMClient)
        client.complete.side_effect = [
            LLMCallResult(response=r, cached=cached, cost_usd=0.0001) for r in responses
        ]
        client.total_spend_usd.return_value = 0.0002
        return client

    def test_accepts_valid_responses_and_writes_them(self, tmp_path: Path) -> None:
        candidates = [_example(id="tpl:1"), _example(id="tpl:2")]
        responses = [
            "order [ORD-99321|ORDER_ID] shipped [yesterday|DATE]",
            "order [ORD-99321|ORDER_ID] went out [yesterday|DATE]",
        ]
        client = self._fake_client(responses)
        output = tmp_path / "ner_v1_paraphrase.jsonl"

        stats = run_paraphrase_pass(client, candidates, output)

        assert stats.accepted == 2
        assert stats.rejected == 0
        written = read_jsonl(output)
        assert [e.id for e in written] == ["para:tpl:1", "para:tpl:2"]

    def test_rejected_responses_are_counted_and_not_written(self, tmp_path: Path) -> None:
        candidates = [_example(id="tpl:1")]
        client = self._fake_client(["order shipped yesterday, no brackets at all"])
        output = tmp_path / "ner_v1_paraphrase.jsonl"

        stats = run_paraphrase_pass(client, candidates, output)

        assert stats.accepted == 0
        assert stats.rejected == 1
        assert not output.exists()

    def test_mixed_accept_and_reject(self, tmp_path: Path) -> None:
        candidates = [_example(id="tpl:1"), _example(id="tpl:2")]
        responses = [
            "order [ORD-99321|ORDER_ID] shipped [yesterday|DATE]",  # accepted
            "no brackets here",  # rejected
        ]
        client = self._fake_client(responses)
        output = tmp_path / "ner_v1_paraphrase.jsonl"

        stats = run_paraphrase_pass(client, candidates, output)

        assert stats.accepted == 1
        assert stats.rejected == 1

    def test_stops_early_when_client_raises(self, tmp_path: Path) -> None:
        candidates = [_example(id="tpl:1"), _example(id="tpl:2"), _example(id="tpl:3")]
        client = MagicMock(spec=LLMClient)
        client.complete.side_effect = RuntimeError("budget exceeded")
        client.total_spend_usd.return_value = 0.0

        stats = run_paraphrase_pass(client, candidates, tmp_path / "out.jsonl")

        assert stats.accepted == 0
        assert stats.rejected == 0
        assert client.complete.call_count == 1

    def test_counts_cached_hits(self, tmp_path: Path) -> None:
        candidates = [_example(id="tpl:1")]
        client = self._fake_client(
            ["order [ORD-99321|ORDER_ID] shipped [yesterday|DATE]"], cached=True
        )

        stats = run_paraphrase_pass(client, candidates, tmp_path / "out.jsonl")

        assert stats.cached_hits == 1

    def test_returns_total_spend_from_client(self, tmp_path: Path) -> None:
        candidates = [_example(id="tpl:1")]
        client = self._fake_client(["order [ORD-99321|ORDER_ID] shipped [yesterday|DATE]"])

        stats = run_paraphrase_pass(client, candidates, tmp_path / "out.jsonl")

        assert isinstance(stats, PassStats)
        assert stats.total_spend_usd == 0.0002

    def test_no_candidates_no_calls(self, tmp_path: Path) -> None:
        client = MagicMock(spec=LLMClient)
        client.total_spend_usd.return_value = 0.0

        stats = run_paraphrase_pass(client, [], tmp_path / "out.jsonl")

        client.complete.assert_not_called()
        assert stats.accepted == 0
        assert stats.rejected == 0


class TestMainRefusesWhenDisabled:
    def test_llm_disabled_returns_without_touching_session_or_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        disabled_settings = Settings(_env_file=None, llm_enabled=False)  # type: ignore[call-arg]
        monkeypatch.setattr(paraphrase_module, "get_settings", lambda: disabled_settings)

        session_local = MagicMock()
        monkeypatch.setattr(paraphrase_module, "SessionLocal", session_local)

        monkeypatch.setattr(
            "sys.argv",
            ["ner-paraphrase", "--input", str(tmp_path / "ner_v1.jsonl")],
        )

        main()

        session_local.assert_not_called()
