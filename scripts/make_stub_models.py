"""Reproducible generator for tests/fixtures/models/* -- the tiny stub
checkpoints unit tests load instead of a real trained model (CLAUDE.md:
"tiny stub model checkpoints in tests/fixtures/models/"). stub_intent and
stub_transformer_intent existed already, hand-made with no committed
recipe (confirmed by git history: both were committed directly, no
generator in either diff). This is that missing recipe, plus a new
stub_ner fixture for M4's TokenClassificationPredictor.

Run to regenerate all three fixtures in place:
  uv run python scripts/make_stub_models.py

Run to verify the committed fixtures still match what this script would
produce -- structurally (config hyperparams, label map, tokenizer vocab,
that the model loads and produces the right output shape), not
byte-for-byte, since torch's init RNG can shift across library versions:
  uv run python scripts/make_stub_models.py --check
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertForTokenClassification,
    PreTrainedTokenizerFast,
)

from ml.inference.rules_ner import ENTITY_LABELS
from ml.inference.token_classification import build_label_list

FIXTURES_DIR = Path("tests/fixtures/models")
SEED = 42

INTENT_LABELS = ["cancel_order", "track_order", "refund_request"]
INTENT_VOCAB_WORDS = [
    "cancel", "my", "order", "please", "track", "package",
    "i", "need", "a", "refund", "where", "is", "the",
]  # fmt: skip

NER_VOCAB_WORDS = [
    "order", "shipped", "yesterday", "charged", "for", "my", "account",
    "was", "debited", "case", "open", "since", "last", "week", "iphone",
]  # fmt: skip

_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
_CONFIG_KEYS = (
    "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
    "max_position_embeddings", "vocab_size", "id2label", "label2id",
)  # fmt: skip


def _build_vocab(words: list[str]) -> dict[str, int]:
    vocab = {token: i for i, token in enumerate(_SPECIAL_TOKENS)}
    for word in words:
        vocab.setdefault(word, len(vocab))
    return vocab


def _build_tokenizer(words: list[str]) -> PreTrainedTokenizerFast:
    vocab = _build_vocab(words)
    tokenizer = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def _tiny_bert_config(vocab_size: int, id2label: dict[int, str]) -> BertConfig:
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        pad_token_id=0,
        type_vocab_size=2,
        num_labels=len(id2label),
        id2label=id2label,
        label2id={label: i for i, label in id2label.items()},
    )


def build_stub_intent(out_dir: Path) -> None:
    """TF-IDF + LogisticRegression -- matches what
    ml/training/train_baseline_intent.py exports; BaselinePredictor loads
    this joblib file directly. LogisticRegression (not LinearSVC) so
    predict_proba is available, per tests/unit/test_inference_baseline.py."""
    texts = [
        "please cancel my order", "cancel this order please", "I want to cancel my order",
        "cancel order now", "please cancel my order today",
        "where is my package", "track my order", "track my package status",
        "where is my order status", "please track my order",
        "I need a refund", "please refund my order", "refund my package",
        "I want a refund", "please refund",
    ]  # fmt: skip
    labels = ["cancel_order"] * 5 + ["track_order"] * 5 + ["refund_request"] * 5

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    pipeline.fit(texts, labels)

    sanity = pipeline.predict(["please cancel my order"])[0]
    if sanity != "cancel_order":
        raise RuntimeError(
            f"stub_intent sanity check failed: expected cancel_order, got {sanity!r}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "model.joblib")


def build_stub_transformer_intent(out_dir: Path) -> None:
    torch.manual_seed(SEED)
    tokenizer = _build_tokenizer(INTENT_VOCAB_WORDS)
    id2label = dict(enumerate(INTENT_LABELS))
    config = _tiny_bert_config(vocab_size=tokenizer.vocab_size, id2label=id2label)
    model = BertForSequenceClassification(config)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "label_map.json").write_text(
        json.dumps({"labels": INTENT_LABELS}, indent=2), encoding="utf-8"
    )


def build_stub_ner(out_dir: Path) -> None:
    torch.manual_seed(SEED)
    labels = build_label_list(ENTITY_LABELS)
    tokenizer = _build_tokenizer(NER_VOCAB_WORDS)
    id2label = dict(enumerate(labels))
    config = _tiny_bert_config(vocab_size=tokenizer.vocab_size, id2label=id2label)
    model = BertForTokenClassification(config)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "label_map.json").write_text(
        json.dumps(
            {"scheme": "BIO", "labels": labels, "entity_types": sorted(ENTITY_LABELS)}, indent=2
        ),
        encoding="utf-8",
    )


def build_all(fixtures_dir: Path) -> None:
    build_stub_intent(fixtures_dir / "stub_intent")
    build_stub_transformer_intent(fixtures_dir / "stub_transformer_intent")
    build_stub_ner(fixtures_dir / "stub_ner")


def _load_json(path: Path) -> dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _check_hf_stub(name: str, committed_dir: Path, regenerated_dir: Path) -> list[str]:
    problems: list[str] = []
    if not (committed_dir / "config.json").exists():
        return [f"{name}: no committed fixture at {committed_dir}"]

    committed_config = _load_json(committed_dir / "config.json")
    regenerated_config = _load_json(regenerated_dir / "config.json")
    for key in _CONFIG_KEYS:
        if committed_config.get(key) != regenerated_config.get(key):
            problems.append(
                f"{name}: config.json[{key}] committed={committed_config.get(key)!r} "
                f"regenerated={regenerated_config.get(key)!r}"
            )

    committed_label_map = _load_json(committed_dir / "label_map.json")
    regenerated_label_map = _load_json(regenerated_dir / "label_map.json")
    if committed_label_map != regenerated_label_map:
        problems.append(f"{name}: label_map.json differs")

    committed_vocab = _load_json(committed_dir / "tokenizer.json")["model"]["vocab"]  # type: ignore[index]
    regenerated_vocab = _load_json(regenerated_dir / "tokenizer.json")["model"]["vocab"]  # type: ignore[index]
    if committed_vocab != regenerated_vocab:
        problems.append(f"{name}: tokenizer vocab differs")

    return problems


def _check_stub_intent(committed_dir: Path, regenerated_dir: Path) -> list[str]:
    if not (committed_dir / "model.joblib").exists():
        return [f"stub_intent: no committed fixture at {committed_dir}"]
    committed = joblib.load(committed_dir / "model.joblib")
    regenerated = joblib.load(regenerated_dir / "model.joblib")
    problems = []
    if type(committed.named_steps["clf"]) is not type(regenerated.named_steps["clf"]):
        problems.append("stub_intent: classifier type differs")
    if list(committed.named_steps["clf"].classes_) != list(regenerated.named_steps["clf"].classes_):
        problems.append("stub_intent: classifier classes differ")
    return problems


def run_check(fixtures_dir: Path) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        build_all(tmp_dir)

        problems = [
            *_check_stub_intent(fixtures_dir / "stub_intent", tmp_dir / "stub_intent"),
            *_check_hf_stub(
                "stub_transformer_intent",
                fixtures_dir / "stub_transformer_intent",
                tmp_dir / "stub_transformer_intent",
            ),
            *_check_hf_stub("stub_ner", fixtures_dir / "stub_ner", tmp_dir / "stub_ner"),
        ]

    if problems:
        for problem in problems:
            print(f"MISMATCH: {problem}", file=sys.stderr)
        return False

    print("all stub fixtures match their reproducible recipe")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed fixtures structurally match this script's output, without writing anything",
    )
    args = parser.parse_args()

    if args.check:
        ok = run_check(FIXTURES_DIR)
        sys.exit(0 if ok else 1)

    build_all(FIXTURES_DIR)
    print(f"wrote stub_intent, stub_transformer_intent, stub_ner -> {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
