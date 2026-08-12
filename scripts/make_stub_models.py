"""Reproducible generator for tests/fixtures/models/* -- the tiny stub
checkpoints unit tests load instead of a real trained model (CLAUDE.md:
"tiny stub model checkpoints in tests/fixtures/models/"). stub_intent and
stub_transformer_intent existed already, hand-made with no committed
recipe (confirmed by git history: both were committed directly, no
generator in either diff). This is that missing recipe, plus stub_ner for
M4's TokenClassificationPredictor and stub_sentiment/stub_emotion (+ their
transformer counterparts) for M5's /predict/sentiment and /predict/emotion
routing tests, and stub_transformer_thread_summary (a tiny
T5ForConditionalGeneration, not a classifier) for M6's SummarizationPredictor.

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
from sklearn.svm import LinearSVC
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertForTokenClassification,
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
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

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_VOCAB_WORDS = [
    "this", "is", "great", "love", "it", "awful", "terrible", "hate",
    "okay", "fine", "the", "product", "service", "worst", "best",
]  # fmt: skip

EMOTION_LABELS = ["anger", "joy", "optimism", "sadness"]
EMOTION_VOCAB_WORDS = [
    "so", "angry", "furious", "happy", "excited", "hopeful", "will",
    "get", "better", "sad", "crying", "miss", "this", "about",
]  # fmt: skip

URGENCY_LABELS = ["low", "medium", "high"]
URGENCY_VOCAB_WORDS = [
    "please", "help", "urgent", "lawyer", "refund", "when", "will",
    "this", "be", "fixed", "just", "wondering", "about", "my", "order",
]  # fmt: skip

_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
_CONFIG_KEYS = (
    "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
    "max_position_embeddings", "vocab_size", "id2label", "label2id",
)  # fmt: skip

THREAD_SUMMARY_VOCAB_WORDS = [
    "order", "shipped", "yesterday", "customer", "agent", "help", "please",
    "refund", "account", "was", "the", "my", "is", "will", "you", "charged",
]  # fmt: skip
_T5_SPECIAL_TOKENS = ["<pad>", "</s>", "<unk>"]
_SEQ2SEQ_CONFIG_KEYS = (
    "vocab_size", "d_model", "d_ff", "num_layers", "num_decoder_layers",
    "num_heads", "d_kv", "decoder_start_token_id", "pad_token_id", "eos_token_id",
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


def build_stub_sentiment(out_dir: Path) -> None:
    """TF-IDF + LogisticRegression, 3-class -- matches what
    ml/training/train_baseline_sentiment.py exports."""
    texts = [
        "this is awful", "I hate it", "terrible service", "the worst", "awful experience",
        "it is okay", "fine I guess", "nothing special", "just okay", "an average day",
        "I love it", "this is great", "best service ever", "so happy with this", "great product",
    ]  # fmt: skip
    labels = ["negative"] * 5 + ["neutral"] * 5 + ["positive"] * 5

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    pipeline.fit(texts, labels)

    sanity = pipeline.predict(["I love it"])[0]
    if sanity != "positive":
        raise RuntimeError(f"stub_sentiment sanity check failed: expected positive, got {sanity!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "model.joblib")


def build_stub_transformer_sentiment(out_dir: Path) -> None:
    torch.manual_seed(SEED)
    tokenizer = _build_tokenizer(SENTIMENT_VOCAB_WORDS)
    id2label = dict(enumerate(SENTIMENT_LABELS))
    config = _tiny_bert_config(vocab_size=tokenizer.vocab_size, id2label=id2label)
    model = BertForSequenceClassification(config)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "label_map.json").write_text(
        json.dumps({"labels": SENTIMENT_LABELS}, indent=2), encoding="utf-8"
    )


def build_stub_emotion(out_dir: Path) -> None:
    """TF-IDF + LinearSVC, 4-class -- matches what
    ml/training/train_baseline_emotion.py exports (LinearSVC won that
    script's val comparison on real tweet_eval data; no predict_proba, so
    this stub also exercises BaselinePredictor's decision_function path,
    unlike stub_intent/stub_sentiment)."""
    texts = [
        "so angry right now", "this makes me furious", "I am so mad", "absolutely enraged", "angry about this",
        "so happy today", "this is amazing news", "feeling joyful", "what a great day", "so excited",
        "things will get better", "hopeful about tomorrow", "looking forward to this", "optimistic about it", "better days ahead",
        "feeling so sad", "this makes me cry", "I miss this so much", "sad about the news", "crying about it",
    ]  # fmt: skip
    labels = ["anger"] * 5 + ["joy"] * 5 + ["optimism"] * 5 + ["sadness"] * 5

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LinearSVC(random_state=SEED)),
        ]
    )
    pipeline.fit(texts, labels)

    sanity = pipeline.predict(["so angry right now"])[0]
    if sanity != "anger":
        raise RuntimeError(f"stub_emotion sanity check failed: expected anger, got {sanity!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "model.joblib")


def build_stub_transformer_emotion(out_dir: Path) -> None:
    torch.manual_seed(SEED)
    tokenizer = _build_tokenizer(EMOTION_VOCAB_WORDS)
    id2label = dict(enumerate(EMOTION_LABELS))
    config = _tiny_bert_config(vocab_size=tokenizer.vocab_size, id2label=id2label)
    model = BertForSequenceClassification(config)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "label_map.json").write_text(
        json.dumps({"labels": EMOTION_LABELS}, indent=2), encoding="utf-8"
    )


def build_stub_transformer_urgency(out_dir: Path) -> None:
    """Low/medium/high -- ml/inference/sentiment_trajectory.py's
    resolution-quality formula requires exactly these three labels
    (ml/data/weak_labels.py::UrgencyLabel), unlike stub_transformer_intent's
    unrelated 3-class label set."""
    torch.manual_seed(SEED)
    tokenizer = _build_tokenizer(URGENCY_VOCAB_WORDS)
    id2label = dict(enumerate(URGENCY_LABELS))
    config = _tiny_bert_config(vocab_size=tokenizer.vocab_size, id2label=id2label)
    model = BertForSequenceClassification(config)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "label_map.json").write_text(
        json.dumps({"labels": URGENCY_LABELS}, indent=2), encoding="utf-8"
    )


def _build_t5_tokenizer(words: list[str]) -> PreTrainedTokenizerFast:
    vocab = {token: i for i, token in enumerate(_T5_SPECIAL_TOKENS)}
    for word in words:
        vocab.setdefault(word, len(vocab))
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>", eos_token="</s>"
    )


def _tiny_t5_config(vocab_size: int) -> T5Config:
    return T5Config(
        vocab_size=vocab_size,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_kv=8,
        # T5 convention: the decoder is seeded with pad_token_id, not a
        # dedicated <bos> -- there isn't one in T5's vocabulary.
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )


def build_stub_transformer_thread_summary(out_dir: Path) -> None:
    """T5ForConditionalGeneration -- no label_map.json, unlike every other
    stub_transformer_* fixture: summarization output is free text, not a
    fixed label set. WordLevel (not WordPiece) vocab: T5's real tokenizer is
    SentencePiece unigram, but WordLevel exercises the same encode/generate/
    decode plumbing ml/inference/summarization.py's SummarizationPredictor
    uses without needing a real .model binary."""
    torch.manual_seed(SEED)
    tokenizer = _build_t5_tokenizer(THREAD_SUMMARY_VOCAB_WORDS)
    config = _tiny_t5_config(vocab_size=tokenizer.vocab_size)
    model = T5ForConditionalGeneration(config)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def build_all(fixtures_dir: Path) -> None:
    build_stub_intent(fixtures_dir / "stub_intent")
    build_stub_transformer_intent(fixtures_dir / "stub_transformer_intent")
    build_stub_ner(fixtures_dir / "stub_ner")
    build_stub_sentiment(fixtures_dir / "stub_sentiment")
    build_stub_transformer_sentiment(fixtures_dir / "stub_transformer_sentiment")
    build_stub_emotion(fixtures_dir / "stub_emotion")
    build_stub_transformer_emotion(fixtures_dir / "stub_transformer_emotion")
    build_stub_transformer_urgency(fixtures_dir / "stub_transformer_urgency")
    build_stub_transformer_thread_summary(fixtures_dir / "stub_transformer_thread_summary")


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


def _check_seq2seq_stub(name: str, committed_dir: Path, regenerated_dir: Path) -> list[str]:
    problems: list[str] = []
    if not (committed_dir / "config.json").exists():
        return [f"{name}: no committed fixture at {committed_dir}"]

    committed_config = _load_json(committed_dir / "config.json")
    regenerated_config = _load_json(regenerated_dir / "config.json")
    for key in _SEQ2SEQ_CONFIG_KEYS:
        if committed_config.get(key) != regenerated_config.get(key):
            problems.append(
                f"{name}: config.json[{key}] committed={committed_config.get(key)!r} "
                f"regenerated={regenerated_config.get(key)!r}"
            )

    committed_vocab = _load_json(committed_dir / "tokenizer.json")["model"]["vocab"]  # type: ignore[index]
    regenerated_vocab = _load_json(regenerated_dir / "tokenizer.json")["model"]["vocab"]  # type: ignore[index]
    if committed_vocab != regenerated_vocab:
        problems.append(f"{name}: tokenizer vocab differs")

    return problems


def _check_stub_baseline(name: str, committed_dir: Path, regenerated_dir: Path) -> list[str]:
    if not (committed_dir / "model.joblib").exists():
        return [f"{name}: no committed fixture at {committed_dir}"]
    committed = joblib.load(committed_dir / "model.joblib")
    regenerated = joblib.load(regenerated_dir / "model.joblib")
    problems = []
    if type(committed.named_steps["clf"]) is not type(regenerated.named_steps["clf"]):
        problems.append(f"{name}: classifier type differs")
    if list(committed.named_steps["clf"].classes_) != list(regenerated.named_steps["clf"].classes_):
        problems.append(f"{name}: classifier classes differ")
    return problems


def run_check(fixtures_dir: Path) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        build_all(tmp_dir)

        problems = [
            *_check_stub_baseline(
                "stub_intent", fixtures_dir / "stub_intent", tmp_dir / "stub_intent"
            ),
            *_check_hf_stub(
                "stub_transformer_intent",
                fixtures_dir / "stub_transformer_intent",
                tmp_dir / "stub_transformer_intent",
            ),
            *_check_hf_stub("stub_ner", fixtures_dir / "stub_ner", tmp_dir / "stub_ner"),
            *_check_stub_baseline(
                "stub_sentiment", fixtures_dir / "stub_sentiment", tmp_dir / "stub_sentiment"
            ),
            *_check_hf_stub(
                "stub_transformer_sentiment",
                fixtures_dir / "stub_transformer_sentiment",
                tmp_dir / "stub_transformer_sentiment",
            ),
            *_check_stub_baseline(
                "stub_emotion", fixtures_dir / "stub_emotion", tmp_dir / "stub_emotion"
            ),
            *_check_hf_stub(
                "stub_transformer_emotion",
                fixtures_dir / "stub_transformer_emotion",
                tmp_dir / "stub_transformer_emotion",
            ),
            *_check_hf_stub(
                "stub_transformer_urgency",
                fixtures_dir / "stub_transformer_urgency",
                tmp_dir / "stub_transformer_urgency",
            ),
            *_check_seq2seq_stub(
                "stub_transformer_thread_summary",
                fixtures_dir / "stub_transformer_thread_summary",
                tmp_dir / "stub_transformer_thread_summary",
            ),
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
    print(
        "wrote stub_intent, stub_transformer_intent, stub_ner, stub_sentiment, "
        "stub_transformer_sentiment, stub_emotion, stub_transformer_emotion, "
        "stub_transformer_urgency, stub_transformer_thread_summary "
        f"-> {FIXTURES_DIR}"
    )


if __name__ == "__main__":
    main()
