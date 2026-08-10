import argparse
import json
import sys
from pathlib import Path

from api.config import get_settings
from api.db.session import SessionLocal

from ml.data.loaders import bitext, twitter
from ml.data.persist import persist_tickets
from ml.data.slice_builder import SliceConfig, build_slice, load_slice, save_slice

_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _ingest_bitext(args: argparse.Namespace) -> None:
    session = SessionLocal()
    try:
        count = persist_tickets(session, bitext.iter_tickets())
        print(f"ingested {count} bitext tickets")
    finally:
        session.close()


def _ingest_twitter(args: argparse.Namespace) -> None:
    settings = get_settings()
    csv_path = Path(args.csv_path or settings.twcs_csv_path)
    slice_path = Path(args.slice_path)

    selected_roots = load_slice(slice_path) if slice_path.exists() else None
    if selected_roots is None:
        print(
            f"warning: no slice manifest at {slice_path}; ingesting the FULL file", file=sys.stderr
        )

    session = SessionLocal()
    try:
        count = persist_tickets(
            session, twitter.iter_tickets(csv_path, selected_roots=selected_roots)
        )
        print(f"ingested {count} twitter tickets")
    finally:
        session.close()


def _build_slice(args: argparse.Namespace) -> None:
    settings = get_settings()
    csv_path = Path(args.csv_path or settings.twcs_csv_path)
    slice_path = Path(args.slice_path)

    conversations = twitter.build_conversations(csv_path)
    config = SliceConfig(
        target_messages=args.target_messages or settings.slice_target_messages,
        max_brand_share=args.max_brand_share or settings.slice_max_brand_share,
        seed=settings.random_seed,
        lang_confidence_threshold=settings.lang_confidence_threshold,
    )
    result = build_slice(conversations, config)
    save_slice(result, slice_path)
    total = sum(result.brand_message_counts.values())
    print(f"selected {len(result.selected_roots)} conversations, {total} messages -> {slice_path}")


def _seed(args: argparse.Namespace) -> None:
    bitext_fixture = _FIXTURES_DIR / "bitext_sample.jsonl"
    twcs_fixture = _FIXTURES_DIR / "twcs_sample.csv"

    with bitext_fixture.open(encoding="utf-8") as f:
        bitext_rows = [json.loads(line) for line in f]

    session = SessionLocal()
    try:
        n_bitext = persist_tickets(session, bitext.iter_tickets(rows=bitext_rows))
        n_twitter = persist_tickets(session, twitter.iter_tickets(twcs_fixture))
        print(f"seeded {n_bitext} bitext + {n_twitter} twitter demo tickets")
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="ml.data.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest-bitext").set_defaults(func=_ingest_bitext)

    ingest_twitter_parser = subparsers.add_parser("ingest-twitter")
    ingest_twitter_parser.add_argument("--csv-path", default=None)
    ingest_twitter_parser.add_argument(
        "--slice-path", default="data/splits/twitter_slice_v1.parquet"
    )
    ingest_twitter_parser.set_defaults(func=_ingest_twitter)

    build_slice_parser = subparsers.add_parser("build-slice")
    build_slice_parser.add_argument("--csv-path", default=None)
    build_slice_parser.add_argument("--slice-path", default="data/splits/twitter_slice_v1.parquet")
    build_slice_parser.add_argument("--target-messages", type=int, default=None)
    build_slice_parser.add_argument("--max-brand-share", type=float, default=None)
    build_slice_parser.set_defaults(func=_build_slice)

    subparsers.add_parser("seed").set_defaults(func=_seed)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
