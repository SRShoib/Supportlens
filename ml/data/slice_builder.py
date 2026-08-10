import json
import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ml.data.loaders.twitter import ConversationSummary

_UNKNOWN_BRAND = "__unknown__"


@dataclass(frozen=True)
class SliceConfig:
    target_messages: int = 150_000
    max_brand_share: float = 0.08
    seed: int = 42


@dataclass(frozen=True)
class SliceResult:
    selected_roots: frozenset[str]
    config: SliceConfig
    brand_message_counts: dict[str, int] = field(default_factory=dict)


def _week_bucket(summary: ConversationSummary) -> str:
    if summary.start_time is None:
        return "unknown"
    iso = summary.start_time.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _capped_shares(natural_shares: Mapping[str, float], cap: float) -> dict[str, float]:
    """Water-filling: brands over the cap get exactly `cap`; the excess is
    redistributed proportionally among the rest, iterating until stable."""
    remaining_brands = set(natural_shares)
    remaining_total_share = 1.0
    result: dict[str, float] = {}

    while remaining_brands:
        remaining_natural_sum = sum(natural_shares[b] for b in remaining_brands)
        if remaining_natural_sum <= 0:
            even = remaining_total_share / len(remaining_brands)
            result.update(dict.fromkeys(remaining_brands, even))
            break

        scale = remaining_total_share / remaining_natural_sum
        over_cap = {b for b in remaining_brands if natural_shares[b] * scale > cap}
        if not over_cap:
            result.update({b: natural_shares[b] * scale for b in remaining_brands})
            break

        result.update(dict.fromkeys(over_cap, cap))
        remaining_total_share -= cap * len(over_cap)
        remaining_brands -= over_cap

    return result


def build_slice(
    conversations: Mapping[str, ConversationSummary], config: SliceConfig | None = None
) -> SliceResult:
    """Sample whole conversations, brand-capped and time-spread. Brand share of the
    message target is min(natural_share, max_brand_share) with the remainder
    redistributed to the long tail; within a brand, conversations are drawn evenly
    across weekly buckets so the slice spans the full corpus date range."""
    config = config or SliceConfig()
    by_brand: dict[str, list[ConversationSummary]] = defaultdict(list)
    for summary in conversations.values():
        by_brand[summary.brand or _UNKNOWN_BRAND].append(summary)

    total_messages = sum(s.message_count for s in conversations.values())
    if total_messages == 0:
        return SliceResult(selected_roots=frozenset(), config=config)

    natural_shares = {
        brand: sum(s.message_count for s in items) / total_messages
        for brand, items in by_brand.items()
    }
    capped_shares = _capped_shares(natural_shares, config.max_brand_share)

    rng = random.Random(config.seed)
    selected: set[str] = set()
    brand_message_counts: dict[str, int] = {}

    for brand in sorted(by_brand):
        brand_target = round(capped_shares[brand] * config.target_messages)
        by_root = {s.root_id: s for s in by_brand[brand]}

        buckets: dict[str, list[str]] = defaultdict(list)
        for summary in by_brand[brand]:
            buckets[_week_bucket(summary)].append(summary.root_id)
        for week in buckets:
            rng.shuffle(buckets[week])

        week_order = sorted(buckets)
        cursors = dict.fromkeys(week_order, 0)
        running = 0

        while running < brand_target:
            progressed = False
            for week in week_order:
                if running >= brand_target:
                    break
                idx = cursors[week]
                if idx >= len(buckets[week]):
                    continue
                root_id = buckets[week][idx]
                cursors[week] += 1
                progressed = True
                selected.add(root_id)
                running += by_root[root_id].message_count
            if not progressed:
                break

        brand_message_counts[brand] = running

    return SliceResult(
        selected_roots=frozenset(selected), config=config, brand_message_counts=brand_message_counts
    )


def save_slice(result: SliceResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"root_id": sorted(result.selected_roots)})
    df.to_parquet(path, engine="pyarrow", index=False)

    meta_path = path.with_suffix(path.suffix + ".json")
    meta_path.write_text(
        json.dumps(
            {
                "target_messages": result.config.target_messages,
                "max_brand_share": result.config.max_brand_share,
                "seed": result.config.seed,
                "num_conversations": len(result.selected_roots),
                "brand_message_counts": result.brand_message_counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_slice(path: Path) -> frozenset[str]:
    df = pd.read_parquet(path, engine="pyarrow")
    return frozenset(df["root_id"].astype(str))
