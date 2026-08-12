"""Fits M7's topic-discovery models from the corpus embeddings artifact
(SPEC M7: "Embed the real-ticket corpus... → BERTopic (UMAP + HDBSCAN)").

Fits BOTH variants in one run, from the same embeddings:

- **KMeans baseline**: MiniBatchKMeans directly on the sentence-transformer
  embedding space (flat, fixed-k clustering) -- CLAUDE.md ground rule #2
  ("baselines before transformers, always... the comparison IS the
  deliverable") applies here even though SPEC M7's text names only
  BERTopic; see docs/decisions.md.
- **BERTopic**: the same embeddings -> UMAP -> HDBSCAN (density clustering,
  produces an outlier cluster and a variable topic count) -> c-TF-IDF
  labels, via the `bertopic` package.

Both variants use c-TF-IDF over the SAME per-ticket documents to derive
each cluster's keyword label, so the comparison isolates the clustering
algorithm, not the labeling method.

Output, per variant, under models/topics_{variant}_v1/:
  topics.json         {"model_version": ..., "topics": [{"topic_key",
                       "label", "keywords", "size"}, ...]}
  assignments.parquet  columns: ticket_id, topic_key, probability
scripts/assign_topics.py reads exactly this pair to write Postgres;
scripts/generate_m7_report.py reads it (plus the embeddings artifact's
documents.parquet) to compute coherence and render the comparison report --
neither of those needs the `topics` dependency group installed, only
fitting does. sentence-transformers/bertopic/umap/hdbscan are lazily
imported inside fit_bertopic() (never at module level) for exactly that
reason -- see docs/decisions.md.

This is a GPU-optional, human-run-locally script (CLAUDE.md ground rule #3)
-- UMAP/HDBSCAN fit on CPU; only the (already-computed) embedding step
benefits from GPU. Never imported by apps/api.

Run (after `make install-topics` and `make embed-tickets`):
  uv run python ml/training/topic_model.py \
      --config ml/training/configs/topics_minilm_bertopic.yaml
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TOP_N_WORDS = 10  # matches ml/evaluation/topic_metrics.py's TOP_N_TERMS
MODELS_DIR = Path("models")


@dataclass
class TopicModelConfig:
    embeddings_path: str = "data/embeddings/tickets_minilm_v1.npy"
    documents_path: str = "data/embeddings/tickets_minilm_v1.parquet"
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    hdbscan_min_cluster_size: int = 75
    hdbscan_min_samples: int = 10
    kmeans_n_clusters: int = 40
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path) -> "TopicModelConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # Same PyYAML scientific-notation gotcha train_transformer.py works
        # around: values like "0.0"/"1e-1" can parse as strings depending
        # on formatting.
        if "umap_min_dist" in data:
            data["umap_min_dist"] = float(data["umap_min_dist"])
        return cls(**data)


@dataclass(frozen=True)
class TopicRecord:
    topic_key: int
    label: str
    keywords: list[str]
    size: int


def _label_from_keywords(keywords: list[str], *, n: int = 4) -> str:
    return ", ".join(keywords[:n]) if keywords else "outliers"


def _ctfidf_keywords(
    documents: list[str], assignments: list[int], topic_key: int, *, top_n: int = TOP_N_WORDS
) -> list[str]:
    """c-TF-IDF over one cluster's member documents vs the whole corpus:
    each cluster is treated as a single pseudo-document, so a term ranks
    high only if it's frequent *within* the cluster and rare *across*
    clusters -- ordinary per-document TF-IDF would just surface generic
    high-frequency words every cluster shares."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    member_docs = [doc for doc, key in zip(documents, assignments, strict=True) if key == topic_key]
    if not member_docs:
        return []

    pseudo_corpus = [" ".join(member_docs), " ".join(documents)]
    vectorizer = TfidfVectorizer(max_features=20_000, stop_words="english", min_df=1)
    tfidf = vectorizer.fit_transform(pseudo_corpus)
    vocab = np.array(vectorizer.get_feature_names_out())
    cluster_scores = np.asarray(tfidf[0].todense()).ravel()
    top_indices = cluster_scores.argsort()[::-1][:top_n]
    return [vocab[i] for i in top_indices if cluster_scores[i] > 0]


def fit_kmeans_baseline(
    embeddings: np.ndarray, documents: list[str], config: TopicModelConfig
) -> tuple[list[int], list[TopicRecord]]:
    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(
        n_clusters=config.kmeans_n_clusters, random_state=config.seed, n_init="auto"
    )
    assignments = [int(t) for t in kmeans.fit_predict(embeddings)]

    records = []
    for topic_key in sorted(set(assignments)):
        keywords = _ctfidf_keywords(documents, assignments, topic_key)
        size = assignments.count(topic_key)
        records.append(TopicRecord(topic_key, _label_from_keywords(keywords), keywords, size))
    return assignments, records


def fit_bertopic(
    embeddings: np.ndarray, documents: list[str], config: TopicModelConfig
) -> tuple[list[int], list[TopicRecord], object]:
    # Lazy import: bertopic/umap/hdbscan live behind the `topics`
    # dependency group (not synced by default or in CI) -- see this
    # module's docstring and pyproject.toml.
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=config.umap_n_neighbors,
        n_components=config.umap_n_components,
        min_dist=config.umap_min_dist,
        metric="cosine",
        random_state=config.seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=TOP_N_WORDS,
        calculate_probabilities=False,
    )
    raw_assignments, _ = topic_model.fit_transform(documents, embeddings=embeddings)
    assignments = [int(t) for t in raw_assignments]

    records = []
    for _, row in topic_model.get_topic_info().iterrows():
        topic_key = int(row["Topic"])
        keywords = [word for word, _score in topic_model.get_topic(topic_key)][:TOP_N_WORDS]
        records.append(
            TopicRecord(topic_key, _label_from_keywords(keywords), keywords, int(row["Count"]))
        )
    return assignments, records, topic_model


def export_variant(
    variant: str,
    ticket_ids: list[str],
    assignments: list[int],
    records: list[TopicRecord],
    probabilities: list[float] | None = None,
) -> Path:
    out_dir = MODELS_DIR / f"topics_{variant}_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_version = f"topics_{variant}_v1"
    (out_dir / "topics.json").write_text(
        json.dumps(
            {"model_version": model_version, "topics": [asdict(r) for r in records]}, indent=2
        ),
        encoding="utf-8",
    )
    probs = probabilities if probabilities is not None else [1.0] * len(ticket_ids)
    pd.DataFrame(
        {"ticket_id": ticket_ids, "topic_key": assignments, "probability": probs}
    ).to_parquet(out_dir / "assignments.parquet")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = TopicModelConfig.from_yaml(args.config)
    embeddings = np.load(config.embeddings_path)
    doc_df = pd.read_parquet(config.documents_path)
    ticket_ids = doc_df["ticket_id"].tolist()
    documents = doc_df["document"].tolist()
    print(f"loaded {len(documents)} documents, embeddings shape={embeddings.shape}")

    print("fitting kmeans baseline...")
    kmeans_assignments, kmeans_records = fit_kmeans_baseline(embeddings, documents, config)
    n_kmeans_topics = len({r.topic_key for r in kmeans_records})
    print(f"  {n_kmeans_topics} clusters")
    kmeans_dir = export_variant("kmeans", ticket_ids, kmeans_assignments, kmeans_records)
    print(f"  wrote {kmeans_dir}")

    print("fitting bertopic...")
    bertopic_assignments, bertopic_records, topic_model = fit_bertopic(
        embeddings, documents, config
    )
    n_bertopic_topics = len({r.topic_key for r in bertopic_records if r.topic_key != -1})
    print(f"  {n_bertopic_topics} topics (+ outlier cluster)")
    bertopic_dir = export_variant("bertopic", ticket_ids, bertopic_assignments, bertopic_records)
    topic_model.save(bertopic_dir / "bertopic_model", serialization="safetensors", save_ctfidf=True)
    print(f"  wrote {bertopic_dir}")


if __name__ == "__main__":
    main()
