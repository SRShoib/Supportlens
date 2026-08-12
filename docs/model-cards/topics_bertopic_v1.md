# Model card: topics_bertopic_v1

**Task:** unsupervised topic discovery (SPEC M7) -- sentence-transformers embeddings (`sentence-transformers/all-MiniLM-L6-v2`) -> UMAP -> HDBSCAN -> c-TF-IDF labels, via `ml/training/topic_model.py`.

## Data

- Corpus: `TicketSource.TWITTER` tickets only (the real-ticket corpus, SPEC §2) -- Bitext is excluded (synthetic, `created_at` always NULL, see `docs/decisions.md`).
- Document unit: the concatenation of each ticket's CUSTOMER messages only (`scripts/compute_embeddings.py`), not the full agent+customer thread.

## Results

- Topics discovered: **54** (SPEC M7 accept: ≥ 30) + 10132 tickets in the HDBSCAN outlier cluster.
- Mean NPMI coherence: **0.2259** (`ml/evaluation/topic_metrics.py`, top-10 c-TF-IDF terms per topic).

## Top topics

| Topic | Size | Keywords |
|---|---|---|
| food, store, just, chicken | 5347 | food, store, just, chicken, mcdonalds, thanks |
| flight, flights, plane, fly | 4074 | flight, flights, plane, fly, flying, gate |
| xbox, game, account, ps4 | 1862 | xbox, game, account, ps4, password, help |
| uber, driver, ride, drivers | 1744 | uber, driver, ride, drivers, lyft, car |
| train, trains, london, ticket | 1588 | train, trains, london, ticket, euston, tickets |
| spotify, music, songs, song | 1022 | spotify, music, songs, song, playlist, album |
| service, customer, hold, chat | 799 | service, customer, hold, chat, worst, number |
| ios, iphone, update, ios11 | 697 | ios, iphone, update, ios11, 11, apps |

## Limitations

- Unsupervised: no held-out test set, no ground-truth topic labels exist for this corpus -- coherence (NPMI over the model's own corpus) is a proxy for quality, not an accuracy number.
- Twitter-only: never fit or evaluated on Bitext's synthetic utterances.
- Topic count and the outlier cluster's size are sensitive to `ml/training/configs/topics_minilm_bertopic.yaml`'s UMAP/HDBSCAN hyperparameters -- not re-tuned per corpus slice.
- Labels shown above are c-TF-IDF keyword joins, the committed default -- SPEC M7's optional LLM naming pass (`ml/data/llm_topic_labels.py`, default off) may have since overwritten some of them; this card doesn't track that provenance.
