from ml.evaluation.latency import LatencyResult


def test_to_metrics_dict_round_trips_every_field() -> None:
    result = LatencyResult(n_runs=20, mean_ms=12.5, p50_ms=11.0, p95_ms=18.0, max_ms=22.0)

    assert result.to_metrics_dict() == {
        "n_runs": 20,
        "mean_ms": 12.5,
        "p50_ms": 11.0,
        "p95_ms": 18.0,
        "max_ms": 22.0,
    }
