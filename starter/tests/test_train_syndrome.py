import numpy as np
import pytest

from quantum_lake_student.stages.part2_helpers import ModelResult, choose_threshold
from quantum_lake_student.stages.train import TaskSpec, check_predictions
from quantum_lake_student.stages.train_syndrome import (
    FEATURE_ORDER,
    LABEL_COLUMN,
    WEIGHT_COLUMN,
    _features,
    run_task_a,
)


SPEC_A = TaskSpec("A", "syndrome", LABEL_COLUMN, WEIGHT_COLUMN, run=run_task_a)


def _row(index: int, split: str, rng: np.random.Generator, *, weight: int | None = None) -> dict:
    bits = rng.integers(0, 2, 16)
    # the label mostly follows the first syndrome value, so the model has something to learn
    label = bool(bits[0]) != bool(rng.random() < 0.1)
    return {
        "example_id": f"{split}-{index}",
        "syndrome_bits": bytes(int(bit) for bit in bits),  # one byte per value, as in the ML table
        LABEL_COLUMN: label,
        WEIGHT_COLUMN: int(rng.integers(1, 20)) if weight is None else weight,
        "data_split": split,
    }


def _splits(seed: int = 0, sizes=(200, 60, 60)) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    return {
        name: [_row(i, name, rng) for i in range(size)]
        for name, size in zip(("train", "validation", "test"), sizes)
    }


def _probabilities(result: ModelResult, split: str) -> list[float]:
    return [p["probability"] for p in result.predictions if p["split"] == split]


def test_features_are_the_16_syndrome_values_in_order() -> None:
    bits = [1, 0, 0, 1] + [0] * 11 + [1]
    matrix = _features([{"syndrome_bits": bytes(bits)}])
    assert matrix.shape == (1, 16)
    assert matrix[0].tolist() == bits
    assert FEATURE_ORDER == [f"syndrome_{i:02d}" for i in range(16)]


def test_features_reject_a_wrong_syndrome_length() -> None:
    with pytest.raises(ValueError, match="exactly 16 binary values"):
        _features([{"syndrome_bits": bytes(15)}])


def test_returns_baseline_and_model_covering_exactly_validation_and_test() -> None:
    splits = _splits()
    results = run_task_a(splits, seed=7)
    assert [r.model_id for r in results] == ["task_a_weighted_majority", "task_a_logistic_regression"]
    weights = {row["example_id"]: row[WEIGHT_COLUMN] for rows in splits.values() for row in rows}
    for result in results:
        assert result.task == "A" and result.distance is None
        check_predictions(result, SPEC_A, splits)  # no train rows, no missing/duplicate rows
        # every prediction carries its row's sample_weight, so metrics are weighted
        assert all(p["weight"] == weights[p["example_id"]] for p in result.predictions)


def test_model_records_its_fit_features_and_threshold() -> None:
    model = run_task_a(_splits(), seed=7)[1]
    assert model.fitted_model.n_features_in_ == 16
    assert model.feature_order == FEATURE_ORDER
    assert all(0.0 <= p["probability"] <= 1.0 for p in model.predictions)
    assert all(p["prediction"] == (p["probability"] >= model.threshold) for p in model.predictions)


def test_sample_weight_counts_like_repeated_rows() -> None:
    splits = _splits(sizes=(60, 20, 20))
    weighted = run_task_a(splits, seed=3)[1]
    # the same training data, with each row written out sample_weight times
    repeated_train = [
        {**row, "example_id": f"{row['example_id']}-{copy}", WEIGHT_COLUMN: 1}
        for row in splits["train"]
        for copy in range(row[WEIGHT_COLUMN])
    ]
    repeated = run_task_a({**splits, "train": repeated_train}, seed=3)[1]
    assert _probabilities(weighted, "test") == pytest.approx(_probabilities(repeated, "test"), abs=1e-3)


def test_threshold_is_the_team_rule_on_weighted_validation_rows() -> None:
    splits = _splits()
    model = run_task_a(splits, seed=5)[1]
    validation = [p for p in model.predictions if p["split"] == "validation"]
    expected = choose_threshold(
        [p["label"] for p in validation],
        [p["probability"] for p in validation],
        [p["weight"] for p in validation],
    )
    assert model.threshold == expected


def test_test_labels_change_neither_the_fit_nor_the_threshold() -> None:
    splits = _splits()
    reference = run_task_a(splits, seed=3)[1]
    flipped = {**splits, "test": [{**r, LABEL_COLUMN: not r[LABEL_COLUMN]} for r in splits["test"]]}
    other = run_task_a(flipped, seed=3)[1]
    assert reference.threshold == other.threshold
    assert [p["probability"] for p in reference.predictions] == [p["probability"] for p in other.predictions]


def test_run_is_repeatable_for_a_fixed_seed() -> None:
    first = run_task_a(_splits(), seed=11)[1]
    second = run_task_a(_splits(), seed=11)[1]
    assert first.threshold == second.threshold
    assert [p["probability"] for p in first.predictions] == [p["probability"] for p in second.predictions]


def test_rejects_an_empty_validation_split() -> None:
    splits = _splits(sizes=(20, 0, 5))
    with pytest.raises(ValueError, match="validation rows"):
        run_task_a(splits, seed=1)


def test_only_the_fit_and_the_test_prediction_are_timed(monkeypatch) -> None:
    from quantum_lake_student.stages import train_syndrome as module

    # only two start/stop pairs exist: fit takes 10, test prediction takes 30
    ticks = iter([0.0, 10.0, 20.0, 50.0])
    monkeypatch.setattr(module, "perf_counter", lambda: next(ticks))
    baseline, model = module.run_task_a(_splits(sizes=(30, 10, 10)), seed=2)
    assert model.train_seconds == 10.0
    assert model.predict_seconds == 30.0
    assert baseline.train_seconds is None and baseline.predict_seconds is None
