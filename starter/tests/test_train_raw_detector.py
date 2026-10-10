import types

import numpy as np
import pytest

pytest.importorskip("sklearn")

from quantum_lake_student.stages.part2_helpers import ModelResult, choose_threshold
from quantum_lake_student.stages.train import TaskSpec, check_predictions
from quantum_lake_student.stages.train_raw_detector import (
    LABEL_COLUMN,
    _features,
    run_task_c,
)


SPEC_C = TaskSpec("C", "google_bounded", LABEL_COLUMN, None, run=run_task_c)


def _row(index: int, split: str, rng: np.random.Generator, *, detector_count: int = 200) -> dict:
    bits = rng.integers(0, 2, detector_count)
    packed = np.packbits(bits, bitorder="little").tobytes()
    # label depends on the first detector so the MLP has something to learn
    return {
        "example_id": f"{split}-{index}",
        "distance": 3,
        "shot_index": index,
        "detector_count": detector_count,
        "detector_event_count": int(bits.sum()),
        "detector_bits": packed,
        LABEL_COLUMN: bool(bits[0]),
        "data_split": split,
    }


def _splits(seed: int = 0, sizes=(200, 60, 60)) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    return {
        name: [_row(i, name, rng) for i in range(size)]
        for name, size in zip(("train", "validation", "test"), sizes)
    }


def test_features_unpack_little_endian_and_discard_padding() -> None:
    row = {"detector_bits": bytes([0b00000101] + [0] * 24), "detector_count": 200}
    matrix = _features([row])
    assert matrix.shape == (1, 200)
    assert matrix[0, :3].tolist() == [1, 0, 1]
    assert matrix.sum() == 2


def test_features_reject_wrong_packed_length() -> None:
    with pytest.raises(ValueError):
        _features([{"detector_bits": bytes(24), "detector_count": 200}])


def test_rejects_non_distance_three_width() -> None:
    splits = _splits(sizes=(10, 4, 4))
    splits["train"][0] = _row(0, "train", np.random.default_rng(1), detector_count=600)
    with pytest.raises(ValueError, match="200 detector bits"):
        run_task_c(splits, seed=1)


def test_returns_baseline_and_mlp_covering_exactly_validation_and_test() -> None:
    splits = _splits()
    results = run_task_c(splits, seed=7)
    assert [r.model_id for r in results] == ["task_c_d3_majority", "task_c_d3_mlp"]
    for result in results:
        assert isinstance(result, ModelResult)
        assert result.task == "C" and result.distance == 3
        check_predictions(result, SPEC_C, splits)  # no train rows, no missing/duplicate rows
        assert {p["split"] for p in result.predictions} == {"validation", "test"}
        assert all(p["weight"] == 1 for p in result.predictions)


def test_mlp_result_records_model_features_threshold_and_times() -> None:
    mlp = run_task_c(_splits(), seed=7)[1]
    assert mlp.fitted_model is not None
    assert mlp.feature_order == [f"detector_{i}" for i in range(200)]
    assert 0.0 < mlp.threshold < 1.0
    assert mlp.train_seconds is not None and mlp.train_seconds >= 0
    assert mlp.predict_seconds is not None and mlp.predict_seconds >= 0
    assert all(p["probability"] is not None and 0.0 <= p["probability"] <= 1.0 for p in mlp.predictions)


def test_prediction_follows_the_chosen_threshold() -> None:
    mlp = run_task_c(_splits(), seed=7)[1]
    assert all(p["prediction"] == (p["probability"] >= mlp.threshold) for p in mlp.predictions)


def test_run_is_repeatable_for_a_fixed_seed() -> None:
    first = run_task_c(_splits(), seed=11)[1]
    second = run_task_c(_splits(), seed=11)[1]
    assert first.threshold == second.threshold
    assert [p["probability"] for p in first.predictions] == [p["probability"] for p in second.predictions]


def test_training_rows_are_not_predicted_and_test_does_not_change_the_fit() -> None:
    splits = _splits()
    reference = run_task_c(splits, seed=3)[1]
    changed = {**splits, "test": [{**r, LABEL_COLUMN: not r[LABEL_COLUMN]} for r in splits["test"]]}
    other = run_task_c(changed, seed=3)[1]
    # test labels must influence neither the fitted model nor the validation threshold
    assert reference.threshold == other.threshold
    assert [p["probability"] for p in reference.predictions] == [p["probability"] for p in other.predictions]


def test_threshold_is_the_team_rule_applied_to_validation_rows_only() -> None:
    splits = _splits()
    mlp = run_task_c(splits, seed=5)[1]
    validation = [p for p in mlp.predictions if p["split"] == "validation"]
    expected = choose_threshold([p["label"] for p in validation], [p["probability"] for p in validation])
    assert mlp.threshold == expected


def test_event_count_matches_the_set_detector_bits() -> None:
    splits = _splits(sizes=(20, 6, 6))
    matrix = _features(splits["train"])
    assert matrix.sum(axis=1).tolist() == [row["detector_event_count"] for row in splits["train"]]


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_wrong_event_count_is_rejected(split: str) -> None:
    splits = _splits(sizes=(20, 6, 6))
    splits[split][0]["detector_event_count"] += 1
    with pytest.raises(ValueError, match=f"{split} rows have detector_event_count"):
        run_task_c(splits, seed=1)


def test_only_the_fit_and_the_test_prediction_are_timed(monkeypatch) -> None:
    from quantum_lake_student.stages import train_raw_detector as module

    # only two start/stop pairs exist: fit takes 10, test prediction takes 30
    ticks = iter([0.0, 10.0, 20.0, 50.0])
    fake_time = types.SimpleNamespace(perf_counter=lambda: next(ticks))
    monkeypatch.setattr(module, "time", fake_time)
    baseline, mlp = module.run_task_c(_splits(sizes=(30, 10, 10)), seed=2)
    assert mlp.train_seconds == 10.0
    assert mlp.predict_seconds == 30.0
    assert baseline.train_seconds is None and baseline.predict_seconds is None

