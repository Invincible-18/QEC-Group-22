import numpy as np
import pytest

from quantum_lake_student.ml import GOOGLE_META_PREDICTION_COLUMNS
from quantum_lake_student.stages.part2_helpers import ModelResult, prediction_rows
from quantum_lake_student.stages.train import TaskSpec, check_predictions
from quantum_lake_student.stages.train_google import (
    FEATURE_ORDER,
    LABEL_COLUMN,
    THRESHOLDS,
    _choose_threshold,
    _features,
    decoder_error_overlap,
    run_task_b,
)


SPEC_B = TaskSpec("B", "google", LABEL_COLUMN, None, run=run_task_b)
DETECTOR_COUNT = {3: 200, 5: 600}
FLIP_SHARE = {3: 0.7, 5: 0.3}
MODEL_NAMES = [
    "majority",
    "belief_matching",
    "correlated_matching",
    "pymatching",
    "tensor_network_contraction",
    "combined",
]


def _row(distance: int, split: str, index: int, rng: np.random.Generator) -> dict:
    flip = bool(rng.random() < FLIP_SHARE[distance])
    row = {
        "example_id": f"d{distance}-{split}-{index}",
        "distance": distance,
        "detector_count": DETECTOR_COUNT[distance],
        "detector_event_count": int(rng.integers(0, DETECTOR_COUNT[distance] // 4)),
        LABEL_COLUMN: flip,
        "data_split": split,
    }
    for column in GOOGLE_META_PREDICTION_COLUMNS:
        row[column] = flip if rng.random() < 0.8 else not flip
    return row


def _splits(seed: int = 0, sizes=(80, 30, 30)) -> dict[str, list[dict]]:
    """Made-up rows with distances 3 and 5 interleaved, like the real input."""
    rng = np.random.default_rng(seed)
    return {
        split: [_row(distance, split, i, rng) for i in range(size) for distance in (3, 5)]
        for split, size in zip(("train", "validation", "test"), sizes)
    }


def test_returns_six_models_per_distance() -> None:
    results = run_task_b(_splits(), seed=1)
    assert [r.model_id for r in results] == [
        f"task_b_d{distance}_{name}" for distance in (3, 5) for name in MODEL_NAMES
    ]
    assert all(r.task == "B" for r in results)
    assert [r.distance for r in results] == [3] * 6 + [5] * 6


def test_each_model_covers_exactly_its_own_distance() -> None:
    splits = _splits()
    for result in run_task_b(splits, seed=1):
        check_predictions(result, SPEC_B, splits)
        assert all(p["example_id"].startswith(f"d{result.distance}-") for p in result.predictions)


def test_baseline_uses_only_its_own_distance_training_rows() -> None:
    splits = _splits()
    results = {r.model_id: r for r in run_task_b(splits, seed=1)}
    shares = {}
    for distance in (3, 5):
        train = [row for row in splits["train"] if row["distance"] == distance]
        shares[distance] = sum(row[LABEL_COLUMN] for row in train) / len(train)
        baseline = results[f"task_b_d{distance}_majority"]
        assert all(p["probability"] == pytest.approx(shares[distance]) for p in baseline.predictions)
        assert all(p["prediction"] == (shares[distance] >= 0.5) for p in baseline.predictions)
    assert shares[3] > 0.5 > shares[5]


def test_supplied_decoders_are_passed_through_unchanged() -> None:
    splits = _splits()
    rows = {row["example_id"]: row for split in ("validation", "test") for row in splits[split]}
    results = {r.model_id: r for r in run_task_b(splits, seed=1)}
    for distance in (3, 5):
        for column in GOOGLE_META_PREDICTION_COLUMNS:
            result = results[f"task_b_d{distance}_{column.removesuffix('_prediction')}"]
            assert result.fitted_model is None and result.threshold is None
            for p in result.predictions:
                assert p["prediction"] == rows[p["example_id"]][column]
                assert p["probability"] is None


def test_features_are_the_helper_inputs_in_feature_order() -> None:
    row = {
        "detector_count": 200,
        "detector_event_count": 30,
        "belief_matching_prediction": True,
        "correlated_matching_prediction": False,
        "pymatching_prediction": True,
        "tensor_network_contraction_prediction": False,
    }
    assert _features([row]).tolist() == [[0.15, 1.0, 0.0, 1.0, 0.0]]
    assert FEATURE_ORDER == ["detector_event_density", *GOOGLE_META_PREDICTION_COLUMNS]


def test_combined_model_records_its_fit_threshold_and_times() -> None:
    combined = run_task_b(_splits(), seed=1)[5]
    assert combined.model_id == "task_b_d3_combined"
    assert combined.fitted_model.n_features_in_ == 5
    assert combined.feature_order == FEATURE_ORDER
    assert combined.threshold in THRESHOLDS
    assert combined.train_seconds >= 0 and combined.predict_seconds >= 0
    for p in combined.predictions:
        assert 0.0 <= p["probability"] <= 1.0
        assert p["prediction"] == (p["probability"] >= combined.threshold)


def test_test_labels_change_neither_the_fit_nor_the_threshold() -> None:
    splits = _splits()
    flipped = {
        **splits,
        "test": [{**row, LABEL_COLUMN: not row[LABEL_COLUMN]} for row in splits["test"]],
    }
    for before, after in zip(run_task_b(splits, seed=1), run_task_b(flipped, seed=1)):
        assert before.threshold == after.threshold
        assert [p["probability"] for p in before.predictions] == [
            p["probability"] for p in after.predictions
        ]


def test_threshold_is_the_best_balanced_accuracy_nearest_one_half() -> None:
    labels = [False, False, True, True]
    # every threshold in (0.2, 0.7] separates the two classes; 0.5 is nearest one half
    assert _choose_threshold(labels, np.array([0.1, 0.2, 0.7, 0.8])) == 0.5
    # other validation probabilities move it
    assert _choose_threshold(labels, np.array([0.1, 0.6, 0.85, 0.9])) == 0.61


def test_run_is_repeatable() -> None:
    first, second = run_task_b(_splits(), seed=1), run_task_b(_splits(), seed=1)
    assert [r.predictions for r in first] == [r.predictions for r in second]
    assert [r.threshold for r in first] == [r.threshold for r in second]


def test_rejects_an_unexpected_or_missing_distance() -> None:
    splits = _splits()
    extra = {**splits, "train": [*splits["train"], {**splits["train"][0], "distance": 7}]}
    with pytest.raises(ValueError, match="expects distances"):
        run_task_b(extra, seed=1)
    only_three = {split: [r for r in rows if r["distance"] == 3] for split, rows in splits.items()}
    with pytest.raises(ValueError, match="expects distances"):
        run_task_b(only_three, seed=1)


def test_rejects_a_distance_with_an_empty_split() -> None:
    splits = _splits()
    splits["validation"] = [r for r in splits["validation"] if r["distance"] == 3]
    with pytest.raises(ValueError, match="distance 5 has no validation rows"):
        run_task_b(splits, seed=1)


def test_decoder_error_overlap_counts_shared_mistakes_on_test_shots_only() -> None:
    answers = {
        "belief_matching": [True, False, True, False],
        "correlated_matching": [True, False, True, False],
        "pymatching": [True, False, False, True],
        "tensor_network_contraction": [True, False, True, True],
        "combined": [True, False, True, False],
    }
    test_rows = [{"example_id": shot, LABEL_COLUMN: True} for shot in "abcd"]
    validation_rows = [{"example_id": "v", LABEL_COLUMN: True}]
    results = []
    for name, predictions in answers.items():
        model_id = f"task_b_d3_{name}"
        rows = prediction_rows(
            validation_rows,
            split="validation",
            model_id=model_id,
            label_column=LABEL_COLUMN,
            predictions=[False],
        )
        rows += prediction_rows(
            test_rows,
            split="test",
            model_id=model_id,
            label_column=LABEL_COLUMN,
            predictions=predictions,
        )
        results.append(ModelResult(model_id=model_id, task="B", distance=3, predictions=rows))

    overlap = decoder_error_overlap(results)
    assert list(overlap) == ["d3"]
    d3 = overlap["d3"]
    assert d3["test_shots"] == 4
    assert d3["all_four_right"] == 1 and d3["all_four_wrong"] == 1 and d3["decoders_disagree"] == 2
    assert d3["error_rate_where_decoders_disagree"] == {
        "belief_matching": 0.5,
        "correlated_matching": 0.5,
        "pymatching": 0.5,
        "tensor_network_contraction": 0.0,
        "combined": 0.5,
    }
    assert d3["combined_wrong_where_all_four_right"] == 0
    assert d3["combined_wrong_where_all_four_wrong"] == 1
    assert d3["all_four_wrong_if_mistakes_were_unrelated"] == pytest.approx(0.125)
