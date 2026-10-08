"""Part II Task B: supplied and combined Google decoders.

For distance 3 and 5 separately: a majority/prior baseline, the four supplied
decoder predictions on the same test shots, and one linear combined decoder on
event density plus the four predictions.
"""

from __future__ import annotations

import math
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from quantum_lake_student.ml import GOOGLE_META_PREDICTION_COLUMNS, google_meta_model_input
from quantum_lake_student.stages.part2_helpers import (
    PREDICTED_SPLITS,
    ModelResult,
    Splits,
    majority_baseline,
    prediction_rows,
)
from quantum_lake_student.stages.part2_metrics import balanced_accuracy


DISTANCES = (3, 5)
LABEL_COLUMN = "actual_observable_flip"
FEATURE_ORDER = ["detector_event_density", *GOOGLE_META_PREDICTION_COLUMNS]
THRESHOLDS = [round(0.05 + 0.01 * step, 2) for step in range(91)]


def _rows_for_distance(splits: Splits, distance: int) -> Splits:
    """The rows of one distance, still grouped by split."""
    selected = {
        split: [row for row in rows if row["distance"] == distance]
        for split, rows in splits.items()
    }
    for split, rows in selected.items():
        if not rows:
            raise ValueError(f"Task B: distance {distance} has no {split} rows")
    return selected


def _supplied_decoder(splits: Splits, distance: int, column: str) -> ModelResult:
    """A supplied decoder's own prediction for every validation and test shot, unchanged."""
    model_id = f"task_b_d{distance}_{column.removesuffix('_prediction')}"
    predictions = []
    for split in PREDICTED_SPLITS:
        rows = splits[split]
        predictions += prediction_rows(
            rows,
            split=split,
            model_id=model_id,
            label_column=LABEL_COLUMN,
            predictions=[row[column] for row in rows],
        )
    # nothing is trained and no probability is given, so threshold, times and
    # Brier score stay empty
    return ModelResult(model_id=model_id, task="B", distance=distance, predictions=predictions)


def _features(rows: list[dict]) -> np.ndarray:
    """The five helper inputs of each shot, in FEATURE_ORDER."""
    return np.asarray([google_meta_model_input(row) for row in rows], dtype=np.float64)


def _labels(rows: list[dict]) -> list[bool]:
    return [bool(row[LABEL_COLUMN]) for row in rows]


def _choose_threshold(labels: list[bool], probabilities: np.ndarray) -> float:
    """The threshold with the best balanced accuracy."""
    weights = [1.0] * len(labels)
    return max(
        THRESHOLDS,
        key=lambda threshold: (
            balanced_accuracy(labels, (probabilities >= threshold).tolist(), weights),
            -abs(threshold - 0.5),
        ),
    )


def _combined_decoder(splits: Splits, distance: int, seed: int) -> ModelResult:
    """Logistic regression on the five helper inputs: fit on train, threshold on validation."""
    model_id = f"task_b_d{distance}_combined"
    model = LogisticRegression(random_state=seed)

    x_train = _features(splits["train"])
    started = time.perf_counter()
    model.fit(x_train, _labels(splits["train"]))
    train_seconds = time.perf_counter() - started
    flip = list(model.classes_).index(True)  # the probability column for "flip"

    validation_probabilities = model.predict_proba(_features(splits["validation"]))[:, flip]
    threshold = _choose_threshold(_labels(splits["validation"]), validation_probabilities)

    x_test = _features(splits["test"])
    started = time.perf_counter()
    test_probabilities = model.predict_proba(x_test)[:, flip]
    predict_seconds = time.perf_counter() - started

    predictions = []
    for split, probabilities in (
        ("validation", validation_probabilities),
        ("test", test_probabilities),
    ):
        predictions += prediction_rows(
            splits[split],
            split=split,
            model_id=model_id,
            label_column=LABEL_COLUMN,
            predictions=(probabilities >= threshold).tolist(),
            probabilities=probabilities.tolist(),
        )
    # the times cover fit() and predicting the test shots, not building the inputs
    return ModelResult(
        model_id=model_id,
        task="B",
        distance=distance,
        predictions=predictions,
        fitted_model=model,
        feature_order=list(FEATURE_ORDER),
        threshold=threshold,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
    )


def run_task_b(splits: Splits, *, seed: int) -> list[ModelResult]:
    found = {row["distance"] for rows in splits.values() for row in rows}
    if found != set(DISTANCES):
        raise ValueError(f"Task B expects distances {DISTANCES}, got {sorted(found)}")

    results: list[ModelResult] = []
    for distance in DISTANCES:
        rows = _rows_for_distance(splits, distance)
        results.append(
            majority_baseline(
                rows,
                model_id=f"task_b_d{distance}_majority",
                task="B",
                label_column=LABEL_COLUMN,
                distance=distance,
            )
        )
        results += [
            _supplied_decoder(rows, distance, column)
            for column in GOOGLE_META_PREDICTION_COLUMNS
        ]
        results.append(_combined_decoder(rows, distance, seed))
    return results


def decoder_error_overlap(results: list[ModelResult]) -> dict[str, dict]:
    """Whether the four supplied decoders get the same test shots wrong, and what
    the combined model does on those shots."""
    decoders = [column.removesuffix("_prediction") for column in GOOGLE_META_PREDICTION_COLUMNS]
    overlap = {}
    for distance in DISTANCES:
        test = {
            result.model_id.removeprefix(f"task_b_d{distance}_"): [
                p for p in result.predictions if p["split"] == "test"
            ]
            for result in results
            if result.task == "B" and result.distance == distance
        }
        if not test:
            continue
        shots = {p["example_id"] for p in test["combined"]}
        wrong = {
            name: {p["example_id"] for p in rows if p["prediction"] != p["label"]}
            for name, rows in test.items()
        }
        all_wrong = set.intersection(*(wrong[name] for name in decoders))
        any_wrong = set.union(*(wrong[name] for name in decoders))
        disagree = any_wrong - all_wrong
        unrelated = len(shots) * math.prod(len(wrong[name]) / len(shots) for name in decoders)
        overlap[f"d{distance}"] = {
            "test_shots": len(shots),
            "all_four_right": len(shots - any_wrong),
            "all_four_wrong": len(all_wrong),
            "all_four_wrong_if_mistakes_were_unrelated": unrelated,
            "decoders_disagree": len(disagree),
            "error_rate_where_decoders_disagree": {
                name: len(wrong[name] & disagree) / len(disagree) if disagree else None
                for name in [*decoders, "combined"]
            },
            "combined_wrong_where_all_four_right": len(wrong["combined"] - any_wrong),
            "combined_wrong_where_all_four_wrong": len(wrong["combined"] & all_wrong),
        }
    return overlap
