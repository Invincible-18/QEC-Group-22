"""Part II Task A: weighted syndrome decoder.

Predict logical_error_label from the 16 ordered syndrome values with a
weighted majority/prior baseline and one weighted linear classifier. Use
sample_weight for fitting where supported; train.py applies it to every metric.
See part2_helpers.py for the input/result types and majority_baseline().
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression

from quantum_lake_student.ml import syndrome_model_input
from quantum_lake_student.stages.part2_helpers import ModelResult, Splits
from quantum_lake_student.stages.part2_helpers import majority_baseline, prediction_rows
from quantum_lake_student.stages.part2_metrics import balanced_accuracy


LABEL_COLUMN = "logical_error_label"
WEIGHT_COLUMN = "sample_weight"
FEATURE_ORDER = [f"syndrome_{index:02d}" for index in range(16)]


def _features(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [syndrome_model_input(row["syndrome_bits"]) for row in rows], dtype=np.float64
    )


def _select_threshold(labels: list[bool], probabilities: np.ndarray, weights: list[float]) -> float:
    unique = np.unique(probabilities)
    candidates = {0.0, 0.5, 1.0, *(float(value) for value in unique)}
    candidates.update(
        float((left + right) / 2)
        for left, right in zip(unique[:-1], unique[1:], strict=True)
    )
    return max(
        candidates,
        key=lambda threshold: (
            balanced_accuracy(
                labels,
                probabilities >= threshold,
                weights,
            ),
            -abs(threshold - 0.5),
            -threshold,
        ),
    )


def run_task_a(splits: Splits, *, seed: int) -> list[ModelResult]:
    train_rows = splits["train"]
    validation_rows = splits["validation"]
    if not train_rows:
        raise ValueError("Task A requires training rows")
    if not validation_rows:
        raise ValueError("Task A requires validation rows for threshold selection")

    baseline_started = perf_counter()
    baseline = majority_baseline(
        splits,
        model_id="task_a_weighted_majority",
        task="A",
        label_column=LABEL_COLUMN,
        weight_column=WEIGHT_COLUMN,
    )
    baseline.train_seconds = 0.0
    baseline.predict_seconds = perf_counter() - baseline_started

    train_started = perf_counter()
    train_features = _features(train_rows)
    train_labels = np.asarray([row[LABEL_COLUMN] for row in train_rows], dtype=bool)
    train_weights = np.asarray([row[WEIGHT_COLUMN] for row in train_rows], dtype=np.float64)
    model = LogisticRegression(random_state=seed, max_iter=1000)
    model.fit(train_features, train_labels, sample_weight=train_weights)
    train_seconds = perf_counter() - train_started

    predict_started = perf_counter()
    validation_features = _features(validation_rows)
    validation_probabilities = model.predict_proba(validation_features)[:, 1]
    validation_labels = [bool(row[LABEL_COLUMN]) for row in validation_rows]
    validation_weights = [float(row[WEIGHT_COLUMN]) for row in validation_rows]
    threshold = _select_threshold(
        validation_labels, validation_probabilities, validation_weights
    )

    predictions = []
    for split in ("validation", "test"):
        rows = splits[split]
        probabilities = model.predict_proba(_features(rows))[:, 1]
        predictions.extend(
            prediction_rows(
                rows,
                split=split,
                model_id="task_a_logistic_regression",
                label_column=LABEL_COLUMN,
                predictions=probabilities >= threshold,
                probabilities=probabilities,
                weight_column=WEIGHT_COLUMN,
            )
        )
    predict_seconds = perf_counter() - predict_started

    linear_model = ModelResult(
        model_id="task_a_logistic_regression",
        task="A",
        distance=None,
        predictions=predictions,
        fitted_model=model,
        feature_order=FEATURE_ORDER.copy(),
        threshold=threshold,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
    )
    return [baseline, linear_model]
