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
from quantum_lake_student.stages.part2_helpers import (
    choose_threshold,
    majority_baseline,
    prediction_rows,
)


LABEL_COLUMN = "logical_error_label"
WEIGHT_COLUMN = "sample_weight"
FEATURE_ORDER = [f"syndrome_{index:02d}" for index in range(16)]


def _features(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [syndrome_model_input(row["syndrome_bits"]) for row in rows], dtype=np.float64
    )


def _labels(rows: list[dict]) -> list[bool]:
    return [bool(row[LABEL_COLUMN]) for row in rows]


def _weights(rows: list[dict]) -> list[float]:
    return [float(row[WEIGHT_COLUMN]) for row in rows]


def run_task_a(splits: Splits, *, seed: int) -> list[ModelResult]:
    train_rows = splits["train"]
    validation_rows = splits["validation"]
    if not train_rows:
        raise ValueError("Task A requires training rows")
    if not validation_rows:
        raise ValueError("Task A requires validation rows for threshold selection")

    # not trained, so it has no times
    baseline = majority_baseline(
        splits,
        model_id="task_a_weighted_majority",
        task="A",
        label_column=LABEL_COLUMN,
        weight_column=WEIGHT_COLUMN,
    )

    # model inputs are built before any timing starts
    x = {split: _features(splits[split]) for split in ("train", "validation", "test")}
    y_train, w_train = _labels(train_rows), _weights(train_rows)
    model = LogisticRegression(random_state=seed, max_iter=1000)

    started = perf_counter()
    model.fit(x["train"], y_train, sample_weight=w_train)
    train_seconds = perf_counter() - started
    error = list(model.classes_).index(True)  # the probability column for a logical error

    # the team's threshold rule, called with validation rows only (not timed)
    validation_probabilities = model.predict_proba(x["validation"])[:, error]
    threshold = choose_threshold(
        _labels(validation_rows), validation_probabilities.tolist(), _weights(validation_rows)
    )

    started = perf_counter()
    test_probabilities = model.predict_proba(x["test"])[:, error]
    predict_seconds = perf_counter() - started

    predictions = []
    for split, probabilities in (
        ("validation", validation_probabilities),
        ("test", test_probabilities),
    ):
        predictions.extend(
            prediction_rows(
                splits[split],
                split=split,
                model_id="task_a_logistic_regression",
                label_column=LABEL_COLUMN,
                predictions=(probabilities >= threshold).tolist(),
                probabilities=probabilities.tolist(),
                weight_column=WEIGHT_COLUMN,
            )
        )

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
