"""Part II Task C: bounded raw-detector prototype.

One small MLP on the 200 unpacked detector bits. train.py passes only the
fixed subset distance = 3 AND shot_index < 12_500, already split. Record
training and prediction time in the ModelResult.
See part2_helpers.py for the input/result types.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.neural_network import MLPClassifier

from quantum_lake_student.ml import unpack_little_endian_bits
from quantum_lake_student.stages.part2_helpers import (
    PREDICTED_SPLITS,
    ModelResult,
    Splits,
    choose_threshold,
    majority_baseline,
    prediction_rows,
)


DISTANCE = 3
DETECTOR_COUNT = 200
LABEL_COLUMN = "actual_observable_flip"
HIDDEN_LAYERS = (32,)
MAX_ITER = 50


def _features(rows: list[dict]) -> np.ndarray:
    """Unpack each row's b8 detector bits into a (rows, detector_count) uint8 matrix."""
    if not rows:
        return np.empty((0, 0), dtype=np.uint8)
    return np.asarray(
        [unpack_little_endian_bits(row["detector_bits"], int(row["detector_count"])) for row in rows],
        dtype=np.uint8,
    )


def _labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([bool(row[LABEL_COLUMN]) for row in rows], dtype=bool)


def _check_event_counts(rows: list[dict], matrix: np.ndarray, split: str) -> None:
    """detector_event_count must equal the number of set detector bits in each row."""
    if len(rows) == 0:
        return
    counted = matrix.sum(axis=1)
    declared = np.asarray([int(row["detector_event_count"]) for row in rows])
    mismatch = int((counted != declared).sum())
    if mismatch:
        raise ValueError(f"{mismatch} {split} rows have detector_event_count different from their set bits")


def run_task_c(splits: Splits, *, seed: int) -> list[ModelResult]:
    rows = {name: list(splits[name]) for name in ("train", "validation", "test")}
    detector_counts = {int(row["detector_count"]) for part in rows.values() for row in part}
    if detector_counts != {DETECTOR_COUNT}:
        raise ValueError(
            f"Task C expects {DETECTOR_COUNT} detector bits per row, got {sorted(detector_counts)}"
        )

    baseline = majority_baseline(
        splits,
        model_id="task_c_d3_majority",
        task="C",
        label_column=LABEL_COLUMN,
        distance=DISTANCE,
    )

    # model inputs are built (and checked) before any timing starts
    x = {name: _features(rows[name]) for name in rows}
    for name in rows:
        _check_event_counts(rows[name], x[name], name)
    y_train = _labels(rows["train"])
    feature_order = [f"detector_{index}" for index in range(DETECTOR_COUNT)]

    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        max_iter=MAX_ITER,
        random_state=seed,
    )
    started = time.perf_counter()
    model.fit(x["train"], y_train)
    train_seconds = time.perf_counter() - started

    positive = list(model.classes_).index(True)  # the probability column for "flip"
    validation_probabilities = model.predict_proba(x["validation"])[:, positive]

    started = time.perf_counter()
    test_probabilities = model.predict_proba(x["test"])[:, positive]
    predict_seconds = time.perf_counter() - started
    probabilities = {"validation": validation_probabilities, "test": test_probabilities}

    # the team's threshold rule, called with validation rows only
    threshold = choose_threshold(
        _labels(rows["validation"]).tolist(), probabilities["validation"].tolist()
    )

    predictions: list[dict] = []
    for split in PREDICTED_SPLITS:
        predictions += prediction_rows(
            rows[split],
            split=split,
            model_id="task_c_d3_mlp",
            label_column=LABEL_COLUMN,
            predictions=(probabilities[split] >= threshold).tolist(),
            probabilities=probabilities[split].tolist(),
        )

    mlp = ModelResult(
        model_id="task_c_d3_mlp",
        task="C",
        distance=DISTANCE,
        predictions=predictions,
        fitted_model=model,
        feature_order=feature_order,
        threshold=threshold,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
    )
    return [baseline, mlp]
