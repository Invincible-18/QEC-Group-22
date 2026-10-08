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
    majority_baseline,
    prediction_rows,
)


DISTANCE = 3
LABEL_COLUMN = "actual_observable_flip"
HIDDEN_LAYERS = (32,)
MAX_ITER = 50
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


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


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    rates = []
    for value in (True, False):
        mask = labels == value
        if mask.any():
            rates.append(float((predictions[mask] == value).mean()))
    return sum(rates) / len(rates)


def _choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Validation threshold with the best balanced accuracy; ties go to the one nearest 0.5."""
    best = max(
        THRESHOLD_GRID,
        key=lambda t: (_balanced_accuracy(labels, probabilities >= t), -abs(float(t) - 0.5)),
    )
    return float(best)


def run_task_c(splits: Splits, *, seed: int) -> list[ModelResult]:
    rows = {name: list(splits[name]) for name in ("train", "validation", "test")}
    detector_counts = {int(row["detector_count"]) for part in rows.values() for row in part}
    if detector_counts != {200}:
        raise ValueError(f"Task C expects 200 detector bits per row, got {sorted(detector_counts)}")

    baseline = majority_baseline(
        splits,
        model_id="task_c_d3_majority",
        task="C",
        label_column=LABEL_COLUMN,
        distance=DISTANCE,
    )

    x_train, y_train = _features(rows["train"]), _labels(rows["train"])
    feature_order = [f"detector_{index}" for index in range(x_train.shape[1])]

    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        max_iter=MAX_ITER,
        random_state=seed,
    )
    started = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - started

    positive = list(model.classes_).index(True)
    probabilities: dict[str, np.ndarray] = {}
    predict_seconds = 0.0
    for split in PREDICTED_SPLITS:
        x = _features(rows[split])
        started = time.perf_counter()
        probabilities[split] = model.predict_proba(x)[:, positive]
        predict_seconds += time.perf_counter() - started

    threshold = _choose_threshold(_labels(rows["validation"]), probabilities["validation"])

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
