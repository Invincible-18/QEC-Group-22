"""What every Part II task uses: the input and result types and shared helpers.

A task receives Splits (rows already grouped by the course data_split) and
returns ModelResults. It fits on "train", chooses any threshold on
"validation", and predicts every "validation" and "test" row. It never computes
metrics, writes files or reads storage; train.py does that for all tasks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from quantum_lake_student.stages.part2_metrics import logical_error_rate


# {"train": [...], "validation": [...], "test": [...]}; each row is one ML-table row
Splits = dict[str, list[dict]]

PREDICTED_SPLITS = ("validation", "test")

THRESHOLD_GRID = tuple(round(0.05 + 0.01 * step, 2) for step in range(91))  # 0.05 .. 0.95


@dataclass
class ModelResult:
    model_id: str                 # task_<a|b|c>_[d<distance>_]<model>, e.g. task_b_d3_pymatching
    task: str                     # "A", "B" or "C"
    distance: int | None          # 3 or 5 for Tasks B and C, None for Task A
    predictions: list[dict]       # built with prediction_rows(), one per validation and test row
    fitted_model: object | None = None   # saved by train.py; None for baselines and supplied decoders
    feature_order: list[str] = field(default_factory=list)
    threshold: float | None = None       # chosen on validation only
    train_seconds: float | None = None
    predict_seconds: float | None = None


def prediction_rows(
    rows: Sequence[dict],
    *,
    split: str,
    model_id: str,
    label_column: str,
    predictions: Sequence[bool],
    probabilities: Sequence[float] | None = None,
    weight_column: str | None = None,
) -> list[dict]:
    """Prediction rows for one split, in the order of ``rows``.

    ``weight_column`` is "sample_weight" for Task A; without it every row has
    weight 1.
    """
    if len(predictions) != len(rows):
        raise ValueError("one prediction is needed per row")
    if probabilities is not None and len(probabilities) != len(rows):
        raise ValueError("one probability is needed per row")
    return [
        {
            "example_id": row["example_id"],
            "model_id": model_id,
            "label": bool(row[label_column]),
            "prediction": bool(prediction),
            "probability": None if probabilities is None else float(probabilities[index]),
            "split": split,
            "weight": int(row[weight_column]) if weight_column else 1,
        }
        for index, (row, prediction) in enumerate(zip(rows, predictions))
    ]


def choose_threshold(
    labels: Sequence[bool],
    probabilities: Sequence[float],
    weights: Sequence[float] | None = None,
) -> float:
    """The team's threshold rule, to be called with validation rows only.

    Picks the THRESHOLD_GRID value with the lowest logical-error rate, weighted
    by ``weights`` (Task A passes sample_weight; Tasks B and C pass nothing).
    Equal error rates keep the lowest such threshold.
    """
    if weights is None:
        weights = [1.0] * len(labels)
    return min(
        THRESHOLD_GRID,
        key=lambda threshold: logical_error_rate(
            labels, [probability >= threshold for probability in probabilities], weights
        ),
    )


def majority_baseline(
    splits: Splits,
    *,
    model_id: str,
    task: str,
    label_column: str,
    distance: int | None = None,
    weight_column: str | None = None,
) -> ModelResult:
    """Predict the more common training label for every row.

    Its probability is the (weighted) training share of logical errors, so a
    Brier score exists for the baseline too.
    """
    train = splits["train"]
    weights = [float(row[weight_column]) if weight_column else 1.0 for row in train]
    prior = sum(w for row, w in zip(train, weights) if row[label_column]) / sum(weights)
    majority = prior >= 0.5
    predictions = []
    for split in PREDICTED_SPLITS:
        rows = splits[split]
        predictions += prediction_rows(
            rows,
            split=split,
            model_id=model_id,
            label_column=label_column,
            predictions=[majority] * len(rows),
            probabilities=[prior] * len(rows),
            weight_column=weight_column,
        )
    return ModelResult(
        model_id=model_id,
        task=task,
        distance=distance,
        predictions=predictions,
        threshold=0.5,
    )
