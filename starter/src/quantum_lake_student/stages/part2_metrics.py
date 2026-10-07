"""Shared Part II metrics, so every model is measured the same way.

Each metric takes per-example weights: Task A passes the physical
sample_weight, Tasks B and C pass equal weights.
"""

from __future__ import annotations

from collections.abc import Sequence

from quantum_lake_student.ml import weighted_logical_error_rate


def logical_error_rate(
    labels: Sequence[bool], predictions: Sequence[bool], weights: Sequence[float]
) -> float:
    """Weighted share of examples whose prediction differs from the actual label."""
    return weighted_logical_error_rate(labels, predictions, weights)


def balanced_accuracy(
    labels: Sequence[bool], predictions: Sequence[bool], weights: Sequence[float]
) -> float:
    """Mean of the weighted recall of each class present in the labels."""
    correct = {True: 0.0, False: 0.0}
    total = {True: 0.0, False: 0.0}
    for label, prediction, weight in zip(labels, predictions, weights, strict=True):
        total[bool(label)] += weight
        if bool(label) == bool(prediction):
            correct[bool(label)] += weight
    recalls = [correct[cls] / total[cls] for cls in (True, False) if total[cls] > 0]
    if not recalls:
        raise ValueError("balanced accuracy needs at least one weighted example")
    return sum(recalls) / len(recalls)


def brier_score(
    labels: Sequence[bool], probabilities: Sequence[float | None], weights: Sequence[float]
) -> float | None:
    """Weighted mean squared error of the predicted probability of a logical error.

    None when any probability is missing: a supplied decoder or a majority
    baseline gives no probability, and one is not invented for it.
    """
    if any(probability is None for probability in probabilities):
        return None
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")
    return sum(
        weight * (probability - float(bool(label))) ** 2
        for label, probability, weight in zip(labels, probabilities, weights, strict=True)
    ) / total_weight
