"""Part II Task A: weighted syndrome decoder.

Predict logical_error_label from the 16 ordered syndrome values with a
weighted majority/prior baseline and one weighted linear classifier. Use
sample_weight for fitting where supported; train.py applies it to every metric.
See part2_helpers.py for the input/result types and majority_baseline().
"""

from __future__ import annotations

from quantum_lake_student.stages.part2_helpers import ModelResult, Splits


def run_task_a(splits: Splits, *, seed: int) -> list[ModelResult]:
    return []
