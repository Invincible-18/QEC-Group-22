"""Part II Task C: bounded raw-detector prototype.

One small MLP on the 200 unpacked detector bits. train.py passes only the
fixed subset distance = 3 AND shot_index < 12_500, already split. Record
training and prediction time in the ModelResult.
See part2_helpers.py for the input/result types.
"""

from __future__ import annotations

from quantum_lake_student.stages.part2_helpers import ModelResult, Splits


def run_task_c(splits: Splits, *, seed: int) -> list[ModelResult]:
    return []
