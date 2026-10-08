"""Required Part II model-training stage.

Consume the two required ML input tables produced by Part I through the
supplied model-input and partition helpers. Publish repeatable model files,
predictions, metrics, run settings, the exact feature order, and the concise
Part II report under ``results/part2/``. Numerical performance is not graded.

This framework reads only the two ML tables, checks them
against their contracts, splits them with the course helper partition_records,
and hands each task its rows already split (see part2_helpers.py). It then
checks every model's predictions against its input, computes all metrics on the
test split with one shared implementation (part2_metrics.py), and writes every
results/part2 file. The tasks themselves are in train_syndrome.py (A),
train_google.py (B) and train_raw_detector.py (C).
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import joblib
import pyarrow as pa
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.io_utils import get_object_bytes, write_local_parquet
from quantum_lake_student.ml import partition_records
from quantum_lake_student.models import StageResult
from quantum_lake_student.stages.build_ml_tables import (
    ML_GOOGLE_OBJECT,
    ML_GOOGLE_SCHEMA,
    ML_SYNDROME_OBJECT,
    ML_SYNDROME_SCHEMA,
)
from quantum_lake_student.stages.part2_helpers import PREDICTED_SPLITS, ModelResult, Splits
from quantum_lake_student.stages.part2_metrics import (
    balanced_accuracy,
    brier_score,
    logical_error_rate,
)
from quantum_lake_student.stages.prepare_data import _code_revision
from quantum_lake_student.stages.train_google import run_task_b
from quantum_lake_student.stages.train_raw_detector import run_task_c
from quantum_lake_student.stages.train_syndrome import run_task_a


SEED = 2026
PART2_ROOT = Path("/workspace/results/part2")
REPORT_DIR = Path(__file__).resolve().parents[3] / "docs" / "part2-report"
RELEASE_MANIFEST_OBJECT = "metadata/course-release/bundle-manifest.json"

TASK_C_DISTANCE = 3
TASK_C_MAX_SHOT_INDEX = 12_500

SPLIT_RULES = {
    "syndrome": "syndrome_data_split: fault rate 0.0005 validation, 0.005 test, the other five train",
    "google": "google_data_split: odd shot_index test; even with shot_index % 10 == 8 validation; other even train",
    "google_bounded": "Task C subset distance = 3 AND shot_index < 12500, then google_data_split",
}

DEPENDENCIES = ("scikit-learn", "numpy", "scipy", "joblib", "pyarrow", "pandas")

PREDICTIONS_SCHEMA = pa.schema(
    [
        ("example_id", pa.string()),
        ("model_id", pa.string()),
        ("task", pa.string()),
        ("distance", pa.int32()),
        ("label", pa.bool_()),
        ("prediction", pa.bool_()),
        ("probability", pa.float64()),
        ("split", pa.string()),
        ("weight", pa.int64()),
    ]
)


@dataclass(frozen=True)
class TaskSpec:
    task: str
    table: str                       # which prepared input: syndrome, google or google_bounded
    label_column: str
    weight_column: str | None        # sample_weight for Task A; None means weight 1
    run: Callable[..., list[ModelResult]]


TASKS = (
    TaskSpec("A", "syndrome", "logical_error_label", "sample_weight", run_task_a),
    TaskSpec("B", "google", "actual_observable_flip", None, run_task_b),
    TaskSpec("C", "google_bounded", "actual_observable_flip", None, run_task_c),
)


# ============================================================================
# inputs
# ============================================================================


def validate_ml_table(name: str, table: pa.Table, schema: pa.Schema) -> None:
    if not table.schema.equals(schema, check_metadata=False):
        raise RuntimeError(
            f"{name} ML table does not match its contract.\nexpected:\n{schema}\ngot:\n{table.schema}"
        )


def load_ml_tables(client, bucket: str) -> dict[str, tuple[pa.Table, str]]:
    """Read and check the two ML tables; returns each table with the SHA-256 of its file."""
    loaded = {}
    for name, (key, schema) in {
        "syndrome": (ML_SYNDROME_OBJECT, ML_SYNDROME_SCHEMA),
        "google": (ML_GOOGLE_OBJECT, ML_GOOGLE_SCHEMA),
    }.items():
        data = get_object_bytes(client, bucket, key)
        table = pq.read_table(io.BytesIO(data))
        validate_ml_table(name, table, schema)
        loaded[name] = (table, hashlib.sha256(data).hexdigest())
    return loaded


def task_inputs(syndrome: pa.Table, google: pa.Table) -> dict[str, Splits]:
    """Group each table by its data_split with the course partition_records helper."""
    google_rows = google.to_pylist()
    bounded = [
        row
        for row in google_rows
        if row["distance"] == TASK_C_DISTANCE and row["shot_index"] < TASK_C_MAX_SHOT_INDEX
    ]
    return {
        "syndrome": partition_records(syndrome.to_pylist()),
        "google": partition_records(google_rows),
        "google_bounded": partition_records(bounded),
    }


# ============================================================================
# checks on what a task returns
# ============================================================================


def check_predictions(result: ModelResult, spec: TaskSpec, splits: Splits) -> None:
    """Raise unless the predictions cover exactly this input's validation and test rows."""
    problems: list[str] = []

    if result.task != spec.task:
        problems.append(f"task is {result.task!r}, expected {spec.task!r}")
    if spec.task == "A" and result.distance is not None:
        problems.append("Task A models have no distance")
    if spec.task != "A" and result.distance not in (3, 5):
        problems.append("Tasks B and C need distance 3 or 5")
    if spec.table == "google_bounded" and result.distance != TASK_C_DISTANCE:
        problems.append(f"Task C uses distance {TASK_C_DISTANCE} only")

    expected = {
        (row["example_id"], split): row
        for split in PREDICTED_SPLITS
        for row in splits[split]
        if result.distance is None or row.get("distance") == result.distance
    }
    seen = Counter((p["example_id"], p["split"]) for p in result.predictions)
    duplicates = sum(count - 1 for count in seen.values() if count > 1)
    missing = expected.keys() - seen.keys()
    extra = seen.keys() - expected.keys()
    if duplicates:
        problems.append(f"{duplicates} rows are predicted more than once")
    if missing:
        problems.append(f"{len(missing)} validation/test rows have no prediction")
    if extra:
        problems.append(
            f"{len(extra)} predictions are not validation/test rows of this input "
            "(a training row, a wrong split or another distance)"
        )

    bad = Counter()
    for prediction in result.predictions:
        row = expected.get((prediction["example_id"], prediction["split"]))
        if row is None:
            continue
        if prediction["model_id"] != result.model_id:
            bad["model_id differs from the result's model_id"] += 1
        if prediction["label"] != bool(row[spec.label_column]):
            bad["label differs from the input row"] += 1
        if prediction["weight"] != (int(row[spec.weight_column]) if spec.weight_column else 1):
            bad["weight differs from the input row"] += 1
        if not isinstance(prediction["prediction"], bool):
            bad["prediction is not a bool"] += 1
        probability = prediction["probability"]
        if probability is not None and not 0.0 <= probability <= 1.0:
            bad["probability is outside 0..1"] += 1
    problems.extend(f"{rule} ({count} rows)" for rule, count in bad.items())

    if problems:
        raise RuntimeError(f"{result.model_id}: " + "; ".join(problems))


def run_tasks(
    inputs: dict[str, Splits], tasks: tuple[TaskSpec, ...] = TASKS, *, seed: int = SEED
) -> list[ModelResult]:
    results: list[ModelResult] = []
    for spec in tasks:
        for result in spec.run(inputs[spec.table], seed=seed):
            check_predictions(result, spec, inputs[spec.table])
            results.append(result)
    repeated = [model_id for model_id, n in Counter(r.model_id for r in results).items() if n > 1]
    if repeated:
        raise RuntimeError(f"model_id used more than once: {repeated}")
    return results


# ============================================================================
# metrics and outputs
# ============================================================================


def evaluate(results: list[ModelResult]) -> dict:
    """Test-split metrics for every model, grouped by task."""
    metrics: dict[str, dict] = {}
    for result in results:
        test = [p for p in result.predictions if p["split"] == "test"]
        labels = [p["label"] for p in test]
        predictions = [p["prediction"] for p in test]
        weights = [p["weight"] for p in test]
        metrics.setdefault(result.task, {})[result.model_id] = {
            "distance": result.distance,
            "test_examples": len(test),
            "test_weight": sum(weights),
            "logical_error_rate": logical_error_rate(labels, predictions, weights),
            "balanced_accuracy": balanced_accuracy(labels, predictions, weights),
            "brier_score": brier_score(labels, [p["probability"] for p in test], weights),
            "train_seconds": result.train_seconds,
            "predict_seconds": result.predict_seconds,
        }
    return metrics


def write_outputs(out_dir: Path, results: list[ModelResult], metrics: dict, run_record: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for old in models_dir.glob("*.joblib"):
        old.unlink()  # a model dropped from the code must not survive from an earlier run

    rows = [
        {**prediction, "task": result.task, "distance": result.distance}
        for result in results
        for prediction in result.predictions
    ]
    write_local_parquet(out_dir / "predictions.parquet", pa.Table.from_pylist(rows, schema=PREDICTIONS_SCHEMA))

    models = {}
    for result in results:
        model_file = None
        if result.fitted_model is not None:
            model_file = f"models/{result.model_id}.joblib"
            joblib.dump(result.fitted_model, out_dir / model_file)
        models[result.model_id] = {
            "task": result.task,
            "distance": result.distance,
            "feature_order": result.feature_order,
            "threshold": result.threshold,
            "model_file": model_file,
            "train_seconds": result.train_seconds,
            "predict_seconds": result.predict_seconds,
        }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "run.json").write_text(json.dumps({**run_record, "models": models}, indent=2))


# ============================================================================
# report.md
# ============================================================================

TASKS_IN_REPORT = ("A", "B", "C")
OVERLAP_KEY = "task_b_decoder_error_overlap"
_PLACEHOLDER = re.compile(r"\{\{\s*([a-z0-9_]+)\s*\}\}")
_COMMENT = re.compile(r"<!--.*?-->\s*", re.DOTALL)


def _cell(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _results_table(models: dict[str, dict], *, with_task: bool = False) -> str:
    """One row per model, from the metrics written to metrics.json."""
    if not models:
        return "_No models in this run._"
    header = ["Model", "Distance", "Test rows", "Test weight", "Logical-error rate",
              "Balanced accuracy", "Brier score", "Train s", "Predict s"]
    if with_task:
        header.insert(0, "Task")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for model_id, m in models.items():
        cells = [
            model_id, _cell(m["distance"]), _cell(m["test_examples"]), _cell(m["test_weight"]),
            _cell(m["logical_error_rate"]), _cell(m["balanced_accuracy"]), _cell(m["brier_score"]),
            _cell(m["train_seconds"], 3), _cell(m["predict_seconds"], 3),
        ]
        if with_task:
            cells.insert(0, m["task"])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _overlap_table(overlap: dict | None) -> str:
    """Task B's decoder-overlap summary: one column per distance, one row per value."""
    if not overlap:
        return "_Not available in this run._"
    columns = list(overlap)
    rows: dict[str, dict[str, object]] = {}
    for column, values in overlap.items():
        for key, value in values.items():
            if isinstance(value, dict):
                for name, inner in value.items():
                    rows.setdefault(f"{key}: {name}", {})[column] = inner
            else:
                rows.setdefault(key, {})[column] = value
    lines = ["| | " + " | ".join(columns) + " |", "|" + "---|" * (len(columns) + 1)]
    for label, by_column in rows.items():
        lines.append(f"| {label.replace('_', ' ')} | "
                     + " | ".join(_cell(by_column.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def report_tables(metrics: dict) -> dict[str, str]:
    """Every placeholder the report templates may use."""
    every_model = {
        model_id: {**values, "task": task}
        for task in TASKS_IN_REPORT
        for model_id, values in metrics.get(task, {}).items()
    }
    return {
        "task_a_results": _results_table(metrics.get("A", {})),
        "task_b_results": _results_table(metrics.get("B", {})),
        "task_c_results": _results_table(metrics.get("C", {})),
        "task_b_overlap": _overlap_table(metrics.get(OVERLAP_KEY)),
        "all_results": _results_table(every_model, with_task=True),
    }


def render_report(report_dir: Path, metrics: dict) -> str:
    """Join the report templates in name order and fill every {{placeholder}}."""
    sections = sorted(report_dir.glob("*.md"))
    if not sections:
        raise RuntimeError(f"no report templates in {report_dir}")
    text = _COMMENT.sub("", "\n\n".join(path.read_text().strip() for path in sections))

    tables = report_tables(metrics)
    unknown = sorted(set(_PLACEHOLDER.findall(text)) - tables.keys())
    if unknown:
        raise RuntimeError(f"unknown report placeholder(s): {unknown}; known: {sorted(tables)}")
    text = _PLACEHOLDER.sub(lambda match: tables[match.group(1)], text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


# ============================================================================
# stage
# ============================================================================


def run(model_run_id: str) -> StageResult:
    result = StageResult(stage="train", run_id=model_run_id)
    started_at = datetime.now(UTC).isoformat()
    clock = time.perf_counter()

    settings = Settings.from_environment()
    client = minio_client(settings)
    loaded = load_ml_tables(client, settings.s3_bucket)
    syndrome, syndrome_sha = loaded["syndrome"]
    google, google_sha = loaded["google"]

    results = run_tasks(task_inputs(syndrome, google), seed=SEED)
    metrics = evaluate(results)

    release = json.loads(get_object_bytes(client, settings.s3_bucket, RELEASE_MANIFEST_OBJECT))
    run_record = {
        "model_run_id": model_run_id,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "total_seconds": time.perf_counter() - clock,
        "code_revision": _code_revision(),
        "data_release": {
            "release_name": release["release_name"],
            "bundle_version": release["bundle_version"],
            "release_date": release["release_date"],
        },
        "input_tables": {
            ML_SYNDROME_OBJECT: {"sha256": syndrome_sha, "rows": syndrome.num_rows},
            ML_GOOGLE_OBJECT: {"sha256": google_sha, "rows": google.num_rows},
        },
        "random_seed": SEED,
        "split_rules": SPLIT_RULES,
        "python": platform.python_version(),
        "dependencies": {name: metadata.version(name) for name in DEPENDENCIES},
    }
    write_outputs(PART2_ROOT, results, metrics, run_record)
    (PART2_ROOT / "report.md").write_text(render_report(REPORT_DIR, metrics))

    result.input_count = syndrome.num_rows + google.num_rows
    result.output_count = len(results)
    result.finish()
    return result
