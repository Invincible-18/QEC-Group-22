import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantum_lake_student.stages.build_ml_tables import ML_SYNDROME_SCHEMA
from quantum_lake_student.stages.part2_helpers import (
    ModelResult,
    choose_threshold,
    majority_baseline,
    prediction_rows,
)
from quantum_lake_student.stages.part2_metrics import (
    balanced_accuracy,
    brier_score,
    logical_error_rate,
)
from quantum_lake_student.stages.train import (
    PREDICTIONS_SCHEMA,
    REPORT_DIR,
    TaskSpec,
    check_predictions,
    evaluate,
    render_report,
    run_tasks,
    task_inputs,
    validate_ml_table,
    write_outputs,
)


LABEL = "logical_error_label"
SPEC_A = TaskSpec("A", "syndrome", LABEL, "sample_weight", run=lambda splits, seed: [])


def _row(index: int, split: str, label: bool, weight: int) -> dict:
    return {"example_id": f"ex-{index}", LABEL: label, "sample_weight": weight, "data_split": split}


def _splits() -> dict[str, list[dict]]:
    return {
        "train": [_row(0, "train", True, 3), _row(1, "train", False, 1)],
        "validation": [_row(2, "validation", False, 2)],
        "test": [_row(3, "test", True, 5), _row(4, "test", False, 5)],
    }


def _result(splits, *, model_id: str = "task_a_model", predict=lambda row: row[LABEL]) -> ModelResult:
    predictions = []
    for split in ("validation", "test"):
        rows = splits[split]
        predictions += prediction_rows(
            rows,
            split=split,
            model_id=model_id,
            label_column=LABEL,
            predictions=[predict(row) for row in rows],
            weight_column="sample_weight",
        )
    return ModelResult(model_id=model_id, task="A", distance=None, predictions=predictions)


# --- metrics ---------------------------------------------------------------------


def test_metrics_use_the_weights() -> None:
    labels, predictions, weights = [True, False, True], [True, True, False], [8, 1, 1]
    assert logical_error_rate(labels, predictions, weights) == pytest.approx(0.2)
    # recall of True = 8/9, recall of False = 0/1
    assert balanced_accuracy(labels, predictions, weights) == pytest.approx((8 / 9) / 2)
    assert brier_score(labels, [0.5, 0.5, 0.5], weights) == pytest.approx(0.25)


def test_brier_score_is_not_applicable_without_probabilities() -> None:
    assert brier_score([True, False], [None, None], [1, 1]) is None


# --- the coverage check -------------------------------------------------------------


def test_check_accepts_exactly_the_validation_and_test_rows() -> None:
    splits = _splits()
    check_predictions(_result(splits), SPEC_A, splits)


def test_check_rejects_a_prediction_on_a_training_row() -> None:
    splits = _splits()
    result = _result(splits)
    result.predictions += prediction_rows(
        splits["train"][:1], split="test", model_id=result.model_id, label_column=LABEL,
        predictions=[True], weight_column="sample_weight",
    )
    with pytest.raises(RuntimeError, match="not validation/test rows"):
        check_predictions(result, SPEC_A, splits)


def test_check_rejects_a_missing_row_and_a_wrong_weight() -> None:
    splits = _splits()
    result = _result(splits)
    result.predictions.pop()
    result.predictions[0]["weight"] = 99
    with pytest.raises(RuntimeError, match="have no prediction.*weight differs"):
        check_predictions(result, SPEC_A, splits)


def test_run_tasks_rejects_a_repeated_model_id() -> None:
    inputs = {"syndrome": _splits()}
    spec = TaskSpec("A", "syndrome", LABEL, "sample_weight", run=lambda splits, seed: [_result(splits), _result(splits)])
    with pytest.raises(RuntimeError, match="used more than once"):
        run_tasks(inputs, (spec,), seed=1)


# --- inputs and the shared baseline ------------------------------------------------


def test_task_c_input_is_the_bounded_distance_three_subset() -> None:
    def google_row(distance: int, shot_index: int) -> dict:
        return {"example_id": f"{distance}-{shot_index}", "distance": distance, "shot_index": shot_index,
                "data_split": "test" if shot_index % 2 else "train"}

    google = pa.Table.from_pylist(
        [google_row(3, 1), google_row(3, 12_499), google_row(3, 12_500), google_row(5, 1)]
    )
    syndrome = pa.Table.from_pylist([_row(0, "train", True, 1)])
    bounded = task_inputs(syndrome, google)["google_bounded"]
    assert sorted(r["example_id"] for rows in bounded.values() for r in rows) == ["3-1", "3-12499"]


def test_majority_baseline_uses_the_weighted_training_prior() -> None:
    splits = _splits()
    result = majority_baseline(
        splits, model_id="task_a_majority", task="A", label_column=LABEL, weight_column="sample_weight"
    )
    check_predictions(result, SPEC_A, splits)
    # weighted training share of True is 3 / (3 + 1)
    assert {p["probability"] for p in result.predictions} == {0.75}
    assert {p["prediction"] for p in result.predictions} == {True}


def test_validate_ml_table_rejects_a_changed_schema() -> None:
    table = pa.Table.from_pylist([{"example_id": "x"}])
    with pytest.raises(RuntimeError, match="does not match its contract"):
        validate_ml_table("syndrome", table, ML_SYNDROME_SCHEMA)


# --- outputs --------------------------------------------------------------------------


def test_write_outputs_writes_every_file_and_drops_old_models(tmp_path) -> None:
    splits = _splits()
    result = _result(splits)
    result.fitted_model = {"weights": [1, 2]}
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "removed_model.joblib").write_text("old")

    write_outputs(tmp_path, [result], evaluate([result]), {"model_run_id": "part2-test"})

    predictions = pq.read_table(tmp_path / "predictions.parquet")
    assert predictions.schema == PREDICTIONS_SCHEMA
    assert predictions.num_rows == 3
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["A"]["task_a_model"]["logical_error_rate"] == 0.0
    run_record = json.loads((tmp_path / "run.json").read_text())
    assert run_record["models"]["task_a_model"]["model_file"] == "models/task_a_model.joblib"
    assert [path.name for path in (tmp_path / "models").iterdir()] == ["task_a_model.joblib"]


# --- the shared threshold rule ---------------------------------------------------


def test_choose_threshold_picks_the_lowest_logical_error_rate() -> None:
    labels = [False, False, True, True]
    # any threshold in (0.30, 0.70] separates the classes; the lowest such grid value is kept
    assert choose_threshold(labels, [0.10, 0.30, 0.70, 0.90]) == 0.31


def test_choose_threshold_uses_the_weights() -> None:
    labels = [False, True]
    probabilities = [0.40, 0.20]
    # unweighted: every threshold makes exactly one mistake, so the lowest is kept
    assert choose_threshold(labels, probabilities) == 0.05
    # the error shot counts 9 times as much, so catching it (threshold <= 0.20) wins
    assert choose_threshold(labels, probabilities, weights=[1, 9]) <= 0.20
    # the fine shot counts 9 times as much, so not flagging it (threshold > 0.40) wins
    assert choose_threshold(labels, probabilities, weights=[9, 1]) > 0.40


# --- report.md ------------------------------------------------------------------------


def test_report_joins_sections_in_name_order_and_fills_tables(tmp_path) -> None:
    (tmp_path / "20_end.md").write_text("## End\n\n{{all_results}}")
    (tmp_path / "10_task.md").write_text("## Task A\n<!-- a note for the writer -->\n{{task_a_results}}")
    splits = _splits()
    metrics = evaluate([_result(splits)])

    report = render_report(tmp_path, metrics)

    assert report.index("## Task A") < report.index("## End")
    assert "a note for the writer" not in report and "{{" not in report
    assert "| task_a_model | n/a | 2 | 10 | 0.0000 | 1.0000 | n/a |" in report


def test_report_rejects_an_unknown_placeholder(tmp_path) -> None:
    (tmp_path / "10_task.md").write_text("{{task_a_reslts}}")
    with pytest.raises(RuntimeError, match="unknown report placeholder"):
        render_report(tmp_path, {})


def test_committed_report_templates_render() -> None:
    report = render_report(REPORT_DIR, {})
    assert "{{" not in report and "<!--" not in report
