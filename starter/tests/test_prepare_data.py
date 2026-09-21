from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from quantum_lake_student.stages.prepare_data import build_syndrome_silver, parse_syndrome


def test_parse_syndrome_returns_sixteen_explicit_bits() -> None:
    value = "((0, 1, 0, 1), (1, 0, 0, 0), (1, 1, 1, 0), (0, 0, 1, 1))"
    assert parse_syndrome(value) == bytes(
        [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
    )


def test_build_syndrome_silver_writes_valid_rows_and_issues(tmp_path: Path) -> None:
    archive = tmp_path / "syndromes_dataset.zip"
    valid = "((0, 1, 0, 1), (1, 0, 0, 0), (1, 1, 1, 0), (0, 0, 1, 1))"
    csv = (
        "labels,syndromes,quantity\n"
        f'0,"{valid}",4\n'
        f'1,"{valid}",2\n'
        '1,"((0, 1),)",3\n'
    )
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("d-3_pfr-0.001000_nb-10M.csv", csv)

    result = build_syndrome_silver(archive, tmp_path / "lake", "run-1")

    assert result.input_count == 3
    assert result.output_count == 2
    assert result.issue_count == 1
    output = pd.read_parquet(
        tmp_path / "lake/silver/qec_syndromes/syndrome_observation.parquet"
    )
    assert list(output.columns) == [
        "source_record_id",
        "experiment_id",
        "physical_fault_rate",
        "syndrome_bits",
        "round_count",
        "check_count",
        "logical_error_label",
        "quantity",
    ]
    assert output["quantity"].tolist() == [4, 2]
    assert output["syndrome_bits"].map(len).tolist() == [16, 16]
    issues = pd.read_parquet(tmp_path / "lake/results/part1/data_issues.parquet")
    assert issues.loc[0, "action"] == "excluded_from_silver"