from quantum_lake_student.io_utils import data_issue_row
from quantum_lake_student.models import QualityFinding, Severity


def _finding(locator: str) -> QualityFinding:
    return QualityFinding(
        rule_id="stabilizer_check_filter",
        severity=Severity.INFO,
        source_system="qasmbench",
        source_record_locator=locator,
        message="terminal data measurement excluded from stabilizer check",
        observed_value="measure q[0] -> c[0];",
    )


def test_issue_id_is_the_same_across_runs() -> None:
    finding = _finding("small/qec_sm_n5/qec_sm_n5.qasm:line=20")
    first = data_issue_row(finding, run_id="part1-run-1", action="filtered")
    second = data_issue_row(finding, run_id="part1-run-2", action="filtered")

    assert first["issue_id"] == second["issue_id"]
    assert (first["run_id"], second["run_id"]) == ("part1-run-1", "part1-run-2")


def test_identical_findings_at_different_places_get_different_issue_ids() -> None:
    source = data_issue_row(_finding("small/qec_sm_n5/qec_sm_n5.qasm:line=20"), run_id="r", action="filtered")
    transpiled = data_issue_row(
        _finding("small/qec_sm_n5/qec_sm_n5_transpiled.qasm:line=20"), run_id="r", action="filtered"
    )

    assert source["issue_id"] != transpiled["issue_id"]
