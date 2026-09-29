import io
from zipfile import ZipFile

from quantum_lake_student.stages import prepare_data


def test_parse_syndrome_returns_sixteen_explicit_bits() -> None:
    value = "((0, 1, 0, 1), (1, 0, 0, 0), (1, 1, 1, 0), (0, 0, 1, 1))"
    assert prepare_data.parse_syndrome(value) == bytes(
        [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
    )


def _zip_bytes(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as bundle:
        for name, content in members.items():
            bundle.writestr(name, content)
    return buffer.getvalue()


def test_prepare_qec_syndromes_writes_valid_rows_and_issues(monkeypatch) -> None:
    valid = "((0, 1, 0, 1), (1, 0, 0, 0), (1, 1, 1, 0), (0, 0, 1, 1))"
    csv = (
        "labels,syndromes,quantity\n"
        f'0,"{valid}",4\n'
        f'1,"{valid}",2\n'
        '1,"((0, 1),)",3\n'
    )
    data = _zip_bytes({"d-3_pfr-0.001000_nb-10M.csv": csv})

    monkeypatch.setattr(prepare_data, "get_object_bytes", lambda client, bucket, key: data)

    records, traces, issues = prepare_data.prepare_qec_syndromes(client=object(), bucket="bucket", run_id="run-1")

    assert len(records) == 2
    assert len(issues) == 3
    assert {issue["action"] for issue in issues} == {"excluded_from_silver", "documented", "flagged"}
    excluded = [issue for issue in issues if issue["action"] == "excluded_from_silver"][0]
    assert excluded["rule_id"] == "syndrome_row_validity"
    documented = [issue for issue in issues if issue["action"] == "documented"][0]
    assert documented["rule_id"] == "labels_column_naming"
    flagged = [issue for issue in issues if issue["action"] == "flagged"][0]
    assert flagged["rule_id"] == "quantity_reconciliation"
    assert [record["quantity"] for record in records] == [4, 2]
    assert all(len(record["syndrome_bits"]) == 16 for record in records)
    assert all(record["source_record_id"] for record in records)
    assert len(traces) == 2
    assert {trace["source_record_id"] for trace in traces} == {r["source_record_id"] for r in records}


def test_prepare_qec_syndromes_flags_quantity_mismatch(monkeypatch) -> None:
    valid = "((0, 1, 0, 1), (1, 0, 0, 0), (1, 1, 1, 0), (0, 0, 1, 1))"
    csv = (
        "labels,syndromes,quantity\n"
        f'0,"{valid}",4000000\n'
        f'1,"{valid}",1000000\n'
    )
    data = _zip_bytes({"d-3_pfr-0.001000_nb-10M.csv": csv})

    monkeypatch.setattr(prepare_data, "get_object_bytes", lambda client, bucket, key: data)

    _, _, issues = prepare_data.prepare_qec_syndromes(client=object(), bucket="bucket", run_id="run-1")

    flagged = [issue for issue in issues if issue["rule_id"] == "quantity_reconciliation"]
    assert len(flagged) == 1
    assert flagged[0]["action"] == "flagged"
    assert "5000000" in flagged[0]["reason"]
    assert "10000000" in flagged[0]["reason"]


def test_prepare_qec_syndromes_accepts_reconciled_quantity(monkeypatch) -> None:
    valid = "((0, 1, 0, 1), (1, 0, 0, 0), (1, 1, 1, 0), (0, 0, 1, 1))"
    csv = (
        "labels,syndromes,quantity\n"
        f'0,"{valid}",6000000\n'
        f'1,"{valid}",4000000\n'
    )
    data = _zip_bytes({"d-3_pfr-0.001000_nb-10M.csv": csv})

    monkeypatch.setattr(prepare_data, "get_object_bytes", lambda client, bucket, key: data)

    _, _, issues = prepare_data.prepare_qec_syndromes(client=object(), bucket="bucket", run_id="run-1")

    assert not [issue for issue in issues if issue["rule_id"] == "quantity_reconciliation"]


def test_prepare_qec_syndromes_source_record_id_is_repeatable(monkeypatch) -> None:
    valid = "((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))"
    csv = f'labels,syndromes,quantity\n0,"{valid}",10\n'
    data = _zip_bytes({"d-3_pfr-0.001000_nb-10M.csv": csv})

    monkeypatch.setattr(prepare_data, "get_object_bytes", lambda client, bucket, key: data)

    records_a, _, _ = prepare_data.prepare_qec_syndromes(client=object(), bucket="bucket", run_id="run-1")
    records_b, _, _ = prepare_data.prepare_qec_syndromes(client=object(), bucket="bucket", run_id="run-2")

    assert records_a[0]["source_record_id"] == records_b[0]["source_record_id"]


_MINIMAL_QASM = """OPENQASM 2.0;
qreg q[1];
creg c[1];
measure q[0] -> c[0];
"""


def test_prepare_qasmbench_accepts_complete_benchmark(monkeypatch) -> None:
    data = _zip_bytes(
        {
            "small/demo/demo.qasm": _MINIMAL_QASM,
            "small/demo/demo_transpiled.qasm": _MINIMAL_QASM,
        }
    )
    monkeypatch.setattr(prepare_data, "get_object_bytes", lambda client, bucket, key: data)

    circuits, _, _, _, _ = prepare_data.prepare_qasmbench(client=object(), bucket="bucket", run_id="run-1")

    assert len(circuits) == 2
    assert {c["variant"] for c in circuits} == {"source", "transpiled"}


def test_prepare_qasmbench_raises_on_missing_transpiled_variant(monkeypatch) -> None:
    data = _zip_bytes({"small/demo/demo.qasm": _MINIMAL_QASM})
    monkeypatch.setattr(prepare_data, "get_object_bytes", lambda client, bucket, key: data)

    try:
        prepare_data.prepare_qasmbench(client=object(), bucket="bucket", run_id="run-1")
        raise AssertionError("expected RuntimeError for missing companion file")
    except RuntimeError as error:
        assert "demo_transpiled.qasm" in str(error)
