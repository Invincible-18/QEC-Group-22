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


# --- Google: shot alignment and tracing -------------------------------------

_GOOGLE_PROPERTIES = """
basis: X
distance: 3
rounds: 1
shots: 2
center_data_qubit_row: 0
center_data_qubit_col: 0
circuit_measurements: 1
circuit_detectors: 1
circuit_sweep_bits: 1
"""


def _google_archive(overrides: dict[str, bytes] | None = None) -> bytes:
    """One 2-shot experiment with 1-bit records, so every b8 record is one byte."""
    two_lines = b"0\n1\n"
    members = {
        "properties.yml": _GOOGLE_PROPERTIES.encode(),
        "measurements.b8": b"\x00\x01",
        "sweep.b8": b"\x00\x01",
        "detection_events.b8": b"\x00\x01",
        "obs_flips_actual.01": two_lines,
        **{name: two_lines for name in prepare_data.GOOGLE_PREDICTION_FILES.values()},
        **(overrides or {}),
    }
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as bundle:
        for name, content in members.items():
            bundle.writestr(f"exp0/{name}", content)
    return buffer.getvalue()


def test_prepare_google_qec_accepts_aligned_companion_files(monkeypatch) -> None:
    data = _google_archive()
    monkeypatch.setattr(prepare_data, "get_object_bytes", lambda client, bucket, key: data)

    experiments, shots, _, issues = prepare_data.prepare_google_qec(
        client=object(), bucket="bucket", run_id="run-1"
    )

    assert len(experiments) == 1
    assert [shot["shot_index"] for shot in shots] == [0, 1]
    assert not issues


def test_prepare_google_qec_stops_on_short_prediction_file(monkeypatch) -> None:
    # properties.yml declares 2 shots, but this prediction file has only 1
    data = _google_archive({"obs_flips_predicted_by_pymatching.01": b"0\n"})
    monkeypatch.setattr(prepare_data, "get_object_bytes", lambda client, bucket, key: data)

    try:
        prepare_data.prepare_google_qec(client=object(), bucket="bucket", run_id="run-1")
        raise AssertionError("expected RuntimeError for a misaligned companion file")
    except RuntimeError as error:
        assert "obs_flips_predicted_by_pymatching.01" in str(error)
        assert "expected 2" in str(error)


def test_every_google_silver_row_is_traced(monkeypatch) -> None:
    data = _google_archive()
    monkeypatch.setattr(prepare_data, "get_object_bytes", lambda client, bucket, key: data)

    experiments, shots, traces, _ = prepare_data.prepare_google_qec(
        client=object(), bucket="bucket", run_id="run-1"
    )

    traced: dict[str, set[str]] = {}
    for trace in traces:
        traced.setdefault(trace["source_record_id"], set()).add(trace["archive_member"])
    assert traced[experiments[0]["source_record_id"]] == {"exp0/properties.yml"}
    for shot in shots:
        # a shot is assembled from all eight aligned files, so all eight are traced
        assert traced[shot["source_record_id"]] == {
            f"exp0/{member}" for member in prepare_data.GOOGLE_SHOT_MEMBERS
        }


# --- QASMBench: tracing ---------------------------------------------------------

# The real small/qec_sm_n5 circuit from the release.
_QEC_SM_N5 = """// Repetition code syndrome measurement
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
qreg a[2];
creg c[3];
creg syn[2];
gate syndrome d1,d2,d3,a1,a2
{
  cx d1,a1; cx d2,a1;
  cx d2,a2; cx d3,a2;
}
x q[0]; // error
barrier q;
syndrome q[0],q[1],q[2],a[0],a[1];
measure a -> syn;
if(syn==1) x q[0];
if(syn==2) x q[2];
if(syn==3) x q[1];
measure q -> c;
"""


def test_every_qasmbench_silver_row_is_traced(monkeypatch) -> None:
    data = _zip_bytes(
        {
            "small/qec_sm_n5/qec_sm_n5.qasm": _QEC_SM_N5,
            "small/qec_sm_n5/qec_sm_n5_transpiled.qasm": _QEC_SM_N5,
        }
    )
    monkeypatch.setattr(prepare_data, "get_object_bytes", lambda client, bucket, key: data)

    circuits, checks, corrections, traces, _ = prepare_data.prepare_qasmbench(
        client=object(), bucket="bucket", run_id="run-1"
    )

    assert (len(circuits), len(checks), len(corrections)) == (2, 4, 6)
    traced_ids = {trace["source_record_id"] for trace in traces}
    for row in [*circuits, *checks, *corrections]:
        assert row["source_record_id"] in traced_ids
