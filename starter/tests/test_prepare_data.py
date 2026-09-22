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
    assert len(issues) == 1
    assert issues[0]["action"] == "excluded_from_silver"
    assert [record["quantity"] for record in records] == [4, 2]
    assert all(len(record["syndrome_bits"]) == 16 for record in records)
    assert all(record["source_record_id"] for record in records)
    assert len(traces) == 2
    assert {trace["source_record_id"] for trace in traces} == {r["source_record_id"] for r in records}


def test_prepare_qec_syndromes_source_record_id_is_repeatable(monkeypatch) -> None:
    valid = "((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))"
    csv = f'labels,syndromes,quantity\n0,"{valid}",10\n'
    data = _zip_bytes({"d-3_pfr-0.001000_nb-10M.csv": csv})

    monkeypatch.setattr(prepare_data, "get_object_bytes", lambda client, bucket, key: data)

    records_a, _, _ = prepare_data.prepare_qec_syndromes(client=object(), bucket="bucket", run_id="run-1")
    records_b, _, _ = prepare_data.prepare_qec_syndromes(client=object(), bucket="bucket", run_id="run-2")

    assert records_a[0]["source_record_id"] == records_b[0]["source_record_id"]
