import pytest

from quantum_lake_student.tracing import (
    GOOGLE_SHOT_MEMBERS,
    google_shot_source_record_id,
    google_shot_trace_rows,
)


BRONZE_OBJECT = "bronze/source=google_qec/google-surface-code-curated.zip"
INPUT_SHA256 = "5d6a24f89f055883a49910979490be4baef54d28bf9a0f8e096a1d6c46d1ea56"
EXPERIMENT_ID = "surface_code_bX_d3_r25_center_3_5"

TRACE_COLUMNS = {
    "source_record_id",
    "source_name",
    "bronze_object",
    "archive_member",
    "record_locator",
    "input_sha256",
}


def identifier(shot_index: int) -> str:
    return google_shot_source_record_id(
        bronze_object=BRONZE_OBJECT,
        experiment_id=EXPERIMENT_ID,
        shot_index=shot_index,
    )


def trace_rows(shot_index: int = 0) -> list[dict[str, str]]:
    return google_shot_trace_rows(
        bronze_object=BRONZE_OBJECT,
        input_sha256=INPUT_SHA256,
        experiment_id=EXPERIMENT_ID,
        shot_index=shot_index,
    )


def test_identifier_matches_the_formula_the_silver_rows_use() -> None:
    # Agreed with the parsing stage. If this value changes, every row in
    # source_trace.parquet stops resolving to its Silver row.
    assert identifier(0) == (
        "cdb13538a4fb6bcb43decdaeaa11999f55eeb77847fb18a8fadefc50efc49902"
    )


def test_identifier_is_repeatable() -> None:
    assert identifier(0) == identifier(0)


def test_identifier_depends_on_the_shot() -> None:
    assert identifier(0) != identifier(1)


def test_identifier_accepts_a_numpy_integer() -> None:
    numpy = pytest.importorskip("numpy")
    assert identifier(numpy.int64(7)) == identifier(7)


def test_one_shot_produces_one_row_per_bronze_member() -> None:
    rows = trace_rows()
    assert len(rows) == len(GOOGLE_SHOT_MEMBERS) == 8
    assert [row["archive_member"] for row in rows] == [
        f"{EXPERIMENT_ID}/{member}" for member in GOOGLE_SHOT_MEMBERS
    ]


def test_every_row_of_a_shot_shares_one_identifier() -> None:
    rows = trace_rows()
    assert {row["source_record_id"] for row in rows} == {identifier(0)}


def test_rows_carry_exactly_the_contract_columns() -> None:
    assert all(set(row) == TRACE_COLUMNS for row in trace_rows())


def test_record_locator_names_the_shot() -> None:
    assert {row["record_locator"] for row in trace_rows(12)} == {"shot=12"}
