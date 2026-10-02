from uuid import uuid4

import psycopg
import pyarrow as pa
import pytest
from psycopg import sql

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import postgres_connection
from quantum_lake_student.stages.build_ml_tables import (
    ML_GOOGLE_SCHEMA,
    build_google_ml_table,
    check_google_examples,
    run_analyses,
)
from quantum_lake_student.stages.load_postgres import rebuild_gold
from quantum_lake_student.stages.prepare_data import (
    GOOGLE_EXPERIMENT_SCHEMA,
    GOOGLE_SHOT_SCHEMA,
)


SHOTS = 10  # shot 8 is the first validation shot, so 10 shots cover all splits
EXPERIMENT_ID = "surface_code_bX_d3_r25_center_3_5"


def google_silver() -> dict[str, pa.Table]:
    experiment = pa.Table.from_pylist(
        [
            {
                "source_record_id": "experiment-record",
                "experiment_id": EXPERIMENT_ID,
                "basis": "X",
                "distance": 3,
                "rounds": 25,
                "shots": SHOTS,
                "center_row": 3,
                "center_col": 5,
                "measurement_count": 12,
                "detector_count": 16,
            }
        ],
        schema=GOOGLE_EXPERIMENT_SCHEMA,
    )
    shots = pa.Table.from_pylist(
        [
            {
                "source_record_id": f"shot-record-{index}",
                "experiment_id": EXPERIMENT_ID,
                "shot_index": index,
                "measurement_bits": b"\x00\x00",
                "sweep_bits": b"\x00\x00",
                # shot i fires detector i (little-endian within each byte)
                "detector_bits": (1 << index).to_bytes(2, "little"),
                "detector_event_count": 1,
                "actual_observable_flip": index % 2 == 0,
                "belief_matching_prediction": index % 2 == 0,
                "correlated_matching_prediction": index % 2 == 1,
                "pymatching_prediction": True,
                "tensor_network_contraction_prediction": index % 3 == 0,
            }
            for index in range(SHOTS)
        ],
        schema=GOOGLE_SHOT_SCHEMA,
    )
    return {"google_experiment": experiment, "google_shot": shots}


@pytest.fixture
def gold(connection):
    schema = f"gold_test_{uuid4().hex[:8]}"
    rebuild_gold(connection, google_silver(), schema=schema)
    yield connection, schema
    connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def connection():
    try:
        connection = postgres_connection(Settings.from_environment())
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL is not reachable")
    connection.autocommit = True
    yield connection
    connection.close()


def test_every_query_file_becomes_a_csv(connection, tmp_path) -> None:
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    (sql_dir / "q9_example.sql").write_text("SELECT 1 AS answer, 'x' AS label")
    (sql_dir / "helper.sql").write_text("SELECT 2")  # not q*.sql, so not an analysis

    written = run_analyses(connection, tmp_path / "out", sql_dir=sql_dir)

    assert [path.name for path in written] == ["q9_example.csv"]
    assert (tmp_path / "out" / "q9_example.csv").read_text().splitlines() == ["answer,label", "1,x"]


def test_google_ml_table_matches_contract_and_gold(gold) -> None:
    connection, schema = gold
    table = build_google_ml_table(connection, schema=schema)

    assert table.schema == ML_GOOGLE_SCHEMA
    assert table.num_rows == SHOTS
    rows = table.to_pylist()
    silver = google_silver()["google_shot"].to_pylist()
    for row, source in zip(rows, silver, strict=True):
        assert row["shot_index"] == source["shot_index"]
        assert row["detector_bits"] == source["detector_bits"]
        for column in (
            "actual_observable_flip",
            "belief_matching_prediction",
            "correlated_matching_prediction",
            "pymatching_prediction",
            "tensor_network_contraction_prediction",
        ):
            assert row[column] == source[column], column
    assert [row["data_split"] for row in rows] == [
        "train", "test", "train", "test", "train", "test", "train", "test", "validation", "test",
    ]
    gold_ids = {
        example_id
        for (example_id,) in connection.execute(
            sql.SQL("SELECT example_id FROM {}.shot").format(sql.Identifier(schema))
        )
    }
    assert {row["example_id"] for row in rows} == gold_ids


def test_google_ml_table_rejects_missing_prediction(gold) -> None:
    connection, schema = gold
    connection.execute(
        sql.SQL(
            "DELETE FROM {}.shot_prediction WHERE shot_index = 0 AND decoder_name = 'pymatching'"
        ).format(sql.Identifier(schema))
    )
    with pytest.raises(RuntimeError, match="missing or not binary"):
        build_google_ml_table(connection, schema=schema)


def _valid_row(shot_index: int) -> dict:
    return {
        "example_id": f"example-{shot_index}",
        "experiment_id": EXPERIMENT_ID,
        "shot_index": shot_index,
        "distance": 3,
        "detector_count": 12,
        "detector_event_count": 1,
        "detector_bits": b"\x01\x00",
        "belief_matching_prediction": False,
        "correlated_matching_prediction": False,
        "pymatching_prediction": False,
        "tensor_network_contraction_prediction": False,
        "actual_observable_flip": False,
        "data_split": ("train", "test", "validation")[shot_index % 3],
    }


def test_check_google_examples_rejects_nonzero_padding() -> None:
    rows = [_valid_row(index) for index in range(3)]
    check_google_examples(rows, gold_shot_count=3)

    rows[0]["detector_bits"] = b"\x01\x80"  # bit 15 lies in the padding of a 12-bit row
    with pytest.raises(RuntimeError, match="padding"):
        check_google_examples(rows, gold_shot_count=3)


def test_check_google_examples_rejects_empty_split() -> None:
    rows = [_valid_row(index) for index in range(3)]
    for row in rows:
        row["data_split"] = "train"
    with pytest.raises(RuntimeError, match="validation split is empty"):
        check_google_examples(rows, gold_shot_count=3)
