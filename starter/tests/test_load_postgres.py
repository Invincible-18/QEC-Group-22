import hashlib
from uuid import uuid4

import psycopg
import pyarrow as pa
import pytest
from psycopg import sql

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import postgres_connection
from gold_test_data import other_sources_silver
from quantum_lake_student.stages.gold_google import GOOGLE_GOLD_TABLES, validate_google
from quantum_lake_student.stages.load_postgres import rebuild_gold
from quantum_lake_student.stages.prepare_data import (
    GOOGLE_EXPERIMENT_SCHEMA,
    GOOGLE_SHOT_SCHEMA,
)


EXPERIMENT_ID = "surface_code_bX_d3_r25_center_3_5"
# each shot fires one detector: 0, 7 and 8 (little-endian, so 0x80 in the first
# byte is detector 7, not detector 0)
DETECTOR_BITS = (b"\x01\x00", b"\x80\x00", b"\x00\x01")
ACTUAL = (True, False, True)


def silver(
    event_counts: tuple[int, ...] = (1, 1, 1),
    detector_bits: tuple[bytes, ...] = DETECTOR_BITS,
) -> dict[str, pa.Table]:
    experiment = pa.Table.from_pylist(
        [
            {
                "source_record_id": "experiment-record",
                "experiment_id": EXPERIMENT_ID,
                "basis": "X",
                "distance": 3,
                "rounds": 25,
                "shots": 3,
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
                "detector_bits": detector_bits[index],
                "detector_event_count": event_counts[index],
                "actual_observable_flip": ACTUAL[index],
                "belief_matching_prediction": ACTUAL[index],
                "correlated_matching_prediction": not ACTUAL[index],
                "pymatching_prediction": True,
                "tensor_network_contraction_prediction": False,
            }
            for index in range(3)
        ],
        schema=GOOGLE_SHOT_SCHEMA,
    )
    return {"google_experiment": experiment, "google_shot": shots, **other_sources_silver()}


@pytest.fixture
def database():
    try:
        connection = postgres_connection(Settings.from_environment())
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL is not reachable")
    connection.autocommit = True
    schema = f"gold_test_{uuid4().hex[:8]}"
    yield connection, schema
    connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
    connection.close()


def query(connection: psycopg.Connection, schema: str, statement: str) -> list[tuple]:
    connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
    return connection.execute(statement).fetchall()


def test_rebuild_loads_one_gold_row_per_silver_row(database) -> None:
    connection, schema = database
    counts = rebuild_gold(connection, silver(), schema=schema)
    assert {table: counts[table] for table in GOOGLE_GOLD_TABLES} == {
        "hardware_experiment": 1,
        "shot": 3,
        "decoder": 4,
        "shot_prediction": 12,
        "detector_position_summary": 16,
    }


def test_packed_detector_bits_are_stored_unchanged(database) -> None:
    # the ML export has to be able to rebuild the packed bytes from Gold alone
    connection, schema = database
    rebuild_gold(connection, silver(), schema=schema)
    rows = query(connection, schema, "SELECT detector_bits FROM shot ORDER BY shot_index")
    assert tuple(bytes(row[0]) for row in rows) == DETECTOR_BITS


def test_validation_notices_a_changed_packed_byte(database) -> None:
    # same row counts and event counts, only one detector byte differs
    connection, schema = database
    rebuild_gold(connection, silver(), schema=schema)
    changed = silver(detector_bits=(b"\x02\x00", b"\x80\x00", b"\x00\x01"))
    connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
    with pytest.raises(RuntimeError, match="detector_bits differs"):
        validate_google(connection, changed["google_experiment"], changed["google_shot"])


def test_detector_positions_are_read_little_endian(database) -> None:
    connection, schema = database
    rebuild_gold(connection, silver(), schema=schema)
    fired = query(
        connection,
        schema,
        "SELECT detector_index FROM detector_position_summary WHERE fired_count > 0 ORDER BY 1",
    )
    assert [row[0] for row in fired] == [0, 7, 8]


def test_each_decoder_prediction_becomes_its_own_row(database) -> None:
    connection, schema = database
    rebuild_gold(connection, silver(), schema=schema)
    rows = query(
        connection,
        schema,
        "SELECT decoder_name, count(*), count(*) FILTER (WHERE predicted_flip) "
        "FROM shot_prediction GROUP BY 1 ORDER BY 1",
    )
    assert rows == [
        ("belief_matching", 3, 2),
        ("correlated_matching", 3, 1),
        ("pymatching", 3, 3),
        ("tensor_network_contraction", 3, 0),
    ]


def test_decoder_error_is_derived_from_prediction_and_actual(database) -> None:
    connection, schema = database
    rebuild_gold(connection, silver(), schema=schema)
    errors = dict(
        query(
            connection,
            schema,
            "SELECT decoder_name, count(*) FILTER (WHERE decoder_error) "
            "FROM decoder_outcome GROUP BY 1",
        )
    )
    assert errors == {
        "belief_matching": 0,
        "correlated_matching": 3,
        "pymatching": 1,
        "tensor_network_contraction": 2,
    }


def test_example_id_is_a_repeatable_hash_of_source_experiment_and_shot(database) -> None:
    connection, schema = database
    rebuild_gold(connection, silver(), schema=schema)
    stored = query(connection, schema, "SELECT example_id FROM shot WHERE shot_index = 0")
    assert stored == [(hashlib.sha256(f"google_qec:{EXPERIMENT_ID}:0".encode()).hexdigest(),)]


def test_rebuilding_twice_gives_identical_gold(database) -> None:
    connection, schema = database
    first_counts = rebuild_gold(connection, silver(), schema=schema)
    first_ids = query(connection, schema, "SELECT example_id, source_record_id FROM shot ORDER BY 1")
    second_counts = rebuild_gold(connection, silver(), schema=schema)
    second_ids = query(connection, schema, "SELECT example_id, source_record_id FROM shot ORDER BY 1")
    assert first_counts == second_counts
    assert first_ids == second_ids


def test_failed_rebuild_keeps_the_previous_gold(database) -> None:
    connection, schema = database
    rebuild_gold(connection, silver(), schema=schema)
    # an event count that doesn't match the packed bits breaks a CHECK halfway
    # through the load
    with pytest.raises(psycopg.errors.CheckViolation):
        rebuild_gold(connection, silver(event_counts=(1, 5, 1)), schema=schema)
    assert query(connection, schema, "SELECT count(*) FROM shot") == [(3,)]
    assert query(connection, schema, "SELECT count(*) FROM shot_prediction") == [(12,)]
