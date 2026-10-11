from pathlib import Path

import pyarrow as pa
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from quantum_lake_student.stages.load_postgres import rebuild_gold
from test_load_postgres import database, silver  # noqa: F401  (fixture reused)


Q3_SQL = Path(__file__).resolve().parents[1] / "sql" / "q3_parity_correction_mapping.sql"


def test_q3_maps_each_correction_to_the_checks_that_fired(database) -> None:
    connection, schema = database
    rebuild_gold(connection, silver(), schema=schema)

    query = Q3_SQL.read_text().replace("gold.", f"{schema}.")
    with connection.cursor(row_factory=dict_row) as cursor:
        rows = cursor.execute(query).fetchall()

    assert [
        (row["condition_value"], row["fired_syndrome_bits"], row["corrected_qubit"])
        for row in rows
    ] == [
        (1, "syn[0]", "q[0]"),
        (2, "syn[1]", "q[2]"),
        (3, "syn[0], syn[1]", "q[1]"),
    ]
    assert all(row["target_in_every_fired_check"] for row in rows)


def test_correction_value_too_large_for_its_register_rolls_back(database) -> None:
    connection, schema = database
    rebuild_gold(connection, silver(), schema=schema)

    broken = silver()
    corrections = broken["qasm_conditional_correction"].to_pylist()
    corrections[0]["condition_value"] = 4  # syn has 2 bits, so 0..3 only
    broken["qasm_conditional_correction"] = pa.Table.from_pylist(
        corrections, schema=broken["qasm_conditional_correction"].schema
    )
    with pytest.raises(RuntimeError, match="value fits their size"):
        rebuild_gold(connection, broken, schema=schema)

    values = connection.execute(
        sql.SQL("SELECT condition_value FROM {}.conditional_correction ORDER BY 1").format(
            sql.Identifier(schema)
        )
    ).fetchall()
    assert values == [(1,), (2,), (3,)]
