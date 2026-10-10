"""Relational integration stage.

Create the student-designed PostgreSQL tables and load them as one all-or-
nothing update. Primary/foreign keys, value checks, and indexes are part of the
deliverable. Repeated loads must not create duplicate records.

How it is done here: one transaction drops the gold schema, creates the tables
again, loads every source from Silver and checks the result. DDL is
transactional in PostgreSQL, so if any step fails the drop is undone too and
the previous gold stays as it was. All keys come from the data, so a rerun
gives the same rows and ids and never adds duplicates.

Each source has its own module (gold_google.py, ...) with its tables, load and
checks; this file only runs them together in that one transaction.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import psycopg
import pyarrow as pa
from psycopg import sql

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client, postgres_connection
from quantum_lake_student.io_utils import read_parquet
from quantum_lake_student.models import StageResult
from quantum_lake_student.stages.gold_google import (
    GOOGLE_SILVER_TABLES,
    GOOGLE_TABLES,
    load_google,
    validate_google,
)
from quantum_lake_student.stages.gold_qasmbench import (
    QASMBENCH_SILVER_TABLES,
    QASMBENCH_TABLES,
    load_qasmbench,
    validate_qasmbench,
)
from quantum_lake_student.stages.gold_syndrome import SYNDROME_SILVER_TABLES, load_syndrome
from quantum_lake_student.stages.prepare_data import _results_root


GOLD_SCHEMA = "gold"

SILVER_TABLES = {**GOOGLE_SILVER_TABLES, **QASMBENCH_SILVER_TABLES, **SYNDROME_SILVER_TABLES}


# ============================================================================
# all-or-nothing rebuild
# ============================================================================


def rebuild_gold(
    connection: psycopg.Connection,
    silver: dict[str, pa.Table],
    *,
    schema: str = GOLD_SCHEMA,
) -> dict[str, int]:
    """Rebuild ``schema`` from Silver in one transaction; if anything fails, nothing changes.

    ``connection`` has to be in autocommit mode so the block below is the only
    transaction. Returns the row count of every Gold table that was loaded.
    """
    with connection.transaction():
        schema_name = sql.Identifier(schema)
        connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(schema_name))
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(schema_name))
        connection.execute(sql.SQL("SET LOCAL search_path TO {}").format(schema_name))

        connection.execute(GOOGLE_TABLES)
        counts = load_google(connection, silver["google_experiment"], silver["google_shot"])
        validate_google(connection, silver["google_experiment"], silver["google_shot"])

        connection.execute(QASMBENCH_TABLES)
        counts |= load_qasmbench(connection, silver)
        validate_qasmbench(connection, silver)

        counts |= load_syndrome(connection, silver)
    return counts


# ============================================================================
# evidence files
# ============================================================================


def _record_counts(run_id: str, silver: dict[str, pa.Table], counts: dict[str, int]) -> None:
    """Add the Silver-to-Gold counts to row_counts.json and the Gold counts to run.json."""
    root = _results_root()
    root.mkdir(parents=True, exist_ok=True)

    row_counts_path = root / "row_counts.json"
    row_counts = json.loads(row_counts_path.read_text()) if row_counts_path.exists() else {}
    if not isinstance(row_counts.get("silver_to_gold"), dict):
        row_counts["silver_to_gold"] = {}
    row_counts["silver_to_gold"]["google_qec"] = {
        "silver/google_qec/experiment.parquet": {
            "silver_rows": silver["google_experiment"].num_rows,
            "gold_table": f"{GOLD_SCHEMA}.hardware_experiment",
            "gold_rows": counts["hardware_experiment"],
        },
        "silver/google_qec/shot.parquet": {
            "silver_rows": silver["google_shot"].num_rows,
            "gold_table": f"{GOLD_SCHEMA}.shot",
            "gold_rows": counts["shot"],
        },
        "derived_in_gold": {
            f"{GOLD_SCHEMA}.shot_prediction": counts["shot_prediction"],
            f"{GOLD_SCHEMA}.detector_position_summary": counts["detector_position_summary"],
            f"{GOLD_SCHEMA}.decoder": counts["decoder"],
        },
    }
    row_counts["silver_to_gold"]["qasmbench"] = {
        QASMBENCH_SILVER_TABLES[silver_name]: {
            "silver_rows": silver[silver_name].num_rows,
            "gold_table": f"{GOLD_SCHEMA}.{gold_table}",
            "gold_rows": counts[gold_table],
        }
        for silver_name, gold_table in (
            ("qasm_circuit", "circuit"),
            ("qasm_stabilizer_check", "stabilizer_check"),
            ("qasm_conditional_correction", "conditional_correction"),
        )
    }
    row_counts["silver_to_gold"]["qec_syndromes"] = {
        SYNDROME_SILVER_TABLES["syndrome_observation"]: {
            "silver_rows": silver["syndrome_observation"].num_rows,
            "gold_table": f"{GOLD_SCHEMA}.syndrome_observation",
            "gold_rows": counts["syndrome_observation"],
        },
        "derived_in_gold": {
            f"{GOLD_SCHEMA}.syndrome_experiment": counts["syndrome_experiment"],
            f"{GOLD_SCHEMA}.syndrome_pattern": counts["syndrome_pattern"],
        },
    }
    row_counts_path.write_text(json.dumps(row_counts, indent=2))

    run_path = root / "run.json"
    run_record = json.loads(run_path.read_text()) if run_path.exists() else {"run_id": run_id}
    run_record.setdefault("output_row_counts", {}).update(
        {f"{GOLD_SCHEMA}.{table}": rows for table, rows in counts.items()}
    )
    run_record["finished_at"] = datetime.now(UTC).isoformat()
    run_path.write_text(json.dumps(run_record, indent=2))


def _google_gold_record(connection: psycopg.Connection, source_record_id: str) -> tuple | None:
    schema = sql.Identifier(GOLD_SCHEMA)
    shot = connection.execute(
        sql.SQL(
            "SELECT example_id, experiment_id, shot_index, actual_observable_flip "
            "FROM {}.shot WHERE source_record_id = %s"
        ).format(schema),
        (source_record_id,),
    ).fetchone()
    if shot is None:
        return None
    example_id, experiment_id, shot_index, actual = shot
    predictions = dict(
        connection.execute(
            sql.SQL(
                "SELECT decoder_name, predicted_flip FROM {}.shot_prediction "
                "WHERE experiment_id = %s AND shot_index = %s ORDER BY decoder_name"
            ).format(schema),
            (experiment_id, shot_index),
        ).fetchall()
    )
    return example_id, {
        "table": f"{GOLD_SCHEMA}.shot",
        "experiment_id": experiment_id,
        "shot_index": shot_index,
        "actual_observable_flip": actual,
        "predictions": predictions,
    }


def _syndrome_gold_record(connection: psycopg.Connection, source_record_id: str) -> tuple | None:
    row = connection.execute(
        sql.SQL(
            "SELECT example.example_id, example.experiment_id, example.syndrome_pattern_id, "
            "       example.logical_error_label, observation.quantity "
            "FROM {schema}.syndrome_example AS example "
            "JOIN {schema}.syndrome_observation AS observation USING (source_record_id) "
            "WHERE example.source_record_id = %s"
        ).format(schema=sql.Identifier(GOLD_SCHEMA)),
        (source_record_id,),
    ).fetchone()
    if row is None:
        return None
    example_id, experiment_id, pattern_id, label, quantity = row
    return example_id, {
        "table": f"{GOLD_SCHEMA}.syndrome_observation",
        "view": f"{GOLD_SCHEMA}.syndrome_example",
        "experiment_id": experiment_id,
        "syndrome_pattern_id": pattern_id,
        "logical_error_label": label,
        "quantity": quantity,
    }


def _add_gold_to_trace_example(connection: psycopg.Connection) -> None:
    """Extend the syndrome and Google trace examples with their Gold records.

    Each chain then reads from the ML example id, through Gold and Silver, down to
    the Bronze files. The prediction step gets added once Part II exists.
    """
    path = _results_root() / "trace_examples.json"
    if not path.exists():
        return
    examples = json.loads(path.read_text())

    for key, gold_record in (
        ("syndrome_prediction", _syndrome_gold_record),
        ("google_prediction", _google_gold_record),
    ):
        example = examples.get(key)
        if not example:
            continue
        found = gold_record(connection, example["source_record_id"])
        if found is None:
            continue
        example_id, record = found
        rest = {k: v for k, v in example.items() if k not in ("example_id", "gold_record")}
        examples[key] = {"example_id": example_id, "gold_record": record, **rest}
    path.write_text(json.dumps(examples, indent=2))


def run(run_id: str) -> StageResult:
    result = StageResult(stage="load_postgres", run_id=run_id)

    settings = Settings.from_environment()
    client = minio_client(settings)
    silver = {
        name: read_parquet(client, settings.s3_bucket, key)
        for name, key in SILVER_TABLES.items()
    }
    result.input_count = sum(table.num_rows for table in silver.values())

    with postgres_connection(settings) as connection:
        connection.autocommit = True
        counts = rebuild_gold(connection, silver)
        _add_gold_to_trace_example(connection)

    _record_counts(run_id, silver, counts)
    result.output_count = sum(counts.values())
    result.finish()
    return result
