"""Load the syndrome Silver table into a normalized PostgreSQL Gold model.

Gold row meanings:
- syndrome_experiment: one simulated experiment at one configured fault rate.
- syndrome_pattern: one distinct 4-round by 4-check syndrome bit pattern.
- syndrome_observation: one original aggregate Silver row, with its label and
  physical-observation quantity, linked to an experiment and pattern.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import pandas as pd
import psycopg
import pyarrow as pa


SYNDROME_SILVER_TABLES = {
    "syndrome_observation": "silver/qec_syndromes/syndrome_observation.parquet",
}

SYNDROME_GOLD_TABLES = ("syndrome_experiment", "syndrome_pattern", "syndrome_observation")

REQUIRED_COLUMNS = {
    "source_record_id",
    "experiment_id",
    "physical_fault_rate",
    "syndrome_bits",
    "round_count",
    "check_count",
    "logical_error_label",
    "quantity",
}

CREATE_EXPERIMENT_TABLE = """
CREATE TABLE IF NOT EXISTS syndrome_experiment (
    experiment_id text PRIMARY KEY,
    physical_fault_rate double precision NOT NULL
        CHECK (physical_fault_rate >= 0 AND physical_fault_rate < 1)
)
"""
CREATE_PATTERN_TABLE = """
CREATE TABLE IF NOT EXISTS syndrome_pattern (
    syndrome_pattern_id text PRIMARY KEY,
    syndrome_bits bytea NOT NULL UNIQUE
        CHECK (octet_length(syndrome_bits) = 16)
)
"""
CREATE_OBSERVATION_TABLE = """
CREATE TABLE IF NOT EXISTS syndrome_observation (
    source_record_id text PRIMARY KEY,
    experiment_id text NOT NULL
        REFERENCES syndrome_experiment (experiment_id),
    syndrome_pattern_id text NOT NULL
        REFERENCES syndrome_pattern (syndrome_pattern_id),
    logical_error_label boolean NOT NULL,
    quantity bigint NOT NULL CHECK (quantity > 0)
)
"""
CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS syndrome_observation_experiment_idx "
    "ON syndrome_observation (experiment_id)",
    "CREATE INDEX IF NOT EXISTS syndrome_observation_pattern_label_idx "
    "ON syndrome_observation (syndrome_pattern_id, logical_error_label)",
)
TABLE_COMMENTS = (
    "COMMENT ON TABLE syndrome_experiment IS "
    "'One simulated syndrome experiment at a physical fault rate.'",
    "COMMENT ON TABLE syndrome_pattern IS "
    "'One distinct 16-value syndrome pattern, ordered by round then check.'",
    "COMMENT ON TABLE syndrome_observation IS "
    "'One original aggregate CSV observation, linked to its experiment and syndrome pattern.'",
)


def _as_syndrome_bytes(value: Any) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError("syndrome_bits must be binary data")
    bits = bytes(value)
    if len(bits) != 16:
        raise ValueError("syndrome_bits must contain exactly 16 values")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("syndrome_bits values must be binary")
    return bits


def prepare_gold_rows(
    silver: pd.DataFrame,
) -> tuple[list[tuple[str, float]], list[tuple[str, bytes]], list[tuple[str, str, str, bool, int]]]:
    """Validate Silver rows and create deduplicated Gold dimensions and facts."""
    missing = REQUIRED_COLUMNS - set(silver.columns)
    if missing:
        raise ValueError(f"Silver table is missing columns: {sorted(missing)}")

    experiments: dict[str, float] = {}
    patterns: dict[str, bytes] = {}
    observations: list[tuple[str, str, str, bool, int]] = []
    seen_source_ids: set[str] = set()

    for row in silver.itertuples(index=False):
        values = row._asdict()
        source_record_id = str(values["source_record_id"])
        experiment_id = str(values["experiment_id"])
        if not source_record_id or source_record_id == "None":
            raise ValueError("source_record_id must be non-empty")
        if source_record_id in seen_source_ids:
            raise ValueError(f"duplicate source_record_id: {source_record_id}")
        seen_source_ids.add(source_record_id)

        physical_fault_rate = float(values["physical_fault_rate"])
        if not math.isfinite(physical_fault_rate) or not 0 <= physical_fault_rate < 1:
            raise ValueError(f"invalid physical_fault_rate for {source_record_id}")
        if int(values["round_count"]) != 4 or int(values["check_count"]) != 4:
            raise ValueError(f"invalid syndrome dimensions for {source_record_id}")

        label_value = values["logical_error_label"]
        if not isinstance(label_value, (bool,)):
            raise ValueError(f"logical_error_label must be boolean for {source_record_id}")
        quantity_value = values["quantity"]
        quantity = int(quantity_value)
        if quantity != quantity_value or quantity <= 0:
            raise ValueError(f"quantity must be a positive integer for {source_record_id}")

        syndrome_bits = _as_syndrome_bytes(values["syndrome_bits"])
        pattern_id = hashlib.sha256(syndrome_bits).hexdigest()
        previous_rate = experiments.setdefault(experiment_id, physical_fault_rate)
        if previous_rate != physical_fault_rate:
            raise ValueError(f"experiment {experiment_id} has conflicting fault rates")
        patterns[pattern_id] = syndrome_bits
        observations.append(
            (source_record_id, experiment_id, pattern_id, bool(label_value), quantity)
        )

    return (
        sorted(experiments.items()),
        sorted(patterns.items()),
        observations,
    )


def load_syndrome(connection: psycopg.Connection, silver: dict[str, pa.Table]) -> dict[str, int]:
    """Replace the syndrome-owned Gold tables from Silver.

    Called by load_postgres.rebuild_gold() inside its single transaction, with
    search_path set to the Gold schema. Returns the row count of each table.
    """
    experiments, patterns, observations = prepare_gold_rows(
        silver["syndrome_observation"].to_pandas()
    )

    for statement in (
        CREATE_EXPERIMENT_TABLE,
        CREATE_PATTERN_TABLE,
        CREATE_OBSERVATION_TABLE,
        *CREATE_INDEXES,
        *TABLE_COMMENTS,
    ):
        connection.execute(statement)

    connection.execute("DELETE FROM syndrome_observation")
    connection.execute("DELETE FROM syndrome_pattern")
    connection.execute("DELETE FROM syndrome_experiment")

    with connection.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO syndrome_experiment "
            "(experiment_id, physical_fault_rate) VALUES (%s, %s)",
            experiments,
        )
        cursor.executemany(
            "INSERT INTO syndrome_pattern "
            "(syndrome_pattern_id, syndrome_bits) VALUES (%s, %s)",
            patterns,
        )
        cursor.executemany(
            "INSERT INTO syndrome_observation "
            "(source_record_id, experiment_id, syndrome_pattern_id, "
            "logical_error_label, quantity) VALUES (%s, %s, %s, %s, %s)",
            observations,
        )
        return {
            table: cursor.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in SYNDROME_GOLD_TABLES
        }