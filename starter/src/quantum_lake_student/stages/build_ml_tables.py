"""Create the required analyses and ML input tables from PostgreSQL Gold.

Write the Gold queries/views, joins, stable example IDs, and export logic.
A thin export step may execute SQL, use the course split helpers, validate the
fixed contracts, and write Parquet. It must not re-read Bronze or Silver.

Analyses: every q*.sql file in sql/ is run against Gold and its result is saved
as a CSV in results/part1/analysis/.

ML tables: each table comes from a committed query in sql/ml_*.sql run against
Gold. Python only adds the course data_split, checks the contract from
required-ml-tables.md, and writes Parquet to the ml/ zone. Nothing is written
unless every check passes. The syndrome table is added once its Gold tables exist.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import psycopg
import pyarrow as pa
from psycopg import sql
from psycopg.rows import dict_row

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client, postgres_connection
from quantum_lake_student.formats import b8_record_bytes, check_padding_zero
from quantum_lake_student.io_utils import write_parquet
from quantum_lake_student.ml import (
    GOOGLE_META_PREDICTION_COLUMNS,
    MODEL_SPLITS,
    google_data_split,
    google_meta_model_input,
    unpack_little_endian_bits,
)
from quantum_lake_student.models import StageResult
from quantum_lake_student.stages.load_postgres import GOLD_SCHEMA
from quantum_lake_student.stages.prepare_data import _results_root


SQL_DIR = Path(__file__).resolve().parents[3] / "sql"

ML_GOOGLE_OBJECT = "ml/ml_google_decoder_example.parquet"

# Column order and types from required-ml-tables.md.
ML_GOOGLE_SCHEMA = pa.schema(
    [
        ("example_id", pa.string()),
        ("experiment_id", pa.string()),
        ("shot_index", pa.int64()),
        ("distance", pa.int32()),
        ("rounds", pa.int32()),
        ("center_row", pa.int32()),
        ("center_col", pa.int32()),
        ("detector_count", pa.int32()),
        ("detector_event_count", pa.int32()),
        ("detector_bits", pa.binary()),
        *((column, pa.bool_()) for column in GOOGLE_META_PREDICTION_COLUMNS),
        ("actual_observable_flip", pa.bool_()),
        ("data_split", pa.string()),
    ]
)


def run_analyses(
    connection: psycopg.Connection, output_dir: Path, *, sql_dir: Path = SQL_DIR
) -> list[Path]:
    """Run every q*.sql file in ``sql_dir`` and write each result to ``output_dir`` as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for query_file in sorted(sql_dir.glob("q*.sql")):
        with connection.cursor() as cursor:
            cursor.execute(query_file.read_text())
            header = [column.name for column in cursor.description]
            rows = cursor.fetchall()
        target = output_dir / f"{query_file.stem}.csv"
        with target.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        written.append(target)
    return written


# ============================================================================
# ml_google_decoder_example
# ============================================================================


def check_google_examples(rows: list[dict], gold_shot_count: int) -> None:
    """Raise if the exported rows break the contract in required-ml-tables.md."""
    failures: list[str] = []

    if len(rows) != gold_shot_count:
        failures.append(f"{len(rows)} rows exported but Gold has {gold_shot_count} shots")
    if len({row["example_id"] for row in rows}) != len(rows):
        failures.append("example_id is not unique")
    if len({(row["experiment_id"], row["shot_index"]) for row in rows}) != len(rows):
        failures.append("an (experiment_id, shot_index) pair appears more than once")

    split_sizes = Counter(row["data_split"] for row in rows)
    failures.extend(f"{split} split is empty" for split in MODEL_SPLITS if not split_sizes[split])

    # distance-3 and distance-5 detector matrices have different widths, so every
    # distance has to use exactly one width
    widths: dict[int, set[int]] = defaultdict(set)
    bad = Counter()
    unpacked_experiments: set[str] = set()
    for row in rows:
        widths[row["distance"]].add(row["detector_count"])
        bits, width = row["detector_bits"], row["detector_count"]

        if len(bits) != b8_record_bytes(width):
            bad["detector_bits length is not ceil(detector_count / 8)"] += 1
            continue
        if not check_padding_zero(bits, width):
            bad["detector_bits padding bits are not zero"] += 1
        if int.from_bytes(bits, "little").bit_count() != row["detector_event_count"]:
            bad["detector_event_count differs from the set detector bits"] += 1

        outcomes = (row["actual_observable_flip"], *(row[c] for c in GOOGLE_META_PREDICTION_COLUMNS))
        if any(not isinstance(value, bool) for value in outcomes):
            bad["actual flip or a decoder prediction is missing or not binary"] += 1
            continue
        try:
            google_meta_model_input(row)
        except (TypeError, ValueError):
            bad["google_meta_model_input rejects the row"] += 1

        # the course unpacking helper on one shot per experiment
        if row["experiment_id"] not in unpacked_experiments:
            unpacked_experiments.add(row["experiment_id"])
            if sum(unpack_little_endian_bits(bits, width)) != row["detector_event_count"]:
                bad["unpack_little_endian_bits disagrees with detector_event_count"] += 1

    failures.extend(
        f"distance {distance} mixes detector widths {sorted(found)}"
        for distance, found in widths.items()
        if len(found) != 1
    )
    failures.extend(f"{rule} ({count} rows)" for rule, count in bad.items())

    if failures:
        raise RuntimeError("ml_google_decoder_example contract failed: " + "; ".join(failures))


def build_google_ml_table(
    connection: psycopg.Connection,
    *,
    schema: str = GOLD_SCHEMA,
    sql_dir: Path = SQL_DIR,
) -> pa.Table:
    """Query Gold for the Google ML table, add data_split, and check the contract."""
    query = (sql_dir / "ml_google_decoder_example.sql").read_text()
    with connection.transaction():
        connection.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
        gold_shot_count = connection.execute("SELECT count(*) FROM shot").fetchone()[0]
        with connection.cursor(row_factory=dict_row) as cursor:
            rows = cursor.execute(query).fetchall()

    for row in rows:
        row["data_split"] = google_data_split(row["shot_index"])
    check_google_examples(rows, gold_shot_count)
    return pa.Table.from_pylist(rows, schema=ML_GOOGLE_SCHEMA)


# ============================================================================
# stage
# ============================================================================


def _record_ml_counts(counts: dict[str, int]) -> None:
    run_path = _results_root() / "run.json"
    if not run_path.exists():
        return
    run_record = json.loads(run_path.read_text())
    run_record.setdefault("output_row_counts", {}).update(counts)
    run_path.write_text(json.dumps(run_record, indent=2))


def run(run_id: str) -> StageResult:
    result = StageResult(stage="build_ml_tables", run_id=run_id)
    settings = Settings.from_environment()

    with postgres_connection(settings) as connection:
        connection.autocommit = True
        run_analyses(connection, _results_root() / "analysis")
        google = build_google_ml_table(connection)

    write_parquet(minio_client(settings), settings.s3_bucket, ML_GOOGLE_OBJECT, google)
    _record_ml_counts({ML_GOOGLE_OBJECT: google.num_rows})

    result.output_count = google.num_rows
    result.finish()
    return result
