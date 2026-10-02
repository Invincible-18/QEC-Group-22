"""Create the required analyses and ML input tables from PostgreSQL Gold.

Write the Gold queries/views, joins, stable example IDs, and export logic.
A thin export step may execute SQL, use the course split helpers, validate the
fixed contracts, and write Parquet. It must not re-read Bronze or Silver.

For now this stage only runs the analysis queries: every q*.sql file in sql/ is
run against Gold and its result is saved as a CSV in results/part1/analysis/.
The ML table export still has to be added here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import psycopg

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import postgres_connection
from quantum_lake_student.models import StageResult
from quantum_lake_student.stages.prepare_data import _results_root


SQL_DIR = Path(__file__).resolve().parents[3] / "sql"


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


def run(run_id: str) -> StageResult:
    result = StageResult(stage="build_ml_tables", run_id=run_id)
    with postgres_connection(Settings.from_environment()) as connection:
        written = run_analyses(connection, _results_root() / "analysis")
    result.output_count = len(written)
    result.finish()
    return result
