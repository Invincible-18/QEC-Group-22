"""Run Part I twice on unchanged input and check nothing changed.

Takes a fingerprint of every output after each run: row counts, duplicate
counts, and a hash of the identifiers or content of each table. The two
fingerprints are compared and written to results/part1/rerun_check.json.
run.json is left out because its run id and times differ by design.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
from psycopg import sql

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client, postgres_connection
from quantum_lake_student.io_utils import read_local_parquet, read_parquet
from quantum_lake_student.stages.build_ml_tables import ML_GOOGLE_OBJECT, ML_SYNDROME_OBJECT
from quantum_lake_student.stages.build_ml_tables import run as run_build_ml_tables
from quantum_lake_student.stages.load_postgres import GOLD_SCHEMA
from quantum_lake_student.stages.load_postgres import run as run_load_postgres
from quantum_lake_student.stages.prepare_data import _results_root
from quantum_lake_student.stages.prepare_data import run as run_prepare_data
from quantum_lake_student.stages.register_sources import run as run_register_sources


SILVER_OBJECTS = (
    "silver/qec_syndromes/syndrome_observation.parquet",
    "silver/google_qec/experiment.parquet",
    "silver/google_qec/shot.parquet",
    "silver/qasmbench/circuit.parquet",
    "silver/qasmbench/stabilizer_check.parquet",
    "silver/qasmbench/conditional_correction.parquet",
)


def _hash_values(values) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(repr(value).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _id_fingerprint(table: pa.Table, id_column: str) -> dict:
    ids = table.column(id_column).to_pylist()
    return {"rows": len(ids), "duplicate_ids": len(ids) - len(set(ids)), "id_hash": _hash_values(ids)}


def _rows_fingerprint(table: pa.Table, columns: tuple[str, ...]) -> dict:
    rows = list(zip(*(table.column(name).to_pylist() for name in columns)))
    return {"rows": len(rows), "content_hash": _hash_values(rows)}


def fingerprint(settings: Settings) -> dict:
    client = minio_client(settings)
    bucket = settings.s3_bucket
    result: dict[str, dict] = {}

    for key in SILVER_OBJECTS:
        result[key] = _id_fingerprint(read_parquet(client, bucket, key), "source_record_id")
    for key in (ML_GOOGLE_OBJECT, ML_SYNDROME_OBJECT):
        result[key] = _id_fingerprint(read_parquet(client, bucket, key), "example_id")

    with postgres_connection(settings) as connection:
        tables = [
            name
            for (name,) in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = %s AND table_type = 'BASE TABLE' ORDER BY table_name",
                (GOLD_SCHEMA,),
            )
        ]
        for name in tables:
            rows, content_hash = connection.execute(
                sql.SQL(
                    "SELECT count(*), md5(coalesce(string_agg(t::text, E'\\n' ORDER BY t::text), '')) "
                    "FROM {}.{} AS t"
                ).format(sql.Identifier(GOLD_SCHEMA), sql.Identifier(name))
            ).fetchone()
            result[f"{GOLD_SCHEMA}.{name}"] = {"rows": rows, "content_hash": content_hash}

    root = _results_root()
    result["results/part1/source_trace.parquet"] = _rows_fingerprint(
        read_local_parquet(root / "source_trace.parquet"),
        ("source_record_id", "archive_member", "record_locator", "input_sha256"),
    )
    issues = read_local_parquet(root / "data_issues.parquet")
    result["results/part1/data_issues.parquet"] = {
        **_rows_fingerprint(issues, ("rule_id", "source_record_id", "observed_value", "action")),
        "issue_id_hash": _hash_values(issues.column("issue_id").to_pylist()),
    }
    for csv_path in sorted((root / "analysis").glob("q*.csv")):
        result[f"results/part1/analysis/{csv_path.name}"] = {
            "content_hash": hashlib.sha256(csv_path.read_bytes()).hexdigest()
        }
    return result


def _run_part1(run_id: str) -> None:
    run_register_sources(run_id)
    run_prepare_data(run_id)
    run_load_postgres(run_id)
    run_build_ml_tables(run_id)


def main() -> int:
    settings = Settings.from_environment()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    _run_part1(f"rerun-check-1-{stamp}")
    first = fingerprint(settings)
    _run_part1(f"rerun-check-2-{stamp}")
    second = fingerprint(settings)

    outputs = {}
    for name in sorted(first.keys() | second.keys()):
        before, after = first.get(name), second.get(name)
        problems = []
        if before != after:
            problems.append("fingerprint changed between runs")
        if after and after.get("duplicate_ids"):
            problems.append(f"{after['duplicate_ids']} duplicate ids")
        outputs[name] = {"same": not problems, "problems": problems, "first": before, "second": after}

    passed = all(entry["same"] for entry in outputs.values())
    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "passed": passed,
        "outputs_compared": len(outputs),
        "outputs": outputs,
    }
    path = _results_root() / "rerun_check.json"
    path.write_text(json.dumps(report, indent=2))

    for name, entry in outputs.items():
        if not entry["same"]:
            print(f"CHANGED {name}: {'; '.join(entry['problems'])}")
    print(f"{'PASSED' if passed else 'FAILED'}: {len(outputs)} outputs compared across two runs -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
