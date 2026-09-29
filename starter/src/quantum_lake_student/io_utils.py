"""Shared read/write helpers for moving tables between pipeline zones.

These wrap the raw MinIO client from ``connections.py`` with the
Parquet-specific plumbing every stage needs: pulling an object's bytes,
writing a PyArrow table back, and turning quality findings into
``data_issues.parquet`` rows. Deciding what belongs in a table, its schema,
and its business meaning remain stage-specific student work.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio

from .models import QualityFinding, stable_record_hash


DATA_ISSUES_SCHEMA = pa.schema(
    [
        ("issue_id", pa.string()),
        ("run_id", pa.string()),
        ("source_record_id", pa.string()),
        ("rule_id", pa.string()),
        ("severity", pa.string()),
        ("observed_value", pa.string()),
        ("action", pa.string()),
        ("reason", pa.string()),
    ]
)


def check_safe_archive_member(name: str) -> bool:
    """Return whether a zip archive member path is safe to extract/read.

    Rejects absolute paths, Windows drive-letter paths, and any ``..``
    path-traversal segment. Required per the brief: "a missing required
    companion file or unsafe archive member must stop the run."
    """
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    first_segment = normalized.split("/", 1)[0]
    if len(first_segment) == 2 and first_segment[1] == ":":
        return False
    parts = normalized.split("/")
    if any(part == ".." for part in parts):
        return False
    return True


def check_required_members(available_members: set[str], prefix: str, required: Iterable[str]) -> list[str]:
    """Return which of the required filenames are missing under ``prefix``.

    ``prefix`` is an archive-member directory (e.g. one experiment or
    circuit directory); ``required`` is a source-specific list of
    filenames expected inside it. Used to enforce the brief's rule that a
    missing required companion file must stop the run, not be silently
    skipped.
    """
    return [name for name in required if f"{prefix}/{name}" not in available_members]


def get_object_bytes(client: Minio, bucket: str, object_name: str) -> bytes:
    """Read one object's full bytes from MinIO."""
    response = client.get_object(bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def write_parquet(
    client: Minio,
    bucket: str,
    object_name: str,
    table: pa.Table,
    *,
    compression: str = "zstd",
) -> int:
    """Write a PyArrow table to MinIO as Parquet. Returns bytes written."""
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression=compression)
    payload = buffer.getvalue()
    client.put_object(
        bucket,
        object_name,
        data=io.BytesIO(payload),
        length=len(payload),
        content_type="application/octet-stream",
    )
    return len(payload)


def read_parquet(client: Minio, bucket: str, object_name: str) -> pa.Table:
    """Read a Parquet object back from MinIO as a PyArrow table."""
    return pq.read_table(io.BytesIO(get_object_bytes(client, bucket, object_name)))


def write_local_parquet(path: Path | str, table: pa.Table, *, compression: str = "zstd") -> int:
    """Write a PyArrow table to a local file. Used for the ``results/`` area,
    which lives outside Bronze/Silver/Gold/ML and is not stored in MinIO."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression=compression)
    return path.stat().st_size


def read_local_parquet(path: Path | str) -> pa.Table:
    """Read a local Parquet file back as a PyArrow table."""
    return pq.read_table(Path(path))


def data_issue_row(
    finding: QualityFinding,
    *,
    run_id: str,
    action: str,
    source_record_id: str | None = None,
) -> dict:
    """Turn a supplied QualityFinding into one data_issues.parquet row.

    ``issue_id`` is a stable hash of the run, rule, source record, and
    observed value, so repeated runs on unchanged input produce the same
    identifier instead of a new one each time.
    """
    issue_id = stable_record_hash(
        {
            "run_id": run_id,
            "rule_id": finding.rule_id,
            "source_record_id": source_record_id,
            "observed_value": finding.observed_value,
        }
    )
    return {
        "issue_id": issue_id,
        "run_id": run_id,
        "source_record_id": source_record_id,
        "rule_id": finding.rule_id,
        "severity": str(finding.severity),
        "observed_value": finding.observed_value,
        "action": action,
        "reason": finding.message,
    }


def write_data_issues(path: Path | str, issues: Iterable[dict]) -> int:
    """Write data-issue rows (see ``data_issue_row``) as a local data_issues.parquet.

    ``results/`` is evidence kept outside the four MinIO/PostgreSQL data
    areas, so this writes to the local filesystem, not to MinIO. (No
    results-specific backend setting or MinIO wiring exists anywhere in
    the platform config; the workspace bind mount is the only storage
    mechanism actually available for this area.)
    """
    table = pa.Table.from_pylist(list(issues), schema=DATA_ISSUES_SCHEMA)
    return write_local_parquet(path, table)
