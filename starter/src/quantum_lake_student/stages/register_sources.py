"""Register and verify the original source files.

Student responsibilities:

- verify checksums from the release description;
- list source files and archive members safely;
- keep the supplied bytes unchanged;
- report missing or unexpected source objects;
- make a second run safe: no duplicate source records.

This stage only reads Bronze and the release manifest; it never writes to
Bronze, Silver, Gold, or ML. Because it produces no persistent business
records of its own (only verification), a second run naturally cannot
create duplicates -- it just re-checks the same read-only inputs.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import bronze_inventory, minio_client
from quantum_lake_student.io_utils import check_safe_archive_member, get_object_bytes
from quantum_lake_student.models import StageResult


# Mounted read-only from datasets/student-bundle/core (see compose.yaml).
MANIFEST_PATH = Path("/course-data/metadata/bundle-manifest.json")


def _load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict:
    return json.loads(manifest_path.read_text())


def _bronze_key(manifest_object_path: str) -> str:
    """Translate a manifest path (raw/source=X/Y) to its Bronze object key.

    The release archive keeps its packaging directory name (``raw/``); the
    platform seeds that same content under ``bronze/`` (see the comment in
    connections.bronze_inventory).
    """
    if not manifest_object_path.startswith("raw/"):
        raise ValueError(f"unexpected manifest object path: {manifest_object_path!r}")
    return "bronze/" + manifest_object_path[len("raw/") :]


def run(run_id: str) -> StageResult:
    result = StageResult(stage="register_sources", run_id=run_id)

    settings = Settings.from_environment()
    client = minio_client(settings)

    manifest = _load_manifest()
    expected_objects = {_bronze_key(obj["path"]): obj for obj in manifest["objects"]}

    actual_sizes = dict(bronze_inventory(settings))
    actual_keys = set(actual_sizes)
    expected_keys = set(expected_objects)

    missing = expected_keys - actual_keys
    if missing:
        raise RuntimeError(f"Missing required Bronze object(s): {sorted(missing)}")

    unexpected = actual_keys - expected_keys
    if unexpected:
        result.issue_count += len(unexpected)
        print(f"WARNING: unexpected Bronze object(s) present: {sorted(unexpected)}")

    for key, spec in expected_objects.items():
        actual_size = actual_sizes[key]
        if actual_size != spec["bytes"]:
            raise RuntimeError(
                f"{key}: size mismatch (expected {spec['bytes']}, got {actual_size})"
            )

        data = get_object_bytes(client, settings.s3_bucket, key)

        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != spec["sha256"]:
            raise RuntimeError(
                f"{key}: checksum mismatch (expected {spec['sha256']}, got {actual_hash})"
            )

        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            unsafe_members = [
                name for name in archive.namelist() if not check_safe_archive_member(name)
            ]
        if unsafe_members:
            raise RuntimeError(f"{key}: unsafe archive member(s): {unsafe_members}")

        result.input_count += 1
        result.output_count += 1

    result.finish()
    return result
