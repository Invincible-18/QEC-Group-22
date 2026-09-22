"""Register and verify the original source files.

Student responsibilities:

- verify checksums from the release description;
- list source files and archive members safely;
- keep the supplied bytes unchanged;
- report missing or unexpected source objects;
- make a second run safe: no duplicate source records.
"""

from quantum_lake_student.models import StageResult

import hashlib
import json
from pathlib import Path
import tarfile
import zipfile
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Reusable Verification Utilities (Can be used by all datasets)
# ---------------------------------------------------------------------------
def is_safe_member_path(member_path: str) -> bool:
    """Verifies archive paths do not attempt directory traversal (../ or absolute)."""
    p = Path(member_path)
    return not (p.is_absolute() or ".." in p.parts)


def compute_sha256(file_path: Path) -> str:
    """Computes stable SHA-256 for a Bronze file."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def scan_archive_source(
    source_dir: Path, source_name: str, target_extension: str
) -> Tuple[List[Dict], List[Dict]]:
    """Generic scanner for Bronze archives (.tar.gz, .tgz, .zip, or raw files).

    Extracts file metadata, calculates SHA-256 hashes, and rejects unsafe paths.
    """
    verified = []
    issues = []

    if not source_dir.exists():
        return verified, issues

    for archive in sorted(source_dir.glob("*")):
        if not archive.is_file():
            continue

        archive_sha = compute_sha256(archive)

        if archive.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive, "r:gz") as tar:
                for member in sorted(tar.getmembers(), key=lambda m: m.name):
                    if not is_safe_member_path(member.name):
                        issues.append({
                            "source_name": source_name,
                            "bronze_object": archive.name,
                            "archive_member": member.name,
                            "issue": "unsafe_path_traversal",
                        })
                        continue
                    if member.name.endswith(target_extension):
                        verified.append({
                            "source_name": source_name,
                            "bronze_object": archive.name,
                            "archive_member": member.name,
                            "input_sha256": archive_sha,
                            "size_bytes": member.size,
                        })
        elif archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as z:
                for info in sorted(z.infolist(), key=lambda i: i.filename):
                    if not is_safe_member_path(info.filename):
                        issues.append({
                            "source_name": source_name,
                            "bronze_object": archive.name,
                            "archive_member": info.filename,
                            "issue": "unsafe_path_traversal",
                        })
                        continue
                    if info.filename.endswith(target_extension):
                        verified.append({
                            "source_name": source_name,
                            "bronze_object": archive.name,
                            "archive_member": info.filename,
                            "input_sha256": archive_sha,
                            "size_bytes": info.file_size,
                        })
        elif archive.suffix == target_extension or archive.name.endswith(target_extension):
            verified.append({
                "source_name": source_name,
                "bronze_object": archive.name,
                "archive_member": archive.name,
                "input_sha256": archive_sha,
                "size_bytes": archive.stat().st_size,
            })

    return verified, issues


# ---------------------------------------------------------------------------
# Stage Runner
# ---------------------------------------------------------------------------
def run(run_id: str) -> StageResult:
    # Set the discovered Bronze path and workspace project root
    raw_qasm_dir = Path("/course-data/raw/source=qasmbench")
    project_root = Path("/workspace")
    results_dir = project_root / "results" / "part1"
    results_dir.mkdir(parents=True, exist_ok=True)

    qasm_verified, qasm_issues = scan_archive_source(
        source_dir=raw_qasm_dir,
        source_name="qasmbench",
        target_extension=".qasm",
    )

    all_verified = qasm_verified
    all_issues = qasm_issues

    reg_file = results_dir / "verified_sources.json"
    reg_file.write_text(json.dumps(all_verified, indent=2))

    res = StageResult(
        stage="register_sources",
        run_id=run_id,
        input_count=len(all_verified),
        output_count=len(all_verified),
        issue_count=len(all_issues),
    )
    res.finish()
    return res