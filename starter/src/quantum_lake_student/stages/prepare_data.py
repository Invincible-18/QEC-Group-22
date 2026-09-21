"""Build the source-specific Silver table for simulated syndrome data."""

from __future__ import annotations

import hashlib
import re
from ast import literal_eval
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from quantum_lake_student.models import StageResult


SYNDROME_COLUMNS = [
    "source_record_id",
    "experiment_id",
    "physical_fault_rate",
    "syndrome_bits",
    "round_count",
    "check_count",
    "logical_error_label",
    "quantity",
]
ISSUE_COLUMNS = [
    "issue_id",
    "run_id",
    "source_record_id",
    "rule_id",
    "severity",
    "observed_value",
    "action",
    "reason",
]
TRACE_COLUMNS = [
    "source_record_id",
    "source_name",
    "bronze_object",
    "archive_member",
    "record_locator",
    "input_sha256",
]


def parse_syndrome(value: str) -> bytes:
    """Parse one nested syndrome into sixteen explicit 0/1 bytes."""
    parsed = literal_eval(value)
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        raise ValueError("syndrome must contain exactly four rounds")

    bits: list[int] = []
    for round_values in parsed:
        if not isinstance(round_values, (list, tuple)) or len(round_values) != 4:
            raise ValueError("every syndrome round must contain four checks")
        for bit in round_values:
            if bit not in (0, 1):
                raise ValueError("syndrome checks must be binary")
            bits.append(int(bit))
    return bytes(bits)


def _fault_rate(csv_name: str) -> float:
    match = re.search(r"_pfr-(?P<rate>\d+(?:\.\d+)?)_", csv_name)
    if match is None:
        raise ValueError("filename does not contain a physical fault rate")
    return float(match.group("rate"))


def _archive_sha256(archive: Path) -> str:
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _issue(
    run_id: str,
    source_record_id: str,
    rule_id: str,
    observed_value: object,
    reason: str,
) -> dict[str, object]:
    issue_key = f"{source_record_id}|{rule_id}|{reason}"
    return {
        "issue_id": hashlib.sha256(issue_key.encode("utf-8")).hexdigest(),
        "run_id": run_id,
        "source_record_id": source_record_id,
        "rule_id": rule_id,
        "severity": "error",
        "observed_value": repr(observed_value),
        "action": "excluded_from_silver",
        "reason": reason,
    }


def build_syndrome_silver(
    archive: Path,
    output_root: Path,
    run_id: str,
) -> StageResult:
    """Read the syndrome ZIP and write Silver, trace, and issue Parquet files."""
    result = StageResult(stage="syndrome_silver", run_id=run_id)
    silver_path = output_root / "silver/qec_syndromes/syndrome_observation.parquet"
    trace_path = output_root / "results/part1/source_trace.parquet"
    issues_path = output_root / "results/part1/data_issues.parquet"
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    input_sha256 = _archive_sha256(archive)
    records: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []

    with ZipFile(archive) as bundle:
        csv_members = sorted(
            member.filename
            for member in bundle.infolist()
            if member.filename.endswith(".csv")
        )
        for member_name in csv_members:
            experiment_id = Path(member_name).stem
            try:
                physical_fault_rate = _fault_rate(member_name)
            except ValueError as error:
                issues.append(_issue(run_id, member_name, "filename_fault_rate", member_name, str(error)))
                continue

            with bundle.open(member_name) as source:
                frame = pd.read_csv(source)
            result.input_count += len(frame)
            if set(frame.columns) != {"labels", "syndromes", "quantity"}:
                issues.append(
                    _issue(
                        run_id,
                        member_name,
                        "syndrome_schema",
                        list(frame.columns),
                        "CSV columns must be labels, syndromes, quantity",
                    )
                )
                continue

            for csv_row_number, (_, row) in enumerate(frame.iterrows(), start=2):
                locator = f"{member_name}#row-{csv_row_number}"
                source_record_id = f"qec_syndromes/{locator}"
                try:
                    label = int(row["labels"])
                    quantity = int(row["quantity"])
                    if label not in (0, 1):
                        raise ValueError("labels must be 0 or 1")
                    if quantity <= 0:
                        raise ValueError("quantity must be greater than zero")
                    syndrome_bits = parse_syndrome(str(row["syndromes"]))
                except (TypeError, ValueError, SyntaxError) as error:
                    issues.append(
                        _issue(
                            run_id,
                            source_record_id,
                            "syndrome_row_validity",
                            row.to_dict(),
                            str(error),
                        )
                    )
                    continue

                records.append(
                    {
                        "source_record_id": source_record_id,
                        "experiment_id": experiment_id,
                        "physical_fault_rate": physical_fault_rate,
                        "syndrome_bits": syndrome_bits,
                        "round_count": 4,
                        "check_count": 4,
                        "logical_error_label": bool(label),
                        "quantity": quantity,
                    }
                )
                traces.append(
                    {
                        "source_record_id": source_record_id,
                        "source_name": "qec_syndromes",
                        "bronze_object": "bronze/source=qec_syndromes/syndromes_dataset.zip",
                        "archive_member": member_name,
                        "record_locator": f"CSV row {csv_row_number}",
                        "input_sha256": input_sha256,
                    }
                )

    pd.DataFrame(records, columns=SYNDROME_COLUMNS).to_parquet(silver_path, index=False)
    pd.DataFrame(traces, columns=TRACE_COLUMNS).to_parquet(trace_path, index=False)
    pd.DataFrame(issues, columns=ISSUE_COLUMNS).to_parquet(issues_path, index=False)
    result.output_count = len(records)
    result.issue_count = len(issues)
    result.finish()
    return result


def run(run_id: str, archive: Path, output_root: Path) -> StageResult:
    return build_syndrome_silver(archive, output_root, run_id)
