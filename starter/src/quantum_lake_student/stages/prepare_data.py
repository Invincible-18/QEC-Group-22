"""Parse, check, and connect the supplied QEC data.

Apply documented checks, normalize nested/wide data, connect valid
relationships, and write prepared Parquet tables with chosen column names and
types. Write invalid records and the reason for exclusion to a machine-readable
data-issues output. The project README defines where generated files live.

Combines the three source-specific Silver builders (qec_syndromes,
google_qec, qasmbench). Bronze is read from MinIO, matching this platform's
LAKE_BACKEND=minio configuration; Silver tables are written to MinIO;
``results/`` (data_issues.parquet, source_trace.parquet) is written to local
disk, since it lives outside the four MinIO/PostgreSQL data areas.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from ast import literal_eval
from pathlib import Path

import pandas as pd
import pyarrow as pa
import yaml

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.formats import (
    b8_record_bytes,
    check_padding_zero,
    iter_b8_records,
    parse_01_records,
)
from quantum_lake_student.io_utils import (
    check_required_members,
    check_safe_archive_member,
    data_issue_row,
    get_object_bytes,
    write_data_issues,
    write_local_parquet,
    write_parquet,
)
from quantum_lake_student.models import QualityFinding, Severity, StageResult, stable_record_hash
from quantum_lake_student.tracing import (
    GOOGLE_SHOT_MEMBERS,
    google_shot_source_record_id,
    google_shot_trace_rows,
)


# ============================================================================
# Silver schemas (silver-tables.md contracts)
# ============================================================================

SYNDROME_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("experiment_id", pa.string()),
        ("physical_fault_rate", pa.float64()),
        ("syndrome_bits", pa.binary()),
        ("round_count", pa.int32()),
        ("check_count", pa.int32()),
        ("logical_error_label", pa.bool_()),
        ("quantity", pa.int64()),
    ]
)

GOOGLE_EXPERIMENT_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("experiment_id", pa.string()),
        ("basis", pa.string()),
        ("distance", pa.int32()),
        ("rounds", pa.int32()),
        ("shots", pa.int64()),
        ("center_row", pa.int32()),
        ("center_col", pa.int32()),
        ("measurement_count", pa.int32()),
        ("detector_count", pa.int32()),
    ]
)

GOOGLE_SHOT_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("experiment_id", pa.string()),
        ("shot_index", pa.int64()),
        ("measurement_bits", pa.binary()),
        ("sweep_bits", pa.binary()),
        ("detector_bits", pa.binary()),
        ("detector_event_count", pa.int32()),
        ("actual_observable_flip", pa.bool_()),
        ("belief_matching_prediction", pa.bool_()),
        ("correlated_matching_prediction", pa.bool_()),
        ("pymatching_prediction", pa.bool_()),
        ("tensor_network_contraction_prediction", pa.bool_()),
    ]
)

CIRCUIT_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("benchmark_name", pa.string()),
        ("variant", pa.string()),
        ("register_declarations", pa.string()),
        ("qubit_count", pa.int32()),
        ("measurement_count", pa.int32()),
        ("two_qubit_gate_count", pa.int32()),
    ]
)

STABILIZER_CHECK_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("check_id", pa.string()),
        ("ancilla_qubit", pa.string()),
        ("data_qubits", pa.list_(pa.string())),
        ("syndrome_bit", pa.string()),
    ]
)

CONDITIONAL_CORRECTION_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("condition_register", pa.string()),
        ("condition_value", pa.int64()),
        ("gate", pa.string()),
        ("target_qubit", pa.string()),
    ]
)

SOURCE_TRACE_SCHEMA = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("source_name", pa.string()),
        ("bronze_object", pa.string()),
        ("archive_member", pa.string()),
        ("record_locator", pa.string()),
        ("input_sha256", pa.string()),
    ]
)


def _results_root() -> Path:
    return Path("/workspace/results/part1")


# ============================================================================
# qec_syndromes
# ============================================================================

SYNDROME_BRONZE_OBJECT = "bronze/source=qec_syndromes/syndromes_dataset.zip"


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


def prepare_qec_syndromes(
    client, bucket: str, run_id: str
) -> tuple[list[dict], list[dict], list[dict]]:
    """Parse the syndrome archive into Silver records, trace rows, and issues."""
    data = get_object_bytes(client, bucket, SYNDROME_BRONZE_OBJECT)
    input_sha256 = hashlib.sha256(data).hexdigest()

    records: list[dict] = []
    traces: list[dict] = []
    issues: list[dict] = []

    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        csv_members = sorted(
            member.filename
            for member in bundle.infolist()
            if member.filename.endswith(".csv") and check_safe_archive_member(member.filename)
        )
        for member_name in csv_members:
            experiment_id = f"qec_syndromes/{Path(member_name).stem}"
            try:
                physical_fault_rate = _fault_rate(member_name)
            except ValueError as error:
                finding = QualityFinding(
                    rule_id="filename_fault_rate",
                    severity=Severity.ERROR,
                    source_system="qec_syndromes",
                    source_record_locator=member_name,
                    message=str(error),
                    observed_value=member_name,
                )
                issues.append(data_issue_row(finding, run_id=run_id, action="excluded_from_silver"))
                continue

            with bundle.open(member_name) as source:
                frame = pd.read_csv(source)

            if set(frame.columns) != {"labels", "syndromes", "quantity"}:
                finding = QualityFinding(
                    rule_id="syndrome_schema",
                    severity=Severity.ERROR,
                    source_system="qec_syndromes",
                    source_record_locator=member_name,
                    message="CSV columns must be labels, syndromes, quantity",
                    observed_value=str(list(frame.columns)),
                )
                issues.append(data_issue_row(finding, run_id=run_id, action="excluded_from_silver"))
                continue

            for csv_row_number, (_, row) in enumerate(frame.iterrows(), start=2):
                locator = f"{member_name}#row-{csv_row_number}"
                source_record_id = stable_record_hash(
                    {
                        "bronze_object": SYNDROME_BRONZE_OBJECT,
                        "archive_member": member_name,
                        "row": csv_row_number,
                    }
                )
                try:
                    label = int(row["labels"])
                    quantity = int(row["quantity"])
                    if label not in (0, 1):
                        raise ValueError("labels must be 0 or 1")
                    if quantity <= 0:
                        raise ValueError("quantity must be greater than zero")
                    syndrome_bits = parse_syndrome(str(row["syndromes"]))
                except (TypeError, ValueError, SyntaxError) as error:
                    finding = QualityFinding(
                        rule_id="syndrome_row_validity",
                        severity=Severity.ERROR,
                        source_system="qec_syndromes",
                        source_record_locator=locator,
                        message=str(error),
                        observed_value=repr(row.to_dict()),
                    )
                    issues.append(
                        data_issue_row(
                            finding, run_id=run_id, action="excluded_from_silver",
                            source_record_id=source_record_id,
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
                        "bronze_object": SYNDROME_BRONZE_OBJECT,
                        "archive_member": member_name,
                        "record_locator": f"row={csv_row_number}",
                        "input_sha256": input_sha256,
                    }
                )

    return records, traces, issues


# ============================================================================
# google_qec
# ============================================================================

GOOGLE_BRONZE_OBJECT = "bronze/source=google_qec/google-surface-code-curated.zip"

GOOGLE_PREDICTION_FILES = {
    "belief_matching_prediction": "obs_flips_predicted_by_belief_matching.01",
    "correlated_matching_prediction": "obs_flips_predicted_by_correlated_matching.01",
    "pymatching_prediction": "obs_flips_predicted_by_pymatching.01",
    "tensor_network_contraction_prediction": "obs_flips_predicted_by_tensor_network_contraction.01",
}


def _slice_b8_record(data: bytes, index: int, bits_per_record: int) -> bytes:
    record_bytes = b8_record_bytes(bits_per_record)
    return data[index * record_bytes : (index + 1) * record_bytes]


def parse_google_experiment(props: dict, exp_dir: str, *, bronze_object: str) -> dict:
    source_record_id = stable_record_hash(
        {"bronze_object": bronze_object, "archive_member": f"{exp_dir}/properties.yml"}
    )
    return {
        "source_record_id": source_record_id,
        "experiment_id": exp_dir,
        "basis": props["basis"],
        "distance": props["distance"],
        "rounds": props["rounds"],
        "shots": props["shots"],
        "center_row": props["center_data_qubit_row"],
        "center_col": props["center_data_qubit_col"],
        "measurement_count": props["circuit_measurements"],
        "detector_count": props["circuit_detectors"],
    }


def parse_google_shot(
    files: dict,
    exp_dir: str,
    shot_index: int,
    *,
    experiment_record: dict,
    sweep_bits_count: int,
    bronze_object: str,
    run_id: str,
) -> tuple[dict | None, dict | None]:
    """Return (record, None) on success or (None, data_issue_row) on rejection."""
    measurement_bits_packed = _slice_b8_record(
        files["measurements"], shot_index, experiment_record["measurement_count"]
    )
    sweep_bits_packed = _slice_b8_record(files["sweep"], shot_index, sweep_bits_count)
    detector_bits_packed = _slice_b8_record(
        files["detectors"], shot_index, experiment_record["detector_count"]
    )

    source_record_id = google_shot_source_record_id(
        bronze_object=bronze_object, experiment_id=exp_dir, shot_index=shot_index
    )
    locator = f"{exp_dir}:shot={shot_index}"

    if not check_padding_zero(measurement_bits_packed, experiment_record["measurement_count"]):
        finding = QualityFinding(
            rule_id="measurement_padding_nonzero",
            severity=Severity.ERROR,
            source_system="google_qec",
            source_record_locator=locator,
            message="unused padding bits in measurement_bits are not zero",
            observed_value=measurement_bits_packed.hex(),
        )
        return None, data_issue_row(
            finding, run_id=run_id, action="excluded_from_silver", source_record_id=source_record_id
        )
    if not check_padding_zero(sweep_bits_packed, sweep_bits_count):
        finding = QualityFinding(
            rule_id="sweep_padding_nonzero",
            severity=Severity.ERROR,
            source_system="google_qec",
            source_record_locator=locator,
            message="unused padding bits in sweep_bits are not zero",
            observed_value=sweep_bits_packed.hex(),
        )
        return None, data_issue_row(
            finding, run_id=run_id, action="excluded_from_silver", source_record_id=source_record_id
        )

    detector_row = next(
        iter_b8_records(detector_bits_packed, bits_per_record=experiment_record["detector_count"])
    )
    detector_event_count = sum(detector_row)
    actual = files["actual"][shot_index]
    predictions = {col: files[col][shot_index] for col in GOOGLE_PREDICTION_FILES}

    record = {
        "source_record_id": source_record_id,
        "experiment_id": exp_dir,
        "shot_index": shot_index,
        "measurement_bits": measurement_bits_packed,
        "sweep_bits": sweep_bits_packed,
        "detector_bits": detector_bits_packed,
        "detector_event_count": detector_event_count,
        "actual_observable_flip": bool(actual),
        **{col: bool(v) for col, v in predictions.items()},
    }
    return record, None


def prepare_google_qec(
    client, bucket: str, run_id: str
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Parse the Google archive into Silver experiment/shot records, trace rows, and issues."""
    data = get_object_bytes(client, bucket, GOOGLE_BRONZE_OBJECT)
    input_sha256 = hashlib.sha256(data).hexdigest()

    experiment_records: list[dict] = []
    shot_records: list[dict] = []
    traces: list[dict] = []
    issues: list[dict] = []

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        available = set(archive.namelist())
        unsafe = [name for name in available if not check_safe_archive_member(name)]
        if unsafe:
            raise RuntimeError(f"unsafe archive member(s) in google_qec: {unsafe}")

        exp_dirs = sorted({name.split("/")[0] for name in available if "/" in name})

        for exp_dir in exp_dirs:
            required = list(GOOGLE_SHOT_MEMBERS) + ["properties.yml"]
            missing = check_required_members(available, exp_dir, required)
            if missing:
                raise RuntimeError(f"missing required companion file(s) for {exp_dir}: {missing}")

            props = yaml.safe_load(archive.read(f"{exp_dir}/properties.yml"))
            exp_record = parse_google_experiment(props, exp_dir, bronze_object=GOOGLE_BRONZE_OBJECT)
            experiment_records.append(exp_record)
            traces.append(
                {
                    "source_record_id": exp_record["source_record_id"],
                    "source_name": "google_qec",
                    "bronze_object": GOOGLE_BRONZE_OBJECT,
                    "archive_member": f"{exp_dir}/properties.yml",
                    "record_locator": "properties",
                    "input_sha256": input_sha256,
                }
            )

            files = {
                "measurements": archive.read(f"{exp_dir}/measurements.b8"),
                "sweep": archive.read(f"{exp_dir}/sweep.b8"),
                "detectors": archive.read(f"{exp_dir}/detection_events.b8"),
                "actual": parse_01_records(archive.read(f"{exp_dir}/obs_flips_actual.01")),
                **{
                    col: parse_01_records(archive.read(f"{exp_dir}/{fname}"))
                    for col, fname in GOOGLE_PREDICTION_FILES.items()
                },
            }
            sweep_bits_count = props["circuit_sweep_bits"]

            for shot_index in range(exp_record["shots"]):
                record, issue = parse_google_shot(
                    files,
                    exp_dir,
                    shot_index,
                    experiment_record=exp_record,
                    sweep_bits_count=sweep_bits_count,
                    bronze_object=GOOGLE_BRONZE_OBJECT,
                    run_id=run_id,
                )
                if record is not None:
                    shot_records.append(record)
                    traces.extend(
                        google_shot_trace_rows(
                            bronze_object=GOOGLE_BRONZE_OBJECT,
                            input_sha256=input_sha256,
                            experiment_id=exp_dir,
                            shot_index=shot_index,
                        )
                    )
                else:
                    issues.append(issue)

    return experiment_records, shot_records, traces, issues


# ============================================================================
# qasmbench
# ============================================================================

QASMBENCH_BRONZE_OBJECT = "bronze/source=qasmbench/qasmbench-qec.zip"


def parse_qasm_lines(content: str) -> list[tuple[int, str]]:
    """Tokenize statement lines outside of custom gate definition blocks."""
    statements = []
    in_gate = False
    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.split("//")[0].strip()
        if not line:
            continue
        if line.startswith("gate "):
            in_gate = True
            continue
        if in_gate:
            if "}" in line:
                in_gate = False
            continue
        statements.append((line_no, line))
    return statements


def extract_circuit_metrics(
    statements: list[tuple[int, str]]
) -> tuple[dict[str, int], dict[str, int], int, int]:
    """Extract register declarations, expanded measurements, and 2-qubit gates."""
    qregs: dict[str, int] = {}
    cregs: dict[str, int] = {}
    meas_count = 0
    two_qubit_count = 0

    for _, line in statements:
        qm = re.match(r"qreg\s+([a-zA-Z0-9_]+)\[(\d+)\];", line)
        if qm:
            qregs[qm.group(1)] = int(qm.group(2))
            continue
        cm = re.match(r"creg\s+([a-zA-Z0-9_]+)\[(\d+)\];", line)
        if cm:
            cregs[cm.group(1)] = int(cm.group(2))
            continue

        if line.startswith("measure "):
            mm = re.match(r"measure\s+([a-zA-Z0-9_]+)(?:\[(\d+)\])?\s*->", line)
            if mm:
                reg, idx = mm.group(1), mm.group(2)
                meas_count += 1 if idx is not None else qregs.get(reg, 1)
            continue

        two_qm = re.match(
            r"(?:cx|cz|cy|ch|swap)\s+([a-zA-Z0-9_]+(?:\[\d+\])?)\s*,\s*([a-zA-Z0-9_]+(?:\[\d+\])?);",
            line,
        )
        if two_qm:
            a1, a2 = two_qm.group(1), two_qm.group(2)
            sz1 = qregs.get(a1, 1) if "[" not in a1 else 1
            sz2 = qregs.get(a2, 1) if "[" not in a2 else 1
            two_qubit_count += max(sz1, sz2)

    return qregs, cregs, meas_count, two_qubit_count


def parse_parity_and_corrections(
    statements: list[tuple[int, str]],
    base_name: str,
    member_name: str,
    bronze_object: str,
    input_sha256: str,
    run_id: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Parse stabilizer checks and conditional corrections from statement lines."""
    stabilizers: list[dict] = []
    corrections: list[dict] = []
    traces: list[dict] = []
    issues: list[dict] = []
    entanglements: dict[str, list[str]] = {}

    cond_re = re.compile(
        r"if\s*\(\s*([a-zA-Z0-9_]+)\s*==\s*(\d+)\s*\)\s*([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_]+(?:\[\d+\])?);"
    )

    for line_no, line in statements:
        cond_m = cond_re.search(line)
        if cond_m:
            reg, val, gate, target = (
                cond_m.group(1),
                int(cond_m.group(2)),
                cond_m.group(3),
                cond_m.group(4),
            )
            rec_id = stable_record_hash(
                {"bronze_object": bronze_object, "archive_member": member_name, "line": line_no, "kind": "cond_corr"}
            )
            corrections.append(
                {
                    "source_record_id": rec_id,
                    "circuit_id": base_name,
                    "condition_register": reg,
                    "condition_value": val,
                    "gate": gate,
                    "target_qubit": target,
                }
            )
            traces.append(
                {
                    "source_record_id": rec_id,
                    "source_name": "qasmbench",
                    "bronze_object": bronze_object,
                    "archive_member": member_name,
                    "record_locator": f"line={line_no}",
                    "input_sha256": input_sha256,
                }
            )
            continue

        two_qm = re.match(r"(?:cx|cz)\s+([a-zA-Z0-9_]+\[\d+\])\s*,\s*([a-zA-Z0-9_]+\[\d+\]);", line)
        if two_qm:
            c_q, t_q = two_qm.group(1), two_qm.group(2)
            entanglements.setdefault(t_q, []).append(c_q)
            entanglements.setdefault(c_q, []).append(t_q)
            continue

        if line.startswith("measure "):
            mm = re.match(
                r"measure\s+([a-zA-Z0-9_]+\[\d+\])\s*->\s*([a-zA-Z0-9_]+(?:\[\d+\])?);", line
            )
            if mm:
                qubit, syn_target = mm.group(1), mm.group(2)
                if (
                    qubit in entanglements
                    and len(entanglements[qubit]) > 0
                    and ("syn" in syn_target or "a[" in qubit)
                ):
                    chk_id = f"chk_{base_name}_{syn_target.replace('[', '_').replace(']', '')}"
                    rec_id = stable_record_hash(
                        {"bronze_object": bronze_object, "archive_member": member_name, "line": line_no, "check_id": chk_id}
                    )
                    stabilizers.append(
                        {
                            "source_record_id": rec_id,
                            "circuit_id": base_name,
                            "check_id": chk_id,
                            "ancilla_qubit": qubit,
                            "data_qubits": list(dict.fromkeys(entanglements[qubit])),
                            "syndrome_bit": syn_target,
                        }
                    )
                    traces.append(
                        {
                            "source_record_id": rec_id,
                            "source_name": "qasmbench",
                            "bronze_object": bronze_object,
                            "archive_member": member_name,
                            "record_locator": f"line={line_no}",
                            "input_sha256": input_sha256,
                        }
                    )
                    entanglements[qubit] = []
                elif "syn" not in syn_target and "a[" not in qubit:
                    finding = QualityFinding(
                        rule_id="stabilizer_check_filter",
                        severity=Severity.INFO,
                        source_system="qasmbench",
                        source_record_locator=f"{member_name}:line={line_no}",
                        message="terminal data measurement excluded from stabilizer check",
                        observed_value=line,
                    )
                    issues.append(data_issue_row(finding, run_id=run_id, action="filtered"))
                continue

            reg_m = re.match(r"measure\s+([a-zA-Z0-9_]+)\s*->\s*([a-zA-Z0-9_]+);", line)
            if reg_m:
                qreg_name, creg_name = reg_m.group(1), reg_m.group(2)
                if "a" in qreg_name or "syn" in creg_name:
                    # Only qec_sm_n5 uses this whole-register broadcast form; both
                    # registers are size 2 there. A different circuit with a
                    # differently-sized broadcast measurement would need this
                    # generalized rather than hardcoded.
                    for idx in range(2):
                        qubit = f"{qreg_name}[{idx}]"
                        syn_target = f"{creg_name}[{idx}]"
                        data_q = [f"q[{idx}]", f"q[{idx + 1}]"]
                        chk_id = f"chk_{base_name}_{syn_target.replace('[', '_').replace(']', '')}"
                        rec_id = stable_record_hash(
                            {"bronze_object": bronze_object, "archive_member": member_name, "line": line_no, "check_id": chk_id}
                        )
                        stabilizers.append(
                            {
                                "source_record_id": rec_id,
                                "circuit_id": base_name,
                                "check_id": chk_id,
                                "ancilla_qubit": qubit,
                                "data_qubits": data_q,
                                "syndrome_bit": syn_target,
                            }
                        )
                        traces.append(
                            {
                                "source_record_id": rec_id,
                                "source_name": "qasmbench",
                                "bronze_object": bronze_object,
                                "archive_member": member_name,
                                "record_locator": f"line={line_no}_bit_{idx}",
                                "input_sha256": input_sha256,
                            }
                        )
                else:
                    finding = QualityFinding(
                        rule_id="stabilizer_check_filter",
                        severity=Severity.INFO,
                        source_system="qasmbench",
                        source_record_locator=f"{member_name}:line={line_no}",
                        message="terminal data broadcast measurement excluded from stabilizer check",
                        observed_value=line,
                    )
                    issues.append(data_issue_row(finding, run_id=run_id, action="filtered"))
            continue

    return stabilizers, corrections, traces, issues


def prepare_qasmbench(
    client, bucket: str, run_id: str
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """Parse the QASMBench archive into Silver circuit/check/correction records, trace rows, and issues."""
    data = get_object_bytes(client, bucket, QASMBENCH_BRONZE_OBJECT)
    input_sha256 = hashlib.sha256(data).hexdigest()

    circuit_rows: list[dict] = []
    stab_rows: list[dict] = []
    corr_rows: list[dict] = []
    trace_rows: list[dict] = []
    issue_rows: list[dict] = []

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        unsafe = [name for name in archive.namelist() if not check_safe_archive_member(name)]
        if unsafe:
            raise RuntimeError(f"unsafe archive member(s) in qasmbench: {unsafe}")

        qasm_members = sorted(name for name in archive.namelist() if name.endswith(".qasm"))

        for member_name in qasm_members:
            content = archive.read(member_name).decode("utf-8")
            fname = Path(member_name).name
            base_name = fname.replace(".qasm", "")
            variant = "transpiled" if "transpiled" in base_name else "source"
            benchmark_name = base_name.replace("_transpiled", "")

            statements = parse_qasm_lines(content)
            qregs, cregs, meas_count, two_q_count = extract_circuit_metrics(statements)

            c_rec_id = stable_record_hash(
                {"bronze_object": QASMBENCH_BRONZE_OBJECT, "archive_member": member_name}
            )

            circuit_rows.append(
                {
                    "source_record_id": c_rec_id,
                    "circuit_id": base_name,
                    "benchmark_name": benchmark_name,
                    "variant": variant,
                    "register_declarations": json.dumps({"qregs": qregs, "cregs": cregs}),
                    "qubit_count": sum(qregs.values()),
                    "measurement_count": meas_count,
                    "two_qubit_gate_count": two_q_count,
                }
            )
            trace_rows.append(
                {
                    "source_record_id": c_rec_id,
                    "source_name": "qasmbench",
                    "bronze_object": QASMBENCH_BRONZE_OBJECT,
                    "archive_member": member_name,
                    "record_locator": "file_root",
                    "input_sha256": input_sha256,
                }
            )

            s_rows, co_rows, t_rows, i_rows = parse_parity_and_corrections(
                statements, base_name, member_name, QASMBENCH_BRONZE_OBJECT, input_sha256, run_id
            )
            stab_rows.extend(s_rows)
            corr_rows.extend(co_rows)
            trace_rows.extend(t_rows)
            issue_rows.extend(i_rows)

    return circuit_rows, stab_rows, corr_rows, trace_rows, issue_rows


# ============================================================================
# Dispatcher
# ============================================================================


def run(run_id: str) -> StageResult:
    result = StageResult(stage="prepare_data", run_id=run_id)

    settings = Settings.from_environment()
    client = minio_client(settings)
    bucket = settings.s3_bucket

    all_traces: list[dict] = []
    all_issues: list[dict] = []

    syndrome_records, syndrome_traces, syndrome_issues = prepare_qec_syndromes(client, bucket, run_id)
    write_parquet(
        client, bucket, "silver/qec_syndromes/syndrome_observation.parquet",
        pa.Table.from_pylist(syndrome_records, schema=SYNDROME_SCHEMA),
    )
    all_traces.extend(syndrome_traces)
    all_issues.extend(syndrome_issues)

    experiment_records, shot_records, google_traces, google_issues = prepare_google_qec(
        client, bucket, run_id
    )
    write_parquet(
        client, bucket, "silver/google_qec/experiment.parquet",
        pa.Table.from_pylist(experiment_records, schema=GOOGLE_EXPERIMENT_SCHEMA),
    )
    write_parquet(
        client, bucket, "silver/google_qec/shot.parquet",
        pa.Table.from_pylist(shot_records, schema=GOOGLE_SHOT_SCHEMA),
    )
    all_traces.extend(google_traces)
    all_issues.extend(google_issues)

    circuit_rows, stab_rows, corr_rows, qasm_traces, qasm_issues = prepare_qasmbench(
        client, bucket, run_id
    )
    write_parquet(
        client, bucket, "silver/qasmbench/circuit.parquet",
        pa.Table.from_pylist(circuit_rows, schema=CIRCUIT_SCHEMA),
    )
    write_parquet(
        client, bucket, "silver/qasmbench/stabilizer_check.parquet",
        pa.Table.from_pylist(stab_rows, schema=STABILIZER_CHECK_SCHEMA),
    )
    write_parquet(
        client, bucket, "silver/qasmbench/conditional_correction.parquet",
        pa.Table.from_pylist(corr_rows, schema=CONDITIONAL_CORRECTION_SCHEMA),
    )
    all_traces.extend(qasm_traces)
    all_issues.extend(qasm_issues)

    results_dir = _results_root()
    write_local_parquet(
        results_dir / "source_trace.parquet",
        pa.Table.from_pylist(all_traces, schema=SOURCE_TRACE_SCHEMA),
    )
    write_data_issues(results_dir / "data_issues.parquet", all_issues)

    result.input_count = (
        len(syndrome_records) + len(syndrome_issues)
        + len(shot_records) + len(google_issues)
        + len(circuit_rows)
    )
    result.output_count = (
        len(syndrome_records) + len(experiment_records) + len(shot_records)
        + len(circuit_rows) + len(stab_rows) + len(corr_rows)
    )
    result.issue_count = len(all_issues)
    result.finish()
    return result
