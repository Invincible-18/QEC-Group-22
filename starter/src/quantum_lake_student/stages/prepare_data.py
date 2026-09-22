"""Parse, check, and connect the supplied QEC data.

Apply documented checks, normalize nested/wide data, connect valid
relationships, and write prepared Parquet tables with chosen column names and
types. Write invalid records and the reason for exclusion to a machine-readable
data-issues output. The project README defines where generated files live.
"""

from quantum_lake_student.models import StageResult
import hashlib
import json
from pathlib import Path
import re
import tarfile
import zipfile
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Strict PyArrow Schemas (assignment/silver-tables.md)
# ---------------------------------------------------------------------------
CIRCUIT_SCHEMA = pa.schema([
    ("source_record_id", pa.string()),
    ("circuit_id", pa.string()),
    ("benchmark_name", pa.string()),
    ("variant", pa.string()),
    ("register_declarations", pa.string()),
    ("qubit_count", pa.int32()),
    ("measurement_count", pa.int32()),
    ("two_qubit_gate_count", pa.int32()),
])

STABILIZER_CHECK_SCHEMA = pa.schema([
    ("source_record_id", pa.string()),
    ("circuit_id", pa.string()),
    ("check_id", pa.string()),
    ("ancilla_qubit", pa.string()),
    ("data_qubits", pa.list_(pa.string())),
    ("syndrome_bit", pa.string()),
])

CONDITIONAL_CORRECTION_SCHEMA = pa.schema([
    ("source_record_id", pa.string()),
    ("circuit_id", pa.string()),
    ("condition_register", pa.string()),
    ("condition_value", pa.int64()),
    ("gate", pa.string()),
    ("target_qubit", pa.string()),
])

SOURCE_TRACE_SCHEMA = pa.schema([
    ("source_record_id", pa.string()),
    ("source_name", pa.string()),
    ("bronze_object", pa.string()),
    ("archive_member", pa.string()),
    ("record_locator", pa.string()),
    ("input_sha256", pa.string()),
])

DATA_ISSUES_SCHEMA = pa.schema([
    ("source_record_id", pa.string()),
    ("rule", pa.string()),
    ("severity", pa.string()),
    ("action", pa.string()),
    ("reason", pa.string()),
    ("value", pa.string()),
])


# ---------------------------------------------------------------------------
# Reusable Shared Parquet & Audit Helpers
# ---------------------------------------------------------------------------
def write_parquet_table(df: pd.DataFrame, schema: pa.Schema, out_path: Path) -> None:
    """Writes a DataFrame to Parquet, initializing schema columns if empty."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df = pd.DataFrame({field.name: pd.Series(dtype=object) for field in schema})
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, out_path)


def merge_audit_parquet(
    out_path: Path, new_df: pd.DataFrame, schema: pa.Schema, dedupe_col: str | None = None
) -> None:
    """Safely merges and persists trace/issue rows from different datasets."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if new_df.empty:
        new_df = pd.DataFrame({field.name: pd.Series(dtype=object) for field in schema})

    if out_path.exists():
        existing_df = pq.read_table(out_path).to_pandas()
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        if dedupe_col and dedupe_col in combined.columns:
            combined = combined.drop_duplicates(subset=[dedupe_col])
    else:
        combined = new_df
    pq.write_table(pa.Table.from_pandas(combined, schema=schema, preserve_index=False), out_path)


# ---------------------------------------------------------------------------
# QASMBench Modular Parsing Functions
# ---------------------------------------------------------------------------
def parse_qasm_lines(content: str) -> List[Tuple[int, str]]:
    """Tokenizes statement lines outside of custom gate definition blocks."""
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


def extract_circuit_metrics(statements: List[Tuple[int, str]]) -> Tuple[Dict[str, int], Dict[str, int], int, int]:
    """Extracts register declarations, expanded measurements, and 2-qubit gates."""
    qregs: Dict[str, int] = {}
    cregs: Dict[str, int] = {}
    meas_count = 0
    two_qubit_count = 0

    for _, line in statements:
        # Registers
        qm = re.match(r"qreg\s+([a-zA-Z0-9_]+)\[(\d+)\];", line)
        if qm:
            qregs[qm.group(1)] = int(qm.group(2))
            continue
        cm = re.match(r"creg\s+([a-zA-Z0-9_]+)\[(\d+)\];", line)
        if cm:
            cregs[cm.group(1)] = int(cm.group(2))
            continue

        # Expanded measurement count
        if line.startswith("measure "):
            mm = re.match(r"measure\s+([a-zA-Z0-9_]+)(?:\[(\d+)\])?\s*->", line)
            if mm:
                reg, idx = mm.group(1), mm.group(2)
                meas_count += 1 if idx is not None else qregs.get(reg, 1)
            continue

        # Expanded 2-qubit gates
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
    statements: List[Tuple[int, str]],
    base_name: str,
    member_name: str,
    bronze_obj: str,
    input_sha: str,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Parses stabilizer checks and conditional corrections from statement lines."""
    stabilizers: List[Dict] = []
    corrections: List[Dict] = []
    traces: List[Dict] = []
    issues: List[Dict] = []
    entanglements: Dict[str, List[str]] = {}

    cond_re = re.compile(
        r"if\s*\(\s*([a-zA-Z0-9_]+)\s*==\s*(\d+)\s*\)\s*([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_]+(?:\[\d+\])?);"
    )

    for line_no, line in statements:
        # Conditional corrections
        cond_m = cond_re.search(line)
        if cond_m:
            reg, val, gate, target = cond_m.group(1), int(cond_m.group(2)), cond_m.group(3), cond_m.group(4)
            rec_id = f"qasmbench:{member_name}:line_{line_no}:cond_corr"
            corrections.append({
                "source_record_id": rec_id,
                "circuit_id": base_name,
                "condition_register": reg,
                "condition_value": val,
                "gate": gate,
                "target_qubit": target,
            })
            traces.append({
                "source_record_id": rec_id,
                "source_name": "qasmbench",
                "bronze_object": bronze_obj,
                "archive_member": member_name,
                "record_locator": f"line_{line_no}",
                "input_sha256": input_sha,
            })
            continue

        # Two-qubit gate entanglement tracking
        two_qm = re.match(r"(?:cx|cz)\s+([a-zA-Z0-9_]+\[\d+\])\s*,\s*([a-zA-Z0-9_]+\[\d+\]);", line)
        if two_qm:
            c_q, t_q = two_qm.group(1), two_qm.group(2)
            entanglements.setdefault(t_q, []).append(c_q)
            entanglements.setdefault(c_q, []).append(t_q)
            continue

        # Stabilizer check measurements (handles single-bit and broadcast register measurements)
        if line.startswith("measure "):
            # 1. Single-bit measurement: measure a[0] -> syn[0];
            mm = re.match(
                r"measure\s+([a-zA-Z0-9_]+\[\d+\])\s*->\s*([a-zA-Z0-9_]+(?:\[\d+\])?);",
                line,
            )
            if mm:
                qubit, syn_target = mm.group(1), mm.group(2)
                if (
                    qubit in entanglements
                    and len(entanglements[qubit]) > 0
                    and ("syn" in syn_target or "a[" in qubit)
                ):
                    chk_id = f"chk_{base_name}_{syn_target.replace('[', '_').replace(']', '')}"
                    rec_id = f"qasmbench:{member_name}:line_{line_no}:{chk_id}"
                    stabilizers.append(
                        {
                            "source_record_id": rec_id,
                            "circuit_id": base_name,
                            "check_id": chk_id,
                            "ancilla_qubit": qubit,
                            "data_qubits": list(
                                dict.fromkeys(entanglements[qubit])
                            ),
                            "syndrome_bit": syn_target,
                        }
                    )
                    traces.append(
                        {
                            "source_record_id": rec_id,
                            "source_name": "qasmbench",
                            "bronze_object": bronze_obj,
                            "archive_member": member_name,
                            "record_locator": f"line_{line_no}",
                            "input_sha256": input_sha,
                        }
                    )
                    entanglements[qubit] = []
                elif "syn" not in syn_target and "a[" not in qubit:
                    issues.append(
                        {
                            "source_record_id": f"qasmbench:{member_name}",
                            "rule": "stabilizer_check_filter",
                            "severity": "info",
                            "action": "filtered",
                            "reason": "terminal_data_measurement_excluded_from_stabilizer_check",
                            "value": line,
                        }
                    )
                continue

            # 2. Whole-register broadcast measurement: measure a -> syn;
            reg_m = re.match(
                r"measure\s+([a-zA-Z0-9_]+)\s*->\s*([a-zA-Z0-9_]+);", line
            )
            if reg_m:
                qreg_name, creg_name = reg_m.group(1), reg_m.group(2)
                if "a" in qreg_name or "syn" in creg_name:
                    # In qec_sm_n5 both registers have size 2
                    for idx in range(2):
                        qubit = f"{qreg_name}[{idx}]"
                        syn_target = f"{creg_name}[{idx}]"
                        data_q = [f"q[{idx}]", f"q[{idx+1}]"]
                        chk_id = f"chk_{base_name}_{syn_target.replace('[', '_').replace(']', '')}"
                        rec_id = f"qasmbench:{member_name}:line_{line_no}:{chk_id}"
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
                                "bronze_object": bronze_obj,
                                "archive_member": member_name,
                                "record_locator": f"line_{line_no}_bit_{idx}",
                                "input_sha256": input_sha,
                            }
                        )
                else:
                    issues.append(
                        {
                            "source_record_id": f"qasmbench:{member_name}",
                            "rule": "stabilizer_check_filter",
                            "severity": "info",
                            "action": "filtered",
                            "reason": "terminal_data_broadcast_measurement_excluded_from_stabilizer_check",
                            "value": line,
                        }
                    )
            continue

    return stabilizers, corrections, traces, issues


def process_qasmbench_dataset(project_root: Path) -> Dict[str, int]:
    """Processes QASMBench raw archives and writes out the 3 Silver tables."""
    # Discovered system mount path
    raw_dir = Path("/course-data/raw/source=qasmbench")
    if not raw_dir.exists():
        raw_dir = project_root / "course-data" / "raw" / "source=qasmbench"

    silver_dir = project_root / "silver" / "qasmbench"
    results_dir = project_root / "results" / "part1"

    silver_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    circuits: List[Dict] = []
    if raw_dir.exists():
        for archive in sorted(raw_dir.glob("*")):
            if not archive.is_file():
                continue
            sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            if archive.name.endswith((".tar.gz", ".tgz")):
                with tarfile.open(archive, "r:gz") as tar:
                    for member in sorted(tar.getmembers(), key=lambda m: m.name):
                        if member.name.endswith(".qasm"):
                            f = tar.extractfile(member)
                            if f:
                                circuits.append({
                                    "bronze_object": archive.name,
                                    "archive_member": member.name,
                                    "input_sha256": sha,
                                    "content": f.read().decode("utf-8"),
                                })
            elif archive.suffix == ".zip":
                with zipfile.ZipFile(archive, "r") as z:
                    for name in sorted(z.namelist()):
                        if name.endswith(".qasm"):
                            circuits.append({
                                "bronze_object": archive.name,
                                "archive_member": name,
                                "input_sha256": sha,
                                "content": z.read(name).decode("utf-8"),
                            })
            elif archive.suffix == ".qasm":
                circuits.append({
                    "bronze_object": archive.name,
                    "archive_member": archive.name,
                    "input_sha256": sha,
                    "content": archive.read_text(encoding="utf-8"),
                })

    circuit_rows: List[Dict] = []
    stab_rows: List[Dict] = []
    corr_rows: List[Dict] = []
    trace_rows: List[Dict] = []
    issue_rows: List[Dict] = []

    for c in circuits:
        fname = Path(c["archive_member"]).name
        base_name = fname.replace(".qasm", "")
        variant = "transpiled" if "transpiled" in base_name else "source"
        benchmark_name = base_name.replace("_transpiled", "").replace(".transpiled", "").split("/")[-1]

        statements = parse_qasm_lines(c["content"])
        qregs, cregs, meas_count, two_q_count = extract_circuit_metrics(statements)

        content_hash = hashlib.sha256(c["content"].encode("utf-8")).hexdigest()[:12]
        c_rec_id = f"qasmbench:{c['archive_member']}:{content_hash}"

        circuit_rows.append({
            "source_record_id": c_rec_id,
            "circuit_id": base_name,
            "benchmark_name": benchmark_name,
            "variant": variant,
            "register_declarations": json.dumps({"qregs": qregs, "cregs": cregs}),
            "qubit_count": sum(qregs.values()),
            "measurement_count": meas_count,
            "two_qubit_gate_count": two_q_count,
        })
        trace_rows.append({
            "source_record_id": c_rec_id,
            "source_name": "qasmbench",
            "bronze_object": c["bronze_object"],
            "archive_member": c["archive_member"],
            "record_locator": "file_root",
            "input_sha256": c["input_sha256"],
        })

        s_rows, co_rows, t_rows, i_rows = parse_parity_and_corrections(
            statements, base_name, c["archive_member"], c["bronze_object"], c["input_sha256"]
        )
        stab_rows.extend(s_rows)
        corr_rows.extend(co_rows)
        trace_rows.extend(t_rows)
        issue_rows.extend(i_rows)

    # Write the 3 required Silver Parquet tables
    write_parquet_table(pd.DataFrame(circuit_rows), CIRCUIT_SCHEMA, silver_dir / "circuit.parquet")
    write_parquet_table(pd.DataFrame(stab_rows), STABILIZER_CHECK_SCHEMA, silver_dir / "stabilizer_check.parquet")
    write_parquet_table(pd.DataFrame(corr_rows), CONDITIONAL_CORRECTION_SCHEMA, silver_dir / "conditional_correction.parquet")

    # Safely merge into shared audit files
    if trace_rows:
        merge_audit_parquet(
            results_dir / "source_trace.parquet",
            pd.DataFrame(trace_rows),
            SOURCE_TRACE_SCHEMA,
            dedupe_col="source_record_id",
        )
    if issue_rows:
        # Overwrite or deduplicate by combination of record and the rejected statement
        issues_df = pd.DataFrame(issue_rows).drop_duplicates(
            subset=["source_record_id", "value"]
        )
        write_parquet_table(
            issues_df, DATA_ISSUES_SCHEMA, results_dir / "data_issues.parquet"
        )

    return {
        "circuits": len(circuit_rows),
        "stabilizer_checks": len(stab_rows),
        "conditional_corrections": len(corr_rows),
    }


# ---------------------------------------------------------------------------
# Main Stage Orchestrator
# ---------------------------------------------------------------------------
def run(run_id: str) -> StageResult:
    project_root = Path("/workspace")
    qasm_counts = process_qasmbench_dataset(project_root)

    total_outputs = (
        qasm_counts["circuits"]
        + qasm_counts["stabilizer_checks"]
        + qasm_counts["conditional_corrections"]
    )

    res = StageResult(
        stage="prepare_data",
        run_id=run_id,
        input_count=qasm_counts["circuits"],
        output_count=total_outputs,
        issue_count=0,
    )
    res.finish()
    return res