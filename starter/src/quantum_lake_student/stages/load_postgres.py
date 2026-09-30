"""Relational integration stage for Gold."""
from __future__ import annotations

import json
from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client, postgres_connection
from quantum_lake_student.io_utils import read_parquet
from quantum_lake_student.models import StageResult

QASM_DDL = """
CREATE TABLE IF NOT EXISTS circuit (
    circuit_id TEXT PRIMARY KEY,
    source_record_id TEXT NOT NULL,
    benchmark_name TEXT NOT NULL,
    variant TEXT NOT NULL CHECK (variant IN ('source', 'transpiled')),
    register_declarations JSONB NOT NULL,
    qubit_count INTEGER NOT NULL CHECK (qubit_count > 0),
    measurement_count INTEGER NOT NULL CHECK (measurement_count >= 0),
    two_qubit_gate_count INTEGER NOT NULL CHECK (two_qubit_gate_count >= 0)
);

CREATE TABLE IF NOT EXISTS stabilizer_check (
    circuit_id TEXT NOT NULL REFERENCES circuit(circuit_id) ON DELETE CASCADE,
    check_id TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    ancilla_qubit TEXT NOT NULL,
    data_qubits TEXT[] NOT NULL,
    syndrome_bit TEXT NOT NULL,
    PRIMARY KEY (circuit_id, check_id)
);

CREATE TABLE IF NOT EXISTS conditional_correction (
    id SERIAL PRIMARY KEY,
    circuit_id TEXT NOT NULL REFERENCES circuit(circuit_id) ON DELETE CASCADE,
    source_record_id TEXT NOT NULL,
    condition_register TEXT NOT NULL,
    condition_value BIGINT NOT NULL,
    gate TEXT NOT NULL,
    target_qubit TEXT NOT NULL,
    UNIQUE (circuit_id, condition_register, condition_value, gate, target_qubit)
);

CREATE INDEX IF NOT EXISTS idx_stabilizer_check_circuit ON stabilizer_check(circuit_id);
CREATE INDEX IF NOT EXISTS idx_conditional_correction_circuit ON conditional_correction(circuit_id);
"""

def load_qasmbench_gold(conn, client, bucket: str) -> int:
    """Load QASMBench Parquet tables from MinIO into Gold PostgreSQL."""
    circuit_t = read_parquet(client, bucket, "silver/qasmbench/circuit.parquet")
    check_t = read_parquet(client, bucket, "silver/qasmbench/stabilizer_check.parquet")
    corr_t = read_parquet(client, bucket, "silver/qasmbench/conditional_correction.parquet")

    with conn.cursor() as cur:
        # 1. Ensure schema exists
        cur.execute(QASM_DDL)

        # 2. Insert circuits (Parent table first)
        cur.executemany(
            """
            INSERT INTO circuit (
                circuit_id, source_record_id, benchmark_name, variant,
                register_declarations, qubit_count, measurement_count, two_qubit_gate_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (circuit_id) DO UPDATE SET
                source_record_id = EXCLUDED.source_record_id,
                benchmark_name = EXCLUDED.benchmark_name,
                variant = EXCLUDED.variant,
                register_declarations = EXCLUDED.register_declarations,
                qubit_count = EXCLUDED.qubit_count,
                measurement_count = EXCLUDED.measurement_count,
                two_qubit_gate_count = EXCLUDED.two_qubit_gate_count;
            """,
            [
                (
                    row["circuit_id"],
                    row["source_record_id"],
                    row["benchmark_name"],
                    row["variant"],
                    row["register_declarations"],
                    row["qubit_count"],
                    row["measurement_count"],
                    row["two_qubit_gate_count"],
                )
                for row in circuit_t.to_pylist()
            ],
        )

        # 3. Insert stabilizer checks (Child table)
        cur.executemany(
            """
            INSERT INTO stabilizer_check (
                circuit_id, check_id, source_record_id, ancilla_qubit, data_qubits, syndrome_bit
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (circuit_id, check_id) DO UPDATE SET
                source_record_id = EXCLUDED.source_record_id,
                ancilla_qubit = EXCLUDED.ancilla_qubit,
                data_qubits = EXCLUDED.data_qubits,
                syndrome_bit = EXCLUDED.syndrome_bit;
            """,
            [
                (
                    row["circuit_id"],
                    row["check_id"],
                    row["source_record_id"],
                    row["ancilla_qubit"],
                    list(row["data_qubits"]),
                    row["syndrome_bit"],
                )
                for row in check_t.to_pylist()
            ],
        )

        # 4. Insert conditional corrections (Child table)
        cur.executemany(
            """
            INSERT INTO conditional_correction (
                circuit_id, source_record_id, condition_register,
                condition_value, gate, target_qubit
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (circuit_id, condition_register, condition_value, gate, target_qubit) DO NOTHING;
            """,
            [
                (
                    row["circuit_id"],
                    row["source_record_id"],
                    row["condition_register"],
                    row["condition_value"],
                    row["gate"],
                    row["target_qubit"],
                )
                for row in corr_t.to_pylist()
            ],
        )

    return circuit_t.num_rows + check_t.num_rows + corr_t.num_rows

def run(run_id: str) -> StageResult:
    result = StageResult(stage="load_postgres", run_id=run_id)
    settings = Settings.from_environment()
    client = minio_client(settings)
    bucket = settings.s3_bucket

    # All-or-nothing transaction block
    with postgres_connection(settings) as conn:
        with conn.transaction():
            total_loaded = load_qasmbench_gold(conn, client, bucket)

    result.output_count = total_loaded
    result.finish()
    return result