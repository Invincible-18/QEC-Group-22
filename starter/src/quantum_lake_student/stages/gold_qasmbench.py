"""QASMBench part of Gold: tables, load from Silver, and checks.

Called by load_postgres.rebuild_gold() inside its single transaction, with
search_path set to the Gold schema. The schema is rebuilt on every run, so
plain INSERTs cannot create duplicates and every key comes from the data.

QASMBench circuits are kept as their own entities: no supplied identifier links
a circuit to a Google or simulated experiment, so there is no foreign key to
either.
"""

from __future__ import annotations

import psycopg
import pyarrow as pa


QASMBENCH_SILVER_TABLES = {
    "qasm_circuit": "silver/qasmbench/circuit.parquet",
    "qasm_stabilizer_check": "silver/qasmbench/stabilizer_check.parquet",
    "qasm_conditional_correction": "silver/qasmbench/conditional_correction.parquet",
}

QASMBENCH_GOLD_TABLES = ("circuit", "stabilizer_check", "conditional_correction")

# Every index comes from a primary key or unique constraint; circuit_id lookups
# on the two child tables use the indexes whose first column is circuit_id.
QASMBENCH_TABLES = """
-- One row represents one parsed QASMBench circuit variant (benchmark x
-- source/transpiled).
CREATE TABLE circuit (
    circuit_id            text    PRIMARY KEY,
    source_record_id      text    NOT NULL UNIQUE,
    benchmark_name        text    NOT NULL,
    variant               text    NOT NULL CHECK (variant IN ('source', 'transpiled')),
    register_declarations jsonb   NOT NULL,
    qubit_count           integer NOT NULL CHECK (qubit_count > 0),
    measurement_count     integer NOT NULL CHECK (measurement_count >= 0),
    two_qubit_gate_count  integer NOT NULL CHECK (two_qubit_gate_count >= 0),
    UNIQUE (benchmark_name, variant)
);

-- One row represents one parity check identified in one circuit: an ancilla,
-- the data qubits it measures, and the classical bit that receives the result.
CREATE TABLE stabilizer_check (
    circuit_id       text   NOT NULL REFERENCES circuit,
    check_id         text   NOT NULL,
    source_record_id text   NOT NULL UNIQUE,
    ancilla_qubit    text   NOT NULL,
    data_qubits      text[] NOT NULL CHECK (cardinality(data_qubits) > 0),
    syndrome_bit     text   NOT NULL,
    PRIMARY KEY (circuit_id, check_id),
    -- each syndrome bit receives the result of exactly one check
    UNIQUE (circuit_id, syndrome_bit)
);

-- One row represents one recovery operation in one circuit that runs when a
-- classical register holds a given value.
CREATE TABLE conditional_correction (
    source_record_id   text   PRIMARY KEY,
    circuit_id         text   NOT NULL REFERENCES circuit,
    condition_register text   NOT NULL,
    condition_value    bigint NOT NULL CHECK (condition_value >= 0),
    gate               text   NOT NULL,
    target_qubit       text   NOT NULL,
    UNIQUE (circuit_id, condition_register, condition_value, gate, target_qubit)
);
"""


def load_qasmbench(connection: psycopg.Connection, silver: dict[str, pa.Table]) -> dict[str, int]:
    """Load QASMBench Silver into the Gold tables on the current search_path.

    Has to run inside a transaction. Returns the row count of each QASMBench table.
    """
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO circuit (circuit_id, source_record_id, benchmark_name, variant,
                                 register_declarations, qubit_count, measurement_count,
                                 two_qubit_gate_count)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s)
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
                for row in silver["qasm_circuit"].to_pylist()
            ],
        )
        cursor.executemany(
            """
            INSERT INTO stabilizer_check (circuit_id, check_id, source_record_id,
                                          ancilla_qubit, data_qubits, syndrome_bit)
            VALUES (%s, %s, %s, %s, %s, %s)
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
                for row in silver["qasm_stabilizer_check"].to_pylist()
            ],
        )
        cursor.executemany(
            """
            INSERT INTO conditional_correction (source_record_id, circuit_id,
                                                condition_register, condition_value,
                                                gate, target_qubit)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    row["source_record_id"],
                    row["circuit_id"],
                    row["condition_register"],
                    row["condition_value"],
                    row["gate"],
                    row["target_qubit"],
                )
                for row in silver["qasm_conditional_correction"].to_pylist()
            ],
        )
        return {
            table: cursor.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in QASMBENCH_GOLD_TABLES
        }


def validate_qasmbench(connection: psycopg.Connection, silver: dict[str, pa.Table]) -> None:
    """Compare the loaded QASMBench Gold with Silver and raise if something is off.

    Raising inside the load transaction rolls the whole Gold load back.
    """
    expected = {
        "circuit rows equal Silver": (
            "SELECT count(*) FROM circuit", silver["qasm_circuit"].num_rows),
        "stabilizer_check rows equal Silver": (
            "SELECT count(*) FROM stabilizer_check", silver["qasm_stabilizer_check"].num_rows),
        "conditional_correction rows equal Silver": (
            "SELECT count(*) FROM conditional_correction",
            silver["qasm_conditional_correction"].num_rows),
        "qubit_count equals the declared quantum registers": (
            """SELECT count(*) FROM circuit
               WHERE qubit_count <> (SELECT sum(size::int)
                                     FROM jsonb_each_text(register_declarations -> 'qregs')
                                          AS q(name, size))""", 0),
        "correction registers are declared and the value fits their size": (
            """SELECT count(*) FROM conditional_correction AS cc
               JOIN circuit AS c USING (circuit_id)
               WHERE NOT (c.register_declarations -> 'cregs') ? cc.condition_register
                  OR cc.condition_value
                     >= 2 ^ ((c.register_declarations -> 'cregs' ->> cc.condition_register)::int)""", 0),
        "syndrome bits name a declared classical register": (
            """SELECT count(*) FROM stabilizer_check AS sc
               JOIN circuit AS c USING (circuit_id)
               WHERE NOT (c.register_declarations -> 'cregs') ? split_part(sc.syndrome_bit, '[', 1)""", 0),
    }
    failures = []
    for rule, (query, want) in expected.items():
        got = connection.execute(query).fetchone()[0]
        if got != want:
            failures.append(f"{rule}: expected {want}, got {got}")
    if failures:
        raise RuntimeError("QASMBench Gold validation failed: " + "; ".join(failures))
