"""Small Silver tables for Gold tests, so every test load includes every source.

rebuild_gold() loads all sources in one transaction, so a test that is about one
source still has to supply valid Silver for the others.
"""

import json

import pyarrow as pa

from quantum_lake_student.stages.prepare_data import (
    CIRCUIT_SCHEMA,
    CONDITIONAL_CORRECTION_SCHEMA,
    STABILIZER_CHECK_SCHEMA,
    SYNDROME_SCHEMA,
)


QASM_CIRCUIT_ID = "qec_sm_n5"


def qasmbench_silver() -> dict[str, pa.Table]:
    """The real qec_sm_n5 circuit: 3 data qubits, 2 parity checks, 3 corrections."""
    circuit = pa.Table.from_pylist(
        [
            {
                "source_record_id": "circuit-record",
                "circuit_id": QASM_CIRCUIT_ID,
                "benchmark_name": QASM_CIRCUIT_ID,
                "variant": "source",
                "register_declarations": json.dumps(
                    {"qregs": {"q": 3, "a": 2}, "cregs": {"c": 3, "syn": 2}}
                ),
                "qubit_count": 5,
                "measurement_count": 5,
                "two_qubit_gate_count": 4,
            }
        ],
        schema=CIRCUIT_SCHEMA,
    )
    checks = pa.Table.from_pylist(
        [
            {
                "source_record_id": f"check-record-{bit}",
                "circuit_id": QASM_CIRCUIT_ID,
                "check_id": f"chk_{QASM_CIRCUIT_ID}_syn_{bit}",
                "ancilla_qubit": f"a[{bit}]",
                "data_qubits": [f"q[{bit}]", f"q[{bit + 1}]"],
                "syndrome_bit": f"syn[{bit}]",
            }
            for bit in (0, 1)
        ],
        schema=STABILIZER_CHECK_SCHEMA,
    )
    corrections = pa.Table.from_pylist(
        [
            {
                "source_record_id": f"correction-record-{value}",
                "circuit_id": QASM_CIRCUIT_ID,
                "condition_register": "syn",
                "condition_value": value,
                "gate": "x",
                "target_qubit": target,
            }
            for value, target in ((1, "q[0]"), (2, "q[2]"), (3, "q[1]"))
        ],
        schema=CONDITIONAL_CORRECTION_SCHEMA,
    )
    return {
        "qasm_circuit": circuit,
        "qasm_stabilizer_check": checks,
        "qasm_conditional_correction": corrections,
    }


def syndrome_silver() -> dict[str, pa.Table]:
    """One syndrome pattern observed under both labels in one experiment."""
    pattern = bytes([0, 1] * 8)
    rows = pa.Table.from_pylist(
        [
            {
                "source_record_id": f"syndrome-record-{label}",
                "experiment_id": "qec_syndromes/d-3_pfr-0.001000_nb-10M",
                "physical_fault_rate": 0.001,
                "syndrome_bits": pattern,
                "round_count": 4,
                "check_count": 4,
                "logical_error_label": label,
                "quantity": quantity,
            }
            for label, quantity in ((False, 7), (True, 2))
        ],
        schema=SYNDROME_SCHEMA,
    )
    return {"syndrome_observation": rows}


def other_sources_silver() -> dict[str, pa.Table]:
    """Silver for every source except Google, for tests that build their own Google rows."""
    return {**qasmbench_silver(), **syndrome_silver()}
