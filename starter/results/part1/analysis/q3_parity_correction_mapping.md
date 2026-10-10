## QASM Analysis Question
How does the repetition-code circuit map data qubits to parity-check ancillas, syndrome bits, and conditional corrections?

## Reproducible SQL Query

`sql/q3_parity_correction_mapping.sql`, run against Gold by the pipeline; its
output is written to `q3_parity_correction_mapping.csv` in this folder. It joins
`circuit`, `conditional_correction` and `stabilizer_check`, matching each
correction only to the checks whose syndrome bit is set in its condition value
(1 -> syn[0], 2 -> syn[1], 3 -> both).

## Detailed Interpretation

This analysis examines how the QASMBench OpenQASM circuits structurally implement quantum error correction (QEC) semantics by mapping physical data qubits to parity-check ancillas, syndrome measurement registers, and conditional feedback operations. 

By executing a three-table join across the `circuit`, `stabilizer_check`, and `conditional_correction` Gold relations, we can trace the complete error-correction loop for the repetition-code benchmark (`qec_sm_n5`):

1. **Parity Acquisition (`stabilizer_check`)**: The circuit uses a specific configuration of data qubits and ancilla measurement qubits. For instance, `qec_sm_n5` initializes 3 data qubits and 2 ancillas. Entangling operations (such as CNOT pairs) measure the parities of adjacent data qubits into designated ancillas, storing the resulting check outcome in a classical syndrome register bit without directly reading the protected data state.
2. **Syndrome-Triggered Feedback (`conditional_correction`)**: The retrieved syndrome bit values act as logical conditions (e.g., `if (syn == 1)` or values 1, 2, and 3) that dynamically trigger corrective gate operations (such as Pauli-X flips) on specific target data qubits. 
3. **Variant Comparison (`source` vs. `transpiled`)**: Comparing the source variant against its transpiled counterpart reveals how high-level logical QEC descriptions are compiled down to target gate sets. While the high-level mapping between data qubits, syndrome bits, and conditional corrections remains logically identical, the transpiled variant expands the statement and gate counts to fit hardware execution constraints.

Overall, this relational model demonstrates that while QASMBench circuits lack runtime shot results, their structural metadata provides a complete blueprint of how parity evidence flows into corrective feedback.

