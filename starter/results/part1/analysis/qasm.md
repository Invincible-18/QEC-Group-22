## QASM Analysis Question
How does the repetition-code circuit map data qubits to parity-check ancillas, syndrome bits, and conditional corrections?

## Reproducible SQL Query

```sql
SELECT 
    c.circuit_id,
    c.benchmark_name,
    c.variant,
    sc.check_id,
    sc.ancilla_qubit,
    sc.data_qubits,
    sc.syndrome_bit,
    cc.condition_value,
    cc.gate AS correction_gate,
    cc.target_qubit AS corrected_qubit
FROM circuit c
JOIN stabilizer_check sc 
    ON c.circuit_id = sc.circuit_id
JOIN conditional_correction cc 
    ON c.circuit_id = cc.circuit_id
WHERE c.benchmark_name = 'qec_sm_n5'
```

## Output:

('qec_sm_n5', 'qec_sm_n5', 'source', 'chk_qec_sm_n5_syn_0', 'a[0]', ['q[0]', 'q[1]'], 'syn[0]', 1, 'x', 'q[0]')
('qec_sm_n5', 'qec_sm_n5', 'source', 'chk_qec_sm_n5_syn_0', 'a[0]', ['q[0]', 'q[1]'], 'syn[0]', 2, 'x', 'q[2]')
('qec_sm_n5', 'qec_sm_n5', 'source', 'chk_qec_sm_n5_syn_0', 'a[0]', ['q[0]', 'q[1]'], 'syn[0]', 3, 'x', 'q[1]')
('qec_sm_n5', 'qec_sm_n5', 'source', 'chk_qec_sm_n5_syn_1', 'a[1]', ['q[1]', 'q[2]'], 'syn[1]', 1, 'x', 'q[0]')
('qec_sm_n5', 'qec_sm_n5', 'source', 'chk_qec_sm_n5_syn_1', 'a[1]', ['q[1]', 'q[2]'], 'syn[1]', 2, 'x', 'q[2]')
('qec_sm_n5', 'qec_sm_n5', 'source', 'chk_qec_sm_n5_syn_1', 'a[1]', ['q[1]', 'q[2]'], 'syn[1]', 3, 'x', 'q[1]')
('qec_sm_n5_transpiled', 'qec_sm_n5', 'transpiled', 'chk_qec_sm_n5_transpiled_syn_0', 'a[0]', ['q[0]', 'q[1]'], 'syn[0]', 1, 'x', 'q[0]')
('qec_sm_n5_transpiled', 'qec_sm_n5', 'transpiled', 'chk_qec_sm_n5_transpiled_syn_0', 'a[0]', ['q[0]', 'q[1]'], 'syn[0]', 2, 'x', 'q[2]')
('qec_sm_n5_transpiled', 'qec_sm_n5', 'transpiled', 'chk_qec_sm_n5_transpiled_syn_0', 'a[0]', ['q[0]', 'q[1]'], 'syn[0]', 3, 'x', 'q[1]')
('qec_sm_n5_transpiled', 'qec_sm_n5', 'transpiled', 'chk_qec_sm_n5_transpiled_syn_1', 'a[1]', ['q[1]', 'q[2]'], 'syn[1]', 1, 'x', 'q[0]')
('qec_sm_n5_transpiled', 'qec_sm_n5', 'transpiled', 'chk_qec_sm_n5_transpiled_syn_1', 'a[1]', ['q[1]', 'q[2]'], 'syn[1]', 2, 'x', 'q[2]')
('qec_sm_n5_transpiled', 'qec_sm_n5', 'transpiled', 'chk_qec_sm_n5_transpiled_syn_1', 'a[1]', ['q[1]', 'q[2]'], 'syn[1]', 3, 'x', 'q[1]')

## Detailed Interpretation

This analysis examines how the QASMBench OpenQASM circuits structurally implement quantum error correction (QEC) semantics by mapping physical data qubits to parity-check ancillas, syndrome measurement registers, and conditional feedback operations. 

By executing a three-table join across the `circuit`, `stabilizer_check`, and `conditional_correction` Gold relations, we can trace the complete error-correction loop for the repetition-code benchmark (`qec_sm_n5`):

1. **Parity Acquisition (`stabilizer_check`)**: The circuit uses a specific configuration of data qubits and ancilla measurement qubits. For instance, `qec_sm_n5` initializes 3 data qubits and 2 ancillas. Entangling operations (such as CNOT pairs) measure the parities of adjacent data qubits into designated ancillas, storing the resulting check outcome in a classical syndrome register bit without directly reading the protected data state.
2. **Syndrome-Triggered Feedback (`conditional_correction`)**: The retrieved syndrome bit values act as logical conditions (e.g., `if (syn == 1)` or values 1, 2, and 3) that dynamically trigger corrective gate operations (such as Pauli-X flips) on specific target data qubits. 
3. **Variant Comparison (`source` vs. `transpiled`)**: Comparing the source variant against its transpiled counterpart reveals how high-level logical QEC descriptions are compiled down to target gate sets. While the high-level mapping between data qubits, syndrome bits, and conditional corrections remains logically identical, the transpiled variant expands the statement and gate counts to fit hardware execution constraints.

Overall, this relational model demonstrates that while QASMBench circuits lack runtime shot results, their structural metadata provides a complete blueprint of how parity evidence flows into corrective feedback.

