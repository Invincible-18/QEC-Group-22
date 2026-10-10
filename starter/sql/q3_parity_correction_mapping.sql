-- Q3: how does the repetition-code circuit map data qubits to parity-check
-- ancillas, syndrome bits, and conditional corrections?
--
-- A correction runs when its classical register holds condition_value. Bit i of
-- that value is syndrome bit <register>[i], so each correction is joined only to
-- the checks whose syndrome bit is 1 in the value (not to every check of the
-- circuit). Joins three Gold tables. target_in_every_fired_check shows whether
-- the corrected qubit is shared by all the checks that fired.
SELECT c.benchmark_name,
       c.variant,
       cc.condition_register,
       cc.condition_value,
       string_agg(sc.syndrome_bit, ', ' ORDER BY sc.syndrome_bit) AS fired_syndrome_bits,
       string_agg(sc.ancilla_qubit || ' checks ' || array_to_string(sc.data_qubits, ' & '),
                  '; ' ORDER BY sc.syndrome_bit) AS fired_checks,
       cc.gate AS correction_gate,
       cc.target_qubit AS corrected_qubit,
       bool_and(cc.target_qubit = ANY (sc.data_qubits)) AS target_in_every_fired_check
FROM gold.circuit AS c
JOIN gold.conditional_correction AS cc
  ON cc.circuit_id = c.circuit_id
JOIN gold.stabilizer_check AS sc
  ON sc.circuit_id = cc.circuit_id
 AND split_part(sc.syndrome_bit, '[', 1) = cc.condition_register
 AND (cc.condition_value >> substring(sc.syndrome_bit FROM '\[(\d+)\]')::int) & 1 = 1
GROUP BY c.benchmark_name, c.variant, cc.source_record_id, cc.condition_register,
         cc.condition_value, cc.gate, cc.target_qubit
ORDER BY c.benchmark_name, c.variant, cc.condition_value;
