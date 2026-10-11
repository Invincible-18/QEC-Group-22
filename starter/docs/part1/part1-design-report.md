# Part I design report: QASMBench

This section covers the QASMBench source from Bronze to Gold. It is based on
`register_sources.py`, `prepare_data.py`, `gold_qasmbench.py` and the files in
`results/part1/`.

## 1. What the source contains

`qasmbench-qec.zip` holds three small benchmark folders: `qec_sm_n5`,
`qec_en_n5` and `error_correctiond3_n5`. Each has a source circuit and a
transpiled circuit (the same circuit rewritten with basic gates), so there are
six `.qasm` files. The archive also has a README, licence files and
`qelib1.inc`, which we do not use.

The three benchmarks differ a lot. Only `qec_sm_n5` has an ancilla register and
`if (syn == k)` corrections. It is a 3-data-qubit repetition code with 2 ancilla
qubits. The other two have no ancilla register and no conditional statements,
so they produce no stabilizer checks or corrections. That is expected, not a
parsing gap. None of the files contains shots or results, only circuits.

## 2. Bronze: checking before reading

`register_sources.py` compares every Bronze object with the release manifest
before anything else runs. The size and SHA-256 must match, and it stops with an
error otherwise. For QASMBench the hash is
`60307f88e34b1f752b94223d6d136da72c629b4b625ac6d1c30e5e2e4f85722a`
(recorded in `results/part1/run.json`).

It also rejects archive members that are absolute paths, drive-letter paths or
contain `..` (`check_safe_archive_member`). `prepare_data.py` repeats the check
and also stops if a benchmark folder lacks its `.qasm` or `_transpiled.qasm`
file. We only read the archive in memory and never change or unpack it.

## 3. Silver

We parse each QASM file line by line, drop comments and skip the body of custom
`gate` definitions. Silver then has three tables:

| table | one row is | rows |
| --- | --- | --- |
| `circuit` | one circuit file (benchmark and source/transpiled) | 6 |
| `stabilizer_check` | one parity check: ancilla, data qubits, syndrome bit | 4 |
| `conditional_correction` | one `if (syn == k) x q[i];` statement | 6 |

The checks and corrections all come from `qec_sm_n5`, two checks and three
corrections in each of its two variants. Column names and types follow
`silver-tables.md`.

Decisions that shaped these tables:

- **Register-wide measurements are expanded.** `silver-tables.md` asks for
  executed operations, not statement lines, so `measure q -> c;` counts as one
  measurement per qubit in the register. Every circuit ends up with
  `measurement_count = 5`.
- **Gates inside `gate` definitions are not counted.** The source `qec_sm_n5`
  hides its CNOTs in a custom gate, so it has `two_qubit_gate_count = 0`. The
  transpiled version spells them out and has 4. The counts are not comparable
  between the two variants.
- **A check is a measurement into the syndrome register.** We track which data
  qubits were hit by a CNOT or CZ before an ancilla is measured. A measurement
  of the ancilla into `syn` is a check. A measurement of a data qubit into `c` is
  a final readout and not a check (see section 4).
- **The source `qec_sm_n5` needs a special case.** It measures both ancillas
  with one line, `measure a -> syn;`, and its CNOTs are hidden in the custom
  gate. We expand that line into two checks and set the data qubits to
  `q[i], q[i+1]`. That pattern is hard-coded for two ancillas, as a comment in
  `prepare_data.py` says. It would need rewriting for a different circuit.
- **Identifiers do not depend on the run.** Each `source_record_id` is a SHA-256
  of the Bronze object, archive member and, for checks and corrections, the line
  number. A rerun on unchanged input therefore gives the same ids.

## 4. Quality findings

- **Readouts are not checks.** The final `measure q[i] -> c[i];` lines measure
  data qubits and say nothing about syndromes, so we do not turn them into
  stabilizer checks. We also do not delete them silently. Each one is written to
  `data_issues.parquet` with rule `stabilizer_check_filter`, severity `info`,
  action `filtered` and the exact QASM line. There are 24 such rows.
  `row_counts.json` reports the same number as
  `measurement_statements_filtered`.
- **The whole issues file has 25 rows.** Those 24 plus one note about the
  syndrome CSV header (`labels`, not `label`). No QASMBench row is rejected.
- **A formatting quirk in the raw files.** Line 98 of
  `error_correctiond3_n5.qasm` is `cx q[1],q[2] ;`, with a space before the
  semicolon. Our first regex required `;` right after the operand, so that CNOT
  was not counted and the source circuit showed 48 two-qubit gates against 49 in
  its transpiled version. We found it by comparing Silver with the raw file. All
  statement patterns now allow whitespace before the semicolon, and a test
  covers it. After the fix both variants give 49.
- **Tracing.** Each of the 16 Silver rows (6 + 4 + 6) has a row in
  `source_trace.parquet` giving the archive member and the line, or `file_root`
  for a circuit. The full trace file is larger because it also covers the other
  two sources.

## 5. Gold

QASMBench keeps its own three tables in PostgreSQL:

- `circuit`: primary key `circuit_id`, unique `source_record_id`, unique
  `(benchmark_name, variant)`. `variant` must be `source` or `transpiled`, and
  the qubit, measurement and gate counts must not be negative.
  `register_declarations` is stored as `jsonb` so registers can be queried.
- `stabilizer_check`: primary key `(circuit_id, check_id)` and a foreign key to
  `circuit`. A syndrome bit is also unique within a circuit, because each bit
  receives exactly one check result. `data_qubits` is a `text[]` and cannot be
  empty.
- `conditional_correction`: primary key `source_record_id`, foreign key to
  `circuit`, and a unique key on `(circuit_id, condition_register,
  condition_value, gate, target_qubit)`.

Every index comes from a primary key or unique constraint. The child tables
look up circuits through their `circuit_id` indexes.

The Gold schema is rebuilt from Silver on every run, inside one transaction, so
a plain `INSERT` cannot create duplicates. Before the transaction ends,
`validate_qasmbench` checks that:

- row counts equal Silver;
- `qubit_count` equals the declared quantum registers;
- every correction names a declared classical register and a value that fits
  its size;
- every syndrome bit names a declared classical register.

Any failure rolls the whole Gold load back.

## 6. A relationship we rejected

We considered linking QASMBench circuits to the Google and simulated syndrome
experiments, on the idea that a circuit documents the code that produced the
shot data. We rejected it for three reasons:

- no source shares an identifier. QASMBench has a benchmark name and variant,
  Google experiments are keyed by basis, distance, rounds and chip position, and
  the syndrome files by fault rate;
- the codes do not match. `qec_sm_n5` is a 3-qubit repetition code with two
  checks, while the Google and syndrome data come from a distance-3 or
  distance-5 surface code (8 checks at distance 3, from the primer);
- `qec_en_n5` and `error_correctiond3_n5` have no syndromes, corrections or
  shots, so there is nothing to join through.

QASMBench therefore stays as a set of reference circuits with no foreign key to
the other sources.

## 7. What the analysis shows

Question 3 joins `circuit`, `stabilizer_check` and `conditional_correction` for
`qec_sm_n5`. The two checks cover qubits `q[0], q[1]` (bit `syn[0]`) and
`q[1], q[2]` (bit `syn[1]`). The three corrections then read like a lookup table:

| `syn` value | checks that fired | correction |
| --- | --- | --- |
| 1 | `syn[0]` only | flip `q[0]` |
| 2 | `syn[1]` only | flip `q[2]` |
| 3 | both | flip `q[1]` |

Qubit `q[1]` belongs to both checks, so only an error on `q[1]` fires both. The
circuit itself starts with `x q[0]` as a deliberate injected error. Only the
first check sees it, so `syn` ends up as 1 and the `syn == 1` correction undoes
the flip. The transpiled variant has the same checks and corrections, only with
different gates and line numbers.

## 8. Limitations

- The source and transpiled gate counts for `qec_sm_n5` cannot be compared (see
  section 3).
- The broadcast expansion and the data qubits of the source checks are written
  for `qec_sm_n5` only.
- Only one of the three circuits has checks and corrections, so the QASMBench
  tables are small and meant as reference material, not for statistics.