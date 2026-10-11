# Part I design report

This report covers Part I of the pipeline: how the three course sources are
checked, cleaned and modelled, from the unchanged archives to the two ML
tables. The overview describes what all sources share. The three source parts
then follow one source each from Bronze to Gold, with its quality findings, a
trace and its analysis question; each part numbers its own sections. The reasoning
behind each decision, with the evidence, is in `notebooks/decision-log.md`.

## Overview

### Architecture

| layer | where | what it holds |
| --- | --- | --- |
| Bronze | MinIO `bronze/` | the three course archives, unchanged |
| Silver | MinIO `silver/` | the six required Parquet tables, cleaned and traced |
| Gold | PostgreSQL schema `gold` | 11 tables and 2 views with keys, constraints and indexes |
| ML | MinIO `ml/` | `ml_syndrome_decoder_example` and `ml_google_decoder_example`, exported from Gold |
| results | local `results/part1/` | run record, row counts, data issues, source trace, trace examples, analyses |

`make run-part1` runs the four stages in order: `register_sources` checks
Bronze, `prepare_data` builds Silver and the issue and trace files,
`load_postgres` rebuilds Gold, and `build_ml_tables` exports the ML tables and
runs the three analysis queries. `run.json` records the code revision, the data
release, the input hashes, counts and timings.

### Rules shared by all sources

- **Bronze is checked first.** Every archive must match the release manifest in
  size and SHA-256, and an unsafe member name (absolute path, drive letter or
  `..`) stops the run; so does a missing required file in a Google experiment
  or QASMBench benchmark folder. Archives are only read in memory, never
  unpacked or changed.
- **Identifiers come from the data.** `source_record_id` is a SHA-256 of where a
  record sits in Bronze, `example_id` a SHA-256 of its business key; neither
  depends on the run or on row order, so a rerun gives the same ids.
- **Nothing is dropped silently.** A record that fails a check is excluded or
  flagged and written to `data_issues.parquet` with its rule, severity, value,
  action and reason. In the final run the file has 25 rows: one documented
  naming note for the syndromes and 24 filtered QASMBench readouts; no record
  was rejected.
- **Gold is rebuilt all-or-nothing.** `load_postgres.rebuild_gold` drops and
  recreates the schema and loads all three sources in one transaction; any
  failure rolls everything back and the previous Gold stays.
- **Reruns are proven safe.** `make rerun-check-part1` runs Part I twice and
  compares 25 outputs (Gold tables, ML tables, Silver and result files); all
  were identical. A full run from a clean environment reproduced every result.
- **The ML tables come only from Gold,** through the committed SQL
  (`sql/ml_*.sql`) and the Gold views, and are checked against their contracts
  before they are written (last section).

### Gold at a glance

Gold keeps each source in its own tables (Syndromes section 4, Google section 4,
QASMBench section 5). The sources share vocabulary (experiment, syndrome, decoder,
check), but no foreign key joins two sources: we examined two such links and
rejected both, hardware to simulated experiments by distance (Google section 8)
and QASMBench circuits to the shot data (QASMBench section 6).

| question | Gold tables joined | where |
| --- | --- | --- |
| Q1: syndromes and logical errors by fault rate | `syndrome_observation`, `syndrome_experiment`, `syndrome_pattern` | Syndromes section 8 |
| Q2: decoder error rates per experiment | `shot_prediction`, `shot`, `hardware_experiment` | Google section 10 |
| Q3: parity checks and corrections | `circuit`, `stabilizer_check`, `conditional_correction` | QASMBench section 7 |

Each query joins three Gold tables; the results are written to
`results/part1/analysis/`. One syndrome and one Google example are traced from
their Part II prediction back to Bronze (Syndromes section 7, Google section 9).


## Part I design report: Simulated syndromes

This part covers the simulated syndrome data from Bronze to Gold. It is based on
the syndrome part of `stages/prepare_data.py`, `stages/gold_syndrome.py`, the
SQL in `sql/` and the files in `results/part1/`.

### 1. What the source contains

`syndromes_dataset.zip` holds seven CSV files and a README, one file per
simulated experiment of a distance-3 code at one physical fault rate, from
0.00001 to 0.01. The fault rate and the number of simulated shots are in the
file name, for example `d-3_pfr-0.005000_nb-10M.csv`.

Each file has three columns, `labels`, `syndromes` and `quantity`. One row is
an aggregate, not a shot: one syndrome with one logical-error label, and
`quantity`, the number of simulated shots that produced exactly that
combination. A syndrome is four rounds of four binary checks, written as a
nested tuple. The quantities of each file add up to 10,000,000, so the 75,598
rows represent 70 million shots.

Files with more noise have more rows: 68 at fault rate 0.00001 and 49,676 at
0.01, because more distinct syndrome and label combinations occur. The same
syndrome can occur twice in one file, once with each label, so a syndrome alone
is not a key; the key of a row is experiment, syndrome and label.

### 2. Bronze: checking before reading

`register_sources.py` checks the archive against the release manifest (size and
SHA-256) before anything else runs, as for the other sources. `prepare_data.py`
then checks every member name again and stops the run on an unsafe one. The
archive is read in memory, never unpacked or changed.

### 3. Silver

| table | one row is | rows |
| --- | --- | ---: |
| `syndrome_observation` | one original aggregate CSV row | 75,598 |

Column names and types follow `silver-tables.md`. Decisions:

- `quantity` stays a weight. Rows are never expanded into single shots; every
  later count is weighted by `quantity`.
- The syndrome is stored as 16 explicit 0/1 bytes, round first and check
  second, with `round_count` and `check_count` both 4.
- `experiment_id` and `physical_fault_rate` come from the file name.
- A row's `source_record_id` is a SHA-256 of the Bronze object, the archive
  member and the CSV row number, and its trace row points to `row=<n>` in that
  member.

The checks, and what happens when one fails:

| check | if it fails |
| --- | --- |
| an archive member has an unsafe name | the run stops |
| the file name contains a fault rate | the file is excluded and recorded |
| the columns are exactly `labels`, `syndromes`, `quantity` | the file is excluded and recorded |
| a row's label is 0 or 1, its quantity is above zero and its syndrome is four rounds of four binary values | the row is excluded and recorded |
| a file's quantities add up to the shot count in its name (within 0.1 %) | the file is kept and the mismatch is flagged |

A quantity mismatch is flagged rather than rejected, because every row in the
file can still be valid; the flag keeps it visible. Every excluded file or row
is written to `data_issues.parquet` with its value, rule, action and reason.

Tests: `tests/test_prepare_data.py` (syndrome parsing, valid and invalid rows
with their issues, a quantity mismatch flagged and a reconciled one accepted,
repeatable `source_record_id`, an unsafe member that stops the run).

### 4. Gold tables

| table | one row represents | primary key | rows |
| --- | --- | --- | ---: |
| `syndrome_experiment` | one simulated experiment at one physical fault rate | `experiment_id` | 7 |
| `syndrome_pattern` | one distinct 16-value syndrome | `syndrome_pattern_id` (SHA-256 of the bits) | 31,941 |
| `syndrome_observation` | one aggregate observation: experiment, pattern, label and `quantity` | `source_record_id` | 75,598 |

The view `syndrome_example` gives each observation its ML `example_id`, the
SHA-256 of `qec_syndromes:<experiment_id>:<syndrome hex>:<label>`, so every ML
example resolves to exactly one observation.

Foreign keys tie each observation to its experiment and its pattern. Checks
enforce the rules we rely on: the fault rate is between 0 and 1, the quantity is
above zero, and a pattern is exactly 16 bytes and unique. Besides the primary
keys, two indexes cover the lookups we do: observations by experiment (Q1, the
ML export) and by pattern and label (Q1's per-pattern counts).

Before inserting, the load checks every Silver row again and stops if an id is
missing or repeated, a value is out of range, a syndrome has the wrong shape or
one experiment has two fault rates. Since this runs inside the single Gold
transaction, any failure rolls the whole Gold load back.

Tests: `tests/test_gold_syndrome.py` (patterns and experiments deduplicated
while every observation and both labels are kept, invalid Silver values and
duplicate ids rejected).

### 5. Design decisions

1. Syndrome patterns are their own table. The same pattern occurs at several
   fault rates; storing it once, with a key, lets Q1 count patterns and compare
   them across experiments with a join instead of comparing byte strings. Cost:
   one more join for every query that needs the bits.
2. Observations keep their aggregate form. One Gold row per CSV row, with
   `quantity`, keeps 75,598 rows instead of 70 million and keeps the link to the
   source row. Cost: every count must be weighted, and a plain `COUNT(*)` gives a
   wrong answer.
3. Simulated experiments get their own table, separate from the hardware
   experiments (Google part, decision 5): a simulated experiment is described only by
   its fault rate.
4. The label stays on the observation, not on the pattern, because the same
   pattern can occur with both labels.

### 6. Quality findings

- All 75,598 rows were accepted and none rejected
  (`results/part1/row_counts.json`). Every quantity total reconciles with its
  file name, so nothing was flagged.
- The dataset description calls the label column `label`, while every CSV
  header says `labels`. We read the actual header and record the difference
  once in `data_issues.parquet` (rule `labels_column_naming`, severity `info`,
  action `documented`).
- The counts reconcile from Silver to Gold: 75,598 Silver rows give 75,598
  observations, 7 experiments and 31,941 distinct patterns.

### 7. One trace

`results/part1/trace_examples.json` follows one syndrome test example: the
all-zero syndrome without a logical error at fault rate 0.005, which stands for
5,309,458 shots. Its `example_id` (`021910f17367…`) resolves through
`gold.syndrome_example` to one `gold.syndrome_observation`, whose
`source_record_id` has one trace row: row 2 of `d-3_pfr-0.005000_nb-10M.csv` in
`syndromes_dataset.zip`. `results/part2/trace_predictions.json` then lists its
two Task A predictions; both predict no logical error, which is correct.

### 8. What the analysis shows

`sql/q1a_syndrome_by_fault_rate.sql` joins `syndrome_observation`,
`syndrome_experiment` and `syndrome_pattern` and weights every total by
`quantity`; `sql/q1b_syndrome_pattern_frequency.sql` gives the same per pattern.

| fault rate | distinct patterns | weighted logical-error rate | share of all-zero syndromes |
| ---: | ---: | ---: | ---: |
| 0.00001 | 68 | 0.0234 % | 99.87 % |
| 0.00005 | 202 | 0.1154 % | 99.36 % |
| 0.0001 | 445 | 0.2291 % | 98.74 % |
| 0.0005 | 1,228 | 1.1440 % | 93.84 % |
| 0.001 | 2,387 | 2.2702 % | 88.06 % |
| 0.005 | 14,643 | 10.3950 % | 53.10 % |
| 0.01 | 31,466 | 18.6527 % | 28.40 % |

As the fault rate grows, logical errors become more common and many more
distinct syndromes occur. The all-zero syndrome (no check fired) is the most
common pattern at every rate, but its share falls from almost all shots to
under a third, and logical errors behind it stay rare (at most 0.0356 %). The
details and definitions are in `results/part1/analysis/q1_syndrome_by_fault_rate.md`.
Counting CSV rows instead of weighting by `quantity` would give very different,
wrong frequencies.

### 9. Limitations

- All seven experiments are one code (distance 3, four rounds), so the results
  say nothing about other sizes.
- The fault rate is known only from the file name; nothing in the rows confirms
  it.
- A syndrome is a pattern, not an error: the same syndrome can come from
  different errors and lead to either label, so it cannot be a key on its own.


<!-- Written by Irina; unchanged except heading levels. -->
## Part I design report: Google QEC

This section covers the Google hardware data from Bronze to Gold: the source,
the checks before and while reading it, the Silver and Gold tables and why they
look like this, one trace and analysis Q2. It is based on
`stages/register_sources.py`, the Google part of `stages/prepare_data.py`,
`tracing.py`, `stages/gold_google.py`, the SQL in `sql/` and the files in
`results/`.

### 1. What the source contains

`google-surface-code-curated.zip` has a README and five experiment folders,
named after their settings: `surface_code_bX_d3_r25_center_3_5` is basis X,
distance 3, 25 rounds, with the centre data qubit at row 3, column 5. Four
experiments are distance 3 at different chip positions and one is distance 5.
All have 25 rounds and 50,000 shots (`properties.yml`).

Each folder has 15 files. We use nine:

| file | content |
| --- | --- |
| `properties.yml` | experiment settings and bit widths |
| `measurements.b8` | raw measurement bits per shot |
| `sweep.b8` | sweep bits that set the starting state of the data qubits |
| `detection_events.b8` | which detectors fired, derived from the measurement and sweep bits with the circuit |
| `obs_flips_actual.01` | whether the logical observable flipped: the ground truth |
| `obs_flips_predicted_by_*.01` | four files, one decoder's predictions each |

The other six (two Stim circuits, three detector error models and
`layout.svg`) describe the circuit, its noise and the chip layout. No required
output needs them.

The files have no shot id: shot *i* is record *i* of each `.b8` file and line
*i* of each `.01` file, so the eight per-shot files belong together by
position. A `.b8` record is one shot's bits, little-endian and padded to whole
bytes. At distance 3 a shot has 209 measurement bits, 9 sweep bits and 200
detector bits (625, 25 and 600 at distance 5), so the measurement and sweep
records end with 7 unused bits and the detector records have none.

In every experiment 49.0 to 50.0 % of shots flip. On average a shot has about
27 to 30 fired detectors at distance 3 and about 96 at distance 5.

Per shot we keep four things apart: measurements, detection events, the actual
flip and the predictions. A decoder error is not in the files. It only exists
when a prediction differs from the actual flip, so a flip on its own is not an
error.

### 2. Bronze: checking before reading

`register_sources.py` checks the archive against the release manifest before
anything else runs. A missing archive, a different size, a different SHA-256 or
an unsafe member name (an absolute path, a drive letter or `..`) stops the run.
The Google hash is recorded in `results/part1/run.json`. The archive is only
read in memory, never unpacked or changed.

`prepare_data.py` repeats the member-name check and also stops if an experiment
folder lacks `properties.yml` or any of the eight per-shot files. The brief
requires this, and it is also the only safe choice: without one of these files
no shot of that experiment can be built.

### 3. Silver

| table | one row is | rows |
| --- | --- | ---: |
| `experiment` | one experiment folder, read from its `properties.yml` | 5 |
| `shot` | one shot: its packed measurement, sweep and detector bits, detector event count, actual flip and four predictions | 250,000 |

Column names and types follow `silver-tables.md`. Decisions:

- The bits stay packed, as bytes, exactly as in the source. Silver should stay
  faithful to the source, and Gold and the ML table need the same packed bytes.
  Only `detector_event_count` is added: the number of set detector bits, read
  little-endian.
- Every bit width is read from the experiment's `properties.yml`, not fixed in
  the code, because the widths differ between distance 3 and 5 (for example 9
  and 25 sweep bits).
- One shot row is built from the eight files of one folder, matched by
  position. This is not a join across sources: all eight files belong to the
  same experiment.
- Identifiers depend only on the source data. An experiment's
  `source_record_id` is a hash of the archive and its `properties.yml` path. A
  shot's is a hash of the archive, the experiment and the shot index
  (`tracing.py`). A rerun on unchanged input gives the same ids.
- In `source_trace.parquet` an experiment has one row and a shot has eight, one
  per file its values came from, all with the same `source_record_id` and the
  locator `shot=<index>`. That gives 2,000,005 Google trace rows
  (5 + 250,000 × 8).

The checks, and what happens when one fails:

| check | if it fails |
| --- | --- |
| each `.b8` file holds as many whole records as `properties.yml` declares shots | the run stops |
| each `.01` file has one line per declared shot | the run stops |
| every `.01` line is `0` or `1` | the run stops |
| the unused padding bits of a shot's measurement and sweep records are zero | that shot is excluded and recorded in `data_issues.parquet` |

A wrong record count means the files no longer line up by position, and a
value other than 0 or 1 means a file is damaged. In both cases no shot of that
experiment can be trusted, so the run stops. Non-zero padding concerns one
record, so only that shot is excluded. Its row in `data_issues.parquet` keeps
the rule, severity, record bytes, action and reason, so it stays visible.

Tests: `tests/test_register_sources.py` (missing archive, size, checksum,
unsafe member), `tests/test_formats.py` (byte-aligned little-endian `b8`
records, a partial record rejected, `01` values validated),
`tests/test_prepare_data.py` (aligned files accepted, a short prediction file
stops the run, every Silver row traced) and `tests/test_tracing.py` (stable
ids, one trace row per file, one shared id, the locator).

### 4. Gold tables

| table | one row represents | primary key | rows |
| --- | --- | --- | ---: |
| `hardware_experiment` | one code configuration run at one chip position | `experiment_id` (folder name) | 5 |
| `shot` | one shot: its packed bits and actual flip | `(experiment_id, shot_index)` | 250,000 |
| `decoder` | one supplied decoder | `decoder_name` | 4 |
| `shot_prediction` | one decoder's prediction for one shot | `(experiment_id, shot_index, decoder_name)` | 1,000,000 |
| `detector_position_summary` | in how many shots of one experiment one detector fired | `(experiment_id, detector_index)` | 1,400 |

The brief asks Gold to support questions about experiments, shots, detector
summaries, decoders and predictions; for this source each of these is one
table. The view `decoder_outcome` puts each prediction next to the actual flip
and adds `decoder_error`. Foreign keys tie shots and summaries to their
experiment, and predictions to their shot and decoder.

The keys come from the data, not from a sequence, so a rerun gives the same
keys. Checks enforce the rules we rely on: `detector_event_count` must equal
`bit_count(detector_bits)`, so a wrong event count can never reach the ML
table; basis is X or Z; counts cannot be negative; `example_id` and
`source_record_id` are unique; and the same code at the same position (basis,
distance, rounds, row, column) cannot be loaded twice under another name.

All indexes come from these primary and unique keys. They cover the lookups we
do: a shot by `example_id` or `source_record_id` (tracing, ML export) and the
joins on the experiment and shot keys. We added no index on `decoder_name`,
because our queries always read the predictions of all four decoders (Q2 groups
by decoder, the ML export takes all four per shot), so no query looks one
decoder up.

### 5. Design decisions

1. Predictions are rows, not columns. Silver has four prediction columns per
   shot. Gold has one `shot_prediction` row per decoder and shot, plus a
   `decoder` table. The database has to answer questions about decoders,
   comparing them becomes one `GROUP BY` (Q2), and a fifth decoder would add
   rows, not columns. Cost: 241 MB instead of about 1 MB for four boolean
   columns, and the ML export has to turn the rows back into columns.

2. Detector bits stay packed, with a small summary. Measured on the full release
   with `sql/detector_storage_size.sql`:

   | option | rows | size |
   | --- | ---: | ---: |
   | packed `detector_bits` in `shot` (the ML export needs them anyway) | 250,000 | 8.8 MB |
   | chosen: `detector_position_summary` | 1,400 | 272 kB |
   | rejected: one row per fired detector | 10,446,925 | 1,613 MB |

   No required output needs single detector events: Q2 uses predictions and
   flips, and the ML export needs the packed bytes. The summary shows which
   detectors fire most for 272 kB instead of 1.6 GB, and `get_bit` still reads
   single bits in Stim's order when needed.

3. The ML `example_id` is stored. It is the SHA-256 of
   `google_qec:<experiment_id>:<shot_index>`, computed once in the load and
   unique in `shot`. The database thereby guarantees that every ML example
   points to exactly one shot, and the ML export reads the id instead of
   recomputing it.

4. The decoder error is a view. Prediction, actual flip and error stay
   separate, and the error cannot go out of date.

5. Hardware and simulated experiments get separate tables. A simulated
   experiment is described only by its fault rate, a hardware one by basis,
   distance, rounds, chip position and bit widths, so one shared table would be
   mostly empty. Both use `experiment_id`, without a foreign key between them
   (section 8).

### 6. Loading and checking

Gold is rebuilt from Silver in one transaction on every run
(`load_postgres.rebuild_gold`). PostgreSQL also rolls back the `DROP` and
`CREATE`, so a failed load leaves the previous Gold in place. We rebuild instead
of updating rows because a fresh load cannot leave duplicates or old rows
behind. The cost is that every run reloads all of Gold.

The Google load inserts the experiments and decoders and copies the Silver
shots into a temporary table with `COPY`, PostgreSQL's bulk-load command. SQL
then builds `shot` and `shot_prediction` from that table and computes
`detector_position_summary` from the bits now in `shot`, all inside the same
transaction. Before committing, `validate_google` checks the row counts against
Silver, the byte length of every measurement and detector record against the
declared bit counts, and an md5 of each experiment's measurement, sweep and
detector bytes against Silver. The md5 check is our proof that the ML export can
rebuild the packed bytes from Gold alone. Any failure rolls the whole load back.
Before writing the ML table, `build_ml_tables.py` checks the byte length,
padding and event count of all 250,000 rows again.

`tests/test_load_postgres.py` tests this on small data, including a changed
byte that the validation catches, little-endian detector positions and a
failed rebuild that keeps the old Gold. Running Part I twice with
`make rerun-check-part1` gave identical results in all 25 compared outputs
(`results/part1/rerun_check.json`).

### 7. Quality findings

- Every Google check passed. Each file had the declared 50,000 records, every
  `.01` value was 0 or 1 and every padding bit was zero, so no shot was
  excluded and `data_issues.parquet` has no Google rows.
- The counts reconcile (`results/part1/row_counts.json`). Silver accepted all
  250,000 declared shots and rejected none, and Gold holds 5 experiments,
  250,000 shots, 1,000,000 predictions (4 per shot) and 1,400 summary rows
  (4 × 200 + 600 detectors). The summary's fired counts add up to 10,446,925,
  the same total as the shots' event counts.
- The files are aligned in content, not only in length. In every experiment
  each decoder is wrong less often than always guessing the more common outcome
  (the worst decoder 0.43 to 0.45 against 0.49 to 0.50, section 10). If a
  prediction file were shifted even one line against the actual flips, its
  predictions would be unrelated to them and wrong about half the time.
- The archive's README names the ground-truth file `obs_flips_actual.b8`, but
  every folder contains `obs_flips_actual.01`, and the same README section says
  the data is stored in `01` format. We read the files that exist and parse
  them as `01`. The syndrome documentation has the same kind of slip (`label`
  against the CSV header `labels`), recorded in `data_issues.parquet` as
  `labels_column_naming` (`info`, `documented`). The Google one is only noted
  here.

### 8. A relationship we rejected

We considered linking the hardware experiments to the simulated syndrome
experiments with the same distance, since both sources contain distance-3
codes (the simulated files are named `d-3_...`). We rejected it. Distance is a
setting, not an identifier: no value marks the same experiment in both sources.
The data also differ in kind: a simulated row is an aggregate count over 4
rounds of 4 checks, a distance-3 hardware shot is one run over 25 rounds of 8
checks. A join on distance would pair every distance-3 syndrome row with every
distance-3 shot, which means nothing. The QASMBench section covers the other
rejected link.

### 9. One trace

`results/part1/trace_examples.json` follows one Google test shot. Its
`example_id` (`75f380e1bb46…`) is shot 1 of `surface_code_bX_d3_r25_center_3_5`
in `gold.shot`. That shot's `source_record_id` has the eight trace rows from
section 3, each pointing to `shot=1` in the archive.
`results/part2/trace_predictions.json` then lists its eight predictions: six
from Task B and two from Task C.

### 10. What the analysis shows

`sql/q2_decoder_error_rates.sql` joins `shot_prediction`, `shot` and
`hardware_experiment` and counts a logical error when a prediction differs from
the actual flip, over all 50,000 shots of each experiment:

| distance | position | tensor network | belief matching | correlated matching | PyMatching |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | (3, 5) | 0.4023 | 0.4025 | 0.4194 | 0.4342 |
| 3 | (5, 3) | 0.4130 | 0.4159 | 0.4252 | 0.4432 |
| 3 | (5, 7) | 0.3839 | 0.3920 | 0.4182 | 0.4306 |
| 3 | (7, 5) | 0.3887 | 0.3884 | 0.4130 | 0.4299 |
| 5 | (5, 5) | 0.3955 | 0.4015 | 0.4257 | 0.4510 |

At every position tensor network and belief matching are the two best decoders,
within 0.01 of each other, then correlated matching, then PyMatching. Across
the distance-3 positions the best rate ranges from 0.384 at (5, 7) to 0.413 at
(5, 3). At distance 5 the two best fall inside that range, while correlated
matching and PyMatching do slightly worse than at any distance-3 position.
Distance 5 only ran at one position, with no distance-3 run there, so distance
and position cannot be separated. All rates are below the 0.49 to 0.50 of
always guessing the more common outcome, but not far below: even the best
decoder is wrong on about 40 % of shots.

### 11. Limitations

- Distance 5 has one chip position only, so distance and position effects
  cannot be told apart.
- Predictions as rows cost about 240 MB more than four columns would.
- Only per-position totals of detector events are stored as a table; single
  events need `get_bit`.
- The release has only basis X and 25 rounds.


<!-- Written by Suganth; unchanged except heading levels. -->
## Part I design report: QASMBench

This section covers the QASMBench source from Bronze to Gold. It is based on
`register_sources.py`, `prepare_data.py`, `gold_qasmbench.py` and the files in
`results/part1/`.

### 1. What the source contains

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

### 2. Bronze: checking before reading

`register_sources.py` compares every Bronze object with the release manifest
before anything else runs. The size and SHA-256 must match, and it stops with an
error otherwise. For QASMBench the hash is
`60307f88e34b1f752b94223d6d136da72c629b4b625ac6d1c30e5e2e4f85722a`
(recorded in `results/part1/run.json`).

It also rejects archive members that are absolute paths, drive-letter paths or
contain `..` (`check_safe_archive_member`). `prepare_data.py` repeats the check
and also stops if a benchmark folder lacks its `.qasm` or `_transpiled.qasm`
file. We only read the archive in memory and never change or unpack it.

### 3. Silver

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

### 4. Quality findings

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

### 5. Gold

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

### 6. A relationship we rejected

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

### 7. What the analysis shows

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

### 8. Limitations

- The source and transpiled gate counts for `qec_sm_n5` cannot be compared (see
  section 3).
- The broadcast expansion and the data qubits of the source checks are written
  for `qec_sm_n5` only.
- Only one of the three circuits has checks and corrections, so the QASMBench
  tables are small and meant as reference material, not for statistics.


## ML tables: from Gold to Part II

The last step of Part I, `stages/build_ml_tables.py`, exports the two tables
Part II reads. Each comes from one committed query against Gold; Python only
adds the course `data_split`, checks the contract from `required-ml-tables.md`
and writes Parquet to MinIO `ml/`. It never reads Bronze or Silver, and nothing
is written unless every check on both tables passes.

| table | one row is | built from | rows |
| --- | --- | --- | ---: |
| `ml_syndrome_decoder_example` | one syndrome-and-label observation in one fault-rate experiment | `sql/ml_syndrome_decoder_example.sql`: view `syndrome_example` joined to `syndrome_observation`, `syndrome_experiment` and `syndrome_pattern` | 75,598 |
| `ml_google_decoder_example` | one hardware shot | `sql/ml_google_decoder_example.sql`: `shot` and `hardware_experiment`, with the four `shot_prediction` rows of a shot turned back into columns | 250,000 |

**Identifiers.** `example_id` is read from Gold, not computed in the export: the
syndrome view builds it from experiment, syndrome and label, and `shot` stores
it as a unique column. Every ML row therefore resolves to exactly one Gold
record (`syndrome_observation` or `shot`), and from there through
`source_record_id` to its Bronze members in `source_trace.parquet`.

**Splits.** `data_split` comes from the course helpers, unchanged:

| table | train | validation | test |
| --- | --- | --- | --- |
| syndromes, by fault rate | five rates: 53,304 rows, 50 million shots | 0.0005: 1,407 rows, 10 million shots | 0.005: 20,887 rows, 10 million shots |
| Google, by shot index | other even shots: 100,000 | even with `shot_index % 10 == 8`: 25,000 | odd shots: 125,000 |

A whole fault-rate experiment stays in one split, and both tables keep all
three splits non-empty.

**Contract checks before writing.** Syndromes: unique `example_id`; 16 binary
syndrome values accepted by `syndrome_model_input`; `round_count` and
`check_count` of 4; a binary label and a positive `sample_weight`; the split
equal to `syndrome_data_split`; and the `sample_weight` total of every
experiment equal to its `quantity` total in Gold. Google: one row per Gold shot
and unique keys; one detector width per distance; `detector_bits` of exactly
`ceil(detector_count / 8)` bytes with zero padding; `detector_event_count` equal
to the set bits (and to `unpack_little_endian_bits` on one shot per
experiment); a binary actual flip and four binary predictions accepted by
`google_meta_model_input`. A shot with a missing prediction is kept by the
query's outer join and then rejected by these checks, so it cannot disappear
silently.

**Repeatability.** Both tables are among the 25 outputs that
`make rerun-check-part1` compares, and were identical across runs. Part II
records the SHA-256 and row count of both files in `results/part2/run.json`, so
every model result names the exact tables it was trained on.

Tests: `tests/test_build_ml_tables.py` (both tables match their contract and
Gold, a missing prediction and non-zero padding are rejected, an empty split or
a fault rate spread over two splits is rejected, weights that differ from Gold
are rejected, syndrome `example_id`s survive a full Gold rebuild, every analysis
query becomes a CSV).
