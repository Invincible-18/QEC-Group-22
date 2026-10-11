# Part I design report: Google QEC

This section covers the Google hardware data from Bronze to Gold: the source,
the checks before and while reading it, the Silver and Gold tables and why they
look like this, one trace and analysis Q2. It is based on
`stages/register_sources.py`, the Google part of `stages/prepare_data.py`,
`tracing.py`, `stages/gold_google.py`, the SQL in `sql/` and the files in
`results/`.

## 1. What the source contains

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

## 2. Bronze: checking before reading

`register_sources.py` checks the archive against the release manifest before
anything else runs. A missing archive, a different size, a different SHA-256 or
an unsafe member name (an absolute path, a drive letter or `..`) stops the run.
The Google hash is recorded in `results/part1/run.json`. The archive is only
read in memory, never unpacked or changed.

`prepare_data.py` repeats the member-name check and also stops if an experiment
folder lacks `properties.yml` or any of the eight per-shot files. The brief
requires this, and it is also the only safe choice: without one of these files
no shot of that experiment can be built.

## 3. Silver

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

## 4. Gold tables

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

## 5. Design decisions

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

## 6. Loading and checking

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

## 7. Quality findings

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

## 8. A relationship we rejected

We considered linking the hardware experiments to the simulated syndrome
experiments with the same distance, since both sources contain distance-3
codes (the simulated files are named `d-3_...`). We rejected it. Distance is a
setting, not an identifier: no value marks the same experiment in both sources.
The data also differ in kind: a simulated row is an aggregate count over 4
rounds of 4 checks, a distance-3 hardware shot is one run over 25 rounds of 8
checks. A join on distance would pair every distance-3 syndrome row with every
distance-3 shot, which means nothing. The QASMBench section covers the other
rejected link.

## 9. One trace

`results/part1/trace_examples.json` follows one Google test shot. Its
`example_id` (`75f380e1bb46…`) is shot 1 of `surface_code_bX_d3_r25_center_3_5`
in `gold.shot`. That shot's `source_record_id` has the eight trace rows from
section 3, each pointing to `shot=1` in the archive.
`results/part2/trace_predictions.json` then lists its eight predictions: six
from Task B and two from Task C.

## 10. What the analysis shows

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

## 11. Limitations

- Distance 5 has one chip position only, so distance and position effects
  cannot be told apart.
- Predictions as rows cost about 240 MB more than four columns would.
- Only per-position totals of detector events are stored as a table; single
  events need `get_bit`.
- The release has only basis X and 25 rounds.
