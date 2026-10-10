# Gold: Google QEC part

This covers the Google hardware data in Gold: what each table holds, the design
decisions and what they cost, how the load works, and analysis Q2. The tables,
the load and the checks are all in `stages/load_postgres.py`.

## Tables

| Table | One row represents | Primary key | Rows |
| --- | --- | --- | ---: |
| `hardware_experiment` | one hardware experiment: one code configuration run at one processor location | `experiment_id` | 5 |
| `shot` | one shot, i.e. one execution of one experiment | `(experiment_id, shot_index)` | 250,000 |
| `decoder` | one of the four supplied decoders | `decoder_name` | 4 |
| `shot_prediction` | one decoder's prediction for one shot | `(experiment_id, shot_index, decoder_name)` | 1,000,000 |
| `detector_position_summary` | how often one detector position fired over all shots of one experiment | `(experiment_id, detector_index)` | 1,400 |

View `decoder_outcome`: one row per prediction, next to the actual flip, with
`decoder_error = predicted_flip <> actual_observable_flip`.

Foreign keys: `shot → hardware_experiment`, `shot_prediction → shot` and
`→ decoder`, `detector_position_summary → hardware_experiment`.

## Keys, constraints and indexes

- **Keys come from the data**, never from a sequence (`experiment_id` is the
  directory name, a shot is `(experiment_id, shot_index)`). A rerun therefore
  produces the same keys.
- **Value checks** on every column where a rule exists: basis is X or Z, counts
  and indexes are not negative. The most useful one is
  `detector_event_count = bit_count(detector_bits)`: PostgreSQL rejects any shot
  whose event count disagrees with its own packed bits. Because `bit_count` also
  counts padding bits, this rejects non-zero padding too.
- **Uniqueness**: `example_id` and `source_record_id` are unique per shot, and
  an experiment's `(basis, distance, rounds, center_row, center_col)` is unique.
- **Indexes**: every index comes from a primary key or unique constraint, and
  each one serves a lookup we need: resolving an `example_id` or a
  `source_record_id` to its shot (tracing), and the joins from predictions and
  summaries to shots and experiments (the foreign keys are prefixes of the
  primary keys). We did not index `shot_prediction.decoder_name`: it has 4
  values over 1,000,000 rows, so PostgreSQL would scan the table anyway.

## Design decisions and what they cost

**1. Predictions are rows, not columns.** Silver has the four predictions as
columns of the shot row. In Gold each prediction is its own row, so the grain
changes from one row per shot to one row per shot and decoder, and the decoder
is a real entity with a foreign key. Analysis Q2 becomes a single `GROUP BY
decoder_name`. *Cost:* `shot_prediction` takes 241 MB, compared to about 1 MB
for four boolean columns, and the ML export has to pivot the rows back into four
columns.

**2. Detector bits stay packed, plus a per-position summary.** We measured both
options on the full release (`sql/detector_storage_size.sql`):

| Design | Rows | Size |
| --- | ---: | ---: |
| packed `detector_bits` per shot (needed by the ML export in any design) | 250,000 | 8.8 MB |
| chosen extra: `detector_position_summary` | 1,400 | 264 kB |
| rejected extra: one row per fired detector | 10,446,925 | 1,613 MB |

None of the required analyses needs individual detector events: Q2 uses
predictions and actual flips, and the ML export needs the packed bytes. The
summary answers "which detector positions fire most" for 264 kB instead of
1.6 GB, so we did not build an event table.

**3. The ML `example_id` is stored in Gold.** It is `sha256("google_qec:" ||
experiment_id || ":" || shot_index)`, computed once during the load and kept in
`shot` with a `UNIQUE` constraint. That makes "every ML example resolves to
exactly one Gold shot" something the database enforces, and the ML export reads
the id instead of recomputing it.

**4. Decoder error is a view, never a stored column.** The prediction, the
actual outcome and the mistake stay three separate things, and the error can
never go stale.

**5. Separate table for hardware experiments.** Simulated and hardware
experiments share only a few columns (id, distance, rounds, number of samples).
The fault rate exists only for simulated runs, and the basis, location and bit
widths only for hardware runs, so one combined table would leave about half of
its columns empty on every row. We keep shared names where the idea is the same (`experiment_id`,
`distance`, `source_record_id`). *Open:* the team draft proposed one combined
`experiment` table; this still has to be agreed with the syndrome Gold part.

## Load: all-or-nothing, and reruns

`load_postgres.rebuild_gold()` runs in a single transaction:

1. drop the `gold` schema and create the tables again;
2. copy the Silver shots into a temporary staging table, then build `shot`,
   `shot_prediction` and `detector_position_summary` from it with SQL;
3. validate: row counts against Silver, byte lengths against `detector_count`
   and `measurement_count`, and an md5 of every packed column per experiment
   against Silver, so the packed bytes in Gold are proven identical to Silver
   on every load.

DDL is transactional in PostgreSQL, so if any step fails, even the drop is
undone and the previous Gold stays. Nothing is ever appended, so a rerun cannot
create duplicates. We checked a full rerun: same ids and counts.

Tests (`tests/test_load_postgres.py`) cover this on a small dataset, including a
load that fails halfway and leaves the earlier Gold untouched, a changed byte
that the validation catches, and little-endian bit order in the summary.

## Relationships we did not create

- **Hardware experiments ↔ simulated experiments by `distance`.** Both have
  distance-3 data, but distance is a shared parameter, not an identifier.
  Joining on it would pair every syndrome row with every distance-3 shot
  without any real connection.
- **Hardware experiments ↔ QASMBench circuits.** No shared identifier exists,
  and the circuits are a different code (5 qubits and 2 checks against 17
  qubits and 8 checks for a distance-3 surface code). The full reasoning is in
  the team decision log, under "Rejected relationships".

## Tracing through Gold

A Google ML example traces back as
`example_id → gold.shot → source_record_id → source_trace.parquet → Bronze`.
`results/part1/trace_examples.json` shows this for shot 0 of
`surface_code_bX_d3_r25_center_3_5`, down to the eight archive members. The
prediction step is added once Part II exists.

## Analysis Q2

*How do the supplied decoder logical-error rates compare by code distance and
distance-three processor location?* Query: `sql/q2_decoder_error_rates.sql`
(joins `shot_prediction`, `shot` and `hardware_experiment`). Output:
`results/part1/analysis/q2_decoder_error_rates.csv`.

| distance | location | tensor network | belief matching | correlated matching | PyMatching |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | (3, 5) | 0.4023 | 0.4025 | 0.4194 | 0.4342 |
| 3 | (5, 3) | 0.4130 | 0.4159 | 0.4252 | 0.4432 |
| 3 | (5, 7) | 0.3839 | 0.3920 | 0.4182 | 0.4306 |
| 3 | (7, 5) | 0.3887 | 0.3884 | 0.4130 | 0.4299 |
| 5 | (5, 5) | 0.3955 | 0.4015 | 0.4257 | 0.4510 |

At every location the two best decoders are tensor-network contraction and
belief matching (within 0.01 of each other), then correlated matching, and
PyMatching makes the most errors. Between distance-3 locations the best rate
varies by about three points, from 0.384 at (5, 7) to 0.413 at (5, 3). At
distance 5 the two best decoders fall inside the distance-3 range, while
correlated matching and PyMatching do slightly worse than at any distance-3
location. Distance 5 was only run at one location, which has no distance-3 run,
so this data cannot separate an effect of distance from an effect of location.
All rates are fairly close to what you would get by always guessing the more
common outcome (0.49 to 0.50 in every experiment): over 25 rounds much of the
protected information is lost.
