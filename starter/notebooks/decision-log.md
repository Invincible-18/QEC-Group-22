# Decision log — QEC data pipeline

Plain-language record of what we found in the data and what we decided, with
the evidence. It feeds the Part I design report (`docs/part1-design-report.md`) and the Part II
report (`docs/part2-report/`).

## Design summary

- **Layers:** Bronze keeps the three course archives unchanged in MinIO; Silver
  is cleaned Parquet in MinIO (the six required tables); Gold is relational, in
  PostgreSQL; the two ML tables are exported from Gold; run records, issues and
  analyses go to local `results/`.
- **One command per part:** `make run-part1` and `make train-part2`; both record
  the code revision, data release, counts and timings in `run.json`.
- **Nothing is trusted before it is checked:** every Bronze archive is checked
  against the release manifest (size and SHA-256), and an unsafe archive member
  or a missing companion file stops the run for every source.
- **Stable identifiers:** `source_record_id`, `example_id` and `issue_id` are
  hashes of where a record comes from and what it is, never of row order in
  Silver or of the run, so a rerun produces the same identifiers.
- **Keep, do not drop:** an invalid row is never silently removed; it goes to
  `data_issues.parquet` with its value, rule, action and reason.
- **Aggregates stay aggregates:** syndrome rows keep `quantity` as a weight and
  are never expanded into single shots.
- **All-or-nothing Gold:** Gold is rebuilt in one transaction; if any load or
  check fails, the previous Gold stays as it was.
- **No invented links:** the three sources share vocabulary (experiment,
  syndrome, decoder, check) but no cross-source keys; the QASMBench-to-shot-data
  relationship was examined and rejected.
- **Storage from measurement:** Google shots are stored packed, one row per
  shot, because a row per fired detector measured about 180 times larger.
- **Repeatable by proof:** `make rerun-check-part1` runs Part I twice and
  compares 25 outputs; a clean-environment run reproduced all results.
- **Part II as a consumer:** a shared framework reads only the two ML tables,
  hands each task its supplied split, and alone computes metrics and writes
  files, so all models are evaluated the same way.


## 1. Discovery

Before writing any pipeline code we explored each archive in notebooks,
without changing or extracting it: archive members and sizes, actual headers
and types, row and shot counts, candidate keys and their collisions, declared
and stored bit widths and padding, and file alignment. The findings below come
from that work. Every check was then rebuilt as tested code in
`prepare_data.py`, so the pipeline repeats it on every run; the notebooks were
exploration only and are not part of the submission.

### qec_syndromes
- 7 CSV files + README.txt in `syndromes_dataset.zip` (358,017 bytes total).
- Columns (actual header): `labels` (int, 0/1), `syndromes` (string,
  nested-tuple literal), `quantity` (int, positive). The documentation calls
  the label column `label`; the difference is recorded as a documented issue.
- Each CSV row is one aggregate observation: one syndrome and label, with
  `quantity` the number of simulated shots that produced it. Every syndrome is
  four rounds of four binary checks.
- Row counts grow with the fault rate: 68 rows at pfr=0.00001, 49,676 rows at
  pfr=0.01 (more noise gives more distinct syndrome/label combinations).
- `quantity` sums to exactly 10,000,000 per file; this is checked for every file.
- Keys: one experiment per CSV file (fault-rate experiment); business key =
  (experiment, syndrome, label). A syndrome alone is NOT a valid key (18,210
  collisions at pfr=0.01).

### google_qec
- 5 experiment directories in `google-surface-code-curated.zip`
  (14,638,673 bytes total): 4x distance 3 (`center_3_5`, `center_5_3`,
  `center_5_7`, `center_7_5`), 1x distance 5 (`center_5_5`).
- Each directory: properties.yml, measurements.b8, sweep.b8,
  detection_events.b8, obs_flips_actual.01, 4x obs_flips_predicted_by_*.01,
  circuit_ideal.stim, circuit_noisy.stim, circuit_detector_error_model.dem,
  layout.svg, and 2 extra pij_*.dem files (undocumented, not used).
- All basis X, 25 rounds, 50,000 shots per experiment -> 250,000 shots.
- Bit widths (declared vs stored with padding), distance 3: measurements
  209 -> 216 bits, sweep 9 -> 16 bits; detectors 200 bits (distance 3) and 600
  bits (distance 5), byte-aligned. Distance-5 sweep is 25 bits (stored in 32);
  the first parser hard-coded the distance-3 width of 9, a bug caught and fixed
  before running on all data.
- Padding bits are really zero (checked, not assumed), and all companion files
  of all 5 experiments align to the declared shot count.
- One shot spans 8 Bronze members (3 .b8 files and 5 .01 files), so
  `source_trace.parquet` has several rows per shot.
- Early finding: on 2,000 shots of `center_3_5` the actual flip is close to a
  coin flip (about 49/51); all four decoders beat the majority baseline but only
  reach 55-60%, and they agree with each other (60-74% by pair) more than with
  the actual flip. They make different mistakes, which motivates Part II Task B.

### qasmbench
- 3 benchmark directories in `qasmbench-qec.zip` (144,172 bytes total):
  `qec_sm_n5`, `qec_en_n5`, `error_correctiond3_n5`, each with a README, 2 PNGs,
  a source .qasm and a transpiled .qasm; plus LICENSE, NOTICE, README and
  qelib1.inc at the top.
- Only `qec_sm_n5` has an ancilla register, a custom check gate and `if`
  corrections (5 qubits: 3 data + 2 ancilla; 2 checks). The other two have no
  ancilla register and no conditional statements, so they have no
  stabilizer-check or correction rows, by design, not by omission.
- The transpiled variant keeps the same checks and corrections, with gates
  inlined and measurements split and reordered.
- One circuit writes a statement with a space before the semicolon
  (`cx q[1],q[2] ;` in `error_correctiond3_n5.qasm`); the first parser missed
  that CNOT (48 instead of 49 two-qubit gates). All statement patterns now allow
  the space, with a test.


## 2. Identifiers

- **`source_record_id`** is a SHA-256 hash of the record's place in Bronze:
  - syndrome row: Bronze object, archive member and CSV row number;
  - Google experiment: Bronze object and its `properties.yml`; Google shot:
    Bronze object, experiment and shot index;
  - QASMBench rows: Bronze object, archive member and statement position.
- **`example_id`** (ML tables) is a hash of the business key, built in Gold
  (the `syndrome_example` view; a column of `shot`): experiment, syndrome and
  label for syndromes; source, experiment and shot for Google. It resolves to
  exactly one Gold record.
- **`issue_id`** is a hash of the rule, source system, record locator, record id
  and observed value. It first included the run id and so changed on every run,
  and collided across locations; both were fixed.
- **Cost:** hashes are not human-readable, so `source_trace.parquet` and the
  trace examples carry the readable locators next to them.


## 3. Bronze and run safety

- Bronze objects are checked against the release manifest (`objects[].sha256`
  and size) at the start of every run; a mismatch stops the run.
- `check_safe_archive_member` rejects absolute paths, drive-letter paths and
  `..`; `check_required_members` checks each source's required files. Both are
  source-agnostic (in `io_utils.py`) and were verified against all 103 real
  archive members (no false positives) and against deliberately bad examples.
  All three sources stop the run on an unsafe member (the syndrome reader first
  skipped such a member silently; changed so all sources behave the same).
- No missing companion files in the real release (5 Google experiments,
  3 QASMBench benchmarks).
- The course dataset bundle is not tracked in git: it is supplied separately,
  and `make bootstrap` seeds it into Bronze.


## 4. Silver

- The three sources stay in separate Silver tables; no cross-source join is
  justified at this layer.
- Syndrome `quantity` is kept as a weight and never expanded into repeated rows
  (70 million represented shots stay 75,598 rows).
- Syndromes are stored as 16 explicit 0/1 bytes, round by round.
- Google bits stay packed (Stim b8, little-endian inside each byte) with their
  declared widths; padding is checked and discarded on unpacking.


## 5. Gold

Final tables (schema `gold`), one sentence each:

| table | one row is |
| --- | --- |
| `syndrome_experiment` | one simulated experiment at one physical fault rate |
| `syndrome_pattern` | one distinct 16-value syndrome (31,941 patterns) |
| `syndrome_observation` | one aggregate CSV observation: experiment, pattern, label and `quantity` |
| `hardware_experiment` | one Google experiment directory and its settings |
| `shot` | one hardware shot: packed measurement, sweep and detector bits, event count, actual flip |
| `decoder` | one supplied decoding algorithm |
| `shot_prediction` | one decoder's prediction for one shot |
| `detector_position_summary` | one detector position in one experiment, with how often it fired |
| `circuit` | one QASMBench circuit variant (benchmark x source/transpiled) |
| `stabilizer_check` | one parity check in one circuit: ancilla, data qubits, syndrome bit |
| `conditional_correction` | one syndrome-conditioned recovery operation in one circuit |

Views: `syndrome_example` and `decoder_outcome` (used by the ML export and Q2).

- **Separate experiment tables:** simulated and hardware experiments have
  different attributes (a fault rate against basis, distance, rounds and chip
  position), so they are two tables rather than one table with a source column
  and mostly empty fields. (The early draft had one shared `experiment` table.)
- **Syndrome patterns as their own table:** the same pattern occurs in several
  experiments; storing it once lets Q1 compare patterns across fault rates with
  a join instead of comparing byte strings.
- **Decoder predictions as rows:** first leaning towards four wide columns on
  `shot` (the shape of the ML table). Chose the normalized pair `decoder` and
  `shot_prediction`: with wide columns the Google tables would be close to a copy
  of Silver, and as rows Q2 is a single GROUP BY with the decoder as an entity
  with a foreign key. Cost: 241 MB instead of about 1 MB, and a pivot in the ML
  export.
- **Detector storage, one row per shot:** per-event storage was estimated from a
  2,000-shot sample (31.08 events per distance-3 shot) and then measured on the
  full release with `sql/detector_storage_size.sql`: 10,446,925 fired detectors
  (95.6 per distance-5 shot), 1,613 MB with its primary key, against 8.8 MB for
  the packed `detector_bits`. Kept one row per shot; the ML table needs exactly
  the packed bits and the event count. The 1,400-row `detector_position_summary`
  (264 kB) makes per-detector questions answerable without an event table.
- **Keys and checks:** every table has a primary key; foreign keys link
  observations to experiments and patterns, shots to experiments, predictions to
  shots and decoders, and checks and corrections to circuits; CHECK constraints
  cover value ranges (fault rate, positive quantity, 16-byte syndromes).
- **All-or-nothing load:** `rebuild_gold` drops and recreates the schema, loads
  and validates every source in one transaction; any failure rolls everything
  back. Validation compares Gold row counts with Silver for every table.


## Rejected relationships

- **Candidate:** join QASMBench circuits (qec_sm_n5, qec_en_n5,
  error_correctiond3_n5) to the Google hardware experiments and/or the
  simulated syndrome experiments, on the theory that a circuit file documents
  the same code the shot data was produced from.
- **Evidence against:**
  - No shared identifier exists in any source. QASMBench files carry only a
    benchmark name and source/transpiled variant; Google experiments are keyed
    by basis/distance/rounds/center coordinates (`properties.yml`); syndrome
    files are keyed by physical fault rate in the filename. None of these
    vocabularies overlap.
  - Structural mismatch: `qec_sm_n5` is a 3-data-qubit / 2-ancilla repetition
    code (2 stabilizer checks). The Google/syndrome data is a distance-3 or
    distance-5 **surface** code -- 9 data qubits and 8 stabilizer checks for
    distance 3. Qubit and check counts don't match, so even a "same code,
    different representation" join isn't defensible.
  - `qec_en_n5` and `error_correctiond3_n5` have no ancilla register, no
    syndrome-conditioned correction and no per-shot data -- single static
    circuits, not repeatable experiments, so there is no shot-level key to join
    through even in principle.
- **Decision:** QASMBench circuits are stored in Gold as independent
  circuit/stabilizer_check/conditional_correction entities, joined to nothing
  else. They document QEC circuit semantics (how ancillas extract syndromes,
  how corrections are applied) as reference material. No cross-source foreign
  key is created.


## 6. ML export and tracing

- Both ML tables are built only from Gold, with the committed SQL in `sql/`
  (`ml_*.sql`) and the Gold views; nothing re-reads Bronze or Silver. Their
  schemas are checked against the contracts before writing.
- The supplied splits are stored in `data_split` exactly as the course helpers
  compute them.
- The two examples in `trace_examples.json` are test rows, so each trace can
  continue to a real Part II prediction (`results/part2/trace_predictions.json`).
  The first version used the first Silver rows, which were training rows and
  are never predicted.


## 7. results/ storage

- **Ambiguity:** no assignment doc states whether `results/` (data_issues.parquet,
  run.json, etc.) should be written to MinIO or the local filesystem; brief.md
  only says it is kept "outside the four data areas."
- **Evidence:** config.py's `lake_backend` setting governs only Bronze, Silver,
  Gold and ML; nothing wires `results/` to MinIO, and compose.yaml mounts only
  the code (`./starter:/workspace`) and the read-only course data.
- **Decision:** results files are written to local disk under
  `/workspace/results/` (`starter/results/` on the host), with plain pyarrow
  (`write_local_parquet`, `read_local_parquet`, `write_data_issues` in
  `io_utils.py`). Generated results are not committed; the run recreates them.


## 8. Part II

- **One framework, three tasks:** `train.py` loads and checks the two ML tables
  (schema and SHA-256), splits them with the course helper, and hands each task
  its rows already split. Tasks only return predictions; the framework checks
  that every model predicts exactly the validation and test rows with the right
  labels and weights, computes all metrics on test with one implementation, and
  writes every file.
- **Threshold rule:** every model with a probability uses the grid value
  0.05-0.95 (step 0.01) with the lowest validation logical-error rate (weighted
  for Task A); ties keep the lowest value. Chosen on validation only.
- **Timing rule:** `train_seconds` covers only `fit`, `predict_seconds` only the
  test prediction; building inputs is not timed. Baselines and supplied decoders
  have no times. The framework rejects results that break this.
- **Report from templates:** `report.md` is generated from
  `docs/part2-report/*.md`, with every table filled from the same run's metrics,
  so the numbers always match `metrics.json`.
- **Task A:** weighted logistic regression is no better than the weighted
  majority on test; the threshold is tuned on the low validation fault rate
  (0.0005) and the fault rate is not an input. Reported as observed.
- **Task C:** the MLP stops at `max_iter=50` with a ConvergenceWarning on
  purpose: with 100-400 iterations training accuracy rose but test error did
  not improve, so longer training only overfits.
- **Dependencies:** scikit-learn, numpy, scipy and joblib are pinned in
  `pyproject.toml`; the versions used are also recorded in `run.json`.
