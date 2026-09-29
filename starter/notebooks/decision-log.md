# Decision log — QEC data pipeline

Plain-language record of discovery findings and design decisions, in the
order they came up. Update this as work continues; it feeds the final
design report.


### qec_syndromes
- 7 CSV files + README.txt in `syndromes_dataset.zip` (358,017 bytes total)
- Columns (actual header): `labels` (int, 0/1), `syndromes` (string,
  nested-tuple literal), `quantity` (int, positive)
- Row counts vary by fault rate: 68 rows at pfr=0.00001, 49,676 rows at
  pfr=0.01 (more noise -> more distinct syndrome/label combinations)
- `quantity` sums to exactly 10,000,000 in both files checked
- Candidate IDs: `experiment_id` = one per CSV file (fault-rate
  experiment); business key = (experiment_id, syndrome_bits, label);
  syndrome alone is NOT a valid key (18,210 collisions at pfr=0.01)
- `source_record_id` candidate: hash of (bronze_object, archive_member,
  CSV row position)

### google_qec
- 5 experiment directories in `google-surface-code-curated.zip`
  (14,638,673 bytes total): 4x distance-3 (`center_3_5`, `center_5_3`,
  `center_5_7`, `center_7_5`), 1x distance-5 (`center_5_5`)
- Each dir: properties.yml, measurements.b8, sweep.b8,
  detection_events.b8, obs_flips_actual.01, 4x
  obs_flips_predicted_by_*.01, circuit_ideal.stim, circuit_noisy.stim,
  circuit_detector_error_model.dem, layout.svg, 2x extra pij_*.dem
  (undocumented, not required for Silver)
- All basis=X, rounds=25, shots=50,000 per experiment -> 250,000 shots
  total
- Bit widths (declared vs stored-with-padding): measurements 209->216
  bits (7 padding), sweep 9->16 bits (7 padding, d3) / d5 sweep=32 bits
  declared, detectors 200 bits/d3 and 600 bits/d5 (0 padding, byte-aligned
  already)
- All 5 experiments x all companion files verified byte/line-count
  aligned to declared shots
- Candidate IDs: `experiment_id` = directory name (or a cleaner
  parsed form: basis/distance/rounds/center_row/center_col);
  `shot_index` = zero-based line/record position within an experiment's
  aligned files
- `source_record_id` candidate: one shot's aligned bytes span multiple
  Bronze members (measurements.b8 + sweep.b8 + detection_events.b8 +
  5x .01 files), so `source_trace.parquet` needs multiple rows per
  `source_record_id` for a single shot, per silver-tables.md's own note
  ## google_qec (additional)

- **Finding:** padding-zero check passes for all sampled shots (measurement
  and sweep records both have genuinely zero unused high bits, not just
  assumed).
- **Finding:** over 2,000 shots of experiment `center_3_5`, actual-flip
  class balance is ~49/51 (near coin-flip); majority baseline gets 51.1%
  agreement; all four decoders beat that baseline but only reach 55-60%.
  Decoders agree with each other more (60-74%, varying by pair) than with
  `actual` -- differentiated pairwise agreement rules out a shared
  indexing bug and instead shows the four decoders make genuinely
  different mistakes on this hard, near-threshold dataset. Useful
  evidence for Part II Task B (combined-decoder motivation).
## google_qec: detector storage design decision

- **Question:** should Gold store one row per shot (packed bytes + event
  count summary) or one row per individual fired detector event?
- **Evidence:** average detector_event_count per shot, measured over a
  2,000-shot real sample of distance-3 data, is 31.08 (out of 200
  detector bits). Extrapolating to the full release (250,000 shots across
  5 experiments, distance 3 and 5 combined):
  - shot-summary design: ~32.5 MB estimated total size
  - per-fired-event design: ~10.9 million rows, ~653 MB estimated total
    size (~20x larger)
- **Decision:** use the shot-summary design (one Gold row per shot, with
  packed measurement/sweep/detector bytes plus detector_event_count) for
  google_shot. The ml_google_decoder_example export needs exactly this
  shape (packed detector_bits + detector_event_count), so a per-event
  table would add ~20x storage cost for data the required ML export
  never queries at that granularity. A per-event table remains available
  as an optional addition later if a specific analysis needs it, but is
  not required for the minimum Gold model.
- **Caveat:** the distance-5 event-rate figure is extrapolated (scaled by
  detector-width ratio) from the distance-3 sample, not independently
  measured -- worth a quick real check against the one distance-5
  experiment before finalizing, though it's unlikely to change the
  conclusion given the 20x margin.


### qasmbench
- 3 benchmark dirs in `qasmbench-qec.zip` (144,172 bytes total):
  `qec_sm_n5`, `qec_en_n5`, `error_correctiond3_n5`; each with README,
  2 PNGs, source .qasm, transpiled .qasm; plus top-level LICENSE, NOTICE,
  README, qelib1.inc
- Only `qec_sm_n5` has explicit ancilla register + custom check gate +
  if-conditioned corrections (2 stabilizer checks, 3 corrections). The
  other two have no ancilla register and no conditional statements at
  all -> 0 stabilizer_check/conditional_correction rows for those, by
  design, not by omission
- `qec_sm_n5`: qubit_count=5 (3 data + 2 ancilla), 2 checks, 3
  corrections; transpiled variant preserves the same check/correction
  content with inlined gates and split/reordered measurement statements
- Candidate IDs: `circuit_id` = (benchmark_name, variant) pair, e.g.
  `qec_sm_n5/source` vs `qec_sm_n5/transpiled`
- `source_record_id` candidate: hash of (bronze_object, archive_member,
  QASM statement position) for circuit-level rows; per-check and
  per-correction rows need their own locator (e.g. statement index of the
  defining gate call / if-statement)



## Rejected relationships

- **Candidate:** join QASMBench circuits (qec_sm_n5, qec_en_n5,
  error_correctiond3_n5) to the Google hardware experiments and/or the
  simulated syndrome experiments, on the theory that a circuit file
  documents the same code the shot data was produced from.
- **Evidence against:**
  - No shared identifier exists in any source. QASMBench files carry only
    a benchmark name and source/transpiled variant; Google experiments are
    keyed by basis/distance/rounds/center-coordinates
    (`properties.yml`); syndrome files are keyed by physical fault rate
    encoded in the filename. None of these vocabularies overlap.
  - Structural mismatch: `qec_sm_n5` is a 3-data-qubit / 2-ancilla
    repetition code (2 stabilizer checks). The Google/syndrome data is a
    distance-3 or distance-5 **surface** code -- 9 data qubits and 8
    stabilizer checks per the primer's figure, for distance 3. Qubit and
    check counts don't match, so even a "same code, different
    representation" join isn't defensible.
  - `qec_en_n5` and `error_correctiond3_n5` have no ancilla register, no
    syndrome-conditioned correction, and no per-shot data at all -- they
    are single static circuits, not repeatable experiments with shots, so
    there is no shot-level key on the QASMBench side to join through even
    in principle.
- **Decision:** QASMBench circuits are stored in Gold as independent
  circuit/stabilizer_check/conditional_correction entities, joined to
  nothing else. They document QEC circuit semantics (how ancillas extract
  syndromes, how corrections are applied) as reference material, not as
  joinable experiment records. No cross-source foreign key is created.


## Gold row-meaning sentences (step 6, draft)

- `experiment`: one row is one distinct QEC run configuration that
  produced shots or aggregate observations -- either one simulated
  physical-fault-rate file or one Google hardware experiment directory
  (a source-discriminator column distinguishes the two).
- `syndrome_observation`: one row is one distinct (experiment, 16-bit
  syndrome, logical-error label) combination from the simulated data,
  carrying its physical shot-count weight.
- `google_shot`: one row is one single hardware shot within one Google
  experiment -- its packed measurement/sweep/detector bits, actual flip,
  and four decoder predictions.
- `decoder`: one row is one named decoding algorithm (belief_matching,
  correlated_matching, pymatching, tensor_network_contraction).
- `decoder_prediction`: one row is one decoder's prediction for one
  Google shot (alternative to wide columns on google_shot -- open
  design choice, see below).
- `circuit`: one row is one parsed QASMBench circuit variant
  (benchmark x source/transpiled).
- `stabilizer_check`: one row is one parity check identified in one
  circuit, linking an ancilla qubit to its data qubits and syndrome bit.
- `conditional_correction`: one row is one syndrome-conditioned recovery
  operation identified in one circuit.

Open design question: keep the 4 decoder predictions as wide boolean
columns on google_shot (matches the ml_google_decoder_example contract
shape directly) vs. a normalized decoder/decoder_prediction pair (more
"relational," easier to add a 5th decoder later, but requires a join to
rebuild the ML table). Leaning wide, since predictions are fixed at
exactly 4 and the ML export needs them as columns anyway.



## Infrastructure decision: results/ storage backend

- **Ambiguity:** no assignment doc explicitly states whether `results/`
  (data_issues.parquet, run.json, etc.) should be written to MinIO or the
  local filesystem. brief.md only says it's kept "outside the four data
  areas."
- **Evidence used to decide:** config.py's `lake_backend` setting (minio
  vs local) governs only Bronze/Silver/Gold/ML; no equivalent backend
  setting or MinIO wiring exists anywhere for `results/`. compose.yaml
  defines only two volumes for the workspace container -- the code bind
  mount (`./starter:/workspace`) and the read-only course-data mount --
  no results-specific storage mechanism at all.
- **Decision:** results/ files are written to local disk (under
  /workspace/results/, i.e. starter/results/ on the host via the bind
  mount), using plain pyarrow.parquet.write_table to a Path, not the
  MinIO client. Verified: a real write from the notebook produced a real
  file on the host filesystem outside the container.
- **io_utils.py additions:** write_local_parquet/read_local_parquet
  (generic local Parquet I/O) and write_data_issues (now writes locally,
  matching this decision).
  - **Correction/confirmation:** distance-5 sweep-bit width is 25 (not 9 --
  the initial parse_shot hardcoded 9 from the distance-3 case, a real bug
  caught and fixed before scaling). Measured average
  detector_event_count for distance-5, from a real 2,000-shot sample, is
  95.12 -- closely matching the earlier extrapolated estimate of 93.24
  used in the storage-design decision above. The shot-summary vs.
  per-event storage conclusion (~20x difference) is confirmed by real
  data, not just extrapolation.

## Shared infra: required-companion-file and archive-safety checks

- Added `check_safe_archive_member` (rejects absolute paths, drive-letter
  paths, `..` traversal) and `check_required_members` (checks a
  source-specific required-filename list against a directory prefix) to
  io_utils.py, since both are genuinely source-agnostic -- verified
  against all 103 real archive members across all three sources (0 false
  positives) and against deliberately unsafe/incomplete examples (all
  correctly caught).
- Confirmed no missing companion files across all 5 Google experiments
  and all 3 QASMBench benchmarks in the real release.

