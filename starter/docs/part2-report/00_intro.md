# Part II report: QEC decoders on the Part I ML tables

<!--
How this report is built: `make train-part2` joins every file in
docs/part2-report/ in name order and writes results/part2/report.md. Each
{{placeholder}} is replaced by a table built from the same run's metrics, so
every number matches results/part2/metrics.json. Write explanations here;
never type result numbers by hand. Comments like this one are removed.
-->

## Inputs

Part II reads only the two Parquet tables Part I exports from Gold:

- `ml/ml_syndrome_decoder_example.parquet` (Task A): one row per distinct
  syndrome-and-label observation in one physical-fault-rate experiment, with
  the 16 ordered syndrome values, the logical-error label, and
  `sample_weight`, the number of simulated shots the row represents.
- `ml/ml_google_decoder_example.parquet` (Tasks B and C): one row per hardware
  shot, with the packed detector bits, the detector-event count, the four
  supplied decoder predictions, and the actual logical flip.

Both tables are checked against their column contracts before training, and
their SHA-256 hashes are recorded in `results/part2/run.json`. Gold, Silver and
the source archives are never read.

## Splits

We use the course splits exactly as Part I stored them in `data_split`:

- Syndrome: fault rate 0.0005 is validation, 0.005 is test, the other five
  fault rates are train. A whole fault-rate experiment stays in one split.
- Google: odd `shot_index` is test, even with `shot_index % 10 == 8` is
  validation, the other even shots are train (40% / 10% / 50%).
- Task C uses only distance 3 with `shot_index < 12_500`, then the same split.

Every model is fitted on train, its threshold is chosen on validation, and all
metrics below are computed once on test.

## Metrics and rules

For every model we report, on the test split: the logical-error rate (the main
metric), balanced accuracy, the Brier score when the model gives a probability
("n/a" otherwise), training time and prediction time. Task A's metrics are
weighted by `sample_weight`; Tasks B and C weight every shot equally.

- **Thresholds:** every model that outputs a probability uses the threshold
  between 0.05 and 0.95 (step 0.01) with the lowest logical-error rate on
  validation (weighted for Task A). Because this rule optimizes the main metric,
  it can lower balanced accuracy where logical errors are rare.
- **Times:** training time covers only fitting the model; prediction time covers
  only predicting the test split. Building the inputs is not timed. Baselines
  and supplied decoders have no times ("n/a").

All randomness uses one fixed seed, recorded in `run.json` with the dependency
versions and the code revision.
