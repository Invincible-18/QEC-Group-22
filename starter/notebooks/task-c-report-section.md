# Task C: bounded raw-detector prototype (report section)

## Input and target

- **Rows:** distance-three Google shots with `shot_index < 12_500`, from
  `ml_google_decoder_example`. This is a fixed subset across the four
  distance-three experiments.
- **Split:** the supplied `data_split` (train / validation / test), unchanged.
- **Input:** the 200 detector bits of each shot, unpacked from the Stim `b8`
  row (little-endian within each byte) in the order `detector_0 … detector_199`.
- **Target:** `actual_observable_flip`.

## Models

- `task_c_d3_majority`: predicts the more common training label, with the
  training share of flips as its probability.
- `task_c_d3_mlp`: one scikit-learn `MLPClassifier` with a single hidden layer
  of 32 units, `max_iter=50` and the fixed seed. The decision threshold is the
  value on a 0.05–0.95 grid that gives the best balanced accuracy on the
  **validation** split. Test shots are used once, for the final metrics.

A small MLP fits this table because it takes a fixed-width binary vector and
needs no hand-built features. This is a pipeline-consumer check, not an
architecture comparison, so no other architecture or hyper-parameter search
was run.

## What the flat 200-bit vector does not show

The MLP receives 200 unrelated numbers. The vector does not state:

- **Detector position:** which detector sits where on the surface-code patch.
  The model cannot tell that two inputs belong to neighbouring checks.
- **Neighbourhood:** which detectors are physically adjacent, so an error chain
  spanning several neighbouring detectors is not visible as one object.
- **Change over QEC rounds:** that the 200 bits come from 25 rounds of the same
  checks, so a detector that fires in consecutive rounds (a measurement error)
  looks the same as two unrelated detectors.

The model must learn all of this from data alone. With a small network and a
bounded training subset it probably learns little more than overall event
density. Fully connected inputs also scale poorly to the 600-bit distance-five
vectors, which Task C does not use.

## Limitations

- Only distance three and `shot_index < 12_500` are used, so conclusions do not
  transfer to distance five or to later shots.
- One seed and one architecture were run, so run-to-run variation is unknown.
- The threshold rule (balanced accuracy on validation) is one choice. The main
  metric, the logical-error rate, can be worse than the majority baseline when
  flips are rare. Report that result as observed and do not hide it.
- Training and prediction times cover the model fit and `predict_proba` only.
  Bit unpacking is not included.

## Results

Copy the observed values for `task_c_d3_majority` and `task_c_d3_mlp` from
`results/part2/metrics.json` (logical-error rate, balanced accuracy, Brier
score, training time, prediction time) after running the training command.

