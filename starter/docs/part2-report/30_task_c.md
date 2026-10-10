## Task C: bounded raw-detector prototype

### Input and target

- **Rows:** distance-three Google shots with `shot_index < 12_500` from
  `ml_google_decoder_example`, a fixed subset across the four distance-three
  experiments. The supplied `data_split` is used unchanged.
- **Input:** the 200 detector bits of each shot, unpacked from the Stim `b8`
  row with `unpack_little_endian_bits` (little-endian inside each byte). The
  feature order is `detector_0` to `detector_199`. Before training, the code
  checks that every row has 200 detectors and that `detector_event_count`
  equals the number of set bits.
- **Target:** `actual_observable_flip`.

### Model choices

- **Baseline:** `task_c_d3_majority` predicts the more common training label,
  with the training share of flips as its probability.
- **Model:** `task_c_d3_mlp` is one scikit-learn `MLPClassifier` with a single
  hidden layer of 32 units, `max_iter=50` and the fixed seed. It takes a
  fixed-width binary vector and needs no hand-built features, so it fits this
  table. The subset is bounded, so a small network keeps the run short. This is
  a pipeline-consumer check, so no other architectures or hyper-parameters
  were tried.
- **Threshold:** chosen with the team rule `choose_threshold`, called with
  validation rows only. It picks the grid value with the lowest validation
  logical-error rate. Test rows never influence the fit or the threshold.
- **Times:** `train_seconds` covers only `model.fit(...)`. `predict_seconds`
  covers only the prediction on the test split. The validation prediction
  (used for the threshold) is not timed. Unpacking the bits and the input
  checks are not timed. The majority baseline has no fitted model, so it has
  no recorded times.

### Results

{{task_c_results}}

The table compares the MLP with the majority baseline on the identical test
shots. Read the logical-error rate first, then balanced accuracy, which shows
whether the MLP detects flips at all or only follows the common label. If the
MLP is not better than the baseline, that is the result and it is reported as
observed.

### Discarded information

A flat vector of 200 bits gives the MLP 200 unrelated inputs. It does not show
explicitly:

- **Detector position:** where each detector sits on the surface-code patch, so
  the model cannot tell which inputs are neighbours.
- **Neighbourhood:** which detectors are adjacent, so a chain of errors across
  neighbouring detectors is not visible as one pattern.
- **Change over QEC rounds:** that the bits come from 25 rounds of the same
  checks, so a detector firing in consecutive rounds looks the same as two
  unrelated detectors.

The network has to learn this structure from the data alone. With a small
network and a bounded subset it may learn little beyond overall event density,
which limits what the prototype can show.

### Limitations

- Only distance three and `shot_index < 12_500` are used, so the result does not
  transfer to distance five or to later shots.
- One seed and one small architecture were run, so run-to-run variation is
  unknown.
- The threshold minimizes the logical-error rate on validation. When flips are
  rare, this can give the same predictions as the majority baseline.
- `predict_seconds` is the time to predict the test split with `predict_proba`
  only. It does not include unpacking the detector bits.
