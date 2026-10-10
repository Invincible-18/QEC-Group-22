## Task A: weighted syndrome decoder

### Input and target

- **Rows:** every row of `ml_syndrome_decoder_example`. One row is one distinct
  syndrome-and-label pattern inside one simulated physical-fault-rate
  experiment, so the same syndrome can appear twice in an experiment, once per
  label. The rows are aggregates and are never expanded into single shots.
- **Weight:** `sample_weight` is the number of simulated shots that produced the
  row. It is used in fitting and in every metric, so each row counts as often
  as it was observed.
- **Input:** the 16 syndrome values from `syndrome_model_input`, in the order
  stored by Part I: four rounds of four checks, round by round. `syndrome_00`
  to `syndrome_03` are round 1, checks 1 to 4, and `syndrome_12` to
  `syndrome_15` are round 4.
- **Target:** `logical_error_label`, whether the shots of that row ended in a
  logical error.
- **Split:** the supplied split by fault rate. Five fault rates are train,
  0.0005 is validation and 0.005 is test, so every fault-rate experiment stays
  whole in one split.

### Model choices

- **Baseline:** `task_a_weighted_majority` predicts the more common label in the
  weighted training rows, with the weighted training share of logical errors
  as its probability. It shows what can be reached without looking at the
  syndrome.
- **Model:** `task_a_logistic_regression` is a scikit-learn logistic
  regression with the fixed seed, fitted with `sample_weight`. It is the
  weighted linear classifier the assignment asks for: it accepts weights
  directly, gives a probability for the Brier score, and its 16 weights can be
  read as how much each check in each round points to a logical error. Apart
  from a higher `max_iter`, so the solver is not stopped early, the
  scikit-learn defaults are kept.
- **Threshold:** chosen with the team rule `choose_threshold` on the validation
  rows only, weighted by `sample_weight`. The value is recorded in `run.json`.
  Test rows never influence the fit or the threshold.
- **Times:** `train_seconds` covers only `model.fit(...)`, and
  `predict_seconds` only the prediction on the test rows. Building the inputs
  and the validation prediction used for the threshold are not timed. The
  baseline has no fitted model, so it has no times.

### Results

{{task_a_results}}

The logistic regression is not better than the majority baseline on the main
metric: both almost always predict "no logical error", so their logical-error
rates are nearly equal and balanced accuracy stays close to one half. Its Brier
score is lower than the baseline's, so its probabilities do carry information
about which syndromes lead to a logical error; it is the yes/no decision that
adds nothing.

The cause is the split by fault rate. Validation is a single low fault rate,
where logical errors are rare, so on validation the lowest logical-error rate
comes from flagging almost nothing, and the rule picks a high threshold. Test is
a fault rate ten times higher, where logical errors are much more common, but
with that threshold the model still flags almost no rows. A threshold tuned on
one noise level does not carry over to another. We report this as observed and
did not tune on test.

In the fitted model, checks 1 and 3 get clearly larger weights than checks 2
and 4 in every round, so the model has learned that some checks are much
stronger evidence of a logical error than others.

### Discarded information

- **The fault rate:** it is not an input, although it changes how likely a
  logical error is for the same syndrome. The model cannot tell a validation
  row from a test row with the same syndrome, which is why one threshold fits
  one split and not the other.
- **Interactions between values:** a linear model adds up one weight per value.
  It cannot express that a check firing in two consecutive rounds means
  something different from the same check firing once, or that two checks
  firing together point to a particular error.
- **Shot-level detail:** the aggregate rows keep only how often a pattern
  occurred, not the order of the shots.

### Limitations

- The same syndrome can occur with both labels, especially at high fault rates,
  so no decoder that sees only the syndrome can be right on every shot.
- Validation is one fault rate, so the threshold is tuned to that noise level
  only. Using a probability directly, or a threshold per fault rate, would need
  the fault rate as an input, which the assignment does not include.
- One model with default settings and one seed; only the threshold was chosen.
- All experiments are one code size (distance 3) with four rounds; other sizes
  or round counts may give different results.
