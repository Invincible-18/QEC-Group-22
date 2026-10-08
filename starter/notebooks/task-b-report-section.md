# Task B: supplied and combined Google decoders

## Data

We use all 250,000 shots of `ml_google_decoder_example` and handle distance 3
(four experiments) and distance 5 (one experiment) separately. The target is
`actual_observable_flip`. A decoder is wrong on a shot when its prediction
differs from this value. A shot's four predictions and actual flip are joined
on `experiment_id` and `shot_index` in `sql/ml_google_decoder_example.sql`, and
every prediction keeps the shot's `example_id` from `gold.shot`.

We keep the supplied split: within each experiment, odd `shot_index` values
are test, `shot_index % 10 == 8` is validation and the other even values are
train.

| | train | validation | test |
| --- | ---: | ---: | ---: |
| distance 3 | 80,000 | 20,000 | 100,000 |
| distance 5 | 20,000 | 5,000 | 25,000 |

## Models

For each distance:

- `majority` always predicts the more common training label. Only 49.2% (d3)
  and 49.9% (d5) of the training shots flip, so it predicts no flip, with that
  share as its probability. It is the reference that uses nothing about the
  shot itself.
- `belief_matching`, `correlated_matching`, `pymatching` and
  `tensor_network_contraction` are Google's predictions, scored as given. They
  have no probability, so their Brier score is not applicable.
- `combined` is a logistic regression (scikit-learn defaults, seed 2026) on the
  five inputs from `google_meta_model_input()`, in this order: event density
  (`detector_event_count / detector_count`) and the four predictions. It is
  fitted on train. The threshold is the value from 0.05 to 0.95 (step 0.01)
  with the best balanced accuracy on validation, as in Tasks A and C. This gave
  0.51 for d3 and 0.55 for d5.

A linear model suits this table because four of the five inputs are yes/no
votes. The model is then a weighted vote plus a term for how many detectors
fired, and its weights are easy to read. The learned weights (d3 / d5) are:
tensor network 0.53 / 0.59, belief matching 0.38 / 0.38, correlated matching
0.29 / 0.26, PyMatching 0.17 / 0.22 and event density 0.38 / 1.10. So the
model trusts the decoders in the order of their accuracy.

## Results

Test split only, from `metrics.json`: 100,000 shots for d3 and 25,000 for d5,
the same shots for every model of a distance. Times are in seconds. Training
time covers `fit()` and prediction time covers the test shots, without
building the inputs. They change slightly from run to run. The baseline is a
constant and Google computed the supplied predictions beforehand, so neither
has times.

| model | distance | logical-error rate | balanced accuracy | Brier | train (s) | predict (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 3 | 0.4929 | 0.5000 | 0.2500 | n/a | n/a |
| belief matching | 3 | 0.4013 | 0.5985 | n/a | n/a | n/a |
| correlated matching | 3 | 0.4220 | 0.5779 | n/a | n/a | n/a |
| PyMatching | 3 | 0.4364 | 0.5635 | n/a | n/a | n/a |
| tensor network | 3 | 0.3967 | 0.6031 | n/a | n/a | n/a |
| combined | 3 | 0.3977 | 0.6019 | 0.2352 | 0.067 | 0.031 |
| majority | 5 | 0.5015 | 0.5000 | 0.2500 | n/a | n/a |
| belief matching | 5 | 0.3994 | 0.6006 | n/a | n/a | n/a |
| correlated matching | 5 | 0.4226 | 0.5774 | n/a | n/a | n/a |
| PyMatching | 5 | 0.4548 | 0.5452 | n/a | n/a | n/a |
| tensor network | 5 | 0.3952 | 0.6048 | n/a | n/a | n/a |
| combined | 5 | 0.3977 | 0.6026 | 0.2341 | 0.022 | 0.001 |

All four decoders beat the baseline, in the same order at both distances. Even
the best one, tensor network, is wrong on about 40% of shots, as each shot runs
25 rounds. The combined model does not improve on it: it is 0.001 (d3) and
0.0025 (d5) behind, a gap small enough to be chance.

## Do the decoders make the same mistakes?

From `task_b_decoder_error_overlap` in `metrics.json`, on the test shots:

| | d3 | d5 |
| --- | ---: | ---: |
| all four right | 30.2% | 24.4% |
| all four wrong | 14.4% | 10.9% |
| all four wrong if their mistakes were unrelated | 2.9% | 3.0% |
| the decoders disagree | 55.4% | 64.8% |
| error rate where they disagree: tensor network | 45.6% | 44.3% |
| error rate where they disagree: combined | 45.7% | 44.6% |
| error rate where they disagree: PyMatching | 52.7% | 53.5% |

Many mistakes are shared. All four decoders are wrong on the same shot about
five times (d3) and 3.6 times (d5) more often than if their mistakes were
unrelated. On those shots every input of the combined model is wrong, and it
gets all of them wrong too (14,440 shots for d3, 2,713 for d5). Where the
decoders disagree, even tensor network is wrong almost half the time, and the
other decoders are not right more often, so their votes add little. The
combined model mostly follows tensor network there and ends with the same
error rate. This is why combining the four decoders gives no gain.

## Discarded information

The combined model only sees the share of fired detectors and four yes/no
answers. It does not see which detectors fired, where on the chip or in which
of the 25 rounds, which is what the decoders themselves used. It gets no
confidence from the decoders either. The four distance-3 experiments sit at
different chip locations but share one model.

## Limitations

- One model with default settings and one seed; nothing was tuned.
- In a linear model, event density can only shift every shot's score. It
  cannot make the model trust the decoders less on noisy shots.
- Distance 5 is one experiment with 5,000 validation shots, so its threshold is
  less certain.
- Choosing the threshold by the lowest validation error gives the same
  thresholds, since flips are close to half of all shots.
- In practice the combined decoder is slower than any single decoder, because
  all four must run first. Their running times are not in the data.
- The results only cover these 25-round X-basis experiments.

`make train` regenerates every number here. Code:
`stages/train_google.py`; tests: `tests/test_train_google.py`.
