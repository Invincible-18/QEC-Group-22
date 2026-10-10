## Task B: supplied and combined Google decoders

### Input and target

We use every shot in `ml_google_decoder_example`. Distance 3 has four
experiments with 200 detectors per shot, and distance 5 has one experiment
with 600. They are different code sizes: distance 5 uses more qubits and has
24 checks per round instead of 8. Every model is therefore built and scored for
each distance separately, and the two are never mixed.

The target is `actual_observable_flip`, which says whether the protected
logical value actually flipped during the shot. A decoder is wrong on a shot
when its prediction differs from this value; a flip on its own is not an error.

Each row holds one shot's actual flip and the four supplied predictions for
that same shot, joined on `experiment_id` and `shot_index` when the ML table
is exported from Gold. Every prediction keeps the shot's `example_id`, so it
traces back through Gold to the source files.

The combined decoder only uses the five values returned by
`google_meta_model_input()`, in this order: detector-event density
(`detector_event_count / detector_count`), then the belief matching,
correlated matching, PyMatching and tensor network predictions. These are the
inputs the assignment prescribes, and the order is recorded in `run.json`.

### Model choices

There are six models per distance:

- `majority` always predicts the more common training label, with the training
  share of flips as its probability. It is the reference, showing what can be
  reached without looking at a shot.
- The four supplied decoders are scored exactly as Google gave them, and the
  combined model is compared against them. Nothing is trained and they give no
  probability, so their Brier score is not applicable.
- `combined` is a logistic regression with scikit-learn defaults and the fixed
  seed, fitted on the training shots. It predicts a flip when its probability
  reaches a threshold, which is chosen on the validation shots with the shared
  rule from the introduction; the value for each distance is in `run.json`. The
  test shots are only used for the final metrics.

The assignment asks for a simple linear combined decoder. We used logistic
regression because four of the five inputs are yes/no votes, so it works as a
weighted vote of the decoders plus a term for the share of detectors that
fired. It learns how much to trust each decoder, its weights can be read
directly, and it gives a probability for the Brier score. With six parameters
and tens of thousands of training shots it has little room to overfit, so we
kept the defaults. Among the four votes, tensor network gets the largest weight
in the saved models and PyMatching the smallest, the same order as their
accuracy.

### Results

{{task_b_results}}

All models of a distance are scored on the same test shots. Every supplied
decoder beats the baseline, and their ranking from best to worst is the same at
both distances: tensor network, belief matching, correlated matching,
PyMatching. Balanced accuracy gives the same ranking, because flips are close
to half of all shots.

The combined model comes second at both distances. It beats belief matching,
correlated matching and PyMatching, but is slightly behind tensor network, by a
gap small enough to be due to chance with this many test shots. Distance 5 has
a quarter of the test shots, so there the gaps between tensor network, belief
matching and the combined model are all small enough to be due to chance, and
their order is not certain. The combined model's Brier score is only a little
below the majority baseline's Brier score, because its probabilities never
move far from 0.5. It trains and predicts in well under a second.

Even the best decoder is wrong on a large share of shots. Each shot covers 25
rounds of checks, so errors have many chances to add up.

### Decoder-error overlap

{{task_b_overlap}}

The decoders make partly different mistakes, since they disagree on more than
half of the shots. But they also fail together: all four are wrong on the same
shot far more often than if their mistakes were unrelated (compare the two
"all four wrong" rows). On those shots every vote the combined model receives
is wrong, so it gets all of them wrong too. Where all four are right, it is
always right.

Where the decoders disagree, even tensor network is wrong almost half the time,
and the other three do no better. Their votes add little there, so the
combined model ends with about the same error rate as tensor network on those
shots. The mistakes are not complementary enough for combining to help,
which is why the combined model does not improve on tensor network.

### Discarded information

The five inputs reduce each shot's 200 or 600 detector bits to one number, the
share that fired. The model does not see which detectors fired, where on the
chip or in which of the 25 rounds, although this pattern is what the decoders
themselves work from. This pattern is also the information that could help
where the votes cannot. On a shot where all four decoders are wrong, the votes
look exactly like those on a shot where all four are right, and the share of
fired detectors is not enough for our model to tell them apart (it gets all
those shots wrong). The full pattern might show which of these shots are hard
ones. Whether a model using it could beat tensor network is open; such a model
would have to be much larger than this one.

We also left out chip location. The four distance-3 experiments share one
model, although `center_row` and `center_col` are in the table. Part I's Q2
analysis (`results/part1/analysis/q2_decoder_error_rates.md`) shows that
decoder error rates differ between these locations but their ranking hardly
changes, so separate weights per location would probably change little.

### Limitations

- One model with default settings and one seed; apart from the threshold,
  nothing was tuned.
- In a linear model, event density can only shift every shot's score. It cannot
  make the model trust the decoders less on noisy shots.
- The supplied decoders only give yes/no answers, so the combined model cannot
  tell a confident decoder from a borderline one, which is what it would need
  on the shots where they disagree.
- Distance 5 is a single experiment at one chip location with fewer validation
  shots, so its threshold is less certain.
- In practice the combined decoder is slower than any single decoder, because
  all four must run first. Their running times are not in the data, so the
  speeds cannot be compared.
- The results only cover these 25-round experiments in the X basis; other
  bases or numbers of rounds may give different results.
