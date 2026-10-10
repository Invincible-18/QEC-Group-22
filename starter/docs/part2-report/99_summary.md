## Summary

{{all_results}}

Only Task B's learned model clearly improves on its majority baseline, and it
does so because its inputs are already the outputs of strong decoders: the
combined model beats three of the four supplied decoders but not tensor
network, because the decoders tend to fail on the same shots. In Task A the
weighted linear model's probabilities are better than the baseline's, but its
yes/no decisions are not, because the threshold is chosen at a low fault rate
and the fault rate is not an input. In Task C the small MLP on raw detector bits
predicts the hardware shots little better than chance, and its probabilities
are less reliable than the baseline's, which matches what the flat 200-bit
vector leaves out. Together, the results show that the useful information sits
in structure the simple inputs do not carry: the noise level for the simulated
syndromes, and the position and timing of detector events for the hardware
shots. Even the best supplied decoder is wrong on a large share of the hardware
shots, so these 25-round experiments are hard to decode with any of the methods
used here.
