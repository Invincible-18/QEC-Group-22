# Google Gold analysis: Question 2

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
