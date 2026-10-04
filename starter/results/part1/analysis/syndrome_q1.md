# Syndrome Gold analysis: Question 1

Question: How do weighted syndrome frequency and logical-error labels change with physical fault rate?

The reproducible PostgreSQL query is in `syndrome_q1.sql`. It joins `gold.syndrome_observation` to `gold.syndrome_experiment`, groups by fault rate and distinct syndrome pattern, and uses `quantity` as the observation weight. Each query result row represents one syndrome pattern at one fault rate.

Definitions:

- `pattern_quantity` is the sum of `quantity` for that pattern, including both labels.
- `weighted_syndrome_frequency` is `pattern_quantity / total quantity at that fault rate`.
- `logical_error_quantity` and `no_logical_error_quantity` are weighted counts, not counts of aggregate CSV rows.
- `weighted_logical_error_rate` is `logical_error_quantity / pattern_quantity` for that pattern and fault rate.
- `distinct_patterns` below counts patterns observed at least once for that fault rate.

## Overall results

| Physical fault rate | Weighted observations | Distinct patterns | Weighted logical errors | Weighted logical-error rate |
| ---: | ---: | ---: | ---: | ---: |
| 0.00001 | 10,000,000 | 68 | 2,340 | 0.0234% |
| 0.00005 | 10,000,000 | 202 | 11,542 | 0.1154% |
| 0.00010 | 10,000,000 | 445 | 22,910 | 0.2291% |
| 0.00050 | 10,000,000 | 1,228 | 114,403 | 1.1440% |
| 0.00100 | 10,000,000 | 2,387 | 227,018 | 2.2702% |
| 0.00500 | 10,000,000 | 14,643 | 1,039,500 | 10.3950% |
| 0.01000 | 10,000,000 | 31,466 | 1,865,274 | 18.6527% |

## Interpretation

Across these experiments, the weighted logical-error rate increases with physical fault rate, and more distinct syndrome patterns are observed. The all-zero pattern is the most frequent pattern at every tested rate, but its weighted frequency decreases from 99.8729% at fault rate 0.00001 to 28.3960% at 0.01. Its logical-error rate remains much lower than the overall rate, rising from 0% at the three lowest rates to 0.0356% at 0.01.

These are associations across the supplied simulated experiments. This analysis does not establish causation. The totals are weighted by `quantity`; using raw aggregate-row counts would give a different and inappropriate frequency estimate.
