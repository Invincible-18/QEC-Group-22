-- Question 1 (summary): How do weighted syndrome frequency and logical-error
-- labels change with physical fault rate?
--
-- One output row is one physical fault rate. Every total is weighted by
-- quantity, not by aggregate CSV row count. The all-zero pattern (no check
-- fired) is reported separately. Per-pattern detail is in
-- q1b_syndrome_pattern_frequency.sql. Joins three Gold tables.
SELECT
    experiment.physical_fault_rate,
    SUM(observation.quantity)::bigint AS weighted_observations,
    COUNT(DISTINCT observation.syndrome_pattern_id) AS distinct_patterns,
    COALESCE(SUM(observation.quantity) FILTER (
        WHERE observation.logical_error_label
    ), 0)::bigint AS weighted_logical_errors,
    COALESCE(SUM(observation.quantity) FILTER (
        WHERE observation.logical_error_label
    ), 0)::double precision / SUM(observation.quantity) AS weighted_logical_error_rate,
    COALESCE(SUM(observation.quantity) FILTER (
        WHERE bit_count(pattern.syndrome_bits) = 0
    ), 0)::double precision / SUM(observation.quantity) AS all_zero_frequency,
    COALESCE(SUM(observation.quantity) FILTER (
        WHERE bit_count(pattern.syndrome_bits) = 0 AND observation.logical_error_label
    ), 0)::double precision / NULLIF(SUM(observation.quantity) FILTER (
        WHERE bit_count(pattern.syndrome_bits) = 0
    ), 0) AS all_zero_logical_error_rate
FROM gold.syndrome_observation AS observation
JOIN gold.syndrome_experiment AS experiment
    USING (experiment_id)
JOIN gold.syndrome_pattern AS pattern
    USING (syndrome_pattern_id)
GROUP BY experiment.physical_fault_rate
ORDER BY experiment.physical_fault_rate;
