-- Question 1: How do weighted syndrome frequency and logical-error labels
-- change with physical fault rate?
--
-- One output row is one syndrome pattern observed at one physical fault rate.
-- Frequencies and label rates are weighted by Silver quantity, not CSV row count.

WITH pattern_label_totals AS (
    SELECT
        experiment.physical_fault_rate,
        observation.syndrome_pattern_id,
        observation.logical_error_label,
        SUM(observation.quantity)::bigint AS weighted_quantity
    FROM gold.syndrome_observation AS observation
    JOIN gold.syndrome_experiment AS experiment
        USING (experiment_id)
    GROUP BY
        experiment.physical_fault_rate,
        observation.syndrome_pattern_id,
        observation.logical_error_label
),
pattern_totals AS (
    SELECT
        physical_fault_rate,
        syndrome_pattern_id,
        SUM(weighted_quantity)::bigint AS pattern_quantity,
        SUM(weighted_quantity) FILTER (
            WHERE logical_error_label
        )::bigint AS logical_error_quantity,
        SUM(weighted_quantity) FILTER (
            WHERE NOT logical_error_label
        )::bigint AS no_logical_error_quantity
    FROM pattern_label_totals
    GROUP BY physical_fault_rate, syndrome_pattern_id
),
experiment_totals AS (
    SELECT
        physical_fault_rate,
        SUM(pattern_quantity)::bigint AS total_quantity
    FROM pattern_totals
    GROUP BY physical_fault_rate
)
SELECT
    pattern_totals.physical_fault_rate,
    pattern_totals.syndrome_pattern_id,
    encode(pattern.syndrome_bits, 'hex') AS syndrome_bits_hex,
    pattern_totals.pattern_quantity,
    experiment_totals.total_quantity,
    pattern_totals.pattern_quantity::double precision
        / experiment_totals.total_quantity AS weighted_syndrome_frequency,
    COALESCE(pattern_totals.no_logical_error_quantity, 0)::bigint
        AS no_logical_error_quantity,
    COALESCE(pattern_totals.logical_error_quantity, 0)::bigint
        AS logical_error_quantity,
    COALESCE(pattern_totals.logical_error_quantity, 0)::double precision
        / pattern_totals.pattern_quantity AS weighted_logical_error_rate
FROM pattern_totals
JOIN experiment_totals
    USING (physical_fault_rate)
JOIN gold.syndrome_pattern AS pattern
    USING (syndrome_pattern_id)
ORDER BY
    pattern_totals.physical_fault_rate,
    weighted_syndrome_frequency DESC,
    pattern_totals.syndrome_pattern_id;
