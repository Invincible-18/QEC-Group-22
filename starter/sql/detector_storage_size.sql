-- Measures the detector storage decision. Not part of the pipeline
--
-- Gold keeps the packed detector bits per shot (the ML export needs them in any
-- design) plus a small per-position summary. The alternative we rejected is one
-- row per fired detector. This builds that alternative in a temp table, prints
-- the sizes and throws it away.
BEGIN;

CREATE TEMP TABLE fired_detector ON COMMIT DROP AS
SELECT s.experiment_id, s.shot_index, position.detector_index
FROM gold.shot AS s
JOIN gold.hardware_experiment AS e USING (experiment_id)
CROSS JOIN LATERAL generate_series(0, e.detector_count - 1) AS position(detector_index)
WHERE get_bit(s.detector_bits, position.detector_index) = 1;

ALTER TABLE fired_detector ADD PRIMARY KEY (experiment_id, shot_index, detector_index);

SELECT 'packed detector_bits (kept in both designs)' AS component,
       count(*) AS row_count,
       pg_size_pretty(sum(pg_column_size(detector_bits))::bigint) AS size
FROM gold.shot
UNION ALL
SELECT 'chosen extra: detector_position_summary',
       (SELECT count(*) FROM gold.detector_position_summary),
       pg_size_pretty(pg_total_relation_size('gold.detector_position_summary'))
UNION ALL
SELECT 'rejected extra: one row per fired detector',
       (SELECT count(*) FROM fired_detector),
       pg_size_pretty(pg_total_relation_size('fired_detector'));

COMMIT;
