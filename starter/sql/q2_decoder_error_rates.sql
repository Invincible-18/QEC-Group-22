-- Q2: how do the supplied decoder logical-error rates compare by code distance
-- and distance-three processor location?
--
-- A decoder makes a logical error on a shot when its prediction differs from
-- the actual flip. Joins three Gold tables. This only describes associations:
-- distance 5 exists at one location only, so distance and location can't be
-- separated in this data.
SELECT e.distance,
       e.center_row,
       e.center_col,
       p.decoder_name,
       count(*) AS shots,
       count(*) FILTER (WHERE p.predicted_flip <> s.actual_observable_flip) AS logical_errors,
       round(avg((p.predicted_flip <> s.actual_observable_flip)::int), 4) AS logical_error_rate
FROM gold.shot_prediction AS p
JOIN gold.shot AS s USING (experiment_id, shot_index)
JOIN gold.hardware_experiment AS e USING (experiment_id)
GROUP BY e.distance, e.center_row, e.center_col, p.decoder_name
ORDER BY e.distance, e.center_row, e.center_col, logical_error_rate;
