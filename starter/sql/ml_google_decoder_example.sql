-- ml_google_decoder_example: one row represents one hardware shot in one Google
-- surface-code experiment (required-ml-tables.md).
--
-- Runs with search_path set to the Gold schema. data_split is added afterwards
-- in Python with the course helper google_data_split. The four shot_prediction
-- rows of a shot are turned back into columns; the LEFT JOIN keeps a shot with a
-- missing prediction as NULL so the export checks reject it instead of dropping
-- the shot silently.
SELECT s.example_id,
       s.experiment_id,
       s.shot_index,
       e.distance,
       e.rounds,
       e.center_row,
       e.center_col,
       e.detector_count,
       s.detector_event_count,
       s.detector_bits,
       bool_or(p.predicted_flip) FILTER (WHERE p.decoder_name = 'belief_matching')
           AS belief_matching_prediction,
       bool_or(p.predicted_flip) FILTER (WHERE p.decoder_name = 'correlated_matching')
           AS correlated_matching_prediction,
       bool_or(p.predicted_flip) FILTER (WHERE p.decoder_name = 'pymatching')
           AS pymatching_prediction,
       bool_or(p.predicted_flip) FILTER (WHERE p.decoder_name = 'tensor_network_contraction')
           AS tensor_network_contraction_prediction,
       s.actual_observable_flip
FROM shot AS s
JOIN hardware_experiment AS e
  ON e.experiment_id = s.experiment_id
LEFT JOIN shot_prediction AS p
  ON p.experiment_id = s.experiment_id
 AND p.shot_index = s.shot_index
GROUP BY s.experiment_id, s.shot_index,
         e.distance, e.rounds, e.center_row, e.center_col, e.detector_count
ORDER BY s.experiment_id, s.shot_index;
