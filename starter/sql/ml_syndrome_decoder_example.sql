-- ml_syndrome_decoder_example: one row represents one distinct syndrome-and-label
-- observation within one physical-fault-rate experiment (required-ml-tables.md).
--
-- Runs with search_path set to the Gold schema. example_id comes from the Gold
-- view syndrome_example. data_split is added afterwards in Python with the course
-- helper syndrome_data_split. round_count and check_count are 4 by the Silver
-- contract: the syndrome load rejects any other shape, and syndrome_pattern only
-- accepts exactly 16 values, so Gold does not store them per row.
SELECT example.example_id,
       example.experiment_id,
       experiment.physical_fault_rate,
       pattern.syndrome_bits,
       4 AS round_count,
       4 AS check_count,
       example.logical_error_label,
       observation.quantity AS sample_weight
FROM syndrome_example AS example
JOIN syndrome_observation AS observation USING (source_record_id)
JOIN syndrome_experiment AS experiment
  ON experiment.experiment_id = example.experiment_id
JOIN syndrome_pattern AS pattern
  ON pattern.syndrome_pattern_id = example.syndrome_pattern_id
ORDER BY experiment.physical_fault_rate, example.example_id;
