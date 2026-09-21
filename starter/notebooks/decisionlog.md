## QEC syndrome discovery decisions

- The Bronze source is `syndromes_dataset.zip`; it remains unchanged and is inspected directly from the archive.
- Each source CSV row represents one aggregate syndrome observation, not one individual shot.
- Each syndrome is expected to contain four rounds with four binary checks per round.
- `quantity` is retained as an observation weight and is not expanded into repeated rows.
- `experiment_id` will be derived from the stable source filename, including its physical fault rate.
- `source_record_id` will include the source filename and CSV row position because a syndrome value alone is not unique.
- Silver will keep `qec_syndromes` separate from Google QEC and QASMBench; no cross-source join is justified at this layer.
