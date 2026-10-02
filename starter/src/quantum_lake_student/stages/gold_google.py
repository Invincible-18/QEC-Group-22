"""Google QEC part of Gold: tables, load from Silver, and checks.

Called by load_postgres.rebuild_gold() inside its single transaction, with
search_path set to the Gold schema.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import psycopg
import pyarrow as pa
from psycopg import sql


GOOGLE_SILVER_TABLES = {
    "google_experiment": "silver/google_qec/experiment.parquet",
    "google_shot": "silver/google_qec/shot.parquet",
}


# ============================================================================
# google_qec: tables
# ============================================================================
# Run with search_path set to the gold schema, so the names are unqualified.

GOOGLE_TABLES = """
-- One row represents one hardware experiment: one code configuration run at
-- one processor location.
CREATE TABLE hardware_experiment (
    experiment_id     text    PRIMARY KEY,
    source_record_id  text    NOT NULL UNIQUE,
    basis             text    NOT NULL CHECK (basis IN ('X', 'Z')),
    distance          integer NOT NULL CHECK (distance > 0),
    rounds            integer NOT NULL CHECK (rounds > 0),
    shots             bigint  NOT NULL CHECK (shots > 0),
    center_row        integer NOT NULL,
    center_col        integer NOT NULL,
    measurement_count integer NOT NULL CHECK (measurement_count > 0),
    detector_count    integer NOT NULL CHECK (detector_count > 0),
    -- the same code at the same location should only appear once
    UNIQUE (basis, distance, rounds, center_row, center_col)
);

-- One row represents one shot, i.e. one execution of one experiment.
-- The bits are kept packed as in the source (Stim b8, little-endian), so the
-- ML export can rebuild them from Gold.
CREATE TABLE shot (
    experiment_id          text    NOT NULL REFERENCES hardware_experiment,
    shot_index             bigint  NOT NULL CHECK (shot_index >= 0),
    example_id             text    NOT NULL UNIQUE,
    measurement_bits       bytea   NOT NULL,
    sweep_bits             bytea   NOT NULL,
    detector_bits          bytea   NOT NULL,
    -- has to match the packed bits; bit_count also counts padding, so this
    -- catches non-zero padding as well
    detector_event_count   integer NOT NULL
        CHECK (detector_event_count = bit_count(detector_bits)),
    actual_observable_flip boolean NOT NULL,
    source_record_id       text    NOT NULL UNIQUE,
    PRIMARY KEY (experiment_id, shot_index)
);

-- One row represents one of the four supplied decoders.
CREATE TABLE decoder (
    decoder_name    text PRIMARY KEY,
    prediction_file text NOT NULL UNIQUE,
    description     text NOT NULL
);

-- One row represents one decoder's prediction for one shot.
CREATE TABLE shot_prediction (
    experiment_id  text    NOT NULL,
    shot_index     bigint  NOT NULL,
    decoder_name   text    NOT NULL REFERENCES decoder,
    predicted_flip boolean NOT NULL,
    PRIMARY KEY (experiment_id, shot_index, decoder_name),
    FOREIGN KEY (experiment_id, shot_index) REFERENCES shot
);

-- One row represents how often one detector position fired over all shots of
-- one experiment.
CREATE TABLE detector_position_summary (
    experiment_id  text    NOT NULL REFERENCES hardware_experiment,
    detector_index integer NOT NULL CHECK (detector_index >= 0),
    fired_count    bigint  NOT NULL CHECK (fired_count >= 0),
    PRIMARY KEY (experiment_id, detector_index)
);

-- A decoder error is when the prediction differs from the actual flip. It is a
-- view and not a stored column, so prediction, actual outcome and error stay
-- three separate things.
CREATE VIEW decoder_outcome AS
SELECT p.experiment_id,
       p.shot_index,
       p.decoder_name,
       p.predicted_flip,
       s.actual_observable_flip,
       p.predicted_flip <> s.actual_observable_flip AS decoder_error
FROM shot_prediction AS p
JOIN shot AS s USING (experiment_id, shot_index);
"""


# ============================================================================
# google_qec: load
# ============================================================================
# Silver has one row per shot with the four predictions as columns. Here every
# prediction gets its own row (shot_prediction), so decoders can be compared
# with one GROUP BY. Shots are copied into a temp staging table first and then
# reshaped with SQL.

DECODERS: tuple[tuple[str, str, str], ...] = (
    ("belief_matching", "obs_flips_predicted_by_belief_matching.01",
     "Belief-matching decoder"),
    ("correlated_matching", "obs_flips_predicted_by_correlated_matching.01",
     "Correlated minimum-weight perfect matching"),
    ("pymatching", "obs_flips_predicted_by_pymatching.01",
     "Minimum-weight perfect matching (PyMatching)"),
    ("tensor_network_contraction", "obs_flips_predicted_by_tensor_network_contraction.01",
     "Tensor-network contraction decoder"),
)

GOOGLE_GOLD_TABLES = (
    "hardware_experiment",
    "shot",
    "decoder",
    "shot_prediction",
    "detector_position_summary",
)

EXPERIMENT_COLUMNS = (
    "experiment_id",
    "source_record_id",
    "basis",
    "distance",
    "rounds",
    "shots",
    "center_row",
    "center_col",
    "measurement_count",
    "detector_count",
)

# Silver shot columns, in the same order as the staging table below.
SHOT_COLUMNS = (
    "source_record_id",
    "experiment_id",
    "shot_index",
    "measurement_bits",
    "sweep_bits",
    "detector_bits",
    "detector_event_count",
    "actual_observable_flip",
    "belief_matching_prediction",
    "correlated_matching_prediction",
    "pymatching_prediction",
    "tensor_network_contraction_prediction",
)

PACKED_COLUMNS = ("measurement_bits", "sweep_bits", "detector_bits")

CREATE_STAGING_SHOT = """
CREATE TEMP TABLE staging_google_shot (
    source_record_id                      text,
    experiment_id                         text,
    shot_index                            bigint,
    measurement_bits                      bytea,
    sweep_bits                            bytea,
    detector_bits                         bytea,
    detector_event_count                  integer,
    actual_observable_flip                boolean,
    belief_matching_prediction            boolean,
    correlated_matching_prediction        boolean,
    pymatching_prediction                 boolean,
    tensor_network_contraction_prediction boolean
) ON COMMIT DROP
"""

LOAD_SHOTS = """
INSERT INTO shot (experiment_id, shot_index, example_id, measurement_bits,
                  sweep_bits, detector_bits, detector_event_count,
                  actual_observable_flip, source_record_id)
SELECT experiment_id,
       shot_index,
       -- example_id for the ML table: hash of source, experiment and shot
       encode(sha256(convert_to(
           'google_qec:' || experiment_id || ':' || shot_index::text, 'UTF8')), 'hex'),
       measurement_bits,
       sweep_bits,
       detector_bits,
       detector_event_count,
       actual_observable_flip,
       source_record_id
FROM staging_google_shot
"""

# the 4 prediction columns of one Silver row become 4 rows here
LOAD_PREDICTIONS = """
INSERT INTO shot_prediction (experiment_id, shot_index, decoder_name, predicted_flip)
          SELECT experiment_id, shot_index, 'belief_matching',
                 belief_matching_prediction FROM staging_google_shot
UNION ALL SELECT experiment_id, shot_index, 'correlated_matching',
                 correlated_matching_prediction FROM staging_google_shot
UNION ALL SELECT experiment_id, shot_index, 'pymatching',
                 pymatching_prediction FROM staging_google_shot
UNION ALL SELECT experiment_id, shot_index, 'tensor_network_contraction',
                 tensor_network_contraction_prediction FROM staging_google_shot
"""

# get_bit counts from the least significant bit of each byte, same as Stim, so
# get_bit(detector_bits, i) is detector i. Computed from the bits already in Gold.
LOAD_DETECTOR_SUMMARY = """
INSERT INTO detector_position_summary (experiment_id, detector_index, fired_count)
SELECT s.experiment_id,
       position.detector_index,
       count(*) FILTER (WHERE get_bit(s.detector_bits, position.detector_index) = 1)
FROM shot AS s
JOIN hardware_experiment AS e USING (experiment_id)
CROSS JOIN LATERAL generate_series(0, e.detector_count - 1) AS position(detector_index)
GROUP BY s.experiment_id, position.detector_index
"""


def load_google(
    connection: psycopg.Connection, experiments: pa.Table, shots: pa.Table
) -> dict[str, int]:
    """Load Google Silver into the Gold tables on the current search_path.

    Has to run inside a transaction. Returns the row count of each Google table.
    """
    with connection.cursor() as cursor:
        cursor.executemany(
            sql.SQL("INSERT INTO hardware_experiment ({}) VALUES ({})").format(
                sql.SQL(", ").join(map(sql.Identifier, EXPERIMENT_COLUMNS)),
                sql.SQL(", ").join(sql.Placeholder() * len(EXPERIMENT_COLUMNS)),
            ),
            [tuple(row[column] for column in EXPERIMENT_COLUMNS) for row in experiments.to_pylist()],
        )
        cursor.executemany("INSERT INTO decoder VALUES (%s, %s, %s)", DECODERS)

        cursor.execute(CREATE_STAGING_SHOT)
        copy_statement = sql.SQL("COPY staging_google_shot ({}) FROM STDIN").format(
            sql.SQL(", ").join(map(sql.Identifier, SHOT_COLUMNS))
        )
        with cursor.copy(copy_statement) as copy:
            for batch in shots.select(list(SHOT_COLUMNS)).to_batches():
                for row in zip(*(column.to_pylist() for column in batch.columns)):
                    copy.write_row(row)

        cursor.execute(LOAD_SHOTS)
        cursor.execute(LOAD_PREDICTIONS)
        cursor.execute(LOAD_DETECTOR_SUMMARY)

        return {
            table: cursor.execute(
                sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
            ).fetchone()[0]
            for table in GOOGLE_GOLD_TABLES
        }


# ============================================================================
# google_qec: checks
# ============================================================================

# md5 of each packed column per experiment, shots in order
GOLD_PACKED_DIGESTS = """
SELECT experiment_id,
       md5(string_agg(measurement_bits, ''::bytea ORDER BY shot_index)),
       md5(string_agg(sweep_bits, ''::bytea ORDER BY shot_index)),
       md5(string_agg(detector_bits, ''::bytea ORDER BY shot_index))
FROM shot
GROUP BY experiment_id
"""


def _silver_packed_digests(shots: pa.Table) -> dict[str, tuple[str, ...]]:
    """md5 of each packed column per experiment, same order as GOLD_PACKED_DIGESTS."""
    rows = sorted(
        zip(*(shots.column(name).to_pylist() for name in ("experiment_id", "shot_index", *PACKED_COLUMNS))),
        key=lambda row: (row[0], row[1]),
    )
    digests: dict[str, list] = defaultdict(lambda: [hashlib.md5() for _ in PACKED_COLUMNS])
    for experiment_id, _, *packed in rows:
        for digest, value in zip(digests[experiment_id], packed):
            digest.update(value)
    return {experiment: tuple(d.hexdigest() for d in ds) for experiment, ds in digests.items()}


def validate_google(
    connection: psycopg.Connection, experiments: pa.Table, shots: pa.Table
) -> None:
    """Compare the loaded Gold with Silver and raise if something is off.

    Raising inside the load transaction rolls the whole Gold load back.
    """
    expected = {
        "hardware_experiment rows equal Silver experiments": (
            "SELECT count(*) FROM hardware_experiment", experiments.num_rows),
        "shot rows equal Silver shots": (
            "SELECT count(*) FROM shot", shots.num_rows),
        "one prediction per shot and decoder": (
            "SELECT count(*) FROM shot_prediction", shots.num_rows * len(DECODERS)),
        "one summary row per detector position": (
            "SELECT count(*) FROM detector_position_summary",
            sum(experiments.column("detector_count").to_pylist())),
        "detector bytes match detector_count": (
            """SELECT count(*) FROM shot JOIN hardware_experiment USING (experiment_id)
               WHERE octet_length(detector_bits) <> (detector_count + 7) / 8""", 0),
        "measurement bytes match measurement_count": (
            """SELECT count(*) FROM shot JOIN hardware_experiment USING (experiment_id)
               WHERE octet_length(measurement_bits) <> (measurement_count + 7) / 8""", 0),
    }
    failures = []
    for rule, (query, want) in expected.items():
        got = connection.execute(query).fetchone()[0]
        if got != want:
            failures.append(f"{rule}: expected {want}, got {got}")

    # the packed bytes in Gold have to be identical to Silver, so the ML export
    # can rebuild them from Gold without going back to Bronze or Silver
    gold_digests = {row[0]: tuple(row[1:]) for row in connection.execute(GOLD_PACKED_DIGESTS)}
    for experiment, silver_digest in _silver_packed_digests(shots).items():
        gold_digest = gold_digests.get(experiment)
        if gold_digest is None:
            failures.append(f"no shots in Gold for {experiment}")
            continue
        for column, want, got in zip(PACKED_COLUMNS, silver_digest, gold_digest):
            if want != got:
                failures.append(f"{column} differs from Silver for {experiment}")

    if failures:
        raise RuntimeError("Google Gold validation failed: " + "; ".join(failures))
