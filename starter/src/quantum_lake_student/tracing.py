"""Source-tracing helpers.

Every Silver row carries a stable ``source_record_id``. The functions here build
that identifier and the matching ``results/part1/source_trace.parquet`` rows,
which map it back to the Bronze bytes the row was produced from.

An identifier depends only on the supplied source data, never on run time or on
row order, so a second run over unchanged input produces the same identifiers.
"""

from __future__ import annotations

from quantum_lake_student.models import stable_record_hash


GOOGLE_SOURCE_NAME = "google_qec"

# One Google shot is assembled from these eight members of a single experiment
# directory. The shot only exists if all eight align, so each member gets its
# own trace row under the shot's identifier.
GOOGLE_SHOT_MEMBERS: tuple[str, ...] = (
    "measurements.b8",
    "sweep.b8",
    "detection_events.b8",
    "obs_flips_actual.01",
    "obs_flips_predicted_by_belief_matching.01",
    "obs_flips_predicted_by_correlated_matching.01",
    "obs_flips_predicted_by_pymatching.01",
    "obs_flips_predicted_by_tensor_network_contraction.01",
)


def google_shot_source_record_id(
    *, bronze_object: str, experiment_id: str, shot_index: int
) -> str:
    """Return the stable identifier of one aligned Google shot.

    ``shot_index`` is passed through ``int`` so a value taken from a pandas
    frame (``numpy.int64``) yields the same identifier as a plain integer; the
    hash helper serialises to JSON and rejects numpy types outright.
    """
    return stable_record_hash(
        {
            "bronze_object": bronze_object,
            "experiment_id": experiment_id,
            "shot_index": int(shot_index),
        }
    )


def google_shot_trace_rows(
    *,
    bronze_object: str,
    input_sha256: str,
    experiment_id: str,
    shot_index: int,
) -> list[dict[str, str]]:
    """Return one ``source_trace`` row per Bronze member behind one shot.

    Every row carries the same ``source_record_id``. That repetition is the
    point: it is how a single Silver shot row resolves to the several archive
    members its values were read from.
    """
    source_record_id = google_shot_source_record_id(
        bronze_object=bronze_object,
        experiment_id=experiment_id,
        shot_index=shot_index,
    )
    return [
        {
            "source_record_id": source_record_id,
            "source_name": GOOGLE_SOURCE_NAME,
            "bronze_object": bronze_object,
            "archive_member": f"{experiment_id}/{member}",
            "record_locator": f"shot={int(shot_index)}",
            "input_sha256": input_sha256,
        }
        for member in GOOGLE_SHOT_MEMBERS
    ]
