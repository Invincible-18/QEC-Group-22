import pandas as pd
import pytest

from quantum_lake_student.stages.gold_syndrome import prepare_gold_rows


def _silver_rows() -> pd.DataFrame:
    pattern = bytes([0, 1] * 8)
    return pd.DataFrame(
        [
            {
                "source_record_id": "source/row-1",
                "experiment_id": "d-3_pfr-0.001000_nb-10M",
                "physical_fault_rate": 0.001,
                "syndrome_bits": pattern,
                "round_count": 4,
                "check_count": 4,
                "logical_error_label": False,
                "quantity": 7,
            },
            {
                "source_record_id": "source/row-2",
                "experiment_id": "d-3_pfr-0.001000_nb-10M",
                "physical_fault_rate": 0.001,
                "syndrome_bits": pattern,
                "round_count": 4,
                "check_count": 4,
                "logical_error_label": True,
                "quantity": 2,
            },
        ]
    )


def test_gold_dimensions_deduplicate_but_keep_observations_and_both_labels() -> None:
    experiments, patterns, observations = prepare_gold_rows(_silver_rows())

    assert len(experiments) == 1
    assert len(patterns) == 1
    assert [row[3] for row in observations] == [False, True]
    assert [row[4] for row in observations] == [7, 2]
    assert len({row[0] for row in observations}) == 2


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("quantity", 0, "positive integer"),
        ("round_count", 3, "dimensions"),
        ("syndrome_bits", bytes([0, 1] * 7 + [2, 1]), "binary"),
    ],
)
def test_gold_rejects_invalid_silver_values(
    column: str, value: object, message: str
) -> None:
    silver = _silver_rows()
    silver.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        prepare_gold_rows(silver)


def test_gold_rejects_duplicate_source_record_ids() -> None:
    silver = _silver_rows()
    silver.loc[1, "source_record_id"] = silver.loc[0, "source_record_id"]

    with pytest.raises(ValueError, match="duplicate source_record_id"):
        prepare_gold_rows(silver)