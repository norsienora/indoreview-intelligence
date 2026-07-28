from pathlib import Path

import pandas as pd
import pytest

from indoreview.data import (
    clean_training_data,
    load_split,
    validate_dataframe,
)


def make_dataframe(
    rows: list[tuple[str, str]],
) -> pd.DataFrame:
    """Create a small sentiment dataframe for testing."""

    return pd.DataFrame(
        rows,
        columns=["text", "label"],
    )


def test_load_split_reads_valid_tsv(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "sample.tsv"

    file_path.write_text(
        "produknya bagus\tpositive\n"
        "biasa saja\tneutral\n",
        encoding="utf-8",
    )

    dataframe = load_split(file_path)

    assert dataframe.shape == (2, 2)
    assert dataframe["label"].tolist() == [
        "positive",
        "neutral",
    ]


def test_validate_dataframe_rejects_invalid_label() -> None:
    dataframe = make_dataframe([
        ("sentimennya tidak jelas", "mixed"),
    ])

    with pytest.raises(
        ValueError,
        match="invalid labels",
    ):
        validate_dataframe(
            dataframe,
            split_name="sample",
        )


def test_clean_training_data_removes_duplicates_and_leakage() -> None:
    train_dataframe = make_dataframe([
        ("produknya bagus", "positive"),
        ("produknya bagus", "positive"),
        ("produknya buruk", "negative"),
        ("muncul di validation", "neutral"),
        ("khusus train", "neutral"),
    ])

    validation_dataframe = make_dataframe([
        ("muncul di validation", "neutral"),
    ])

    test_dataframe = make_dataframe([
        ("khusus test", "negative"),
    ])

    clean_dataframe = clean_training_data(
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )

    assert clean_dataframe["text"].tolist() == [
        "produknya bagus",
        "produknya buruk",
        "khusus train",
    ]

    assert clean_dataframe["text"].duplicated().sum() == 0


def test_clean_training_data_rejects_conflicting_labels() -> None:
    train_dataframe = make_dataframe([
        ("teks yang sama", "positive"),
        ("teks yang sama", "negative"),
    ])

    validation_dataframe = make_dataframe([
        ("khusus validation", "neutral"),
    ])

    test_dataframe = make_dataframe([
        ("khusus test", "negative"),
    ])

    with pytest.raises(
        ValueError,
        match="conflicting labels",
    ):
        clean_training_data(
            train_dataframe,
            validation_dataframe,
            test_dataframe,
        )