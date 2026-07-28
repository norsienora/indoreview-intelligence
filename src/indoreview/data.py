from pathlib import Path

import pandas as pd


COLUMN_NAMES = ["text", "label"]

VALID_LABELS = {
    "positive",
    "neutral",
    "negative",
}

SPLIT_FILENAMES = {
    "train": "train.tsv",
    "validation": "validation.tsv",
    "test": "test.tsv",
}


def validate_dataframe(
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:
    """Validate the schema and basic quality of a dataset split."""

    if list(dataframe.columns) != COLUMN_NAMES:
        raise ValueError(
            f"{split_name}: expected columns {COLUMN_NAMES}, "
            f"but received {list(dataframe.columns)}."
        )

    if dataframe["text"].isna().any():
        raise ValueError(f"{split_name}: missing text values found.")

    if dataframe["label"].isna().any():
        raise ValueError(f"{split_name}: missing label values found.")

    blank_text_count = (
        dataframe["text"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if blank_text_count > 0:
        raise ValueError(
            f"{split_name}: {blank_text_count} blank texts found."
        )

    invalid_labels = (
        set(dataframe["label"]) - VALID_LABELS
    )

    if invalid_labels:
        raise ValueError(
            f"{split_name}: invalid labels found: "
            f"{sorted(invalid_labels)}."
        )


def load_split(file_path: Path) -> pd.DataFrame:
    """Load and validate one TSV dataset split."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=COLUMN_NAMES,
    )

    validate_dataframe(
        dataframe,
        split_name=file_path.stem,
    )

    return dataframe


def load_raw_splits(
    data_directory: Path,
) -> dict[str, pd.DataFrame]:
    """Load the train, validation, and test splits."""

    return {
        split_name: load_split(
            data_directory / filename
        )
        for split_name, filename
        in SPLIT_FILENAMES.items()
    }


def clean_training_data(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Remove duplicate training texts and evaluation leakage."""

    validate_dataframe(train_dataframe, "train")
    validate_dataframe(validation_dataframe, "validation")
    validate_dataframe(test_dataframe, "test")

    labels_per_text = (
        train_dataframe
        .groupby("text")["label"]
        .nunique()
    )

    conflicting_duplicate_count = int(
        (labels_per_text > 1).sum()
    )

    if conflicting_duplicate_count > 0:
        raise ValueError(
            "Training data contains "
            f"{conflicting_duplicate_count} duplicated texts "
            "with conflicting labels."
        )

    evaluation_texts = (
        set(validation_dataframe["text"])
        | set(test_dataframe["text"])
    )

    clean_dataframe = (
        train_dataframe.loc[
            ~train_dataframe["text"].isin(evaluation_texts)
        ]
        .drop_duplicates(
            subset="text",
            keep="first",
        )
        .reset_index(drop=True)
    )

    validate_dataframe(
        clean_dataframe,
        "clean_train",
    )

    if clean_dataframe["text"].duplicated().any():
        raise ValueError(
            "Duplicate texts remain after cleaning."
        )

    remaining_overlap = (
        set(clean_dataframe["text"])
        & evaluation_texts
    )

    if remaining_overlap:
        raise ValueError(
            "Evaluation leakage remains after cleaning."
        )

    return clean_dataframe