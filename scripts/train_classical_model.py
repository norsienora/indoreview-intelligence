"""Train and save the selected classical sentiment model."""

from pathlib import Path

from indoreview.data import (
    clean_training_data,
    load_raw_splits,
)
from indoreview.model import (
    save_classical_model,
    train_classical_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = (
    PROJECT_ROOT / "data" / "raw"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "classical_model.joblib"
)


def main() -> None:
    """Train and persist the classical model."""
    raw_splits = load_raw_splits(
        RAW_DATA_DIR
    )

    training_dataframe = clean_training_data(
        raw_splits["train"],
        raw_splits["validation"],
        raw_splits["test"],
    )

    model = train_classical_model(
        training_dataframe
    )

    saved_path = save_classical_model(
        model,
        MODEL_PATH,
    )

    print(
        f"Training rows: "
        f"{len(training_dataframe):,}"
    )
    print(
        f"Model saved to: {saved_path}"
    )


if __name__ == "__main__":
    main()