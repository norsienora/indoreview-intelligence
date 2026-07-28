"""Reusable classical sentiment model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import validate_dataframe


LABEL_ORDER = (
    "negative",
    "neutral",
    "positive",
)


def build_classical_model() -> Pipeline:
    """Build the selected TF-IDF Logistic Regression model."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 1),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def train_classical_model(
    training_dataframe: pd.DataFrame,
) -> Pipeline:
    """Train the classical model using a validated dataframe."""
    validate_dataframe(
        training_dataframe,
        split_name="training",
    )

    observed_labels = set(
        training_dataframe["label"]
    )
    missing_labels = (
        set(LABEL_ORDER) - observed_labels
    )

    if missing_labels:
        missing_labels_text = ", ".join(
            sorted(missing_labels)
        )
        raise ValueError(
            "Training data is missing labels: "
            f"{missing_labels_text}"
        )

    model = build_classical_model()

    model.fit(
        training_dataframe["text"],
        training_dataframe["label"],
    )

    return model

def save_classical_model(
    model: Pipeline,
    model_path: Path,
) -> Path:
    """Save a trained model to disk."""
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )

    return model_path


def load_classical_model(
    model_path: Path,
) -> Pipeline:
    """Load a trained model from disk."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(model_path)

    if not isinstance(model, Pipeline):
        raise TypeError(
            "The loaded artifact is not "
            "a scikit-learn Pipeline."
        )

    return model