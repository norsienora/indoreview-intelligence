import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from indoreview.model import (
    LABEL_ORDER,
    build_classical_model,
    train_classical_model,
)


def create_training_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "text": [
            "produk bagus memuaskan",
            "layanan bagus memuaskan",
            "produk buruk mengecewakan",
            "layanan buruk mengecewakan",
            "produk biasa saja",
            "layanan biasa saja",
        ],
        "label": [
            "positive",
            "positive",
            "negative",
            "negative",
            "neutral",
            "neutral",
        ],
    })


def test_build_classical_model() -> None:
    model = build_classical_model()

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == [
        "tfidf",
        "classifier",
    ]

    assert (
        model.named_steps["tfidf"].ngram_range
        == (1, 1)
    )
    assert model.named_steps["tfidf"].min_df == 2

    assert (
        model.named_steps["classifier"].max_iter
        == 1000
    )


def test_train_classical_model() -> None:
    training_dataframe = (
        create_training_dataframe()
    )

    model = train_classical_model(
        training_dataframe
    )

    predictions = model.predict([
        "produk bagus",
        "produk buruk",
        "produk biasa",
    ])

    assert len(predictions) == 3
    assert set(predictions).issubset(
        set(LABEL_ORDER)
    )
    assert tuple(
        model.named_steps["classifier"].classes_
    ) == LABEL_ORDER


def test_training_rejects_missing_labels() -> None:
    incomplete_dataframe = pd.DataFrame({
        "text": [
            "produk bagus",
            "layanan bagus",
            "produk buruk",
            "layanan buruk",
        ],
        "label": [
            "positive",
            "positive",
            "negative",
            "negative",
        ],
    })

    with pytest.raises(
        ValueError,
        match="missing labels: neutral",
    ):
        train_classical_model(
            incomplete_dataframe
        )