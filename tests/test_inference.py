import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from indoreview.inference import predict_sentiment
from indoreview.model import (
    LABEL_ORDER,
    train_classical_model,
)


@pytest.fixture
def trained_model() -> Pipeline:
    training_dataframe = pd.DataFrame(
        {
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
        }
    )

    return train_classical_model(
        training_dataframe
    )


def test_predict_sentiment_returns_probabilities(
    trained_model: Pipeline,
) -> None:
    result = predict_sentiment(
        trained_model,
        "  produk bagus  ",
    )

    assert result["text"] == "produk bagus"
    assert result["label"] in LABEL_ORDER

    assert set(result["probabilities"]) == set(
        LABEL_ORDER
    )

    assert sum(
        result["probabilities"].values()
    ) == pytest.approx(1.0)

    assert result["confidence"] == pytest.approx(
        max(result["probabilities"].values())
    )


@pytest.mark.parametrize(
    "blank_text",
    ["", "   "],
)
def test_predict_sentiment_rejects_blank_text(
    trained_model: Pipeline,
    blank_text: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be blank",
    ):
        predict_sentiment(
            trained_model,
            blank_text,
        )


def test_predict_sentiment_rejects_non_string(
    trained_model: Pipeline,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        predict_sentiment(
            trained_model,
            None,  # type: ignore[arg-type]
        )