"""Inference utilities for sentiment classification."""

from typing import TypedDict

from sklearn.pipeline import Pipeline


class SentimentPrediction(TypedDict):
    """Structured sentiment prediction result."""

    text: str
    label: str
    confidence: float
    probabilities: dict[str, float]


def predict_sentiment(
    model: Pipeline,
    text: str,
) -> SentimentPrediction:
    """Predict sentiment and class probabilities."""
    if not isinstance(text, str):
        raise TypeError(
            "Prediction text must be a string."
        )

    clean_text = text.strip()

    if not clean_text:
        raise ValueError(
            "Prediction text must not be blank."
        )

    probability_values = (
        model.predict_proba([clean_text])[0]
    )

    classifier = model.named_steps[
        "classifier"
    ]
    class_labels = classifier.classes_

    probabilities = {
        str(label): float(probability)
        for label, probability in zip(
            class_labels,
            probability_values,
            strict=True,
        )
    }

    predicted_index = int(
        probability_values.argmax()
    )
    predicted_label = str(
        class_labels[predicted_index]
    )
    confidence = float(
        probability_values[predicted_index]
    )

    return {
        "text": clean_text,
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
    }