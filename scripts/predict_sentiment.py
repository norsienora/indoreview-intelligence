"""Predict Indonesian sentiment from the command line."""

import argparse
import json
from pathlib import Path

from indoreview.inference import predict_sentiment
from indoreview.model import load_classical_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "classical_model.joblib"
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Predict the sentiment of Indonesian text."
        )
    )

    parser.add_argument(
        "text",
        help="Indonesian text to classify.",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model artifact.",
    )

    return parser.parse_args()


def main() -> None:
    """Load the model and print its prediction."""
    arguments = parse_arguments()

    model = load_classical_model(
        arguments.model_path
    )

    result = predict_sentiment(
        model,
        arguments.text,
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()