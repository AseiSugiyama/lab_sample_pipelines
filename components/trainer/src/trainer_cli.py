"""Command-Line Interface (CLI) adapter for the Trainer component.

This module provides a standalone command-line entrypoint for training execution
within containerized tasks. It parses model hyperparameters and file destinations,
delegating to `train_model`.
"""

import argparse
from trainer import train_model


def main() -> None:
    """Parse command-line arguments and run the model training workflow."""
    parser = argparse.ArgumentParser(
        description="Trainer CLI for Palmer Penguins classification pipeline."
    )
    parser.add_argument(
        "--transformed-train-data",
        type=str,
        required=True,
        help="Input filesystem path for preprocessed training CSV.",
    )
    parser.add_argument(
        "--trained-model",
        type=str,
        required=True,
        help="Destination filesystem path where trained model pickle will be written.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_xf",
        help="Feature column name suffix (default: '_xf').",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of decision trees in RandomForestClassifier (default: 100).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility (default: 42).",
    )
    args = parser.parse_args()

    metrics = train_model(
        train_data_path=args.transformed_train_data,
        model_output_path=args.trained_model,
        suffix=args.suffix,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    print(f"Model training complete: {metrics}")


if __name__ == "__main__":
    main()
