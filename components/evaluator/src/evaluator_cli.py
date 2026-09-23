"""Command-Line Interface (CLI) adapter for the Evaluator component.

This module provides a standalone command-line entrypoint for model evaluation.
It parses paths for the trained model, evaluation data, and optional artifact
destinations (confusion matrix image, metrics JSON), delegating to `evaluate_model`.
"""

import argparse
from evaluator import evaluate_model


def main() -> None:
    """Parse command-line arguments and run the model evaluation workflow."""
    parser = argparse.ArgumentParser(
        description="Evaluator CLI for Palmer Penguins classification pipeline."
    )
    parser.add_argument(
        "--trained-model",
        type=str,
        required=True,
        help="Input filesystem path for trained model pickle file.",
    )
    parser.add_argument(
        "--transformed-eval-data",
        type=str,
        required=True,
        help="Input filesystem path for preprocessed evaluation CSV.",
    )
    parser.add_argument(
        "--confusion-matrix-path",
        type=str,
        default=None,
        help="Optional destination path for rendered confusion matrix PNG plot.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Optional destination path for metrics JSON file.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_xf",
        help="Feature column name suffix (default: '_xf').",
    )
    args = parser.parse_args()

    results = evaluate_model(
        trained_model_path=args.trained_model,
        transformed_eval_data_path=args.transformed_eval_data,
        confusion_matrix_output_path=args.confusion_matrix_path,
        metrics_output_path=args.metrics_path,
        suffix=args.suffix,
    )
    print(f"Model evaluation complete: accuracy={results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
