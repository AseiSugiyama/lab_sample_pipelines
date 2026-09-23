"""Command-Line Interface (CLI) adapter for the Transform component.

This module provides a standalone command-line entrypoint for containerized execution
and debugging. It parses input/output file paths and delegates execution to `transform_data`.
"""

import argparse
from transform import transform_data


def main() -> None:
    """Parse command-line arguments and run the transform workflow."""
    parser = argparse.ArgumentParser(
        description="Transform CLI for Palmer Penguins classification pipeline."
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Input filesystem path for raw training CSV.",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        required=True,
        help="Input filesystem path for raw evaluation CSV.",
    )
    parser.add_argument(
        "--transformed-train-data",
        type=str,
        required=True,
        help="Output filesystem path for transformed training CSV.",
    )
    parser.add_argument(
        "--transformed-eval-data",
        type=str,
        required=True,
        help="Output filesystem path for transformed evaluation CSV.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_xf",
        help="Suffix string to append to column headers (default: '_xf').",
    )
    args = parser.parse_args()

    metrics = transform_data(
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        transformed_train_output_path=args.transformed_train_data,
        transformed_eval_output_path=args.transformed_eval_data,
        suffix=args.suffix,
    )
    print(f"Data transformation complete: {metrics}")


if __name__ == "__main__":
    main()
