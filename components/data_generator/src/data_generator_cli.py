"""Command-Line Interface (CLI) adapter for the Data Generator component.

This module provides a standalone command-line entrypoint for containerized execution
and debugging. It parses arguments via standard library `argparse` and delegates
execution directly to the core `generate_data` domain function.
"""

import argparse
from data_generator import PENGUIN_DATASET_URI, generate_data


def main() -> None:
    """Parse command-line arguments and run the data generator workflow."""
    parser = argparse.ArgumentParser(
        description="Data Generator CLI for Palmer Penguins classification pipeline."
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Destination filesystem path where training CSV will be written.",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        required=True,
        help="Destination filesystem path where evaluation CSV will be written.",
    )
    parser.add_argument(
        "--dataset-url",
        type=str,
        default=PENGUIN_DATASET_URI,
        help="HTTP/HTTPS source URL or local file path for Palmer Penguins CSV.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Evaluation partition ratio (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic dataset splitting (default: 42).",
    )
    args = parser.parse_args()

    metrics = generate_data(
        train_data_output_path=args.train_data,
        eval_data_output_path=args.eval_data,
        dataset_url=args.dataset_url,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Data generation complete: {metrics}")


if __name__ == "__main__":
    main()
