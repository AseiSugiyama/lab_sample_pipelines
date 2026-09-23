"""Pure logic implementation for the Data Generator component.

This module is responsible for fetching the Palmer Penguins dataset from a remote
or local source, validating the target column, and splitting the data into training
and evaluation subsets. It operates independently of KFP and CLI dependencies,
enabling fast in-memory execution and unit testing.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

# Public Palmer Penguins processed dataset hosted by TensorFlow Datasets / Google Cloud Storage.
PENGUIN_DATASET_URI = (
    "https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins_processed.csv"
)
TARGET_COLUMN = "species"


def fetch_dataset(dataset_url: str = PENGUIN_DATASET_URI) -> pd.DataFrame:
    """Fetch the Palmer Penguins dataset from a remote URL or local file path.

    Args:
        dataset_url: HTTP/HTTPS URL or local filesystem path pointing to the CSV file.
            Defaults to the official Palmer Penguins processed dataset on GCS.

    Returns:
        pd.DataFrame: Loaded dataset containing penguin morphological measurements
            and species classification labels.

    Raises:
        ValueError: If the file cannot be fetched or parsed as a valid CSV.
    """
    return pd.read_csv(dataset_url)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset into training and evaluation partitions.

    Args:
        df: Input DataFrame containing the target classification column.
        test_size: Proportion of the dataset to allocate to the evaluation split (0.0 to 1.0).
            Defaults to 0.2 (20%).
        random_state: Random seed to ensure deterministic splitting across runs.
            Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of (train_df, eval_df).

    Raises:
        ValueError: If TARGET_COLUMN ('species') is missing from the DataFrame columns.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )
    train_df, eval_df = train_test_split(  # pyright: ignore[reportAssignmentType]
        df,
        test_size=test_size,
        random_state=random_state,
    )
    return train_df, eval_df  # pyright: ignore[reportReturnType]


def generate_data(
    train_data_output_path: str,
    eval_data_output_path: str,
    dataset_url: str = PENGUIN_DATASET_URI,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Generate training and evaluation datasets and persist them to disk.

    This function coordinates dataset retrieval, validation, splitting, and writing
    to target output paths. Directory hierarchies for output paths are automatically
    created if they do not exist.

    Args:
        train_data_output_path: Filesystem path where training CSV will be written.
        eval_data_output_path: Filesystem path where evaluation CSV will be written.
        dataset_url: Source URL or local path for the raw dataset.
        test_size: Ratio of the evaluation split (default: 0.2).
        random_state: Random seed for deterministic reproducibility (default: 42).

    Returns:
        Dict[str, object]: Summary metadata containing:
            - "train_rows" (int): Number of rows in the training split.
            - "eval_rows" (int): Number of rows in the evaluation split.
            - "columns" (List[str]): List of all feature and target column names.
    """
    df = fetch_dataset(dataset_url)
    train_df, eval_df = split_dataset(df, test_size=test_size, random_state=random_state)

    # Ensure target parent directories exist before persisting artifacts.
    # On Vertex AI Pipelines, the runtime creates parent directories for Output[Dataset],
    # but defensive directory creation ensures robust execution in standalone CLI environments.
    train_path = Path(train_data_output_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)

    eval_path = Path(eval_data_output_path)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(eval_path, index=False)

    return {
        "train_rows": len(train_df),
        "eval_rows": len(eval_df),
        "columns": list(df.columns),
    }
