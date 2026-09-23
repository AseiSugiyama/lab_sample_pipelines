"""Pure logic implementation for the Transform component.

This module applies feature column name transformations (appending a configurable
suffix, such as `_xf`) to simulate feature engineering and schema differentiation
across pipeline stages. It operates completely independently of KFP and CLI frameworks.
"""

from pathlib import Path
from typing import Dict, List
import pandas as pd


def transform_dataframe(df: pd.DataFrame, suffix: str = "_xf") -> pd.DataFrame:
    """Append a suffix to all feature and target column names in a DataFrame.

    Renaming columns across pipeline transformations prevents ambiguous column collisions
    in downstream feature stores and clearly demarcates preprocessed feature sets.

    Args:
        df: Input pandas DataFrame to transform.
        suffix: Suffix string to append to each column header (default: "_xf").

    Returns:
        pd.DataFrame: New DataFrame with renamed column headers.
    """
    return df.rename(columns={col: f"{col}{suffix}" for col in df.columns})


def transform_data(
    train_data_path: str,
    eval_data_path: str,
    transformed_train_output_path: str,
    transformed_eval_output_path: str,
    suffix: str = "_xf",
) -> Dict[str, object]:
    """Read train/eval datasets, append suffix to columns, and persist to output paths.

    Args:
        train_data_path: Filesystem path to the input training CSV.
        eval_data_path: Filesystem path to the input evaluation CSV.
        transformed_train_output_path: Destination path for transformed training CSV.
        transformed_eval_output_path: Destination path for transformed evaluation CSV.
        suffix: Suffix string to append to all column headers (default: "_xf").

    Returns:
        Dict[str, object]: Transformation metadata containing:
            - "train_rows" (int): Number of transformed training rows.
            - "eval_rows" (int): Number of transformed evaluation rows.
            - "columns" (List[str]): List of transformed column headers.
            - "suffix" (str): Applied suffix string.
    """
    train_df = pd.read_csv(train_data_path)
    eval_df = pd.read_csv(eval_data_path)

    transformed_train_df = transform_dataframe(train_df, suffix=suffix)
    transformed_eval_df = transform_dataframe(eval_df, suffix=suffix)

    # Ensure target parent directories exist before writing
    out_train_path = Path(transformed_train_output_path)
    out_train_path.parent.mkdir(parents=True, exist_ok=True)
    transformed_train_df.to_csv(out_train_path, index=False)

    out_eval_path = Path(transformed_eval_output_path)
    out_eval_path.parent.mkdir(parents=True, exist_ok=True)
    transformed_eval_df.to_csv(out_eval_path, index=False)

    return {
        "train_rows": len(transformed_train_df),
        "eval_rows": len(transformed_eval_df),
        "columns": list(transformed_train_df.columns),
        "suffix": suffix,
    }
