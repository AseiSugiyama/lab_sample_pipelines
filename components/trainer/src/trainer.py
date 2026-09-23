"""Pure logic implementation for the Trainer component.

This module fits a scikit-learn RandomForestClassifier on the transformed Palmer Penguins
dataset and serializes the trained estimator into a Python pickle file. It operates
completely independently of KFP and CLI frameworks, enabling fast local execution.
"""

from pathlib import Path
import pickle
from typing import Dict, List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TARGET_COLUMN = "species"


def train_model(
    train_data_path: str,
    model_output_path: str,
    suffix: str = "_xf",
    n_estimators: int = 100,
    random_state: int = 42,
) -> Dict[str, object]:
    """Train a RandomForestClassifier on transformed penguin dataset and serialize model.

    This function reads the preprocessed training dataset, extracts the target classification
    column (accounting for the feature transformation suffix), trains an ensemble
    classifier, and serializes the resulting model artifact to disk.

    Args:
        train_data_path: Filesystem path to the transformed training CSV.
        model_output_path: Destination path where the model pickle file will be saved.
        suffix: Feature transformation suffix appended to column names (default: "_xf").
        n_estimators: Number of trees in the random forest (default: 100).
        random_state: Random seed for deterministic reproducibility (default: 42).

    Returns:
        Dict[str, object]: Training metadata containing:
            - "framework" (str): "scikit-learn"
            - "algorithm" (str): "RandomForestClassifier"
            - "n_samples" (int): Number of training examples fitted.
            - "n_features" (int): Number of input features used for training.
            - "feature_names" (List[str]): List of training feature names.

    Raises:
        ValueError: If the target column is missing from the dataset.
    """
    df = pd.read_csv(train_data_path)
    target_name = f"{TARGET_COLUMN}{suffix}"
    if target_name not in df.columns:
        raise ValueError(
            f"Target column '{target_name}' not found in train dataset. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_name])
    y = df[target_name]

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)

    # Write output model artifact.
    # When pipeline runs, runtime gives path to save dir for each outputPath placeholder.
    # For more detail, see:
    # https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline#compare
    out_model_path = Path(model_output_path)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    with out_model_path.open("wb") as f:
        pickle.dump(model, f)

    return {
        "framework": "scikit-learn",
        "algorithm": "RandomForestClassifier",
        "n_samples": len(df),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
    }
