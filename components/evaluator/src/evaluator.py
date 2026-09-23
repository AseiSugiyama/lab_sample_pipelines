"""Pure logic implementation for the Evaluator component.

This module evaluates a trained scikit-learn model on the transformed evaluation dataset.
It computes classification accuracy, generates a confusion matrix with human-readable
class labels (Adelie, Chinstrap, Gentoo), and optionally outputs visualization images
and metrics JSON files for standalone CLI execution.
"""

import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

TARGET_COLUMN = "species"
CLASS_NAMES = ["Adelie", "Chinstrap", "Gentoo"]


def evaluate_model(
    trained_model_path: str,
    transformed_eval_data_path: str,
    confusion_matrix_output_path: Optional[str] = None,
    metrics_output_path: Optional[str] = None,
    suffix: str = "_xf",
) -> Dict[str, Any]:
    """Evaluate trained model on evaluation dataset, generating metrics and confusion matrix.

    Args:
        trained_model_path: Filesystem path to the serialized model pickle file.
        transformed_eval_data_path: Filesystem path to the preprocessed evaluation CSV.
        confusion_matrix_output_path: Optional path to save confusion matrix PNG plot (for CLI).
        metrics_output_path: Optional path to save scalar metrics JSON (for CLI).
        suffix: Feature transformation suffix appended to column names (default: "_xf").

    Returns:
        Dict[str, object]: Evaluation metrics dictionary containing:
            - "accuracy" (float): Classification accuracy score (0.0 to 1.0).
            - "confusion_matrix" (List[List[int]]): 2D confusion matrix array.
            - "categories" (List[str]): List of class label strings.
            - "n_eval_samples" (int): Number of evaluation instances.

    Raises:
        ValueError: If the target column is missing from the evaluation dataset.
    """
    with Path(trained_model_path).open("rb") as f:
        model = pickle.load(f)

    eval_df = pd.read_csv(transformed_eval_data_path)
    target_name = f"{TARGET_COLUMN}{suffix}"
    if target_name not in eval_df.columns:
        raise ValueError(
            f"Target column '{target_name}' not found in eval dataset. "
            f"Available columns: {list(eval_df.columns)}"
        )

    X_eval = eval_df.drop(columns=[target_name])
    y_eval = eval_df[target_name]

    y_pred = model.predict(X_eval)
    accuracy = float(accuracy_score(y_eval, y_pred))

    # Confusion matrix computation with human-readable class names
    cm = confusion_matrix(y_eval, y_pred)
    categories: List[str] = [
        CLASS_NAMES[int(cls_idx)] if int(cls_idx) < len(CLASS_NAMES) else str(cls_idx)
        for cls_idx in sorted(y_eval.unique())
    ]

    # Optional file outputs for CLI / Container usage (when not using KFP first-class artifacts)
    if confusion_matrix_output_path:
        cm_path = Path(confusion_matrix_output_path)
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues")
        fig.savefig(cm_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    if metrics_output_path:
        m_path = Path(metrics_output_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        m_path.write_text(
            json.dumps(
                {
                    "metrics": [
                        {
                            "name": "accuracy",
                            "numberValue": accuracy,
                            "format": "PERCENTAGE",
                        }
                    ]
                },
                indent=2,
            )
        )

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "categories": categories,
        "n_eval_samples": len(eval_df),
    }
