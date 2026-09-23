"""Tests for evaluator pure logic."""

from pathlib import Path
import pickle
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from evaluator import evaluate_model


@pytest.fixture
def trained_model_and_eval_data(tmp_path):
    # Create simple dataset
    df = pd.DataFrame(
        {
            "species_xf": [0, 1, 2, 0, 1, 2],
            "f1_xf": [1.0, 2.0, 3.0, 1.1, 2.1, 3.1],
        }
    )
    eval_csv = tmp_path / "eval_xf.csv"
    df.to_csv(eval_csv, index=False)

    # Train a quick model
    X = df.drop(columns=["species_xf"])
    y = df["species_xf"]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    model_pkl = tmp_path / "model.pkl"
    with open(model_pkl, "wb") as f:
        pickle.dump(model, f)

    return str(model_pkl), str(eval_csv)


def test_evaluate_model(tmp_path, trained_model_and_eval_data):
    model_path, eval_path = trained_model_and_eval_data
    cm_path = tmp_path / "cm.png"
    metrics_path = tmp_path / "metrics.json"

    res = evaluate_model(
        trained_model_path=model_path,
        transformed_eval_data_path=eval_path,
        confusion_matrix_output_path=str(cm_path),
        metrics_output_path=str(metrics_path),
        suffix="_xf",
    )

    assert "accuracy" in res
    assert res["accuracy"] >= 0.0
    assert "confusion_matrix" in res
    assert cm_path.exists()
    assert metrics_path.exists()
