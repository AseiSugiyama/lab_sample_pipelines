"""Tests for trainer pure logic."""

from pathlib import Path
import pickle
import pandas as pd
import pytest
from trainer import train_model


@pytest.fixture
def sample_train_data(tmp_path):
    train_file = tmp_path / "transformed_train.csv"
    df = pd.DataFrame(
        {
            "species_xf": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            "feature1_xf": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature2_xf": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )
    df.to_csv(train_file, index=False)
    return str(train_file)


def test_train_model(tmp_path, sample_train_data):
    model_out = tmp_path / "model.pkl"
    meta = train_model(
        train_data_path=sample_train_data,
        model_output_path=str(model_out),
        suffix="_xf",
        n_estimators=10,
        random_state=42,
    )

    assert meta["framework"] == "scikit-learn"
    assert meta["algorithm"] == "RandomForestClassifier"
    assert meta["n_samples"] == 10
    assert meta["n_features"] == 2
    assert model_out.exists()

    with open(model_out, "rb") as f:
        loaded_model = pickle.load(f)
    assert hasattr(loaded_model, "predict")
