"""Tests for trainer KFP component adapter."""

from pathlib import Path
from typing import Any
import pandas as pd
from trainer_component import train_model_op
from kfp.dsl import Dataset, Model


def test_trainer_component_spec():
    """Verify component specification, inputs, outputs, and base image."""
    # KFP's @dsl.component dynamically attaches .component_spec and .python_func at runtime
    op: Any = train_model_op
    spec = op.component_spec
    assert "train" in spec.name
    assert "transformed_train_data" in spec.inputs
    assert "trained_model" in spec.outputs
    assert "Model" in spec.outputs["trained_model"].type
    assert "kfp-sample-trainer" in spec.implementation.container.image


def test_trainer_component_execution(tmp_path):
    """Execute trainer component logic locally."""
    train_in = tmp_path / "train_xf.csv"
    model_out = tmp_path / "model.pkl"

    df = pd.DataFrame(
        {
            "species_xf": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            "f1_xf": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    df.to_csv(train_in, index=False)

    train_ds = Dataset(name="transformed_train_data", uri=str(train_in))
    trained_model = Model(name="trained_model", uri=str(model_out))

    # Access underlying python_func dynamically attached by @dsl.component
    op: Any = train_model_op
    op.python_func(
        transformed_train_data=train_ds,
        trained_model=trained_model,
        suffix="_xf",
        n_estimators=10,
        random_state=42,
    )

    assert model_out.exists()
    assert trained_model.metadata["framework"] == "scikit-learn"
    assert trained_model.metadata["algorithm"] == "RandomForestClassifier"
    assert trained_model.metadata["n_samples"] == 10
