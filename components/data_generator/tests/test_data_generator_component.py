"""Tests for data generator KFP component adapter."""

import pandas as pd
from data_generator import TARGET_COLUMN
from data_generator_component import data_generator_op
from kfp.dsl import Dataset


def test_data_generator_component_spec():
    """Verify component specification, inputs, outputs, and base image."""
    spec = data_generator_op.component_spec
    assert "data-generator" in spec.name or "data_generator" in spec.name
    assert "train_data" in spec.outputs
    assert "eval_data" in spec.outputs
    assert "Dataset" in spec.outputs["train_data"].type
    assert "Dataset" in spec.outputs["eval_data"].type
    assert "kfp-sample-data-generator" in spec.implementation.container.image


def test_data_generator_component_execution(tmp_path):
    """Execute component logic locally using component function."""
    source_df = pd.DataFrame(
        {
            TARGET_COLUMN: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    source_file = tmp_path / "dummy_source.csv"
    source_df.to_csv(source_file, index=False)

    train_ds = Dataset(name="train_data", uri=str(tmp_path / "train.csv"))
    eval_ds = Dataset(name="eval_data", uri=str(tmp_path / "eval.csv"))

    # Direct execution of the underlying component function
    data_generator_op.python_func(
        train_data=train_ds,
        eval_data=eval_ds,
        dataset_url=str(source_file),
        test_size=0.2,
        random_state=42,
    )

    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "eval.csv").exists()
    assert train_ds.metadata["rows"] == 8
    assert eval_ds.metadata["rows"] == 2
