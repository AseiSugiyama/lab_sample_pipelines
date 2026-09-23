"""Tests for transform KFP component adapter."""

from typing import Any
import pandas as pd
from transform_component import transform_op
from kfp.dsl import Dataset


def test_transform_component_spec():
    """Verify component specification, inputs, outputs, and base image."""
    # KFP's @dsl.component dynamically attaches .component_spec and .python_func at runtime
    op: Any = transform_op
    spec = op.component_spec
    assert "transform" in spec.name
    assert "train_data" in spec.inputs
    assert "eval_data" in spec.inputs
    assert "transformed_train_data" in spec.outputs
    assert "transformed_eval_data" in spec.outputs
    assert "Dataset" in spec.outputs["transformed_train_data"].type
    assert "Dataset" in spec.outputs["transformed_eval_data"].type
    assert "kfp-sample-transform" in spec.implementation.container.image


def test_transform_component_execution(tmp_path):
    """Execute transform component logic locally."""
    train_in = tmp_path / "train.csv"
    eval_in = tmp_path / "eval.csv"
    train_out = tmp_path / "train_xf.csv"
    eval_out = tmp_path / "eval_xf.csv"

    pd.DataFrame({"species": [0, 1], "feat": [1.0, 2.0]}).to_csv(train_in, index=False)
    pd.DataFrame({"species": [1, 2], "feat": [3.0, 4.0]}).to_csv(eval_in, index=False)

    train_ds = Dataset(name="train_data", uri=str(train_in))
    eval_ds = Dataset(name="eval_data", uri=str(eval_in))
    transformed_train_ds = Dataset(name="transformed_train_data", uri=str(train_out))
    transformed_eval_ds = Dataset(name="transformed_eval_data", uri=str(eval_out))

    # Access underlying python_func dynamically attached by @dsl.component
    op: Any = transform_op
    op.python_func(
        train_data=train_ds,
        eval_data=eval_ds,
        transformed_train_data=transformed_train_ds,
        transformed_eval_data=transformed_eval_ds,
        suffix="_xf",
    )

    assert train_out.exists()
    assert eval_out.exists()
    assert transformed_train_ds.metadata["rows"] == 2
    assert transformed_train_ds.metadata["suffix"] == "_xf"
