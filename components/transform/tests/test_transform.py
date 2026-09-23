"""Tests for transform pure logic."""

import pandas as pd
import pytest
from transform import transform_data, transform_dataframe


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "species": [0, 1],
            "feature_a": [1.0, 2.0],
            "feature_b": [10.0, 20.0],
        }
    )


def test_transform_dataframe(sample_data):
    transformed = transform_dataframe(sample_data, suffix="_test")
    assert list(transformed.columns) == ["species_test", "feature_a_test", "feature_b_test"]
    assert len(transformed) == 2


def test_transform_data_io(tmp_path, sample_data):
    train_in = tmp_path / "train.csv"
    eval_in = tmp_path / "eval.csv"
    sample_data.to_csv(train_in, index=False)
    sample_data.to_csv(eval_in, index=False)

    train_out = tmp_path / "train_xf.csv"
    eval_out = tmp_path / "eval_xf.csv"

    res = transform_data(
        train_data_path=str(train_in),
        eval_data_path=str(eval_in),
        transformed_train_output_path=str(train_out),
        transformed_eval_output_path=str(eval_out),
        suffix="_xf",
    )

    assert res["train_rows"] == 2
    assert res["eval_rows"] == 2
    assert res["suffix"] == "_xf"
    assert train_out.exists()
    assert eval_out.exists()

    df_out = pd.read_csv(train_out)
    assert "species_xf" in df_out.columns
