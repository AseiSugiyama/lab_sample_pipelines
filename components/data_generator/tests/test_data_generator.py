"""Tests for data generator pure logic."""

import pandas as pd
import pytest
from data_generator import generate_data, split_dataset, TARGET_COLUMN


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            TARGET_COLUMN: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            "culmen_length_mm": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "body_mass_g": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        }
    )


def test_split_dataset(sample_dataframe):
    train_df, eval_df = split_dataset(sample_dataframe, test_size=0.2, random_state=42)
    assert len(train_df) == 8
    assert len(eval_df) == 2
    assert TARGET_COLUMN in train_df.columns
    assert TARGET_COLUMN in eval_df.columns


def test_generate_data_with_local_source(tmp_path, sample_dataframe):
    source_file = tmp_path / "source.csv"
    sample_dataframe.to_csv(source_file, index=False)

    train_out = tmp_path / "train.csv"
    eval_out = tmp_path / "eval.csv"

    metrics = generate_data(
        train_data_output_path=str(train_out),
        eval_data_output_path=str(eval_out),
        dataset_url=str(source_file),
        test_size=0.3,
        random_state=42,
    )

    assert metrics["train_rows"] == 7
    assert metrics["eval_rows"] == 3
    assert train_out.exists()
    assert eval_out.exists()

    loaded_train = pd.read_csv(train_out)
    assert len(loaded_train) == 7
