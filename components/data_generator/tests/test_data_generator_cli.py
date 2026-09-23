"""Tests for data generator CLI adapter."""

import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
from data_generator import TARGET_COLUMN


def test_data_generator_cli_execution(tmp_path):
    source_df = pd.DataFrame(
        {
            TARGET_COLUMN: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    source_file = tmp_path / "dummy_source.csv"
    source_df.to_csv(source_file, index=False)

    train_out = tmp_path / "train.csv"
    eval_out = tmp_path / "eval.csv"

    src_dir = str(Path(__file__).parents[1] / "src")
    env = dict(os.environ, PYTHONPATH=src_dir)

    cmd = [
        sys.executable,
        "-m",
        "data_generator_cli",
        "--train-data",
        str(train_out),
        "--eval-data",
        str(eval_out),
        "--dataset-url",
        str(source_file),
        "--test-size",
        "0.2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    assert res.returncode == 0
    assert "Data generation complete" in res.stdout
    assert train_out.exists()
    assert eval_out.exists()
