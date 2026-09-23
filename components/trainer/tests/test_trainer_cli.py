"""Tests for trainer CLI adapter."""

import os
from pathlib import Path
import subprocess
import sys
import pandas as pd


def test_trainer_cli_execution(tmp_path):
    train_in = tmp_path / "train_xf.csv"
    model_out = tmp_path / "model.pkl"

    df = pd.DataFrame(
        {
            "species_xf": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            "f1_xf": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    df.to_csv(train_in, index=False)

    src_dir = str(Path(__file__).parents[1] / "src")
    env = dict(os.environ, PYTHONPATH=src_dir)

    cmd = [
        sys.executable,
        "-m",
        "trainer_cli",
        "--transformed-train-data",
        str(train_in),
        "--trained-model",
        str(model_out),
        "--suffix",
        "_xf",
        "--n-estimators",
        "10",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    assert res.returncode == 0
    assert "Model training complete" in res.stdout
    assert model_out.exists()
