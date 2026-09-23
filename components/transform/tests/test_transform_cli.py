"""Tests for transform CLI adapter."""

import os
from pathlib import Path
import subprocess
import sys
import pandas as pd


def test_transform_cli_execution(tmp_path):
    train_in = tmp_path / "train.csv"
    eval_in = tmp_path / "eval.csv"
    train_out = tmp_path / "train_xf.csv"
    eval_out = tmp_path / "eval_xf.csv"

    pd.DataFrame({"species": [0, 1], "feat": [1.0, 2.0]}).to_csv(train_in, index=False)
    pd.DataFrame({"species": [1, 2], "feat": [3.0, 4.0]}).to_csv(eval_in, index=False)

    src_dir = str(Path(__file__).parents[1] / "src")
    env = dict(os.environ, PYTHONPATH=src_dir)

    cmd = [
        sys.executable,
        "-m",
        "transform_cli",
        "--train-data",
        str(train_in),
        "--eval-data",
        str(eval_in),
        "--transformed-train-data",
        str(train_out),
        "--transformed-eval-data",
        str(eval_out),
        "--suffix",
        "_xf",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    assert res.returncode == 0
    assert "Data transformation complete" in res.stdout
    assert train_out.exists()
    assert eval_out.exists()

    df = pd.read_csv(train_out)
    assert "species_xf" in df.columns
    assert "feat_xf" in df.columns
