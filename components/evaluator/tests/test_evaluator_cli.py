"""Tests for evaluator CLI adapter."""

import os
from pathlib import Path
import pickle
import subprocess
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def test_evaluator_cli_execution(tmp_path):
    df = pd.DataFrame(
        {
            "species_xf": [0, 1, 2, 0, 1, 2],
            "f1_xf": [1.0, 2.0, 3.0, 1.1, 2.1, 3.1],
        }
    )
    eval_csv = tmp_path / "eval_xf.csv"
    df.to_csv(eval_csv, index=False)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(df.drop(columns=["species_xf"]), df["species_xf"])
    model_pkl = tmp_path / "model.pkl"
    with open(model_pkl, "wb") as f:
        pickle.dump(model, f)

    cm_path = tmp_path / "cm.png"
    metrics_path = tmp_path / "metrics.json"

    src_dir = str(Path(__file__).parents[1] / "src")
    env = dict(os.environ, PYTHONPATH=src_dir)

    cmd = [
        sys.executable,
        "-m",
        "evaluator_cli",
        "--trained-model",
        str(model_pkl),
        "--transformed-eval-data",
        str(eval_csv),
        "--confusion-matrix-path",
        str(cm_path),
        "--metrics-path",
        str(metrics_path),
        "--suffix",
        "_xf",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    assert res.returncode == 0
    assert "Model evaluation complete" in res.stdout
    assert cm_path.exists()
    assert metrics_path.exists()
