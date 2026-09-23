"""Tests for evaluator KFP component adapter."""

from pathlib import Path
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from evaluator_component import evaluate_model_op
from kfp.dsl import Dataset, Model, Metrics, ClassificationMetrics


def test_evaluator_component_spec():
    """Verify component specification, inputs, outputs, and base image."""
    spec = evaluate_model_op.component_spec
    assert "evaluate" in spec.name
    assert "trained_model" in spec.inputs
    assert "transformed_eval_data" in spec.inputs
    assert "metrics" in spec.outputs
    assert "classification_metrics" in spec.outputs
    assert "Metrics" in spec.outputs["metrics"].type
    assert "ClassificationMetrics" in spec.outputs["classification_metrics"].type
    assert "kfp-sample-evaluator" in spec.implementation.container.image


def test_evaluator_component_execution(tmp_path):
    """Execute evaluator component logic locally."""
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

    trained_model = Model(name="trained_model", uri=str(model_pkl))
    eval_ds = Dataset(name="transformed_eval_data", uri=str(eval_csv))
    metrics = Metrics(name="metrics", uri=str(tmp_path / "metrics.json"))
    classification_metrics = ClassificationMetrics(
        name="classification_metrics", uri=str(tmp_path / "cm.json")
    )

    evaluate_model_op.python_func(
        trained_model=trained_model,
        transformed_eval_data=eval_ds,
        metrics=metrics,
        classification_metrics=classification_metrics,
        suffix="_xf",
    )

    assert "accuracy" in metrics.metadata
    assert "confusionMatrix" in classification_metrics.metadata
