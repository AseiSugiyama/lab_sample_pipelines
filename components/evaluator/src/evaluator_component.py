"""Kubeflow Pipelines (KFP) Component adapter for the Evaluator component.

This module defines `evaluate_model_op` using `@dsl.component`.
It connects input artifacts (`Input[Model]`, `Input[Dataset]`) to `evaluate_model`
and records scalar performance metrics via `Output[Metrics]` and interactive
confusion matrix diagrams via `Output[ClassificationMetrics]`, which render
natively in the Vertex AI Pipelines web UI.
"""

import os
from kfp import dsl
from kfp.dsl import ClassificationMetrics, Dataset, Input, Metrics, Model, Output

REGISTRY_BASE = os.environ.get(
    "KFP_REGISTRY_BASE",
    "asia-northeast1-docker.pkg.dev/your-sample-pipeline-project/kfp-sample",
)
DEFAULT_EVALUATOR_IMAGE = f"{REGISTRY_BASE}/kfp-sample-evaluator:latest"



@dsl.component(base_image=DEFAULT_EVALUATOR_IMAGE)
def evaluate_model_op(
    trained_model: Input[Model],
    transformed_eval_data: Input[Dataset],
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    suffix: str = "_xf",
):
    """Evaluate trained model and log metrics & interactive confusion matrix.

    This component logs evaluation outcomes directly into Vertex AI Pipelines:
    1. `metrics.log_metric("accuracy", ...)` stores the scalar accuracy in Vertex MLMD,
       allowing pipeline runs to be compared side-by-side in the Vertex AI console.
    2. `classification_metrics.log_confusion_matrix(...)` renders an interactive
       confusion matrix widget directly within the Vertex AI Pipelines execution graph UI.

    Args:
        trained_model: Input trained model artifact handle.
        transformed_eval_data: Input preprocessed evaluation dataset artifact handle.
        metrics: Output scalar metrics artifact handle for MLMD tracking.
        classification_metrics: Output visualization artifact handle for interactive UI.
        suffix: Feature transformation suffix (default: '_xf').
    """
    # In-function import pattern ensures container runtime isolation and clean AST compilation.
    from evaluator import evaluate_model

    results = evaluate_model(
        trained_model_path=trained_model.path,
        transformed_eval_data_path=transformed_eval_data.path,
        suffix=suffix,
    )

    # 1. Log scalar accuracy metric to Vertex ML Metadata
    metrics.log_metric("accuracy", results["accuracy"])

    # 2. Log interactive confusion matrix to Vertex AI visualization tab
    classification_metrics.log_confusion_matrix(
        results["categories"],
        results["confusion_matrix"],
    )
