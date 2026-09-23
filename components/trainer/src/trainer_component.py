"""Kubeflow Pipelines (KFP) Component adapter for the Trainer component.

This module defines `train_model_op` using `@dsl.component`.
It connects KFP input `Input[Dataset]` and output `Output[Model]` artifacts to `train_model`,
and logs algorithm hyperparameters and ML framework metadata directly into Vertex AI MLMD.
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output

DEFAULT_TRAINER_IMAGE = (
    "asia-northeast1-docker.pkg.dev/your-sample-pipeline-project/kfp-sample/kfp-sample-trainer:latest"
)


@dsl.component(base_image=DEFAULT_TRAINER_IMAGE)
def train_model_op(
    transformed_train_data: Input[Dataset],
    trained_model: Output[Model],
    suffix: str = "_xf",
    n_estimators: int = 100,
    random_state: int = 42,
):
    """Train RandomForestClassifier on preprocessed training data and record model metadata.

    The KFP orchestrator generates a designated GCS URI for `Output[Model]`. During execution,
    the container receives a local mount path (`trained_model.path`) where the pickle is serialized,
    which is then automatically uploaded to Cloud Storage by the KFP node daemon.

    Args:
        transformed_train_data: Input preprocessed dataset artifact.
        trained_model: Output model artifact handle.
        suffix: Feature transformation suffix (default: '_xf').
        n_estimators: Number of trees in RandomForest (default: 100).
        random_state: Random seed for reproducibility (default: 42).
    """
    # In-function import pattern ensures container runtime isolation and clean AST compilation.
    from trainer import train_model

    meta = train_model(
        train_data_path=transformed_train_data.path,
        model_output_path=trained_model.path,
        suffix=suffix,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    # Populate Vertex AI ML Metadata (MLMD) properties.
    # These properties are recorded in the Vertex AI Model Registry / Metadata store.
    trained_model.metadata["framework"] = meta["framework"]
    trained_model.metadata["algorithm"] = meta["algorithm"]
    trained_model.metadata["n_samples"] = meta["n_samples"]
    trained_model.metadata["n_estimators"] = n_estimators
    trained_model.metadata["random_state"] = random_state
