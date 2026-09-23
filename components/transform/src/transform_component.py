"""Kubeflow Pipelines (KFP) Component adapter for the Transform component.

This module defines `transform_op` using `@dsl.component`.
It accepts KFP first-class `Input[Dataset]` artifacts from upstream data generation,
executes column suffix transformations via `transform_data`, writes the results
to `Output[Dataset]`, and records transformation metadata into Vertex AI MLMD.
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Output

DEFAULT_TRANSFORM_IMAGE = (
    "asia-northeast1-docker.pkg.dev/your-sample-pipeline-project/kfp-sample/kfp-sample-transform:latest"
)


@dsl.component(base_image=DEFAULT_TRANSFORM_IMAGE)
def transform_op(
    train_data: Input[Dataset],
    eval_data: Input[Dataset],
    transformed_train_data: Output[Dataset],
    transformed_eval_data: Output[Dataset],
    suffix: str = "_xf",
):
    """Transform dataset column names with a specified suffix and log metadata.

    Args:
        train_data: Input training dataset artifact produced by upstream component.
        eval_data: Input evaluation dataset artifact produced by upstream component.
        transformed_train_data: Output artifact handle for preprocessed training dataset.
        transformed_eval_data: Output artifact handle for preprocessed evaluation dataset.
        suffix: Suffix string appended to all column headers (default: '_xf').
    """
    # In-function import pattern ensures component self-containment in isolated container environments.
    from transform import transform_data

    metrics = transform_data(
        train_data_path=train_data.path,
        eval_data_path=eval_data.path,
        transformed_train_output_path=transformed_train_data.path,
        transformed_eval_output_path=transformed_eval_data.path,
        suffix=suffix,
    )

    # Attach operational metadata to the output artifacts for lineage tracking in Vertex AI
    transformed_train_data.metadata["rows"] = metrics["train_rows"]
    transformed_train_data.metadata["suffix"] = suffix
    transformed_eval_data.metadata["rows"] = metrics["eval_rows"]
    transformed_eval_data.metadata["suffix"] = suffix
