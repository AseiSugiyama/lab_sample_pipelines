"""Kubeflow Pipelines (KFP) Component adapter for the Data Generator component.

This module defines the KFP Python Component (`data_generator_op`) using `@dsl.component`.
It connects the KFP first-class artifact system (`Output[Dataset]`) to the underlying
core `generate_data` domain logic and records execution metadata (such as dataset row counts)
into Vertex AI ML Metadata (MLMD).
"""

from kfp import dsl
from kfp.dsl import Dataset, Output

DEFAULT_DATA_GENERATOR_IMAGE = (
    "asia-northeast1-docker.pkg.dev/your-sample-pipeline-project/kfp-sample/kfp-sample-data-generator:latest"
)
DEFAULT_DATASET_URI = (
    "https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins_processed.csv"
)


@dsl.component(base_image=DEFAULT_DATA_GENERATOR_IMAGE)
def data_generator_op(
    train_data: Output[Dataset],
    eval_data: Output[Dataset],
    dataset_url: str = DEFAULT_DATASET_URI,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Fetch Palmer Penguins dataset, partition into train/eval sets, and log row metadata.

    When running on Vertex AI Pipelines, the orchestrator allocates a unique Cloud Storage
    staging path for each `Output[Dataset]` placeholder before container launch and passes
    it via `train_data.path` and `eval_data.path`.

    Args:
        train_data: Output artifact handle where training partition CSV will be written.
        eval_data: Output artifact handle where evaluation partition CSV will be written.
        dataset_url: Remote URL or path to download raw dataset.
        test_size: Evaluation split ratio (default: 0.2).
        random_state: Random seed for reproducible partitioning (default: 42).
    """
    # In-function import pattern: keeps core logic isolated and prevents host environment
    # dependency pollution during pipeline DAG compilation (AST serialization).
    from data_generator import generate_data

    metrics = generate_data(
        train_data_output_path=train_data.path,
        eval_data_output_path=eval_data.path,
        dataset_url=dataset_url,
        test_size=test_size,
        random_state=random_state,
    )

    # Record operational metadata into Vertex AI ML Metadata (MLMD).
    # These properties become visible and queryable in the Vertex Pipelines UI.
    train_data.metadata["rows"] = metrics["train_rows"]
    eval_data.metadata["rows"] = metrics["eval_rows"]
