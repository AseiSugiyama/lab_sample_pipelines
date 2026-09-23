"""Route 1: General-purpose Container & CLI Pipeline.

Executes each component container via command-line arguments using @dsl.container_component.
"""

import os
from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Model, Output

REGISTRY_BASE = os.environ.get(
    "KFP_REGISTRY_BASE",
    "asia-northeast1-docker.pkg.dev/your-sample-pipeline-project/kfp-sample",
)


@dsl.container_component
def data_generator_container_op(
    train_data: Output[Dataset],
    eval_data: Output[Dataset],
    dataset_url: str = "https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins_processed.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Run data_generator container with CLI arguments."""
    return dsl.ContainerSpec(
        image=f"{REGISTRY_BASE}/kfp-sample-data-generator:latest",
        command=["python", "-m", "data_generator_cli"],
        args=[
            "--train-data",
            train_data.path,
            "--eval-data",
            eval_data.path,
            "--dataset-url",
            dataset_url,
            "--test-size",
            test_size,
            "--random-state",
            random_state,
        ],
    )


@dsl.container_component
def transform_container_op(
    train_data: Input[Dataset],
    eval_data: Input[Dataset],
    transformed_train_data: Output[Dataset],
    transformed_eval_data: Output[Dataset],
    suffix: str = "_xf",
):
    """Run transform container with CLI arguments."""
    return dsl.ContainerSpec(
        image=f"{REGISTRY_BASE}/kfp-sample-transform:latest",
        command=["python", "-m", "transform_cli"],
        args=[
            "--train-data",
            train_data.path,
            "--eval-data",
            eval_data.path,
            "--transformed-train-data",
            transformed_train_data.path,
            "--transformed-eval-data",
            transformed_eval_data.path,
            "--suffix",
            suffix,
        ],
    )


@dsl.container_component
def trainer_container_op(
    transformed_train_data: Input[Dataset],
    trained_model: Output[Model],
    suffix: str = "_xf",
    n_estimators: int = 100,
    random_state: int = 42,
):
    """Run trainer container with CLI arguments."""
    return dsl.ContainerSpec(
        image=f"{REGISTRY_BASE}/kfp-sample-trainer:latest",
        command=["python", "-m", "trainer_cli"],
        args=[
            "--transformed-train-data",
            transformed_train_data.path,
            "--trained-model",
            trained_model.path,
            "--suffix",
            suffix,
            "--n-estimators",
            n_estimators,
            "--random-state",
            random_state,
        ],
    )


@dsl.container_component
def evaluator_container_op(
    trained_model: Input[Model],
    transformed_eval_data: Input[Dataset],
    confusion_matrix: Output[Dataset],
    metrics: Output[Dataset],
    suffix: str = "_xf",
):
    """Run evaluator container with CLI arguments."""
    return dsl.ContainerSpec(
        image=f"{REGISTRY_BASE}/kfp-sample-evaluator:latest",
        command=["python", "-m", "evaluator_cli"],
        args=[
            "--trained-model",
            trained_model.path,
            "--transformed-eval-data",
            transformed_eval_data.path,
            "--confusion-matrix-path",
            confusion_matrix.path,
            "--metrics-path",
            metrics.path,
            "--suffix",
            suffix,
        ],
    )


@dsl.pipeline(
    name="penguin-classification-container-pipeline",
    description="Palmer Penguins classification pipeline using CLI container components",
)
def penguin_container_pipeline(
    dataset_url: str = "https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins_processed.csv",
    suffix: str = "_xf",
    n_estimators: int = 100,
    random_state: int = 42,
):
    """Pipeline connecting container CLI steps."""
    gen_step = data_generator_container_op(
        dataset_url=dataset_url,
        random_state=random_state,
    )

    xf_step = transform_container_op(
        train_data=gen_step.outputs["train_data"],
        eval_data=gen_step.outputs["eval_data"],
        suffix=suffix,
    )

    train_step = trainer_container_op(
        transformed_train_data=xf_step.outputs["transformed_train_data"],
        suffix=suffix,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    evaluator_container_op(
        trained_model=train_step.outputs["trained_model"],
        transformed_eval_data=xf_step.outputs["transformed_eval_data"],
        suffix=suffix,
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=penguin_container_pipeline,
        package_path="kfp_container_pipeline.yaml",
    )
    print("Compiled container pipeline to kfp_container_pipeline.yaml")
