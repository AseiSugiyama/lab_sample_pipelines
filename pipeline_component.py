"""Route 2: KFP Native Component Pipeline.

Connects native KFP components leveraging Vertex AI lineage, metadata, and rich visualizations.
"""

from components.data_generator import data_generator_op
from components.evaluator import evaluate_model_op
from components.trainer import train_model_op
from components.transform import transform_op
from kfp import compiler, dsl


@dsl.pipeline(
    name="penguin-classification-component-pipeline",
    description="Palmer Penguins classification pipeline using native KFP v2 components and visualizations",
)
def penguin_component_pipeline(
    dataset_url: str = "https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins_processed.csv",
    suffix: str = "_xf",
    n_estimators: int = 100,
    random_state: int = 42,
):
    """Pipeline connecting native KFP v2 components."""
    gen_step = data_generator_op(
        dataset_url=dataset_url,
        random_state=random_state,
    )

    xf_step = transform_op(
        train_data=gen_step.outputs["train_data"],
        eval_data=gen_step.outputs["eval_data"],
        suffix=suffix,
    )

    train_step = train_model_op(
        transformed_train_data=xf_step.outputs["transformed_train_data"],
        suffix=suffix,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    evaluate_model_op(
        trained_model=train_step.outputs["trained_model"],
        transformed_eval_data=xf_step.outputs["transformed_eval_data"],
        suffix=suffix,
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=penguin_component_pipeline,
        package_path="kfp_component_pipeline.yaml",
    )
    print("Compiled component pipeline to kfp_component_pipeline.yaml")
