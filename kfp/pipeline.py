""""KFP penguin classification sample pipeline"""

import os
from pathlib import Path
from typing import Union
from string import Template
from enum import Enum

import kfp
from kfp.v2 import compiler

#
# CONSTANTS
# ------------------------------------------------------------------------------

PIPELINE_NAME = "kfp-sample-pipeline"
COMPONENT_PREFIX = "kfp-sample"
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCP_GCR_ENDPOINT = os.getenv("GCP_GCR_ENDPOINT", "")
GCP_GCS_PIPELINE_ROOT = os.getenv("GCP_GCS_PIPELINE_ROOT")


class GeneratedData(Enum):
    TrainData = "train_data_path"
    EvalData = "eval_data_path"
    TransformedTrainData = "transformed_train_data_path"
    TransformedEvalData = "transformed_eval_data_path"
    TrainedModel = "trained_model_path"


#
# SUB FUNCTIONS
# ------------------------------------------------------------------------------


def get_version_from_toml(toml_path: str) -> Union[str, None]:
    path = Path(toml_path)
    lines = path.read_text().split("\n")
    for line in lines:
        if line.startswith("version = "):
            _, right = line.split("=")
            return right.replace('"', "").strip()
    return "latest"


def get_component_spec(name: str) -> str:
    base_dir = f"components/{name.replace('-', '_')}"
    version = get_version_from_toml(f"{base_dir}/pyproject.toml")
    tag = f"v{version}"
    image = f"{GCP_GCR_ENDPOINT}/{GCP_PROJECT_ID}/{COMPONENT_PREFIX}-{name}:{tag}"
    path = Path(f"{base_dir}/src/{name.replace('-', '_')}.yaml")
    template = Template(path.read_text())
    return template.substitute(tagged_name=image)


#
# COMPONENTS
# ------------------------------------------------------------------------------


def _data_generator_op() -> kfp.dsl.ContainerOp:
    name = "data-generator"
    component_spec = get_component_spec(name)
    data_generator_op = kfp.components.load_component_from_text(component_spec)
    return data_generator_op()


def _transform_op(
    train_data_path: str, eval_data_path: str, suffix: str
) -> kfp.dsl.ContainerOp:
    name = "transform"
    component_spec = get_component_spec(name)
    data_generator_op = kfp.components.load_component_from_text(component_spec)
    return data_generator_op(
        train_data_path=train_data_path, eval_data_path=eval_data_path, suffix=suffix
    )


def _trainer_op(transformed_train_data_path: str, suffix: str) -> kfp.dsl.ContainerOp:
    name = "trainer"
    component_spec = get_component_spec(name)
    trainer_op = kfp.components.load_component_from_text(component_spec)
    return trainer_op(
        transformed_train_data_path=transformed_train_data_path, suffix=suffix
    )


def _evaluator_op(
    trained_model_path: str, transformed_eval_data_path: str, suffix: str
) -> kfp.dsl.ContainerOp:
    name = "evaluator"
    component_spec = get_component_spec(name)
    evaluator_op = kfp.components.load_component_from_text(component_spec)
    return evaluator_op(
        trained_model_path=trained_model_path,
        transformed_eval_data_path=transformed_eval_data_path,
        suffix=suffix,
    )


#
# PIPELINE
# ------------------------------------------------------------------------------


@kfp.dsl.pipeline(
    name=PIPELINE_NAME,
    pipeline_root=f"gs://{GCP_GCS_PIPELINE_ROOT}/",
)
def kfp_sample_pipeline(suffix: str):
    data_generator = _data_generator_op()
    transform = _transform_op(
        train_data_path=data_generator.outputs[GeneratedData.TrainData.value],
        eval_data_path=data_generator.outputs[GeneratedData.EvalData.value],
        suffix=suffix,
    )
    trainer = _trainer_op(
        transformed_train_data_path=transform.outputs[
            GeneratedData.TransformedTrainData.value
        ],
        suffix=suffix,
    )
    _ = _evaluator_op(
        trained_model_path=trainer.outputs[GeneratedData.TrainedModel.value],
        transformed_eval_data_path=transform.outputs[
            GeneratedData.TransformedEvalData.value
        ],
        suffix=suffix,
    )


# Compile the pipeline with V2 SDK to test the compatibility between V1 and V2 SDK
compiler.Compiler().compile(
    pipeline_func=kfp_sample_pipeline,
    package_path="kfp_sample_pipeline.json",
)
