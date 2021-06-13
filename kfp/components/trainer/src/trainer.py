"""Data Generator implementation."""

import argparse
import pickle
from dataclasses import dataclass
from logging import getLogger
from os.path import dirname
from typing import get_type_hints

import gcsfs
import numpy as np
import sklearn
from artifact import KfpArtifactFactory, load_artifact, save_artifact
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

TARGET = "species"
logging = getLogger(__file__)

#
# COMPONENT ARGUMENTS
# ------------------------------------------------------------------------------


@dataclass
class ComponentArguments:
    """Argument of the component. Note: Data Generator has no inputs."""

    transformed_train_data_path: str
    suffix: str


@dataclass
class OutputDestinations:
    """Outputs of the component."""

    trained_model_path: str


@dataclass
class Artifacts:
    component_arguments: ComponentArguments
    output_destinations: OutputDestinations

    @classmethod
    def arg_parser(cls) -> argparse.ArgumentParser:
        """Parse component argument and return as ComponentArguments."""
        parser = argparse.ArgumentParser()
        # generate argument parser based on ComponentArgument's definition
        for artifact in get_type_hints(cls).values():
            for arg_name, arg_type in get_type_hints(artifact).items():
                parser.add_argument(arg_name, type=arg_type)

        return parser

    @classmethod
    def from_args(cls) -> "Artifacts":
        args = vars(cls.arg_parser().parse_args())

        artifacts = {}
        for key, artifact_cls in get_type_hints(cls).items():
            existed_keys = get_type_hints(artifact_cls).keys()
            filtered_vars = {k: v for k, v in args.items() if k in existed_keys}

            artifacts[key] = artifact_cls(**filtered_vars)
        # parse args and convert into PipelineArguments
        return cls(**artifacts)


#
# MAIN FUNCTION
# ------------------------------------------------------------------------------


def main(train_data_path, suffix) -> BaseEstimator:
    label_key = TARGET + suffix
    logging.debug(f"load data from {train_data_path}")
    data = load_dataset(train_data_path, label_key)
    logging.debug("train model")
    return train(data, label_key)


#
# SUB FUNCTIONS
# ------------------------------------------------------------------------------


def train(data, label_key) -> BaseEstimator:

    if data.dtype.names is None:
        raise ValueError("Column names are missing")
    else:
        keys = data.dtype.names

    feature_keys = [key for key in keys if key != label_key]
    x_train = data[feature_keys].tolist()
    y_train = data[label_key]

    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    return model


def load_dataset(source: str, label_key: str) -> np.ndarray:
    fs = gcsfs.GCSFileSystem()
    with fs.open(source, "r") as f:
        data = np.genfromtxt(f, names=True, delimiter=",")

    if label_key not in data.dtype.names:
        raise IndexError(f"{label_key} is not in column names as {data.dtype.names}")

    types = [
        (name, "<i8") if name == label_key else (name, "<f8")
        for name in data.dtype.names
    ]

    return data.astype(types)


def write_pickle(destination: str, model: BaseEstimator):
    fs = gcsfs.GCSFileSystem()
    with fs.open(destination, "wb") as f:
        pickle.dump(model, f)


#
# ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    artifacts = Artifacts.from_args()

    logging.debug("Create Model Artifact")
    model_artifact = KfpArtifactFactory.create_artifact(
        "model",
        "Sklearn-RandomForest",
        f"{dirname(artifacts.output_destinations.trained_model_path)}/model.pkl",
    )
    model_artifact.framework = "sklearn"
    model_artifact.framework_version = f"{sklearn.__version__}"

    print(
        f"Load dataset artifact of train dataset from transform component.\nuri is here: {artifacts.component_arguments.transformed_train_data_path}"
    )
    train_ds_artifact = load_artifact(
        artifacts.component_arguments.transformed_train_data_path
    )
    print(f"URI:{train_ds_artifact.uri}")
    model = main(train_ds_artifact.uri, artifacts.component_arguments.suffix)

    # Write output.
    # When pipeline runs, runtime gives path to save dir for each outputPath
    # placeholder. For more detail, see
    # https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline#compare
    logging.debug(f"save model to {model_artifact.uri}")
    write_pickle(model_artifact.uri, model)
    logging.debug(
        "save model artifact to {artifacts.output_destinations.trained_model_path}"
    )
    save_artifact(artifacts.output_destinations.trained_model_path, model_artifact)
