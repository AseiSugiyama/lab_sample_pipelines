"""Data Generator implementation."""

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Tuple, get_type_hints

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

TARGET = "species"


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


def main(args: ComponentArguments) -> BaseEstimator:
    label_key = TARGET + args.suffix
    with Path(args.transformed_train_data_path).open() as f:
        data = load_dataset(f, label_key)

    if data.dtype.names is None:
        raise ValueError("Column names are missing")
    else:
        keys = data.dtype.names

    feature_keys = [key for key in keys if key != label_key]
    x_train, y_train = data[feature_keys], data[label_key]

    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    return model


#
# SUB FUNCTIONS
# ------------------------------------------------------------------------------


def load_dataset(source: IO, label_key: str) -> np.ndarray:
    data = np.genfromtxt(source, names=True, delimiter=",")

    if TARGET not in data.dtype.names:
        raise IndexError(f"{label_key} is not in column names as {data.dtype.names}")

    types = [
        (name, "<i8") if name == label_key else (name, "<f8")
        for name in data.dtype.names
    ]

    return data.astype(types)


def write_pickle(destination: str, model: BaseEstimator):
    path = Path(destination)
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("wb+") as f:
        pickle.dump(model, f)


#
# ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    artifacts = Artifacts.from_args()

    model = main(artifacts.component_arguments)

    # Write output.
    # When pipeline runs, runtime gives path to save dir for each outputPath
    # placeholder. For more detail, see
    # https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline#compare
    write_pickle(artifacts.output_destinations.trained_model_path, model)
