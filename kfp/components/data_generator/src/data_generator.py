"""Data Generator implementation."""

import argparse
from dataclasses import dataclass
from typing import IO, Tuple, get_type_hints
from urllib import request

import numpy as np
from sklearn.model_selection import train_test_split

PENGUIN_DATASET_URI = "https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins_processed.csv"
TARGET = "species"


#
# COMPONENT ARGUMENTS
# ------------------------------------------------------------------------------


@dataclass
class ComponentArguments:
    """Argument of the component. Note: Data Generator has no inputs."""


@dataclass
class OutputDestinations:
    """Outputs of the component."""

    train_data_path: str
    eval_data_path: str


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


def fetch_dataset(source: IO) -> np.ndarray:
    data = np.genfromtxt(source, names=True, delimiter=",")

    if TARGET not in data.dtype.names:
        raise IndexError(f"{TARGET} is not in column names as {data.dtype.names}")

    types = [
        (name, "<i8") if name == TARGET else (name, "<f8") for name in data.dtype.names
    ]

    return data.astype(types)


def main(args: ComponentArguments) -> Tuple[np.ndarray, np.ndarray]:
    with request.urlopen(PENGUIN_DATASET_URI) as f:
        data = fetch_dataset(f)
    # train-test split
    train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, eval_data


#
# SUB FUNCTIONS
# ------------------------------------------------------------------------------


def write_csv(destination: str, data: np.ndarray):
    if data.dtype.names:
        header = ",".join((name for name in data.dtype.names))
    else:
        raise ValueError("Column names are missing")

    np.savetxt(destination, data, delimiter=",", header=header, comments="")


#
# ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    artifacts = Artifacts.from_args()

    train_data, eval_data = main(artifacts.component_arguments)

    # Write output.
    # When pipeline runs, runtime gives path to save dir for each outputPath
    # placeholder. For more detail, see
    # https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline#compare
    write_csv(artifacts.output_destinations.train_data_path, train_data)
    write_csv(artifacts.output_destinations.eval_data_path, eval_data)
