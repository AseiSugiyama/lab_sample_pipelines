"""Data Generator implementation."""

import argparse
from dataclasses import dataclass
from typing import IO, Tuple, get_type_hints
from pathlib import Path

import numpy as np

TARGET = "species"


#
# COMPONENT ARGUMENTS
# ------------------------------------------------------------------------------


@dataclass
class ComponentArguments:
    """Argument of the component. Note: Data Generator has no inputs."""

    train_data_path: str
    eval_data_path: str
    suffix: str


@dataclass
class OutputDestinations:
    """Outputs of the component."""

    transformed_train_data_path: str
    transformed_eval_data_path: str


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


def main(args: ComponentArguments) -> Tuple[np.ndarray, np.ndarray]:
    def download(data: str):
        with Path(data).open() as f:
            return fetch_dataset(f)

    datasets = map(download, [args.train_data_path, args.eval_data_path])

    def _transform(data):
        return transform(data, args.suffix)

    train_data, eval_data = map(_transform, datasets)

    return train_data, eval_data


#
# SUB FUNCTIONS
# ------------------------------------------------------------------------------


def transform(data: np.ndarray, suffix: str) -> np.ndarray:
    if data.dtype.names:
        names = (name + suffix for name in data.dtype.names)
        formats = (data[name].dtype for name in data.dtype.names)
    else:
        raise ValueError("Column names are missing")

    dtypes = list(zip(names, formats))

    return np.array(data, dtype=dtypes)


def fetch_dataset(source: IO) -> np.ndarray:
    data = np.genfromtxt(source, names=True, delimiter=",")

    if TARGET not in data.dtype.names:
        raise IndexError(f"{TARGET} is not in column names as {data.dtype.names}")

    types = [
        (name, "<i8") if name == TARGET else (name, "<f8") for name in data.dtype.names
    ]

    return data.astype(types)


def write_csv(destination: str, data: np.ndarray):
    if data.dtype.names:
        header = ",".join((name for name in data.dtype.names))
    else:
        raise ValueError("Column names are missing")
    path = Path(destination)
    path.parent.mkdir(exist_ok=True, parents=True)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


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
    write_csv(artifacts.output_destinations.transformed_train_data_path, train_data)
    write_csv(artifacts.output_destinations.transformed_eval_data_path, eval_data)
