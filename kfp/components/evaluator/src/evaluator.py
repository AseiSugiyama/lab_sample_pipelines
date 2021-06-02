"""Data Generator implementation."""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional, Protocol, Tuple, get_type_hints
from matplotlib import pyplot as plt

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix

TARGET = "species"


class Scorable(Protocol):
    def score(
        self: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        ...


#
# COMPONENT ARGUMENTS
# ------------------------------------------------------------------------------


@dataclass
class ComponentArguments:
    """Argument of the component. Note: Data Generator has no inputs."""

    trained_model_path: str
    transformed_eval_data_path: str
    suffix: str


@dataclass
class OutputDestinations:
    """Outputs of the component."""

    confusion_matrix_path: str
    mlpipeline_metrics: str


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


def main(args: ComponentArguments) -> Tuple[ConfusionMatrixDisplay, float]:
    label_key = TARGET + args.suffix
    model = load_model(args.trained_model_path)

    with Path(args.transformed_eval_data_path).open() as f:
        data = load_dataset(f, label_key)

    if data.dtype.names is None:
        raise ValueError("Column names are missing")
    else:
        keys = data.dtype.names

    feature_keys = [key for key in keys if key != label_key]
    x_eval = data[feature_keys].tolist()
    y_eval = data[label_key]

    score = model.score(x_eval, y_eval)
    matrix = plot_confusion_matrix(
        model, x_eval, y_eval, cmap=plt.cm.Blues, normalize="true"
    )

    return matrix, score


#
# SUB FUNCTIONS
# ------------------------------------------------------------------------------


def load_model(model_path: str) -> Scorable:
    with Path(model_path).open("rb") as f:
        model = pickle.load(f)
    return model


def load_dataset(source: IO, label_key: str) -> np.ndarray:
    data = np.genfromtxt(source, names=True, delimiter=",")

    if label_key not in data.dtype.names:
        raise IndexError(f"{label_key} is not in column names as {data.dtype.names}")

    types = [
        (name, "<i8") if name == label_key else (name, "<f8")
        for name in data.dtype.names
    ]

    return data.astype(types)


def write_confusion_matrix(destination: str, matrix: ConfusionMatrixDisplay):
    path = Path(destination)
    path.parent.mkdir(exist_ok=True, parents=True)
    matrix.figure_.savefig(path, dpi=200, bbox_inches="tight")


def write_text(destination: str, payload: str):
    textfile = Path(destination)
    textfile.parent.mkdir(exist_ok=True, parents=True)
    textfile.write_text(payload)


#
# ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    artifacts = Artifacts.from_args()

    confusion_matrix, score = main(artifacts.component_arguments)

    # Write output.
    # When pipeline runs, runtime gives path to save dir for each outputPath
    # placeholder. For more detail, see
    # https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline#compare
    write_confusion_matrix(
        artifacts.output_destinations.confusion_matrix_path, confusion_matrix
    )

    # Write metric value.
    write_text(
        artifacts.output_destinations.mlpipeline_metrics,
        json.dumps(
            {
                "metrics": [
                    {
                        "name": "accuracy",
                        "numberValue": score,
                        "format": "RAW",  # RAW or PERCENTAGE
                    }
                ]
            }
        ),
    )
