"""Data Generator implementation."""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Tuple, get_type_hints
from pathlib import Path

import gcsfs
import numpy as np
from artifact import KfpArtifactFactory, load_artifact, save_artifact
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
from kfp.dsl.metrics_utils import ConfusionMatrix

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
    metadata: str


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


def main(ds_uri, model_uri, suffix) -> Tuple[ConfusionMatrixDisplay, float]:
    label_key = TARGET + suffix
    model = load_model(model_uri)

    data = load_dataset(ds_uri, label_key)

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
    fs = gcsfs.GCSFileSystem()

    with fs.open(model_path) as f:
        model = pickle.load(f)
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
    metadata = json.loads(artifacts.output_destinations.metadata)
    print(f"metadata: {metadata}")
    model_artifact = load_artifact(artifacts.component_arguments.trained_model_path)
    eval_dataset_artifact = load_artifact(
        artifacts.component_arguments.transformed_eval_data_path
    )
    confusion_matrix, score = main(
        eval_dataset_artifact.uri,
        model_artifact.uri,
        artifacts.component_arguments.suffix,
    )

    artifact_schema = metadata["outputs"]["artifacts"]["confusion_matrix_path"]["artifacts"][0]
    cm_artifact = KfpArtifactFactory.create_artifact(
        "classification_metrics", artifact_schema["name"], artifact_schema["uri"]
    )
    cm = ConfusionMatrix()
    cm.load_matrix(
        [str(s) for s in confusion_matrix.display_labels.tolist()],
        [r for r in confusion_matrix.confusion_matrix.tolist()]
    )
    cm_artifact.confusionMatrix = cm.get_metrics()

    artifact_schema = metadata["outputs"]["artifacts"]["mlpipeline_metrics"]["artifacts"][0]
    score_artifact = KfpArtifactFactory.create_artifact(
        "metrics", artifact_schema["name"], artifact_schema["uri"]
    )

    score_artifact.accuracy = score
    save_artifact(artifacts.output_destinations.confusion_matrix_path, cm_artifact)
    save_artifact(artifacts.output_destinations.mlpipeline_metrics, score_artifact)

    output_metadata = {}
    output_metadata["artifacts"] = {}
    for name, artifact in zip(["confusion_matrix_path", "mlpipeline_metrics"], [cm_artifact, score_artifact]):
        runtime_artifact = {
            "name": artifact.runtime_artifact.name,
            "uri": artifact.uri,
            "metadata": artifact.metadata
        }
        output_metadata["artifacts"][name] = {"artifacts": [runtime_artifact]}

    print(f"作成されたoutput_metadata:{output_metadata}")
    path = Path(metadata["outputs"]["outputFile"])
    print(f"Directoryを作成:{str(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Directory作成 Done")

    print("Metadataの書き込み")
    with open(path, "w") as f:
        json.dump(output_metadata, f)
    print("Metadataの書き込みDone")
    # print("Metadataの書き込み Done")
    # # Write output.
    # When pipeline runs, runtime gives path to save dir for each outputPath
    # placeholder. For more detail, see
    # https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline#compare
    # write_confusion_matrix(
    #     artifacts.output_destinations.confusion_matrix_path, confusion_matrix
    # )

    # # Write metric value.
    # write_text(
    #     artifacts.output_destinations.mlpipeline_metrics,
    #     json.dumps(
    #         {
    #             "metrics": [
    #                 {
    #                     "name": "accuracy",
    #                     "numberValue": score,
    #                     "format": "RAW",  # RAW or PERCENTAGE
    #                 }
    #             ]
    #         }
    #     ),
    # )
