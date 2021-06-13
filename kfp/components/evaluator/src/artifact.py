import gcsfs

from kfp.dsl.artifact import Artifact
from kfp.dsl.artifact_utils import read_schema_file
from kfp.dsl.type_utils import _ARTIFACT_CLASSES_MAPPING

_TYPE_SCHEMA_MAPPER = {
    "classificationmetrics": "classification_metrics",
    "slicedclassificationmetrics": "confidence_metrics",
}
ARTIFACT_CLASSES = [
    _TYPE_SCHEMA_MAPPER[k] if k in _TYPE_SCHEMA_MAPPER else k
    for k in _ARTIFACT_CLASSES_MAPPING.keys()
]


class KfpArtifactFactory:
    @classmethod
    def create_artifact(
        cls, artifact_type: str, name: str = None, uri: str = None
    ) -> Artifact:
        if artifact_type in ARTIFACT_CLASSES:
            artifact = Artifact(
                instance_schema=read_schema_file(f"{artifact_type}.yaml")
            )
            artifact.name = name
            artifact.uri = uri
            return artifact
        else:
            raise ValueError(f"this name is not in Artifact type: {ARTIFACT_CLASSES}")


def save_artifact(uri: str, artifact: Artifact) -> None:
    fs = gcsfs.GCSFileSystem()
    with fs.open(uri, "w") as f:
        f.write(artifact.serialize())


def load_artifact(uri: str) -> Artifact:
    fs = gcsfs.GCSFileSystem()
    with fs.open(uri, "r") as f:
        return Artifact.deserialize(f.read())
