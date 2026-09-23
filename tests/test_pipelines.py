"""Tests for KFP pipeline compilation (Container and Component routes)."""

from pathlib import Path
import yaml
from kfp import compiler
from pipeline_component import penguin_component_pipeline
from pipeline_container import penguin_container_pipeline


def test_container_pipeline_compilation(tmp_path):
    """Verify that container pipeline compiles into valid KFP v2 PipelineSpec YAML."""
    out_yaml = tmp_path / "kfp_container_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=penguin_container_pipeline,  # pyright: ignore[reportArgumentType]
        package_path=str(out_yaml),
    )

    assert out_yaml.exists()
    spec = yaml.safe_load(out_yaml.read_text())

    assert "pipelineInfo" in spec
    assert spec["pipelineInfo"]["name"] == "penguin-classification-container-pipeline"

    root_dag = spec["root"]["dag"]["tasks"]
    assert "data-generator-container-op" in root_dag
    assert "transform-container-op" in root_dag
    assert "trainer-container-op" in root_dag
    assert "evaluator-container-op" in root_dag


def test_component_pipeline_compilation(tmp_path):
    """Verify that component pipeline compiles into valid KFP v2 PipelineSpec YAML."""
    out_yaml = tmp_path / "kfp_component_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=penguin_component_pipeline,  # pyright: ignore[reportArgumentType]
        package_path=str(out_yaml),
    )

    assert out_yaml.exists()
    spec = yaml.safe_load(out_yaml.read_text())

    assert "pipelineInfo" in spec
    assert spec["pipelineInfo"]["name"] == "penguin-classification-component-pipeline"

    root_dag = spec["root"]["dag"]["tasks"]
    assert "data-generator-op" in root_dag
    assert "transform-op" in root_dag
    assert "train-model-op" in root_dag
    assert "evaluate-model-op" in root_dag
