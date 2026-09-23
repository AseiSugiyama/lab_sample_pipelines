"""Tests for run_pipeline.py runner script."""

from unittest.mock import MagicMock, patch
import pytest

from run_pipeline import main, parse_args


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test parse_args uses environment variable defaults when flags are omitted."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    monkeypatch.setenv("GCP_REGION", "asia-northeast1")
    args = parse_args([])
    assert args.template == "kfp_container_pipeline.yaml"
    assert args.project == "test-project"
    assert args.location == "asia-northeast1"
    assert args.pipeline_root is None
    assert args.enable_caching is True


def test_parse_args_explicit_flags() -> None:
    """Test parse_args accepts explicit command line flags."""
    args = parse_args([
        "--template",
        "custom_pipeline.yaml",
        "--project",
        "custom-proj",
        "--location",
        "us-central1",
        "--pipeline-root",
        "gs://my-bucket/root",
        "--no-enable-caching",
    ])
    assert args.template == "custom_pipeline.yaml"
    assert args.project == "custom-proj"
    assert args.location == "us-central1"
    assert args.pipeline_root == "gs://my-bucket/root"
    assert args.enable_caching is False


def test_main_missing_project(capsys: pytest.CaptureFixture) -> None:
    """Test main exits with error when project is not provided."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error: GCP Project ID must be specified" in captured.err


def test_main_missing_pipeline_root(capsys: pytest.CaptureFixture) -> None:
    """Test main exits with error when pipeline root is not provided."""
    with patch.dict("os.environ", {"GCP_PROJECT_ID": "test-project"}, clear=True):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Pipeline root must be specified" in captured.err


@patch("run_pipeline.aiplatform.PipelineJob")
@patch("run_pipeline.aiplatform.init")
def test_main_successful_submission(
    mock_init: MagicMock,
    mock_pipeline_job_cls: MagicMock,
) -> None:
    """Test main successfully initializes aiplatform and submits PipelineJob."""
    mock_job_instance = MagicMock()
    mock_pipeline_job_cls.return_value = mock_job_instance

    main([
        "--project",
        "test-proj",
        "--location",
        "asia-northeast1",
        "--template",
        "kfp_container_pipeline.yaml",
        "--pipeline-root",
        "gs://test-bucket/pipeline_root",
    ])

    mock_init.assert_called_once_with(project="test-proj", location="asia-northeast1")
    mock_pipeline_job_cls.assert_called_once_with(
        display_name="kfp_container_pipeline-run",
        template_path="kfp_container_pipeline.yaml",
        pipeline_root="gs://test-bucket/pipeline_root",
        enable_caching=True,
    )
    mock_job_instance.submit.assert_called_once()
