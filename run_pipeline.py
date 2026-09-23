#!/usr/bin/env python3
"""Sample runner script for submitting KFP pipelines to Vertex AI Pipelines."""

from __future__ import annotations

import argparse
import os
import sys
from google.cloud import aiplatform


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for pipeline submission."""
    parser = argparse.ArgumentParser(
        description="Submit compiled KFP v2 pipeline specification to Vertex AI Pipelines."
    )
    parser.add_argument(
        "--template",
        type=str,
        default="kfp_container_pipeline.yaml",
        help="Path to compiled KFP pipeline specification YAML file.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get("GCP_PROJECT_ID"),
        help="Google Cloud Project ID (defaults to GCP_PROJECT_ID env var).",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=os.environ.get("GCP_REGION", "asia-northeast1"),
        help="Google Cloud Region (defaults to GCP_REGION or asia-northeast1).",
    )
    parser.add_argument(
        "--pipeline-root",
        type=str,
        default=None,
        help="Cloud Storage pipeline root URI (e.g., gs://bucket/pipeline_root).",
    )
    parser.add_argument(
        "--display-name",
        type=str,
        default=None,
        help="Display name for the Vertex AI PipelineJob.",
    )
    parser.add_argument(
        "--enable-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable execution caching for pipeline steps.",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> None:
    """Entrypoint to submit a pipeline job to Vertex AI Pipelines."""
    args = parse_args(argv)

    if not args.project:
        print(
            "Error: GCP Project ID must be specified via --project or GCP_PROJECT_ID environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    pipeline_root = args.pipeline_root
    if not pipeline_root:
        bucket_name = os.environ.get("BUCKET_NAME")
        if bucket_name:
            clean_bucket = bucket_name.removeprefix("gs://").strip("/")
            pipeline_root = f"gs://{clean_bucket}/pipeline_root"
        else:
            print(
                "Error: Pipeline root must be specified via --pipeline-root or BUCKET_NAME environment variable.",
                file=sys.stderr,
            )
            sys.exit(1)

    template_name = os.path.splitext(os.path.basename(args.template))[0]
    display_name = args.display_name or f"{template_name}-run"

    print("================================================================================")
    print(f"Submitting Pipeline Job to Vertex AI Pipelines")
    print(f"Project:       {args.project}")
    print(f"Location:      {args.location}")
    print(f"Template:      {args.template}")
    print(f"Pipeline Root: {pipeline_root}")
    print(f"Display Name:  {display_name}")
    print(f"Caching:       {args.enable_caching}")
    print("================================================================================")

    aiplatform.init(project=args.project, location=args.location)

    job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=args.template,
        pipeline_root=pipeline_root,
        enable_caching=args.enable_caching,
    )

    job.submit()
    print("Pipeline job successfully submitted to Vertex AI Pipelines!")


if __name__ == "__main__":
    main()
