#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Build and Push KFP Component Containers to Google Cloud Artifact Registry
# Target Architecture: linux/amd64 (Intel / Vertex AI Pipelines compatible)
# ------------------------------------------------------------------------------

GCP_PROJECT_ID=${GCP_PROJECT_ID:-"your-sample-pipeline-project"}
GCP_REGION=${GCP_REGION:-"asia-northeast1"}
ARTIFACT_REGISTRY_REPO=${ARTIFACT_REGISTRY_REPO:-"kfp-sample"}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "================================================================================"
echo "Project ID:           ${GCP_PROJECT_ID}"
echo "Region:               ${GCP_REGION}"
echo "Artifact Registry:    ${ARTIFACT_REGISTRY_REPO}"
echo "Target Architecture:  linux/amd64"
echo "================================================================================"

for COMPONENT in 'data_generator' 'transform' 'trainer' 'evaluator'
do
    COMPONENT_DIR="${SCRIPT_DIR}/components/${COMPONENT}"
    PYPROJECT_TOML="${COMPONENT_DIR}/pyproject.toml"

    # Extract name and version from PEP 621 pyproject.toml
    IMAGE_NAME=$(awk -F'[ ="]+' '$1 == "name" { print $2 }' "${PYPROJECT_TOML}")
    IMAGE_VERSION=$(awk -F'[ ="]+' '$1 == "version" { print $2 }' "${PYPROJECT_TOML}")

    REPO_ENDPOINT="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}"
    IMAGE_LATEST="${REPO_ENDPOINT}/${IMAGE_NAME}:latest"
    IMAGE_VERSIONED="${REPO_ENDPOINT}/${IMAGE_NAME}:v${IMAGE_VERSION}"

    echo ""
    echo ">>> Building and pushing: ${IMAGE_NAME} (${COMPONENT})..."

    # Cross-build for linux/amd64 using Docker Buildx and push directly
    docker buildx build \
        --platform linux/amd64 \
        --no-cache \
        -t "${IMAGE_LATEST}" \
        -t "${IMAGE_VERSIONED}" \
        --push \
        "${COMPONENT_DIR}"

    echo ">>> Successfully deployed: ${IMAGE_LATEST}"
done

echo ""
echo "================================================================================"
echo "All component containers have been successfully built and pushed for linux/amd64!"
echo "================================================================================"
