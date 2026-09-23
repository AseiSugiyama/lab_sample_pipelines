# How to deploy

This document explains how to build, deploy, and run the Penguin Classification Pipeline on Google Cloud Vertex Pipelines and Kubeflow Pipelines using `uv` and KFP SDK v2.

## Pre-requirements

Followings are required:

- Python >= 3.14
- uv (Fast Python package installer and resolver)
- Docker
- Google Cloud CLI (`gcloud`)
- A Google Cloud Project with billing enabled

## Server-side setups

### 1. Cloud Storage (GCS) Bucket

Create a GCS Bucket to store pipeline execution artifacts:

```shell
gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=asia-northeast1
```

### 2. Artifact Registry

Google Container Registry (`*.gcr.io`) is deprecated. Create a Docker repository in Google Cloud Artifact Registry:

```shell
gcloud artifacts repositories create pipeline-components \
    --repository-format=docker \
    --location=asia-northeast1 \
    --description="Docker repository for KFP pipeline components"
```

### 3. Vertex AI Pipelines / IAM Permissions

Ensure the following APIs are enabled:

```shell
gcloud services enable \
    compute.googleapis.com \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com \
    storage.googleapis.com
```

Grant the default Compute Engine service account (or your dedicated Vertex Pipelines runner service account) the following roles:
- `Vertex AI User` (`roles/aiplatform.user`)
- `Storage Object Admin` (`roles/storage.objectAdmin`)
- `Artifact Registry Reader` (`roles/artifactregistry.reader`)

## Client-side setups

### 1. Install Dependencies with uv

Install the required environment locally using `uv`:

```shell
uv sync
```

### 2. Docker Authentication

Authenticate Docker to push to your Artifact Registry:

```shell
gcloud auth configure-docker asia-northeast1-docker.pkg.dev
```

### 3. Build & Push Component Images (linux/amd64 Target)

Vertex AI Pipelines runs on Intel/AMD x86_64 (`linux/amd64`). If building on Apple Silicon (M-series Mac), images must be cross-compiled for `--platform linux/amd64` to prevent `exec format error` at runtime.

Before running the deployment script, ensure you have an active Docker Buildx builder:

```shell
docker buildx create --name mybuilder --use
```

The script `deploy_all_component.sh` uses three configuration variables: `GCP_PROJECT_ID`, `GCP_REGION`, and `ARTIFACT_REGISTRY_REPO`. You can configure them either by exporting environment variables in your shell or by editing the script directly:

**Method A: Export environment variables in your terminal (Recommended)**

```shell
export GCP_PROJECT_ID=$(gcloud config get-value project)
export GCP_REGION="asia-northeast1"
export ARTIFACT_REGISTRY_REPO="kfp-sample"

./deploy_all_component.sh
```

**Method B: Edit the script directly**

Open `deploy_all_component.sh` and set your values at the top of the file:

```shell
GCP_PROJECT_ID=${GCP_PROJECT_ID:-"your-actual-project-id"}
GCP_REGION=${GCP_REGION:-"asia-northeast1"}
ARTIFACT_REGISTRY_REPO=${ARTIFACT_REGISTRY_REPO:-"kfp-sample"}
```

Then run:

```shell
./deploy_all_component.sh
```

*(Optional: Google Cloud Build)*
Alternatively, you can build remotely on Google Cloud native x86_64 workers without local Docker emulation:

```shell
gcloud builds submit --tag ${IMAGE_URI} components/{component_name}
```

## Compile Pipelines

This repository provides two pipeline definitions (`pipeline_container.py` and `pipeline_component.py`). Both pipelines execute the exact same machine learning workflow (Palmer Penguins data retrieval, feature suffix transformation, Random Forest classifier training, and performance evaluation). However, they differ in orchestration style and integration level:

- Container Pipeline (`pipeline_container.py`):
  Uses `@dsl.container_component` to invoke each component's CLI entrypoint (`{name}_cli.py`) with explicit command-line arguments. This route has high affinity with standard containerized tools and workflows.
- Native KFP Component Pipeline (`pipeline_component.py`):
  Uses native `@dsl.component` decorators and first-class KFP Artifacts (`Dataset`, `Model`, `Metrics`, `ClassificationMetrics`). This route enables seamless metadata tracking in Vertex MLMD and rich interactive visualizations (such as confusion matrix diagrams) directly within the Vertex AI console.

Compile either or both pipelines into the KFP v2 Pipeline Spec (IR YAML) as needed:

### 1. Compile Container Pipeline

Set `KFP_REGISTRY_BASE` to match your Artifact Registry repository, then compile:

```shell
export KFP_REGISTRY_BASE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}"
uv run python pipeline_container.py
# -> Output: kfp_container_pipeline.yaml
```

### 2. Compile Native KFP Component Pipeline

Compile the native KFP component pipeline into `kfp_component_pipeline.yaml`:

```shell
uv run python pipeline_component.py
# -> Output: kfp_component_pipeline.yaml
```

## Run on Vertex AI Pipelines

### Option A: Using Google Cloud CLI

```shell
gcloud ai pipelines run \
    --display-name="penguin-classification-run" \
    --template-path="kfp_component_pipeline.yaml" \
    --location="asia-northeast1" \
    --pipeline-root="gs://YOUR_BUCKET_NAME/pipeline_root" \
    --parameters="suffix=_xf"
```

### Option B: Using Python SDK

```python
from google.cloud import aiplatform

aiplatform.init(project="your-project-id", location="asia-northeast1")

job = aiplatform.PipelineJob(
    display_name="penguin-classification-run",
    template_path="kfp_component_pipeline.yaml",
    pipeline_root="gs://YOUR_BUCKET_NAME/pipeline_root",
    parameter_values={
        "suffix": "_xf",
        "n_estimators": 100,
        "random_state": 42,
    },
    enable_caching=True,
)

job.submit()
```
