# How to deploy

## Pre-requirements

Followings are required to deploy kfp sample pipeline.

- Python ^3.8
- poetry
- docker

It is also required to make a GCP project and enable billing for your project.

## Server-side setups

### GCS Bucket

Create GCS Bucket to save kfp sample pipeline's artifacts. For more detail, check [Creating storage buckets](https://cloud.google.com/storage/docs/creating-buckets).

### Container Registry

It is required to setup some container registry to run this kfp sample pipeline. To setup registory, check [GCR documentation](https://cloud.google.com/container-registry/docs/quickstart) or [Artifact Registry documentation](https://cloud.google.com/artifact-registry/docs/quickstarts).

### Pipelines

Before deploy this pipeline to Kubeflow Pipelines or Managed Services such as Vertex AI, we need to setup the pipeline runtime environment.

To setup Vertex Pipelines, check [Google Cloud configuration guide for Vertex Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/configure-project).

To setup Kubeflow Pipelines via AI Platform Pipelines, check [Google Cloud setting up AI Platform Pipelines guide](https://cloud.google.com/ai-platform/pipelines/docs/setting-up).

To setup Kubeflow Pipelines for your Kubernetes cluster, check [Kubeflow Pipelines install guide](https://www.kubeflow.org/docs/components/pipelines/installation/).

Note: We need to make some change to IAM permission during the settings.

## Client-side setups

### Kubeflow Pipelines SDK

We need to install Kubeflow pipelines SDK to compile pipeline Python dsl into pipeline spec yaml (v1) and json (v2). To install the sdk, `poetry install` at same directory of `pipeline.py`.

### Docker Authentication

To deploy our component container image, it is required to authenticate our local docker. For more detail, check [Authentication methods](https://cloud.google.com/container-registry/docs/advanced-authentication).

### Push image to GCR

All component container image are required to push the container registry. `deploy_all_component.sh` can help to push images.

First, fill the two variables at first two lines of `deploy_all_component.sh`.

```shell
GCP_PROJECT_ID=your-sample-pipeline-project # Enter your GCP Project ID
GCP_GCR_ENDPOINT=us.gcr.io # Enter your GCR endpoint like `asia.gcr.io`
```

Then, we can deploy components as following command.

```shell
# chmod +x deploy_all_component.sh
deploy_all_component.sh
```
## Compile

### Fill the pipeline DSL

Kubeflow Pipelines SDK require to GCS bucket as `pipeline_root` parameter of `@kfp.dsl.pipeline` decorator. It is also required to GCP project ID and container registry endpoint to pull the images of components.

Enter the GCP project ID, container registry endpoint, and GCS bucket in `pipeline.py`

```python
GCP_PROJECT_ID = "your-sample-pipeline-project" # Enter your GCP project ID
GCP_GCR_ENDPOINT = "us.gcr.io"  # Enter your GCR endpoint
GCP_GCS_PIPELINE_ROOT = "your-bucket-name"  # Enter your GCS bucket without gs://
```

### Compile pipeline Python dsl

Run `dsl-compile` to compile the kfp sample pipeline.

```shell
dsl-compile --py pipeline.py --output kfp_sample_pipeline.yaml
```

The `dsl-compile` command will execute the pipeline.py to build `kfp_sample_pipeline.yaml` as v1 pipeline spec. In the same time, `kfp.v2.compiler.Compile` compiles same pipeline and generate `kfp_sample_pipeline.json` as v2 pipeline spec.
