# Evaluator

The Evaluator component calculates accuracy as a score of the trained model and make a confusion matrix for visualization.

## Runtime Inputs and Returns

### CLI Interface (evaluator_cli.py / Container Pipeline)
- Inputs:
  - `--transformed-eval-data` [str]: Path to preprocessed evaluation CSV.
  - `--trained-model` [str]: Path to trained model pickle.
  - `--suffix` [str]: Column name suffix (default: `_xf`).
- Outputs:
  - `--confusion-matrix-path` [str]: Path to save static confusion matrix plot (PNG).
  - `--metrics-path` [str]: Path to save scalar metrics JSON (`{"metrics": [...]}`).

### Native KFP Component Interface (evaluator_component.py / Component Pipeline)
- Inputs:
  - `trained_model`: Input[Model]
  - `transformed_eval_data`: Input[Dataset]
  - `suffix`: str (default: `_xf`)
- Outputs:
  - `metrics`: Output[Metrics] (logs accuracy scalar to Vertex MLMD)
  - `classification_metrics`: Output[ClassificationMetrics] (renders interactive confusion matrix widget in Vertex AI console)

### Note on Confusion Matrix and ClassificationMetrics
- CLI-based container components receive simple filesystem path strings (`str`) rather than Python KFP object handles. Calling `classification_metrics.log_confusion_matrix(...)` directly from the CLI is not supported without embedding KFP-internal JSON schemas or KFP dependencies into the standalone container.
- To maintain container portability (allowing `evaluator_cli.py` to run independently in plain Docker or local environments), the CLI outputs a standard PNG image and metrics JSON.
- If rich interactive confusion matrix visualization in the Vertex AI console is required, use the native KFP component adapter (`evaluator_component.py` via `pipeline_component.py`).

## Files

```console
$ tree .
.
├── Dockerfile      -- to generate the container image of the component
├── README.md       -- this file
├── pyproject.toml  -- describing project properies and dependencies used by poetry
├── src             -- component source code
└── tests           -- test code
```

## Pre-requirements

- Python ^3.9
- poetry
- Docker

We also use GCR as a docker registry. It is required to set up GCR and it's registration. See [documentation of GCR](https://cloud.google.com/container-registry/docs/quickstart).

## Install

We can start to develop components by `poetry install`.

## Test

For testing framework, we hire [pytest](https://docs.pytest.org/en/6.2.x/). We can test this component by `poetry run pytest`.

## Run locally

```shell
poetry run python src/evaluator.py \
  ./tmp/model.pkl \
  ./tmp/eval_xf.csv \
  "_xf" \
  ./tmp/confusion_matrix.png \
  ./tmp/mlpipeline-metrics.json
```

## Build dockerfile

To build a container image with same version described in `pyproject.toml`, use following;

```shell
docker build --platform amd64 --target production -t $(awk -F'[ ="]+' '$1 == "name" { print $2 }' pyproject.toml | sed 's/_/-/g'):latest .
```

## Run docker

```shell
docker run \
  --mount type=bind,source="$(pwd)"/tmp,target=/component/tmp \
  kfp-sample-evaluator \
  poetry run python src/evaluator.py \
  ./tmp/model.pkl \
  ./tmp/eval_xf.csv \
  "_xf" \
  ./tmp/confusion_matrix.png \
  ./tmp/mlpipeline-metrics.json
```

## Deploy to GCR

First set your GCP project ID and GCR endpoint.

```shell
GCP_PROJECT_ID= # Enter your GCP Project ID
GCP_GCR_ENDPOINT= # Enter your GCR endpoint like `asia.gcr.io`
```

Then run the following commands.

```shell
IMAGE_NAME=$(awk -F'[ ="]+' '$1 == "name" { print $2 }' pyproject.toml | sed 's/_/-/g')
IMAGE_VERSION_TAG=v$(awk -F'[ ="]+' '$1 == "version" { print $2 }' pyproject.toml)
GCR_IMAGE_NAME_VERSIONED=${GCP_GCR_ENDPOINT}/${GCP_PROJECT_ID}/${IMAGE_NAME}:${IMAGE_VERSION_TAG}
docker tag ${IMAGE_NAME}:latest ${GCR_IMAGE_NAME_VERSIONED}
docker push ${GCR_IMAGE_NAME_VERSIONED}
```

If fails, check followings;

- Check your default project with `gcloud project list`
- Ensure `gcloud auth configure-docker` has been executed
