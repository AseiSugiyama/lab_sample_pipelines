# Trainer

The Trainer component train the machine learning model using preprocessed data. In this component, we use scikit-learn randomforest, however, any models can be used in same manner.

## Runtime Inputs and Returns

- Inputs:
  - transformed_train_data_path [str]: Path to saved train data.
  - suffix [str]: suffix to add each column name.
- Outputs:
  - trained_model_path [str]: Path to trained model.
- Metrics
  - N/A

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
poetry run python src/trainer.py \
  ./tmp/train_xf.csv \
  "_xf" \
  ./tmp/model.pkl
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
  kfp-sample-trainer \
  poetry run python src/trainer.py \
  ./tmp/train_xf.csv \
  "_xf" \
  ./tmp/model.pkl
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
