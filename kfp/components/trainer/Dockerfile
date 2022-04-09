# ------------------------------------------------------------------------------
# KUBEFLOW PIPELINE COMPONENT CONTAINER BUILD SCRIPT v1.1
# (c) Repro Inc.
# ------------------------------------------------------------------------------

#
# BUILD BASE IMAGE (USED IN BOTH OF TEST/PROD)
# ------------------------------------------------------------------------------
FROM amd64/python:3.9.12-slim-bullseye AS build_base

# install dependent packages
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    # for install
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install poetry
WORKDIR /home
ENV HOME /home
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
ENV PATH $PATH:/home/.poetry/bin
RUN poetry config virtualenvs.create false

# specify working directory and install dependency files
# without dev dependencies (they are required only on testing.)
WORKDIR /component
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-dev

# copy base source code and add path into PYTHONPATH
ENV PYTHONPATH $PYTHONPATH:/component/src

# to utilize the cache system in docker, copying srcs has been moved to 
# subsequent stages instead of do it in here.

#
# BUILD AND EXECUTE UNIT TEST
# ------------------------------------------------------------------------------
FROM build_base AS test_runner

# install full dependencies (includes dev-dependencies)
RUN poetry install

# copy test directory and run test
COPY src/ src/
COPY tests/ tests/

# run unit tests
RUN poetry run pytest

#
# BUILD PRODUCTION IMAGE
# ------------------------------------------------------------------------------
FROM build_base AS production

# apply the cache system, copy codes
COPY src/ src/
