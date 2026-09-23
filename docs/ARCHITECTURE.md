# Architecture & Technical Design Specification

This document defines the architecture, technology selections, component design patterns, and codebase specifications for the Palmer Penguins classification pipeline running on Kubeflow Pipelines (KFP SDK v2) and Google Cloud Vertex AI Pipelines.

---

## 1. Project Objectives and Dual Value Propositions

This repository serves as a production-grade reference implementation addressing common engineering challenges encountered when building machine learning pipelines with Kubeflow Pipelines SDK v2 and Vertex AI Pipelines. To accommodate different organizational maturity levels and existing asset constraints, this project simultaneously provides two distinct pipeline architectures:

### 1.1 Container Pipeline Focused on Generality and Extensibility
Provides a standardized pattern for operationalizing existing Dockerfiles, pre-built images, or third-party CLI tools into a pipeline with minimal friction. Each component provides a standalone Command-Line Interface (CLI) entrypoint invoked via container orchestration (`pipeline_container.py`). This demonstrates how to rapidly integrate and orchestrate arbitrary containerized workloads on Vertex AI Pipelines regardless of the underlying programming language or framework.

### 1.2 Native KFP Component Pipeline Leveraging Artifacts and Vertex AI Visualizations
Provides a modern Pythonic pattern that directly manipulates KFP SDK v2 first-class type abstractions (`Input[Dataset]`, `Output[Model]`, `Output[Metrics]`, `Output[ClassificationMetrics]`) and ML Metadata (MLMD). By importing component operational definitions (`*_op`) exported from component package facades (`pipeline_component.py`), developers learn how to leverage platform-native features such as end-to-end lineage tracking, dynamic metadata logging, and rich interactive confusion matrix visualizations.

### 1.3 Official Reference Documentation
- [Vertex AI Pipelines Overview](https://cloud.google.com/vertex-ai/docs/pipelines/overview)
- [Kubeflow Pipelines v2 Migration Guide](https://www.kubeflow.org/docs/components/pipelines/v2/migration/)

---

## 2. Technology Stack Selection and Rationale

This section details the primary technology stack selected to ensure long-term maintainability, rapid container builds, fast test feedback loops, and robust observability on Vertex AI Pipelines.

### 2.1 Python Version: Python 3.14

Python 3.14 is adopted as the runtime language for this project. Below are the architectural justifications based on official release lifecycles, container autonomy, and machine learning ecosystem compatibility.

#### Positioning Based on the Official Python Lifecycle
According to the official Python Developer's Guide ([Status of Python versions](https://devguide.python.org/versions/)), releases transition through active maintenance (bugfix), security-only, and end-of-life (EOL) phases.

Python 3.14 (released October 2025, EOL: October 2030) and Python 3.13 (EOL: October 2029) reside in the primary bugfix (active maintenance) phase, receiving regular bug fixes and security patches. In contrast, Python 3.12, 3.11, and 3.10 have entered the security-only phase where no general bug fixes are provided, and Python 3.9 and earlier have reached end-of-life.

To guarantee forward compatibility and multi-year maintainability for greenfield pipelines, books, and reference architectures, aligning with an actively maintained release (Python 3.14) adheres to standard software engineering best practices.

#### Container Autonomy via Custom Container Images
This pipeline is not constrained by Vertex AI's pre-built training container schedules. Each component is packaged as an independent custom container based on official Debian-based images (`python:3.14-slim-bookworm`). This decoupling allows independent runtime upgrades without waiting for cloud provider container deprecation schedules.

#### Ecosystem Compatibility Verification
All core dependencies utilized across this project (`kfp==2.17.0`, `google-cloud-aiplatform==2.1.3`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `numpy==2.5.3`, etc.) have been verified for full dependency resolution and wheel compatibility under Python 3.14 using `uv pip compile -p 3.14`.

#### Official References
- [Status of Python versions](https://devguide.python.org/versions/)
- [PEP 745 – Python 3.14 Release Schedule](https://peps.python.org/pep-0745/)

### 2.2 Package and Environment Management: uv

#### Selection Rationale
`uv` is an extremely fast Python package manager and resolver developed by Astral, implemented in Rust. It fully complies with the PEP 621 standard (`[project]` table in `pyproject.toml`) and guarantees cross-environment reproducibility via `uv.lock`. Within container builds, executing `uv pip install --system` reduces image build times from multiple minutes (typical of legacy Poetry installations) down to mere seconds.

#### Official References
- [Astral uv Documentation](https://docs.astral.sh/uv/)
- [uv Projects Guide](https://docs.astral.sh/uv/guides/projects/)
- [Using uv in Docker](https://docs.astral.sh/uv/guides/docker/)
- [PEP 621 – Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)

### 2.3 Pipeline Orchestration: Kubeflow Pipelines SDK v2 (kfp >= 2.8.0)

#### Selection Rationale
KFP SDK v2 implements a standardized Intermediate Representation (IR YAML) specification shared across open-source Kubeflow Pipelines and Google Cloud Vertex AI Pipelines. It introduces first-class artifact abstractions (`Dataset`, `Model`, `Metrics`), pipeline parameter validation, and robust ML metadata lineage tracking.

#### Official References
- [Kubeflow Pipelines v2 Documentation](https://www.kubeflow.org/docs/components/pipelines/v2/)
- [Build a pipeline using KFP SDK](https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline)

### 2.4 Container Registry: Google Cloud Artifact Registry

#### Selection Rationale
Google Container Registry (`*.gcr.io`) has been officially deprecated. Google Cloud Artifact Registry (`{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE}:{TAG}`) is the current standard enterprise container registry, providing regional caching, granular IAM access control, and automated vulnerability scanning.

#### Official References
- [Transition from Container Registry to Artifact Registry](https://cloud.google.com/artifact-registry/docs/transition/transition-from-gcr)
- [Artifact Registry Docker Quickstart](https://cloud.google.com/artifact-registry/docs/docker/quickstart)

### 2.5 Testing Framework: pytest

#### Selection Rationale
`pytest` is the de facto standard test framework in Python. Utilizing the `tmp_path` fixture enables fully isolated filesystem tests (CSVs, pickles, YAMLs) without leaving temporary artifacts. Furthermore, KFP SDK v2 allows direct in-memory execution of component functions via `component.python_func`, enabling sub-second unit and integration test runs across all components (`uv run pytest`) without requiring live cloud infrastructure.

#### Official References
- [pytest: helps you write better programs](https://docs.pytest.org/)

---

## 3. Component Architecture (Containerization, Separation of Concerns, and Builds)

At its core, a Kubeflow Pipelines component is an isolated container execution unit that reads input data from durable storage (such as Google Cloud Storage), performs deterministic computations, and persists output artifacts back to durable storage.

Accordingly, robust component design must harmonize internal Python module architecture, container build configuration, entrypoint flexibility, and KFP orchestrator contract constraints.

### 3.1 Three-File Component Architecture (Core / CLI / Component)

Following the principle of Separation of Concerns (SoC), each component (`data_generator`, `transform`, `trainer`, `evaluator`) is structured into three dedicated source files. Decoupling core algorithmic logic from CLI argument parsing and KFP orchestrator bindings maximizes local testability while preserving pipeline reusability.

```text
components/{name}/
├── Dockerfile               # Container image build definition
├── pyproject.toml           # Component-specific dependencies (managed by uv)
├── __init__.py              # Package facade: exports public {name}_op
├── src/
│   ├── {name}.py            # Pure domain logic: no KFP/CLI dependencies, top-level imports, pytest target
│   ├── {name}_cli.py        # CLI adapter: parses argparse flags and executes {name}.py (for Track 1)
│   └── {name}_component.py  # KFP adapter: binds KFP SDK Artifacts and invokes {name}.py (for Track 2)
└── tests/
    ├── test_{name}.py           # Fast pytest unit tests for domain logic
    ├── test_{name}_cli.py       # Unit tests for CLI flags, defaults, and exit codes
    └── test_{name}_component.py # Unit tests for KFP Component spec definitions and execution
```

#### Roles and Implementation Examples (Trainer Component)

Using the model training component (`trainer`) as a concrete example, the following sections demonstrate how the pure domain logic, CLI adapter, KFP component adapter, and automated tests interact.

##### 1. Pure Domain Logic (`components/trainer/src/trainer.py`)
A self-contained Python function with zero dependencies on KFP or CLI parsers. It accepts raw filesystem paths and primitive hyperparameters, returning execution summaries for metadata capture.

```python
# src/trainer.py
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(
    train_data_path: str,
    model_output_path: str,
    suffix: str = "_xf",
    n_estimators: int = 100,
    random_state: int = 42,
) -> dict:
    df = pd.read_csv(train_data_path)
    X = df.drop(columns=[f"species{suffix}"])
    y = df[f"species{suffix}"]

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X, y)

    with open(model_output_path, "wb") as f:
        pickle.dump(clf, f)

    return {
        "framework": "scikit-learn",
        "algorithm": "RandomForestClassifier",
        "n_samples": len(df),
    }
```

##### 2. CLI Adapter (`components/trainer/src/trainer_cli.py`)
The command-line entrypoint for Track 1 (Container Pipeline). It parses terminal arguments via `argparse` and delegates to `train_model`.

```python
# src/trainer_cli.py
import argparse
from trainer import train_model

def main():
    parser = argparse.ArgumentParser(description="Train penguin model via CLI")
    parser.add_argument("--transformed-train-data", type=str, required=True)
    parser.add_argument("--trained-model", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="_xf")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_model(
        train_data_path=args.transformed_train_data,
        model_output_path=args.trained_model,
        suffix=args.suffix,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

if __name__ == "__main__":
    main()
```

##### 3. KFP Component Adapter (`components/trainer/src/trainer_component.py`)
The KFP SDK v2 adapter for Track 2 (Native Component Pipeline). It accepts first-class KFP artifacts (`Input[Dataset]`, `Output[Model]`), delegates execution to `train_model`, and records lineage metadata into Vertex MLMD.

```python
# src/trainer_component.py
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model

@dsl.component(base_image="asia-northeast1-docker.pkg.dev/.../trainer:latest")
def train_model_op(
    transformed_train_data: Input[Dataset],
    trained_model: Output[Model],
    suffix: str = "_xf",
    n_estimators: int = 100,
    random_state: int = 42,
):
    from trainer import train_model

    meta = train_model(
        train_data_path=transformed_train_data.path,
        model_output_path=trained_model.path,
        suffix=suffix,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    trained_model.metadata["framework"] = meta["framework"]
    trained_model.metadata["algorithm"] = meta["algorithm"]
    trained_model.metadata["n_samples"] = meta["n_samples"]
    trained_model.metadata["n_estimators"] = n_estimators
    trained_model.metadata["random_state"] = random_state
```

##### 4. KFP Component Unit Tests (`components/trainer/tests/test_trainer_component.py`)
Validates both the interface contract (`component_spec`) and local in-memory execution (`python_func`) without calling cloud APIs.

```python
# tests/test_trainer_component.py
from pathlib import Path
from kfp.dsl import Dataset, Model
from components.trainer.src.trainer_component import train_model_op

def test_train_model_op_spec():
    """Verify component input/output interface definitions (Spec)."""
    spec = train_model_op.component_spec
    assert "transformed_train_data" in spec.inputs
    assert "trained_model" in spec.outputs
    assert spec.outputs["trained_model"].type == "system.Model"

def test_train_model_op_execution(tmp_path: Path):
    """Verify local execution of the component and metadata logging."""
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("culmen_length_mm_xf,species_xf\n1.0,0\n")
    model_pkl = tmp_path / "model.pkl"

    train_ds = Dataset(name="transformed_train_data", uri=str(train_csv))
    trained_model = Model(name="trained_model", uri=str(model_pkl))

    train_model_op.python_func(
        transformed_train_data=train_ds,
        trained_model=trained_model,
        suffix="_xf",
        n_estimators=10,
        random_state=42,
    )

    assert model_pkl.exists()
    assert trained_model.metadata["framework"] == "scikit-learn"
    assert trained_model.metadata["algorithm"] == "RandomForestClassifier"
    assert trained_model.metadata["n_samples"] == 1
```

---

### 3.2 Container Images and Docker & uv Build Design

Component container images adopt a standardized multi-stage build design optimized for rapid layer caching, minimal image size, and flexible entrypoints.

#### Standard Dockerfile Structure
The following standardized `Dockerfile` is placed at the root of each component directory. It extracts the official `uv` binary via multi-stage build and installs dependencies directly into the system environment on top of `python:3.14-slim-bookworm`:

```dockerfile
# 1. Extract uv binary
FROM ghcr.io/astral-sh/uv:latest AS uv_bin

# 2. Base runtime image
FROM python:3.14-slim-bookworm

# Copy uv binary
COPY --from=uv_bin /uv /uvx /bin/

# Copy dependencies and install into system Python (caching layer)
WORKDIR /component
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy source code
COPY src/ src/
ENV PYTHONPATH=/component/src

# Default command (CMD fallback without rigid ENTRYPOINT)
CMD ["python", "-m", "trainer_cli", "--help"]
```

#### Official References
- [Using uv in Docker](https://docs.astral.sh/uv/guides/docker/)
- [Docker Multi-stage builds](https://docs.docker.com/build/building/multi-stage/)

#### Entrypoint Design and Multi-Target Execution
By avoiding a fixed `ENTRYPOINT` in the `Dockerfile`, the orchestrator retains full authority over command execution. This single container image seamlessly supports all target modes:

- Track 1 (Container Pipeline: `pipeline_container.py`):
  Invokes CLI entrypoints explicitly via `@dsl.container_component` (`command=["python", "-m", "trainer_cli"]`, `args=[...]`).
- Track 2 (Component Pipeline: `pipeline_component.py`):
  Executes KFP SDK's dynamically injected executor wrapper (`python3 -m kfp.dsl.executor_main ...`) without collisions with a fixed entrypoint script.
- Local Interactive Debugging:
  - `docker run image`: Runs default `CMD` and outputs CLI usage help.
  - `docker run -it image bash`: Spawns an interactive bash shell for immediate container inspection.
- Automated In-Memory Testing:
  Executes `uv run pytest` across `tests/` in sub-second test runs.

---

### 3.3 Implementation Constraints and Design Decisions for KFP Component Adapters (`*_op`)

Packaging Python functions with KFP SDK v2 requires specific implementation patterns that diverge from standard Python application conventions. Below are the architectural rationales for two critical design decisions.

#### Technical Reasons for Inside-Function Imports
While standard Python style guides (PEP 8) prescribe top-level imports, KFP v2 `@dsl.component` adapters place module imports inside the function body (`from data_generator import generate_data`). This pattern is mandated by two architectural properties of KFP:

##### 1. AST Serialization and Ephemeral Execution Model
During pipeline compilation, KFP v2 extracts only the Abstract Syntax Tree (AST) and raw source code of the function decorated with `@dsl.component`, embedding it directly into the Pipeline Spec (IR YAML).
At runtime inside the container, the KFP executor (`kfp.dsl.executor_main`) writes this function snippet out as an isolated temporary file (`ephemeral_component.py`) and invokes it in a standalone Python process.
Any module-level imports or global constants defined outside the function are lost during extraction, causing runtime `NameError` exceptions. Scoping imports inside the function guarantees that all dependencies resolve reliably within the ephemeral script.

##### 2. Host Environment Path Pollution Prevention
If top-level imports (e.g., `from data_generator import generate_data`) were present in component files, importing `data_generator_op` on the developer's workstation or CI runner during pipeline definition (`pipeline_component.py`) would force the host Python interpreter to resolve component modules from the host `sys.path`, resulting in `ModuleNotFoundError`.
Encapsulating imports inside the component function maintains complete physical decoupling between the host compilation environment and the in-container runtime environment (`PYTHONPATH=/component/src`).

#### Rationale for Passing Outputs as Function Arguments (`Output[Dataset]`, etc.)
In standard Python, functions return outputs via the `return` statement. In KFP v2, output destinations are injected as function arguments (e.g., `train_data: Output[Dataset]`, `trained_model: Output[Model]`). The architectural reasons for this design include:

##### 1. Deterministic Storage Pre-allocation by the Orchestrator
Vertex AI Pipelines and Kubeflow Pipelines pre-allocate unique Cloud Storage directories (e.g., `gs://bucket/root/run_id/task_id/train_data/`) for every execution step prior to running the container.
This allocated path is injected into the container as `train_data.path`. Component logic simply writes data to this path, allowing the platform to manage storage hierarchies and access policies automatically.

##### 2. Streaming Large Datasets/Models and Avoiding Out-Of-Memory (OOM) Errors
Machine learning datasets and trained models often span hundreds of megabytes, gigabytes, or terabytes.
If components returned full datasets via `return`, the orchestrator would be forced to serialize and hold massive payloads in container process memory, triggering OOM failures.
By providing filesystem paths (`.path`), libraries like pandas, pickle, PyTorch, and TensorFlow can stream artifacts directly to local disk or cloud-mounted storage.

##### 3. Dynamic MLMD Metadata Logging During Execution
`Output[Artifact]` classes (`Dataset`, `Model`, `Metrics`, `ClassificationMetrics`) are first-class Vertex AI ML Metadata (MLMD) entities. Passing them into the function allows developers to enrich artifacts with dynamic metadata during task execution:
- `train_data.metadata["rows"] = 1000` (record row counts)
- `trained_model.metadata["framework"] = "scikit-learn"` (record ML framework)
- `metrics.log_metric("accuracy", 0.95)` (record evaluation metrics)
- `classification_metrics.log_confusion_matrix(labels, matrix)` (render interactive confusion matrices in the Vertex AI console)

##### 4. Static Lineage Resolution in the Directed Acyclic Graph (DAG)
When authoring the pipeline (`pipeline_component.py`), passing `gen_step.outputs["train_data"]` as an input to downstream tasks allows the KFP compiler to statically construct data lineage and dependency edges. The complete execution DAG and artifact flow can thus be validated prior to launching cloud jobs.

---

### 3.4 Implementation Considerations: Three Core Responsibilities of Container Processing and Design Balance

Conceptually, processing inside an ML container decomposes into three primary stages:

1. **Reading from Storage**: Opening input data (CSVs, model pickles) from Cloud Storage or local volumes and loading it into Python in-memory structures (pandas DataFrames, NumPy arrays, scikit-learn models).
2. **Processing In-Memory Data**: Executing core algorithmic logic (feature transformations, model training, evaluation metrics computation) independently of external storage systems.
3. **Writing Results to Storage**: Serializing computed outputs (transformed CSVs, model binaries, evaluation metrics JSONs) and saving them to designated destination paths.

#### Testability Benefits of Three-Tier Separation
Separating these three responsibilities into distinct functions offers clear architectural advantages. In particular, isolating in-memory data processing as pure functions allows sub-second, deterministic unit testing without requiring disk I/O, temporary directory mocking, or cloud access. It also allows storage backends to evolve without modifying core algorithmic logic.

#### Design Decision in this Implementation (Avoiding Over-Engineering)
In this reference project, each component (`data_generator.py`, `transform.py`, `trainer.py`, `evaluator.py`) encapsulates concise logic spanning 20 to 40 lines of code.
Decomposing such compact logic into separate classes or multiple layers of functions would introduce unnecessary boilerplate, argument forwarding, and cognitive overhead.
Consequently, this implementation chooses to keep data loading, processing, and saving sequentially within a single clean function per component, prioritizing developer clarity and readability while maintaining fast test execution via `tmp_path`.

---

## 4. Pipeline Architecture (Dual Orchestration Topologies)

To provide maximum flexibility for different operational needs, this repository provides two pipeline definitions: `pipeline_container.py` and `pipeline_component.py`.

### 4.1 Track 1: General Container Pipeline (`pipeline_container.py`)

Targeted at developers seeking to integrate existing Dockerfiles, legacy scripts, or third-party CLI tools with minimal changes.
Uses KFP v2 `@dsl.container_component` to invoke each component's `src/{name}_cli.py` entrypoint. Demonstrates a decoupled orchestration pattern passing parameters and data paths via command-line flags.

```python
# pipeline_container.py (Structural Example)
from kfp import dsl

@dsl.container_component
def trainer_container_op(
    transformed_train_data: dsl.Input[dsl.Dataset],
    trained_model: dsl.Output[dsl.Model],
    suffix: str = "_xf",
    n_estimators: int = 100,
    random_state: int = 42,
):
    return dsl.ContainerSpec(
        image="asia-northeast1-docker.pkg.dev/.../trainer:latest",
        command=["python", "-m", "trainer_cli"],
        args=[
            "--transformed-train-data", transformed_train_data.path,
            "--trained-model", trained_model.path,
            "--suffix", suffix,
            "--n-estimators", str(n_estimators),
            "--random-state", str(random_state),
        ],
    )
```

### 4.2 Track 2: Native KFP Artifact Pipeline (`pipeline_component.py`)

Targeted at teams looking to maximize KFP SDK v2 first-class type safety and Vertex AI Pipelines UI integrations.
Imports component operations (`*_op`) directly from package facades (`components.<name>`). Demonstrates native artifact handoffs (`Dataset`, `Model`, `Metrics`, `ClassificationMetrics`), dynamic metadata enrichment, and interactive confusion matrix rendering in the Google Cloud Console.

```python
# pipeline_component.py (Structural Example)
from kfp import dsl
from components.data_generator import data_generator_op
from components.transform import transform_op
from components.trainer import train_model_op
from components.evaluator import evaluate_model_op

@dsl.pipeline(
    name="penguin-classification-component-pipeline",
    description="Penguin classification pipeline using native KFP components and artifacts",
)
def penguin_component_pipeline(
    dataset_url: str = "...",
    n_estimators: int = 100,
    random_state: int = 42,
):
    gen_step = data_generator_op(dataset_url=dataset_url)
    xf_step = transform_op(raw_dataset=gen_step.outputs["train_data"])
    train_step = train_model_op(
        transformed_train_data=xf_step.outputs["transformed_train_data"],
        n_estimators=n_estimators,
        random_state=random_state,
    )
    eval_step = evaluate_model_op(
        trained_model=train_step.outputs["trained_model"],
        transformed_eval_data=xf_step.outputs["transformed_eval_data"],
    )
```

### 4.3 Evaluation Metrics and Confusion Matrix Visualization Design (ClassificationMetrics Constraints and Trade-offs)

The handling of evaluation metrics and confusion matrices highlights the architectural distinction between the Container Pipeline and the Component Pipeline.

#### 1. Mechanism of ClassificationMetrics in KFP v2
To render an interactive confusion matrix widget in the Vertex AI console, KFP v2 requires invoking `classification_metrics.log_confusion_matrix(categories, matrix)` on an output artifact of type `kfp.dsl.ClassificationMetrics`.
Under the hood, this method constructs a specific metadata payload conforming to Vertex AI MLMD schema specifications (`{"confusionMatrix": {"annotationSpecs": [...], "rows": [...]}}`) and records it directly into the artifact's metadata dictionary.

#### 2. Structural Constraints in CLI / Container Components
In a container-based component using `@dsl.container_component`, the following architectural constraints apply:
- **String Path Arguments**: The CLI entrypoint (`evaluator_cli.py`) receives a destination path via terminal arguments (`--confusion-matrix-path /path/to/confusion_matrix`). Because no Python `ClassificationMetrics` object handle exists inside the CLI script, helper methods like `log_confusion_matrix()` cannot be invoked.
- **Portability vs. Framework Coupling**: To generate the Vertex AI interactive widget directly from a container CLI, the CLI script would need to either import KFP SDK libraries or manually construct Vertex AI's proprietary MLMD JSON dictionary format. Doing so undermines the portability principle of keeping the CLI a pure, framework-agnostic Python tool that runs anywhere (including local terminals and plain Docker).

#### 3. Architectural Decision and Separation
Based on these trade-offs, this repository establishes a clear separation of concerns:
- **Container Pipeline (`pipeline_container.py` / CLI Mode)**: Prioritizes container and CLI portability. Outputs a standard static plot (PNG format via matplotlib) and scalar metrics JSON (`{"metrics": [...]}`), easily inspectable in local environments or image viewers.
- **Component Pipeline (`pipeline_component.py` / Native Mode)**: Prioritizes full Vertex AI UI integration. Leverages `Output[ClassificationMetrics]` and `log_confusion_matrix()` to render rich interactive confusion matrix widgets in the Google Cloud Console.

### 4.4 Pipeline Compilation and IR YAML Generation
Using `kfp.compiler.Compiler`, both Python pipeline DAG definitions compile into standardized KFP v2 Intermediate Representation (IR YAML) files.

- Compiling `pipeline_container.py` produces `kfp_container_pipeline.yaml`.
- Compiling `pipeline_component.py` produces `kfp_component_pipeline.yaml`.

Compilation runs entirely locally (`uv run python pipeline_component.py`) and validates DAG syntax and parameter compatibility without cloud connectivity.

---

## 5. Deployment and Execution Lifecycle

The deployment lifecycle comprises four clear stages: Cloud Infrastructure Provisioning, Cross-Platform Container Image Building, Pipeline Compilation, and Job Submission to Vertex AI Pipelines.

### 5.1 Cloud Infrastructure Provisioning (GCS / Artifact Registry / IAM)

Prior to deployment, the following Google Cloud resources and IAM permissions must be configured:

#### 1. Cloud Storage (GCS) Bucket (Pipeline Root)
Stores intermediate datasets, preprocessed data, trained model artifacts, and evaluation metrics:
```bash
gcloud storage buckets create gs://${BUCKET_NAME} --location=asia-northeast1
```

#### 2. Google Cloud Artifact Registry
A Docker repository storing component container images:
```bash
gcloud artifacts repositories create pipeline-components \
    --repository-format=docker \
    --location=asia-northeast1 \
    --description="Docker repository for KFP pipeline components"
```

#### 3. Vertex AI Pipelines Service Account and IAM Roles
Grants the pipeline execution service account permissions to read/write GCS artifacts and pull container images:
- `Vertex AI User` (`roles/aiplatform.user`)
- `Storage Object Admin` (`roles/storage.objectAdmin`)
- `Artifact Registry Reader` (`roles/artifactregistry.reader`)

### 5.2 Container Image Building and Registry Registration

Vertex AI Pipelines worker nodes run on Intel/AMD x86_64 (`linux/amd64`) architecture. When developing on Apple Silicon Macs (ARM64), images must be compiled for `linux/amd64` to prevent `exec format error` runtime failures.

#### Strategy 1: Local Cross-Compilation (Docker Buildx targeting linux/amd64)
Leverages Docker BuildKit / QEMU emulation to compile and push images from local workstations:
```bash
docker buildx build --platform linux/amd64 --no-cache \
    -t asia-northeast1-docker.pkg.dev/${PROJECT_ID}/pipeline-components/${COMPONENT_NAME}:latest \
    --push components/${COMPONENT_NAME}
```
The included `deploy_all_component.sh` script automates cross-compilation and pushing for all four components (`data_generator`, `transform`, `trainer`, `evaluator`).

#### Strategy 2: Cloud-Native Build (Google Cloud Build)
Delegates container compilation to native x86_64 cloud workers, eliminating local emulation overhead:
```bash
gcloud builds submit \
    --tag asia-northeast1-docker.pkg.dev/${PROJECT_ID}/pipeline-components/${COMPONENT_NAME}:latest \
    components/${COMPONENT_NAME}
```

### 5.3 Pipeline Compilation and YAML Generation

After registering container images, compile the target pipeline into its execution specification (IR YAML):

```bash
# Compile Track 1 (Container Pipeline)
uv run python pipeline_container.py

# Compile Track 2 (Component Pipeline)
uv run python pipeline_component.py
```

### 5.4 Vertex AI Pipelines Submission and Execution (PipelineJob)

Submit the compiled pipeline specification to Vertex AI Pipelines using the Google Cloud Vertex AI Python SDK (`run_pipeline.py` or inline Python script):

```python
from google.cloud import aiplatform

aiplatform.init(
    project="your-gcp-project-id",
    location="asia-northeast1",
)

job = aiplatform.PipelineJob(
    display_name="penguin-classification-run",
    template_path="kfp_component_pipeline.yaml",
    pipeline_root="gs://your-pipeline-root-bucket/pipeline_root",
    parameter_values={
        "n_estimators": 100,
        "random_state": 42,
    },
    enable_caching=True,
)

job.submit()
```

#### Execution Monitoring and Vertex AI UI Observability
Once submitted, the Vertex AI Pipelines dashboard in the Google Cloud Console provides real-time visualization of:
- Visual graph execution progress and task state transitions.
- Input and output artifact URIs and Cloud Storage bucket mappings.
- ML Metadata (MLMD) tracking execution parameters and custom metadata (`rows`, `framework`, `n_samples`).
- Interactive confusion matrix rendering for the Evaluator component.

### 5.5 Official Reference Documentation
- [Run a pipeline](https://cloud.google.com/vertex-ai/docs/pipelines/run-pipeline)
- [PipelineJob API reference](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.PipelineJob)
- [Push and pull Docker images](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling)
- [Cloud Build Overview](https://cloud.google.com/build/docs/overview)
