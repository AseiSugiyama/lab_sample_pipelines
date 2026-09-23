FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM python:3.14-slim-bookworm

# Copy uv binary
COPY --from=uv_bin /uv /uvx /bin/

# Install dependencies
WORKDIR /component
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application source
COPY src/ src/
ENV PYTHONPATH=/component/src

# Default fallback command
CMD ["python", "-m", "transform_cli", "--help"]
