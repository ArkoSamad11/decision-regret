# API image: FastAPI + DuckDB, serving a pre-built artifacts/xdr.duckdb.
#
# SPEC.md §14: "The API image bakes artifacts/xdr.duckdb and the model
# weights. It is stateless and scales to zero." Build context is the REPO
# ROOT, not api/, so this Dockerfile can copy configs/ and artifacts/
# alongside api/ -- build with:
#
#   docker build -f api/Dockerfile -t xdr-api .
#
# artifacts/ is gitignored (binary model weights don't belong in git).
# Run `make reproduce` locally before building this image; there is nothing
# to bake in otherwise.

FROM python:3.12-slim

# libgomp1: lightgbm's OpenMP runtime dependency, not present in the slim
# base image by default -- without it, `import lightgbm` fails at import
# time, not at first use, so this is easy to miss until the container
# actually starts.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/pyproject.toml api/pyproject.toml
COPY api/src api/src
RUN pip install --no-cache-dir /app/api

COPY configs configs
COPY artifacts artifacts

# pip installs xdr into site-packages, so the package can no longer find the
# repo root by walking up from its own __file__ -- that walk lands in the
# interpreter's lib/ directory. Without this, /health reports "no_artifacts"
# and every route returns empty against a perfectly good baked-in database.
ENV XDR_ROOT=/app
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "xdr.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
