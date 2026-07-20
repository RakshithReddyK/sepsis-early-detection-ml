# Reproducible training environment for the sepsis-ml package.
#
# Build:
#   docker build -t sepsis-ml .
# Run training (mounts the repo's data/ and writes models/ + reports/ back out):
#   docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" \
#       -v "$(pwd)/reports:/app/reports" sepsis-ml
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY src ./src

RUN pip install --upgrade pip && pip install .

COPY data ./data
COPY tests ./tests

# Default: run training against the committed balanced_sepsis.csv.
ENTRYPOINT ["sepsis-ml", "train"]
