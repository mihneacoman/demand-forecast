"""MLflow runs tagged with what produced them: code, data, features, environment.

The tracking server comes from MLFLOW_TRACKING_URI, which MLflow reads itself:
run with `uv run --env-file .env`.
"""

import hashlib
import os
import platform
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager

import mlflow
import numpy as np
import pandas as pd

from demand_forecast.data.download import EXPECTED_SHA256

# One experiment for every model, so all of them compare in one place.
EXPERIMENT = "m5-forecasting"


def git_output(*args: str) -> str:
    """Return the output of a git command, stripped."""
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def dataset_version() -> str:
    """Return one hash for the raw files, from the SHA256 they are verified by."""
    files = "\n".join(
        f"{name}:{digest}" for name, digest in sorted(EXPECTED_SHA256.items())
    )
    return hashlib.sha256(files.encode()).hexdigest()


def run_context(feature_set_version: int) -> dict[str, str]:
    """Return the tags that tie a run to its code, data, features and environment."""
    return {
        "git_commit": git_output("rev-parse", "HEAD"),
        # Uncommitted changes mean the commit alone does not reproduce the run.
        "git_dirty": str(bool(git_output("status", "--porcelain"))).lower(),
        "dataset_version": dataset_version(),
        "feature_set_version": str(feature_set_version),
        # Run times only compare on the same machine.
        "platform": platform.platform(),
        "cpu_count": str(os.cpu_count()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "mlflow_version": mlflow.__version__,
    }


@contextmanager
def tracked_run(
    run_name: str, feature_set_version: int, nested: bool = False
) -> Iterator[mlflow.ActiveRun]:
    """Start an MLflow run in the shared experiment, tagged with its run context."""
    if not nested:
        mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.set_tags(run_context(feature_set_version))
        yield run
