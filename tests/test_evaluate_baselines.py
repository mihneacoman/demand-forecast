from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from demand_forecast.models.evaluate_baselines import BASELINES, main
from demand_forecast.utils.tracking import EXPERIMENT


def test_cli_logs_every_baseline_under_one_parent_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Enough days for three folds and the test split, 28 days each.
    n_days = 5 * 28 + 10
    rng = np.random.default_rng(0)
    table = pd.DataFrame(
        {
            "id": pd.Categorical(np.repeat(["A", "B"], n_days)),
            "d": np.tile(np.arange(1, n_days + 1, dtype="int16"), 2),
            "sales": rng.poisson(3, 2 * n_days).astype("int16"),
            "sell_price": np.float32(2.5),
        }
    )
    features = tmp_path / "features.parquet"
    table.to_parquet(features, index=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    main(["--features", str(features)])

    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], output_format="list")
    parent = next(run for run in runs if run.info.run_name == "baselines")
    children = [run for run in runs if run is not parent]
    assert sorted(run.info.run_name for run in children) == sorted(BASELINES)
    for child in children:
        assert child.data.tags["mlflow.parentRunId"] == parent.info.run_id
        assert {
            "wrmsse",
            "wrmsse_fold_std",
            "fold_3_wrmsse",
            "test_wrmsse",
            "seconds",
            "test_seconds",
        } <= set(child.data.metrics)
    artifacts = mlflow.MlflowClient().list_artifacts(parent.info.run_id)
    assert {artifact.path for artifact in artifacts} == {
        "baseline_scores.csv",
        "wrmsse_by_fold.png",
    }
