import re
from pathlib import Path

import mlflow
import pytest

from demand_forecast.utils.tracking import dataset_version, tracked_run


def test_runs_are_tagged_with_their_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    with (
        tracked_run("parent", feature_set_version=2) as parent,
        tracked_run("child", feature_set_version=2, nested=True) as child,
    ):
        pass

    client = mlflow.MlflowClient()
    for run in (parent, child):
        tags = client.get_run(run.info.run_id).data.tags
        assert re.fullmatch(r"[0-9a-f]{40}", tags["git_commit"])
        assert tags["git_dirty"] in {"true", "false"}
        assert tags["dataset_version"] == dataset_version()
        assert tags["feature_set_version"] == "2"
        assert {"python_version", "pandas_version", "mlflow_version"} <= set(tags)
    child_tags = client.get_run(child.info.run_id).data.tags
    assert child_tags["mlflow.parentRunId"] == parent.info.run_id
