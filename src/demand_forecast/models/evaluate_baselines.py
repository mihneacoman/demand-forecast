"""Score the baselines on the walk-forward folds and the test split, in MLflow."""

import argparse
from collections.abc import Callable
from pathlib import Path

import mlflow
import pandas as pd
from matplotlib.figure import Figure

from demand_forecast.data.build_features import FEATURE_SET_VERSION, FEATURES_PATH
from demand_forecast.data.splits import HORIZON, N_FOLDS, Split, split_days
from demand_forecast.models.baselines import moving_average, naive, seasonal_naive
from demand_forecast.models.metrics import evaluate
from demand_forecast.utils.tracking import tracked_run

BASELINES: dict[str, Callable[[pd.DataFrame, Split], pd.Series]] = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "moving_average": moving_average,
}
# Only what the baselines and the metrics read, not the whole feature table.
COLUMNS = ["id", "d", "sales", "sell_price"]
FOLD_METRICS = ["wape", "mase", "rmsse", "wrmsse"]


def score_baselines(table: pd.DataFrame) -> pd.DataFrame:
    """Return the metrics of every baseline on every fold and on the test split."""
    folds, test = split_days(int(table["d"].max()))
    splits = {f"fold_{i}": fold for i, fold in enumerate(folds, start=1)}
    splits["test"] = test
    rows = [
        {
            "baseline": name,
            "split": split_name,
            **evaluate(table, split, forecast(table, split)),
        }
        for name, forecast in BASELINES.items()
        for split_name, split in splits.items()
    ]
    return pd.DataFrame(rows)


def run_metrics(scores: pd.DataFrame) -> dict[str, float]:
    """Flatten one baseline's scores into MLflow metrics.

    Each split's metrics are prefixed by its name; the unprefixed ones are means
    over the folds, the numbers models are compared by, with WRMSSE's spread.
    """
    metrics = {
        f"{row.split}_{name}": float(getattr(row, name))
        for row in scores.itertuples()
        for name in [*FOLD_METRICS, "series_excluded"]
    }
    folds = scores[scores["split"] != "test"]
    metrics |= {name: float(folds[name].mean()) for name in FOLD_METRICS}
    metrics["wrmsse_fold_std"] = float(folds["wrmsse"].std())
    return metrics


def wrmsse_figure(scores: pd.DataFrame) -> Figure:
    """Plot each baseline's WRMSSE across the folds."""
    folds = scores[scores["split"] != "test"]
    figure = Figure(figsize=(7, 4))
    axes = figure.subplots()
    for name, group in folds.groupby("baseline", sort=False):
        axes.plot(group["split"], group["wrmsse"], marker="o", label=name)
    axes.set(xlabel="Validation fold", ylabel="WRMSSE", title="Baselines by fold")
    axes.legend()
    figure.tight_layout()
    return figure


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Score the baselines on the walk-forward folds and the test split."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=FEATURES_PATH,
        help=f"path of the feature table Parquet file (default: {FEATURES_PATH})",
    )
    args = parser.parse_args(argv)

    table = pd.read_parquet(args.features, columns=COLUMNS)
    scores = score_baselines(table)

    with tracked_run("baselines", FEATURE_SET_VERSION):
        mlflow.log_params({"horizon": HORIZON, "n_folds": N_FOLDS})
        for name, group in scores.groupby("baseline", sort=False):
            with tracked_run(str(name), FEATURE_SET_VERSION, nested=True):
                mlflow.log_param("baseline", name)
                mlflow.log_metrics(run_metrics(group))
        mlflow.log_text(scores.to_csv(index=False), "baseline_scores.csv")
        mlflow.log_figure(wrmsse_figure(scores), "wrmsse_by_fold.png")
    print(scores.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
