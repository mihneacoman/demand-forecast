"""Forecast metrics: WAPE, MASE and RMSSE, scaled per series as in the M5."""

import numpy as np
import pandas as pd

from demand_forecast.data.splits import Split, split_masks


def series_scales(table: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Return each series' naive one-day-ahead error on the split's training days.

    `mase_scale` is the mean absolute error and `rmsse_scale` the mean squared
    error of forecasting each day by the day before, from the series' first sale
    on. Rows must be in series-day order; a day missing from the table (Christmas)
    is skipped, so the day before it forecasts the day after. Series with fewer
    than two days from their first sale have no scale and are left out.
    """
    train, _ = split_masks(table, split)
    history = table.loc[train, ["id", "sales"]]
    sold = (history["sales"] > 0).groupby(history["id"]).cummax()
    history = history[sold]
    errors = history.groupby("id")["sales"].diff().dropna()
    ids = history.loc[errors.index, "id"]
    return pd.DataFrame(
        {
            "mase_scale": errors.abs().groupby(ids, observed=True).mean(),
            "rmsse_scale": (errors**2).groupby(ids, observed=True).mean(),
        }
    )


def wape(forecast: pd.DataFrame) -> float:
    """Return the absolute errors summed over all rows, over the sales summed."""
    errors = forecast["prediction"] - forecast["sales"]
    return float(errors.abs().sum() / forecast["sales"].sum())


def series_errors(forecast: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    """Return each series' MASE and RMSSE over the forecast days.

    Series with no scale, or a zero one, are left out: dividing by it would make
    their errors undefined.
    """
    errors = forecast["prediction"] - forecast["sales"]
    means = pd.DataFrame({"abs": errors.abs(), "squared": errors**2})
    means = means.groupby(forecast["id"], observed=True).mean()
    scaled = means.join(scales[scales["rmsse_scale"] > 0], how="inner")
    return pd.DataFrame(
        {
            "mase": scaled["abs"] / scaled["mase_scale"],
            "rmsse": np.sqrt(scaled["squared"] / scaled["rmsse_scale"]),
        }
    )
