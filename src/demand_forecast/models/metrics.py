"""Forecast metrics: WAPE, MASE and RMSSE, scaled per series as in the M5."""

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
