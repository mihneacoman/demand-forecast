"""Baseline forecasts: every model has to beat these to be worth its complexity.

Each baseline forecasts a split's validation rows from its training days only,
indexed like those rows. A series with no history to forecast from (launched
during the validation days) is forecast as 0.
"""

import numpy as np
import pandas as pd

from demand_forecast.data.splits import HORIZON, Split, split_masks

SEASON = 7


def spread_over_rows(ids: pd.Series, values: pd.Series) -> pd.Series:
    """Repeat each series' value on its rows, 0 for series without one."""
    spread = values.reindex(ids).to_numpy(dtype="float64")
    return pd.Series(spread, index=ids.index).fillna(0.0)


def naive(table: pd.DataFrame, split: Split) -> pd.Series:
    """Forecast every validation day by the last training day."""
    train, valid = split_masks(table, split)
    # Every launched series has a row each day but Christmas, so its last
    # training day is in the last week; reading only that week saves copying.
    last_week = train & (table["d"] > split.train_end - SEASON)
    last = table.loc[last_week].groupby("id", observed=True)["sales"].last()
    return spread_over_rows(table.loc[valid, "id"], last)


def seasonal_naive(table: pd.DataFrame, split: Split) -> pd.Series:
    """Forecast each validation day by the same weekday in the last training week.

    A day missing from that week (before launch, or Christmas) is forecast as 0.
    """
    train, valid = split_masks(table, split)
    rows = table.loc[valid, ["id", "d"]]
    weeks_back = np.ceil((rows["d"] - split.train_end) / SEASON).astype("int16")
    source = rows.assign(d=(rows["d"] - SEASON * weeks_back).astype(rows["d"].dtype))
    last_week = train & (table["d"] > split.train_end - SEASON)
    history = table.loc[last_week, ["id", "d", "sales"]]
    matched = source.merge(history, on=["id", "d"], how="left", validate="many_to_one")
    sales = matched["sales"].to_numpy(dtype="float64")
    return pd.Series(sales, index=rows.index).fillna(0.0)


def moving_average(
    table: pd.DataFrame, split: Split, window: int = HORIZON
) -> pd.Series:
    """Forecast every validation day by the mean of the last `window` training days."""
    train, valid = split_masks(table, split)
    recent = train & (table["d"] > split.train_end - window)
    mean = table.loc[recent].groupby("id", observed=True)["sales"].mean()
    return spread_over_rows(table.loc[valid, "id"], mean)
