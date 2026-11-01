"""Build the modeling tables from the raw M5 files."""

import numpy as np
import pandas as pd

from demand_forecast.data.load import SALES_ID_COLUMNS

CALENDAR_COLUMNS = [
    "d",
    "date",
    "wm_yr_wk",
    "event_name_1",
    "event_name_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]
PRICE_KEYS = ["store_id", "item_id", "wm_yr_wk"]


def to_long(sales: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide sales into one row per series and day."""
    day_columns = [column for column in sales.columns if column.startswith("d_")]
    days = np.array(
        [int(column.removeprefix("d_")) for column in day_columns], dtype="int16"
    )
    n_series, n_days = len(sales), len(days)
    return pd.DataFrame(
        {
            **{
                column: sales[column].repeat(n_days).array
                for column in SALES_ID_COLUMNS
            },
            "d": np.tile(days, n_series),
            "sales": sales[day_columns].to_numpy().ravel(),
        },
        copy=False,
    )


def add_calendar(long: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Add the date, Walmart week, events and SNAP flags to each sales row."""
    days = calendar[CALENDAR_COLUMNS].assign(
        d=lambda df: df["d"].astype(str).str.removeprefix("d_").astype("int16")
    )
    merged = long.merge(days, on="d", how="left", validate="many_to_one")
    assert merged["date"].notna().all(), "sales days missing from the calendar"
    return merged


def add_prices(long: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Add the weekly price to each row, left missing while off the shelf."""
    # Categoricals merged with different category sets turn into object columns.
    aligned = prices.astype({column: long[column].dtype for column in PRICE_KEYS[:2]})
    return long.merge(aligned, on=PRICE_KEYS, how="left", validate="many_to_one")


def drop_pre_launch(long: pd.DataFrame) -> pd.DataFrame:
    """Keep each series from its first priced day, which marks its launch."""
    # Rows must be in day order within a series: a row counts as launched once
    # any row up to it has a price, so later gaps in the price are kept.
    launched = long["sell_price"].notna().groupby(long["id"]).cummax()
    return long[launched].reset_index(drop=True)
